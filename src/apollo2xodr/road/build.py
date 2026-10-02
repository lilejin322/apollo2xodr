"""
Group lanes into OpenDRIVE roads and compute each road's reference line, lane widths and elevation.

Plain roads hold lanes outside junctions that run side by side and share their boundaries. Everything OpenDRIVE can
only express inside a junction is connecting material: Apollo junction lanes, and roads that split or merge outside
Apollo junctions. Connecting material that touches forms one junction region. A path road follows one path through a
region, from a lane of a plain road that enters it to a lane of a plain road that leaves it, with one lane section per
Apollo lane on the way; lanes where paths fork or merge are written once per path.

A road ends square to its reference line, so where Apollo lane ends are staggered its lanes cannot all end where
Apollo's do. Plain roads therefore stop where all their lanes still exist, unless less than half the road would remain
(see _trim), and whatever continues from a road is attached to its written lane ends. A plain road that continues one
already built moves its boundaries there, its whole cross-section starting (ending) on the neighbour's square end. A
path road starts on the written centre of its incoming lane (and ends on that of its outgoing lane) with the
neighbour's reference-line heading, lane width and height there; without an incoming (outgoing) lane it keeps Apollo's
end. Path roads follow the Apollo centre line and Apollo's width samples; a lane offset of half the width keeps their
lane centred on the reference line, except on a wide, tight turn where fit_curved_path_borders fits the lane offset
towards Apollo's drawn borders instead. Where a lane border would fold on a tight turn, settle_surface may smooth the
reference line and, on a path road, narrow the lane.
"""

from collections import Counter
import numpy as np
import shapely
from shapely.geometry import LineString
from typing import Dict, List, Optional, Set
from .. import geometry
from ..cycles import split_cycles, split_connecting_entries
from ..geometry.hermite import HermiteCurve, blend_heading, blend_length
from ..geometry.polyline import move_endpoint, slice_polyline
from ..model import BoundaryRef, Lane, MapData
from ..reference_line import ReferenceLine
from .model import Road
from .planning import (parting_pairs, plan_roads, connecting_plans, junction_regions, connecting_paths,
                            lane_order, lane_groups)
from .surface import plain_widths, path_widths, settle_surface, fit_curved_path_borders, fit_elevation
from ..sketch import smooth_sketched_lanes

__all__ = ['Road', 'build_roads', 'lane_order', 'lane_groups']

######################################## Parameters #######################################

SHORT_ROAD = 8.0
"""m, a road up to this long held at both ends may become one curve between its ends"""
CONTINUATION = 10.0
"""m of the lanes beyond a free road end fitted with the road to find its heading there"""
CONTINUATION_GAP = 2.0
"""m, a lane beyond continues a boundary only if it starts this close to its end"""

###########################################################################################

def build_roads(data: MapData, tolerance: float = geometry.FIT_TOLERANCE) -> List[Road]:
    """
    Group the map's lanes into roads and build each road's reference line, lane widths and elevation.

    ``data`` is changed in place: a hand-drawn map has its lanes smoothed, cycles in connecting material are split
    (``data.lanes`` then holds the pieces, with links and signal overlaps rewritten). A branching junction at the map
    edge also gets ordinary entry segments cut from its starting lanes. In addition, the boundary ends of a plain road
    that continues one already built are moved, height included, onto its written lane ends.

    :param MapData data: the map as read, mutated as described
    :param float tolerance: the most, in m, a source vertex may lie from a reference line as first fitted; attached
                            ends, smoothing and surface repair can move the final line farther. Also the least
                            smoothing allowance for a hand-drawn map, though never more than SKETCH_MAX (0.5 m)
    :returns: the roads of side-by-side lanes first, then the path roads; each road's id is its index
    :rtype: List[Road]
    """
    # Neighbours that part are judged on Apollo's own centre lines, before smoothing redraws them.
    parting = parting_pairs(data)
    smooth_sketched_lanes(data, tolerance)
    plans = plan_roads(data, parting)
    plan_of = {lane: plan for plan in plans for lane in plan.lanes}
    connecting = connecting_plans(plans, plan_of)
    # Paths cannot be followed around a cycle. Splitting one replaces its lanes with head, core and tail pieces, and
    # the cores are ordinary lanes outside any junction, so plans and connecting material are worked out again.
    if split_cycles(data, {lane for plan in connecting for lane in plan.lanes}):
        plans = plan_roads(data, parting)
        plan_of = {lane: plan for plan in plans for lane in plan.lanes}
        connecting = connecting_plans(plans, plan_of)
    region = junction_regions(plans, plan_of, connecting)  # connecting plan -> name of its junction region
    paths = connecting_paths(plans, connecting)
    entries = _entry_lanes(paths, plan_of, region)
    if entries:
        split_connecting_entries(data, entries)
        plans = plan_roads(data, parting)
        plan_of = {lane: plan for plan in plans for lane in plan.lanes}
        connecting = connecting_plans(plans, plan_of)
        region = junction_regions(plans, plan_of, connecting)
        paths = connecting_paths(plans, connecting)

    # Every other plan becomes a plain road, numbered from 0 in plan order. road_of holds only their lanes: paths
    # and later plain roads attach to these roads.
    roads = [Road(i, plan.right, plan.left, None) for i, plan in enumerate(p for p in plans if p not in connecting)]
    road_of = {lane: road for road in roads for lane in road.lanes}
    # Each route through connecting material becomes a path road of its own, numbered after the plain roads.
    entered = {region[plan_of[lanes[0]]] for incoming, lanes, _ in paths if incoming is not None}  # regions entered
    for incoming, lanes, outgoing in paths:
        # Ambiguous unentered paths received plain entry segments above. A remaining unentered chain, or an isolated
        # path touching no plain road, can be written outside a junction.
        junction = region[plan_of[lanes[0]]]
        if junction not in entered or (incoming is None and outgoing is None):
            junction = None
        roads.append(Road(len(roads), lanes, [], junction, path=True, incoming=incoming, outgoing=outgoing))

    # Plain roads are built in order, each attaching at an end only to a single plain road built before it. Path roads
    # come after all of them, so the plain roads they attach to are always built.
    built = set()
    for road in roads:
        if road.path:
            _build_path(road, road_of, tolerance)
        else:
            _build_plain(road, road_of, built, tolerance)
            built.add(road)
    return roads

def _entry_lanes(paths, plan_of, region) -> Set[Lane]:
    """Sources needing plain entry segments before otherwise unentered paths can form a junction.

    Paths may stay outside junctions only when no source lane is duplicated and every ordinary road end touches at
    most one path. Ambiguous groups need ordinary incoming segments instead; disconnected single chains keep their
    original representation.

    :param paths: (incoming lane, connecting lanes, outgoing lane) routes from connecting_paths
    :param plan_of: lane to the road plan holding it
    :param region: connecting plan to its junction region name
    :returns: starting lanes of ambiguous groups; each has no predecessors
    """
    entered = {region[plan_of[lanes[0]]] for incoming, lanes, _ in paths if incoming is not None}
    groups = {}
    for incoming, lanes, outgoing in paths:
        name = region[plan_of[lanes[0]]]
        groups.setdefault(name, []).append((incoming, lanes, outgoing))
    entries = set()
    for name, group in groups.items():
        occurrences = Counter(lane for _, lanes, _ in group for lane in lanes)
        exits = Counter((plan_of[outgoing], outgoing in plan_of[outgoing].right)
                        for _, _, outgoing in group if outgoing is not None)
        for incoming, lanes, outgoing in group:
            if name in entered and (incoming is not None or outgoing is not None):
                continue
            ambiguous = any(occurrences[lane] > 1 for lane in lanes)
            if outgoing is not None:
                ambiguous |= exits[plan_of[outgoing], outgoing in plan_of[outgoing].right] > 1
            if ambiguous:
                entries.update(lane for lane in lanes if not lane.predecessors)
    return entries

def _build_plain(road: Road, road_of: Dict[Lane, Road], built: Set[Road], tolerance: float) -> None:
    """
    Build a road of side-by-side lanes: its reference line, lane widths and elevation.

    An end that meets a single plain road built before it is attached to that road's written lane ends, with the whole
    cross-section on that road's square end. Any other end is free, and is cut back to where all lanes of the road
    exist, unless less than half the road would remain (see _trim).

    :param Road road: a road that is not a path, not built yet. road.reference, road.widths and road.elevation are
                      written, and road.geometry_adjustment when a fold repair moves the reference line. At attached
                      ends the boundary ends of its lanes are moved, height included, in the shared map data
    :param Dict[Lane, Road] road_of: the plain road holding each lane; lanes of connecting material are absent
    :param Set[Road] built: plain roads already built, the only ones an end may attach to; not changed here
    :param float tolerance: the most, in m, a source vertex may lie from the reference line as first fitted
    """
    # Attached ends only: at_end -> heading of the reference line there (along s), and
    # (lane, at_end) -> width of the neighbour's lane where this lane meets it.
    headings, end_widths = {}, {}
    for at_end in (False, True):
        ends = road.end_lanes(at_end)
        # Roads of the lanes linked across this end; None stands for a connecting lane, which no plain road holds.
        beyond = {road_of.get(n) for lane, starts in ends for n in (lane.predecessors if starts else lane.successors)}
        # Attach only to one plain road that is already built. Not to a junction, several roads, nothing (the map
        # ends here), or a road not built yet: that one may attach to this road when its turn comes.
        if len(beyond) != 1 or next(iter(beyond)) not in built:
            continue
        other = beyond.pop()
        for lane, starts in ends:
            # the neighbour's lanes this lane continues from (where it starts) or into (where it ends)
            linked = [n for n in (lane.predecessors if starts else lane.successors) if road_of.get(n) is other]
            if not linked:
                continue
            # Move this lane's boundary ends, height included, onto the borders its neighbour is written with.
            left, right = other.lane_end(linked[0], not starts)
            z = other.elevation_at(other.lane_end_s(linked[0], not starts))
            _move_boundary_end(lane.left, starts, left, z)
            _move_boundary_end(lane.right, starts, right, z)
            end_widths[lane, at_end] = float(np.hypot(*(right - left)))
            # The neighbour's lane drives the same way as this one; its heading, turned by pi for a left lane (which
            # drives against s), is the reference line's. The first lane linked at this end sets it.
            heading = other.lane_heading(linked[0], not starts)
            headings.setdefault(at_end, heading if lane in road.right else heading + np.pi)
            on_end, end_z = left, z  # every moved border lies on the neighbour's square road end
        # The reference line follows the left boundary of the innermost right lane, which has not moved if that lane
        # has no link here (it may start a little later than the lane beside it). Its end goes onto the neighbour's
        # square road end too, the line through the moved borders across the heading there, so that the whole
        # cross-section starts (ends) where the neighbour's does.
        inner, at_start = road.right[0], not at_end  # right lanes start at s = 0
        point = inner.left.points[0 if at_start else -1, :2]
        along = np.array([np.cos(headings[at_end]), np.sin(headings[at_end])])
        if abs((point - on_end) @ along) > 1e-9:
            _move_boundary_end(inner.left, at_start, point - ((point - on_end) @ along) * along, end_z)

    # The reference line follows the left boundary of the innermost right lane: the centre line of a two-way road,
    # the left edge of a one-way road.
    lane = road.right[0]
    if lane.sketched_roundabout and lane.width_samples is not None:
        # A smoothed roundabout centre is sampled densely, but its separately
        # drawn boundary may still have polygon corners. Use the centre and
        # Apollo width to give the plain road the same smooth course as the
        # intervening junction paths.
        center = lane.center[:, :2]
        along = geometry.arc_lengths(center)
        tangent = np.gradient(center, along, axis=0, edge_order=2 if len(center) > 2 else 1)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # unit normal pointing left
        width = np.interp(along / along[-1], lane.width_samples[:, 0], lane.width_samples[:, 1])
        # the left boundary: the centre moved left by half the width
        reference = np.column_stack([center + width[:, None] * normal / 2, lane.center[:, 2]])
        # attached ends start exactly where the boundary ends were moved above
        for at_end in headings:
            index = -1 if at_end else 0
            reference[index] = lane.left.points[index]
        height_source = reference
    else:
        height_source = lane.left.points
        reference = geometry.remove_end_kinks(geometry.dedupe(height_source))
    trimmed = _trim(reference, road, set(headings))  # free ends cut back to where all lanes exist, within limits
    if trimmed is not reference:
        # Crop the original height profile at the written free ends, without dropping the vertices removed only
        # to clean up the XY reference. Projection maps the cleaned endpoints back to their source heights.
        height_line = LineString(height_source[:, :2])
        lo, hi = [height_line.project(shapely.Point(*point[:2])) for point in (trimmed[0], trimmed[-1])]
        height_source = slice_polyline(height_source, lo, hi)
    source = _attach(trimmed, headings)  # attached ends turned gradually into the neighbour's heading
    # End headings for the fit: the neighbour's at attached ends; at a free end the curve's own, where it goes on
    # beyond the end (see _natural_headings). Elsewhere the fit follows the end segment.
    ends = {**_natural_headings(road, reference, source, set(headings), tolerance), **headings}
    road.reference = ReferenceLine.from_polyline(source, tolerance, start_heading=ends.get(False),
                                                 end_heading=ends.get(True))
    # Widths meet the neighbour's at attached ends. A fold repair refits them on the moved reference line; plain
    # roads are never narrowed, so its narrow flag is ignored here.
    plain_widths(road, end_widths)
    settle_surface(road, lambda narrow: plain_widths(road, end_widths))
    fit_elevation(road, height_source)  # neither end-kink removal nor _attach should redraw the source heights

def _build_path(road: Road, road_of: Dict[Lane, Road], tolerance: float) -> None:
    """
    Build a path road: its reference line, lane sections, widths and elevation.

    The reference line follows the Apollo centre lines of the path's lanes, one after another. Where the path meets
    its incoming (outgoing) lane, it starts (ends) on that lane's written centre, with the neighbour's reference-line
    heading turned to the lane's driving direction, the lane's width and the road's height there. That heading is not
    the tangent of the written lane where the neighbour's width or lane offset varies (see Road.lane_heading). Its
    ends are never trimmed; an end without a neighbour stays where Apollo puts it, and the fit there follows the
    direction of the end segment.

    :param Road road: a path road, not built yet. road.reference, road.section_starts, road.widths, road.lane_offset
                      and road.elevation are written, and road.geometry_adjustment when a repair moves the reference
                      line. The map data is not changed
    :param Dict[Lane, Road] road_of: the plain road holding each lane; the roads holding road.incoming and
                                     road.outgoing must already be built
    :param float tolerance: the most, in m, a source vertex may lie from the reference line as first fitted
    """
    # the Apollo centre lines of the path's lanes, one after another
    center = np.vstack([lane.center for lane in road.right])
    # at_end -> the neighbour's reference-line heading where the path meets it, and its lane's width there
    headings, end_widths = {}, {}
    # The incoming lane ends where the path starts and the outgoing lane starts where the path ends, so at_end also
    # serves as the neighbour's at_start below.
    for at_end, lane in ((False, road.incoming), (True, road.outgoing)):
        if lane is None:  # nothing leads in (out) here: the end stays where Apollo puts it
            continue
        other = road_of[lane]
        # Start (end) on the neighbour's written lane centre, at the neighbour's height there.
        left, right = other.lane_end(lane, at_end)
        center = move_endpoint(center, not at_end, (left + right) / 2)
        center[-1 if at_end else 0, 2] = other.elevation_at(other.lane_end_s(lane, at_end))
        # Path lanes are right lanes, driving along s, so the neighbour's reference heading there, turned to its
        # lane's driving direction, becomes this reference line's (not the lane's tangent where widths vary).
        headings[at_end] = other.lane_heading(lane, at_end)
        end_widths[at_end] = float(np.hypot(*(right - left)))

    # Ends turned gradually into the neighbours' headings, then fitted with those headings held.
    reference = _attach(geometry.remove_end_kinks(geometry.dedupe(center)), headings)
    road.reference = ReferenceLine.from_polyline(reference, tolerance, start_heading=headings.get(False),
                                                 end_heading=headings.get(True))
    # Sections and centred widths that meet the neighbours' widths. A fold repair may narrow tight turns; a wide,
    # tight path may then be refitted towards Apollo's drawn borders.
    path_widths(road, end_widths)
    settle_surface(road, lambda narrow: path_widths(road, end_widths, narrow))
    fit_curved_path_borders(road, end_widths)
    fit_elevation(road, center)  # heights from the centre lines, not from the ends _attach redraws

def _natural_headings(road: Road, reference: np.ndarray, source: np.ndarray, attached: Set[bool],
                      tolerance: float) -> Dict[bool, float]:
    """
    Headings for the fit at the road's free ends, where the road is a curve that goes on beyond them.

    Left to itself, the fit leaves a polyline in the direction of its end segment. On a coarsely sampled curve that
    direction misses the curve's by about half the turn at the vertex before the end, and the road, with everything
    attached to it later, would wiggle there. The end segment's direction is therefore turned by what two estimates
    of the curve's own heading at the end agree on:

    1. Circle: the circle through the three vertices of ``source`` nearest the end (its first three at the start, its
       last three at the end). Its tangent at the end is approximated by turning the end segment by the turn at the
       vertex before it, times the end segment's share of the two segments' length; this is exact for equal segments
       and a small-angle approximation otherwise.
    2. Fit: a reference line fitted through ``reference`` extended by the boundary of a lane beyond the end (see
       ``_continuation``; of several, the one turning least), read where ``source`` ends.

    An end is turned only where both estimates turn the same way, and by the smaller of the two. Where they turn
    opposite ways or either does not turn, the end gets no heading, so a straight road between junctions stays
    straight whatever the lanes beyond do.

    :param Road road: the road being built; its reference line follows the left boundary of road.right[0], whose
                      predecessors and successors give the lanes beyond
    :param np.ndarray reference: the reference polyline before trimming, shape (N, 3); only x, y are used
    :param np.ndarray source: the polyline the reference line is fitted to, trimmed and attached, shape (M, 3);
                              only x, y are used
    :param Set[bool] attached: the ends attached to another road (True for s = length); they get no heading
    :param float tolerance: fit tolerance, in m, of the extended reference line
    :returns: at_end to the heading in radians along s, for each free end that is turned; empty when none is.
              A free end also gets none when ``source`` has fewer than three vertices or no lane continues there
    :rtype: Dict[bool, float]
    """
    lane = road.right[0]  # the reference line runs along its left boundary
    pts = reference[:, :2]  # untrimmed: the lanes beyond continue from its ends
    ends = geometry.dedupe(source)[:, :2]  # what the fit sees: its end vertices give the circle estimate
    # Only a free end gets a continuation, and only when there are three vertices for the circle.
    before = None if False in attached or len(ends) < 3 else _continuation(pts, lane.predecessors, False)
    after = None if True in attached or len(ends) < 3 else _continuation(pts, lane.successors, True)
    if before is None and after is None:
        return {}
    # Fitted together with what lies beyond, the curve runs on through the end instead of stopping on its chord.
    parts = [part for part in (before, pts, after) if part is not None]
    natural = ReferenceLine.from_polyline(np.vstack(parts), tolerance)
    result = {}
    for at_end, beyond in ((False, before), (True, after)):
        if beyond is None:
            continue
        # Both estimates are turns from the end segment's direction, walking out towards this end.
        p0, p1, p2 = (ends[-3], ends[-2], ends[-1]) if at_end else (ends[2], ends[1], ends[0])
        near, far = p2 - p1, p1 - p0  # the end segment and the one before it, walking towards the end
        chord = np.arctan2(near[1], near[0])  # where the fit would leave the end on its own
        bend = _angle(np.arctan2(near[1], near[0]) - np.arctan2(far[1], far[0]))  # turn at the vertex before the end
        circle = bend * np.hypot(*near) / (np.hypot(*near) + np.hypot(*far))  # the end segment's share of it
        # The extended fit's heading where source ends; walking out of the start runs against s, hence pi.
        s, _ = natural.project(ends[-1 if at_end else 0])
        fitted = _angle(float(natural.at(s)[2]) + (0 if at_end else np.pi) - chord)
        if circle * fitted > 0:  # both turn the same way
            turn = np.sign(fitted) * min(abs(fitted), abs(circle))
            result[at_end] = float(chord + turn + (0 if at_end else np.pi))  # back to the heading along s
    return result

def _angle(a: float) -> float:
    """
    Wrap an angle into [-pi, pi).

    :param float a: angle in radians, of any size
    :returns: the angle of the same direction in [-pi, pi); pi itself comes back as -pi. Rounding can return pi for
              an angle a hair below -pi
    :rtype: float
    """
    return float((a + np.pi) % (2 * np.pi) - np.pi)

def _continuation(pts: np.ndarray, lanes: List[Lane], at_end: bool) -> Optional[np.ndarray]:
    """
    Up to CONTINUATION metres of the left boundary of the lane beyond an end that turns least.

    A lane beyond is a candidate when the near end of its left boundary lies within CONTINUATION_GAP of the end.
    Boundaries of consecutive lanes need not meet exactly, so each is shifted to start exactly at the end. Its
    vertices more than FIT_KNOT_GAP and at most CONTINUATION metres along it are kept, so no kept vertex shares a
    knot with the end. The candidate turning least is the one whose first kept vertex lies most nearly straight
    ahead of the end; of equals, the first in ``lanes``.

    :param np.ndarray pts: the reference polyline, x, y only, shape (N, 2)
    :param List[Lane] lanes: the lanes beyond that end: predecessors of the road's lane at the start, successors at
                             the end
    :param bool at_end: True for the end at the last point, False for the first
    :returns: the K >= 1 kept vertices of the chosen boundary, shape (K, 2), in the reference line's direction, so
              they stack before ``pts`` at the start and after it at the end; None when no lane qualifies
    :rtype: Optional[np.ndarray]
    """
    end = pts[-1] if at_end else pts[0]
    direction = geometry.end_direction(pts, not at_end)  # along the polyline at this end
    best = None  # (how straight on the boundary runs, its kept vertices)
    for other in lanes:
        line = geometry.dedupe(other.left.points)[:, :2]
        # Walk away from the end: a lane before the start drives towards it, so its boundary is reversed.
        if not at_end:
            line = line[::-1]
        # too short, or its near end too far from this end to continue it
        if len(line) < 2 or np.hypot(*(line[0] - end)) > CONTINUATION_GAP:
            continue
        line = line + (end - line[0])  # shifted to start exactly at the end
        along = geometry.arc_lengths(line)
        # Drop vertices that would share a knot with the end, and stop CONTINUATION metres on.
        beyond = line[(along > geometry.FIT_KNOT_GAP) & (along <= CONTINUATION)]
        if not len(beyond):
            continue
        # Cosine between the first step and straight ahead, which at the start runs against the polyline:
        # the larger, the less the boundary turns.
        step = beyond[0] - end
        cos = float(step @ direction) / np.hypot(*step) * (1 if at_end else -1)
        if best is None or cos > best[0]:  # of equals, the earlier lane stays
            # back in the reference line's direction: before the start, that means reversed again
            best = (cos, beyond if at_end else beyond[::-1])
    return None if best is None else best[1]

def _trim(reference: np.ndarray, road: Road, attached: Set[bool]) -> np.ndarray:
    """
    Cut the reference polyline to where all lanes of the road exist, at ends not attached to another road.

    Lanes of one road can end metres apart along it (a slanted stop line). Each free end therefore moves in to the
    lane end lying farthest in from it, taking the centre line ends of the lanes there projected onto the polyline.
    A square road end at that point keeps every lane within its Apollo extent, and the junction attached there takes
    up the rest. Nothing is cut when the rest would be shorter than half the polyline, or when both ends would move
    less than 5 cm.

    :param np.ndarray reference: the reference polyline, shape (N, 3); x, y give the stations, z is kept
    :param Road road: the road being built, whose lanes end at its two ends
    :param Set[bool] attached: the ends attached to another road (True for s = length); they are never cut
    :returns: the polyline between the two cuts, shape (M, 3), with heights interpolated at the cuts; when nothing
              is cut, ``reference`` itself, so callers can tell with ``is``
    :rtype: np.ndarray
    """
    line = LineString(reference[:, :2])
    lo, hi = 0.0, line.length  # stations of the two cuts; an end stays put until moved in below
    for at_end in (False, True):
        if at_end in attached:  # it already lies on the written end of the road it continues
            continue
        # station on the polyline of every lane's centre line end at this end
        ends = [line.project(shapely.Point(*lane.center[0 if starts else -1, :2]))
                for lane, starts in road.end_lanes(at_end)]
        # Every lane exists inside the lane end farthest in: the smallest station at the end, the largest at the start.
        if at_end:
            hi = min(ends)
        else:
            lo = max(ends)
    # Keep the polyline whole when less than half of it would remain, or when both ends would move less than 5 cm.
    if hi - lo < 0.5 * line.length or (lo < 0.05 and hi > line.length - 0.05):
        return reference  # the very array passed in: callers test for a cut with `is`
    return slice_polyline(reference, lo, hi)  # cut by x, y length, with z interpolated at the new ends

def _move_boundary_end(ref: BoundaryRef, at_start: bool, xy: np.ndarray, z: float) -> None:
    """
    Move one end of a lane's boundary, in the lane's driving direction, to a given point and height.

    The end moves to ``xy`` as with move_endpoint: the line is cut back to where ``xy`` projects within its first
    15 m from that end, or its end vertex is replaced when ``xy`` lies beyond it. The new end then takes height
    ``z`` instead of the interpolated one. The result goes back into the shared Boundary in its stored order, so a
    neighbouring lane sharing the boundary moves with it; the marks of ``ref`` are left as they are.

    :param BoundaryRef ref: the boundary as seen from the lane; its points are (N, 3) and become (M, 3), M <= N
    :param bool at_start: move the end where the lane starts when true, otherwise the one where it ends
    :param np.ndarray xy: x, y of the new end, shape (2,)
    :param float z: height of the new end, in m
    """
    points = move_endpoint(ref.points, at_start, xy)
    points[0 if at_start else -1, 2] = z
    ref.boundary.points = points[::-1] if ref.reversed else points

def _attach(points: np.ndarray, headings: Dict[bool, float]) -> np.ndarray:
    """
    Let the reference polyline turn gradually into the given headings at its start (end).

    Each end with a heading has its end part replaced by a curve that leaves (arrives) in that heading, as with
    blend_heading: about 10 m per radian of turn and at least 1 m, but never more than 40% of the line, so a 1 m line
    has only 0.4 m replaced. A short road held at both ends (up to SHORT_ROAD) whose two turns do not both fit
    becomes a single cubic Hermite curve between its end points in those headings instead, which spreads the turning
    over its whole length. The end points never move.
    Heights on a replaced part are linear between its ends, so elevation is fitted from the polyline before this.

    :param np.ndarray points: the reference polyline, shape (N, 3)
    :param Dict[bool, float] headings: at_end (True for the last point) to the heading in radians along the polyline
                                       there; an end not listed keeps its own direction
    :returns: the redrawn polyline, shape (M, 3), without repeated points; with no headings, ``points`` deduplicated
    :rtype: np.ndarray
    """
    length = geometry.arc_lengths(points)[-1] if len(points) > 1 else 0.0  # x, y length
    # Held at both ends, a short road whose two turns need more room than it has (blend lengths before the 40% cap)
    # becomes one curve from end to end.
    if len(headings) == 2 and 0 < length <= SHORT_ROAD and \
            sum(blend_length(points, h, not at_end) for at_end, h in headings.items()) > length:
        directions = [np.array([np.cos(headings[end]), np.sin(headings[end])]) for end in (False, True)]
        # evenly in t, one point per 0.25 m of chord (not of curve), with z linear from end to end
        return HermiteCurve.fit_directions(points[0], points[-1], directions[0], directions[1]).sample()
    # Otherwise one end after the other; the second blend works on the line the first one returned.
    for at_end, heading in headings.items():
        points = blend_heading(points, heading, not at_end)  # blend_heading takes at_start, hence not at_end
    return geometry.dedupe(points)  # without repeated points
