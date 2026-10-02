"""
Read an Apollo map into :class:`~apollo2xodr.model.MapData`.

Besides converting points to the local frame, this resolves lane links and lets neighbouring lanes whose boundaries
coincide share one boundary.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import shapely
from shapely import STRtree
from shapely.geometry import LineString
from . import geometry
from .frames import latlon_to_utm, utm_zone
from .model import Boundary, BoundaryRef, Lane, MapData, TrafficControl
from .proto import load_map

log = logging.getLogger(__name__)

__all__ = ['read_map']

############################################## Apollo Map Configuration ##############################################

BOUNDARY_TYPES = {1: 'DOTTED_YELLOW', 2: 'DOTTED_WHITE', 3: 'SOLID_YELLOW', 4: 'SOLID_WHITE', 5: 'DOUBLE_YELLOW',
                  6: 'CURB'}
LANE_TYPES = {1: 'none', 2: 'driving', 3: 'biking', 4: 'sidewalk', 5: 'parking', 6: 'shoulder'}
MAX_LINK_GAP = 1.0         # m, end points closer than this may be linked when Apollo lists no link for them
SHARE_TOLERANCE = 1.0      # m, neighbouring lanes share a boundary only where their own boundaries are this close

#######################################################################################################################

def read_map(path, simplify_tolerance: float = 0.025) -> MapData:
    """
    Read an Apollo map from a Protobuf file and convert it to :class:`~apollo2xodr.model.MapData` in a local frame.

    Every point is shifted by the map origin. Lane links are resolved, and neighbouring lanes of one road share a
    boundary where their own boundaries coincide.

    :param path: Apollo map file, binary (``base_map.bin``) or text format (``.txt``)
    :param float simplify_tolerance: max x/y and z deviation, in m, of the simplified lane centre lines and
                                     boundaries; 0 only drops consecutive points closer than 1e-6 m in x/y
    :returns: lanes, traffic controls, and the origin and UTM zone of the local frame
    :rtype: MapData
    """
    hd_map = load_map(path)
    origin, zone, projection = _origin(hd_map)
    to_local = _LocalFrame(origin)          # map points minus the origin keep coordinates small

    # lane id -> (road id, junction id) from the road sections; a lane listed by two roads keeps the first
    road_of: Dict[str, Tuple[str, Optional[str]]] = {}
    road_order: List[str] = []
    for road in hd_map.road:
        road_order.append(road.id.id)
        junction = road.junction_id.id if road.HasField('junction_id') and road.junction_id.id else None
        for section in road.section:
            for lane_id in section.lane_id:
                road_of.setdefault(lane_id.id, (road.id.id, junction))

    lanes: Dict[str, Lane] = {}
    for lane in hd_map.lane:
        # a lane that no road lists becomes a road of its own
        road, junction = road_of.get(lane.id.id, (f'lane:{lane.id.id}', None))
        # the road's junction comes first; the lane's own junction id fills in when the road has none
        if junction is None and lane.HasField('junction_id') and lane.junction_id.id:
            junction = lane.junction_id.id
        lanes[lane.id.id] = Lane(
            id=lane.id.id,
            center=geometry.simplify(to_local.curve(lane.central_curve), simplify_tolerance),
            left=_boundary(lane.left_boundary, to_local, simplify_tolerance),
            right=_boundary(lane.right_boundary, to_local, simplify_tolerance),
            speed_limit=float(lane.speed_limit) if lane.HasField('speed_limit') else 20.0,  # m/s, 72 km/h if unset
            road=road,
            junction=junction,
            apollo_predecessors=[x.id for x in lane.predecessor_id],
            apollo_successors=[x.id for x in lane.successor_id],
            # Apollo may list several neighbours on one side; only the first is kept
            left_forward=_first(lane.left_neighbor_forward_lane_id),
            right_forward=_first(lane.right_neighbor_forward_lane_id),
            left_reverse=_first(lane.left_neighbor_reverse_lane_id),
            kind=LANE_TYPES.get(lane.type, 'driving'),     # a type missing from LANE_TYPES counts as driving
            width_samples=_width_samples(lane),
        )

    # both look lanes up by id, so they run once every Lane exists
    _link_lanes(lanes)
    _share_boundaries(lanes)
    controls = _read_controls(hd_map, to_local)
    # Presence of source heights must not depend on the requested simplification tolerance.
    has_heights = any(p.z != 0 for lane in hd_map.lane for seg in lane.central_curve.segment
                      for p in seg.line_segment.point)
    return MapData(lanes=lanes, road_order=road_order, junction_order=[j.id.id for j in hd_map.junction],
                   controls=controls, origin=origin, utm_zone=zone, projection=projection, has_heights=has_heights)

def _origin(hd_map) -> Tuple[Tuple[float, float], int, str]:
    """
    Map origin (E0, N0) from the header's lat/lon extent, the UTM zone and the projection string.

    The origin is the UTM point at the centre of the extent, or the first lane point, rounded to whole metres, when
    the header has no UTM projection or no extent.

    :param hd_map: Apollo map message, as returned by :func:`~apollo2xodr.proto.load_map`
    :returns: (E0, N0) in m, the UTM zone number, and the projection string ('' if the header has none)
    :rtype: Tuple[Tuple[float, float], int, str]
    """
    header = hd_map.header
    projection = header.projection.proj if header.HasField('projection') else ''
    # proj is a str in the bundled descriptor, while the other text fields of the header are bytes; decode to be safe
    projection = projection.decode() if isinstance(projection, bytes) else projection
    # centre of the extent: top/bottom are latitudes, left/right longitudes, all 0 when the header has no extent
    lat = (header.top + header.bottom) / 2
    lon = (header.left + header.right) / 2
    # a proj4 string such as '+proj=utm +zone=10 ...' names its zone; otherwise derive it from the centre
    zone_match = re.search(r'\+zone=(\d+)', projection)
    zone = int(zone_match.group(1)) if zone_match else utm_zone(lat, lon)
    # without a UTM projection or an extent the header cannot place the map, so take the origin from the map itself
    if 'utm' not in projection or (lat == 0 and lon == 0):
        log.warning('No UTM projection or lat/lon extent in the map header; using the first lane point as origin')
        first = next(iter(hd_map.lane), None)
        point = first.central_curve.segment[0].line_segment.point[0] if first else None
        # (0, 0) when the map has no lanes
        return ((round(point.x), round(point.y)) if point else (0.0, 0.0)), zone, projection
    return latlon_to_utm(lat, lon, zone), zone, projection

class _LocalFrame:
    """
    A local frame centred on the map origin: x/y are Apollo's minus (E0, N0), z is Apollo's unchanged.
    """
    def __init__(self, origin):
        """
        Constructor, store the origin as (E0, N0, 0), shape (3,)

        :param origin: map origin (E0, N0) in m, from :func:`_origin`
        """
        self.origin = np.array([origin[0], origin[1], 0.0])

    def points(self, points) -> np.ndarray:
        """
        Apollo points in the local frame.

        :param points: N Apollo ``PointENU`` messages, any iterable
        :returns: the points, shape (N, 3); shape (0, 3) if there are none
        :rtype: np.ndarray
        """
        # with no points np.array gives shape (0,), which cannot broadcast against the origin; reshape makes it (0, 3)
        return np.array([(p.x, p.y, p.z) for p in points], dtype=float).reshape(-1, 3) - self.origin

    def curve(self, curve) -> np.ndarray:
        """
        The points of an Apollo curve in the local frame.

        :param curve: Apollo ``Curve`` message; the points of its segments are joined in order
        :returns: the points, shape (N, 3); shape (0, 3) if there are none
        :rtype: np.ndarray
        """
        return self.points(p for seg in curve.segment for p in seg.line_segment.point)

def _boundary(boundary, to_local: _LocalFrame, tolerance: float) -> BoundaryRef:
    """
    One Apollo lane boundary in the local frame, with its marking types along it.

    The marks (point, type) come from Apollo's ``boundary_type`` records, placed at their stations ``s``, the x/y arc
    length from the start of the boundary.

    :param boundary: Apollo ``LaneBoundary`` message; its curve needs at least one point
    :param _LocalFrame to_local: the frame to convert the points into
    :param float tolerance: simplification tolerance of the stored points, in m
    :returns: a new, unshared boundary in Apollo's point order. ``boundary.points`` has shape (M, 3); ``marks`` are
              sorted by station, the first at the boundary's start, each point shape (3,)
    :rtype: BoundaryRef
    """
    # kind follows Apollo's first record in file order, before any sorting, and only that record's first type
    kind = 'UNKNOWN'
    if boundary.virtual:
        kind = 'VIRTUAL'
    elif boundary.boundary_type and boundary.boundary_type[0].types:
        kind = BOUNDARY_TYPES.get(boundary.boundary_type[0].types[0], 'UNKNOWN')
    points = to_local.curve(boundary.curve)
    s = geometry.arc_lengths(points)        # x/y arc length at every point, the scale of Apollo's stations
    # (station, type) per record; a type outside BOUNDARY_TYPES or a record without types is UNKNOWN.
    # A virtual boundary keeps one VIRTUAL record at the start, whatever Apollo lists
    records = [(0.0, kind)] if boundary.virtual else [
        (b.s, BOUNDARY_TYPES.get(b.types[0], 'UNKNOWN') if b.types else 'UNKNOWN')
        for b in boundary.boundary_type]
    records = sorted(dict(records).items())  # duplicate stations: Apollo's final record wins
    # the marks must cover the boundary from its start
    if not records or records[0][0] > 0:
        records.insert(0, (0.0, 'UNKNOWN'))
    # each station's point, on the points before simplification; np.interp clamps stations past either end
    marks = [(np.array([np.interp(station, s, points[:, k]) for k in range(3)]), value)
             for station, value in records]
    # only the stored points are simplified, so the marks keep their exact positions
    return BoundaryRef(Boundary(geometry.simplify(points, tolerance), kind), marks=marks)

def _width_samples(lane) -> Optional[np.ndarray]:
    """
    Apollo's lane width samples as (fraction of the lane's length, width), or None without samples on both sides.

    ``left_sample`` and ``right_sample`` are the distances from the centre line to each boundary; the width is their
    sum at every station of either side.

    :param lane: Apollo ``Lane`` message
    :returns: rows sorted by fraction, shape (K, 2); the fraction is ``s / lane.length`` clipped to [0, 1], the
              width in m. None also if ``lane.length <= 0``
    :rtype: Optional[np.ndarray]
    """
    if not len(lane.left_sample) or not len(lane.right_sample) or lane.length <= 0:
        return None
    # sorted by station, as np.interp needs ascending stations
    left = np.array(sorted((x.s, x.width) for x in lane.left_sample))
    right = np.array(sorted((x.s, x.width) for x in lane.right_sample))
    s = np.unique(np.concatenate([left[:, 0], right[:, 0]]))
    # each side is interpolated at the other's stations and held at its own ends
    width = np.interp(s, left[:, 0], left[:, 1]) + np.interp(s, right[:, 0], right[:, 1])
    return np.column_stack([np.clip(s / lane.length, 0.0, 1.0), width])

def _first(ids) -> Optional[str]:
    """
    The first id of a repeated Apollo ``Id`` field, such as ``lane.left_neighbor_forward_lane_id``.

    :param ids: repeated Apollo ``Id`` messages
    :returns: the first id string, or None if ``ids`` is empty
    :rtype: Optional[str]
    """
    return ids[0].id if len(ids) else None

def _add_link(pred: Lane, succ: Lane) -> None:
    """
    Link two lanes both ways: ``succ`` joins ``pred.successors`` and ``pred`` joins ``succ.predecessors``, unless
    already there.

    :param Lane pred: the lane driven first
    :param Lane succ: the lane driven next, from the end of ``pred``
    """
    # Apollo usually lists a link on both lanes, so the same pair arrives twice; the checks keep it once.
    # Lane has eq=False, so `in` compares identity and never the numpy fields
    if succ not in pred.successors:
        pred.successors.append(succ)
    if pred not in succ.predecessors:
        succ.predecessors.append(pred)

def _link_lanes(lanes: Dict[str, Lane]) -> None:
    """
    Apollo links, plus links between touching end points that Apollo leaves unlinked.

    An end and a start touch when they are closer in x/y than ``gap``, the smaller of ``MAX_LINK_GAP`` and half the
    shortest start-to-end chord of a lane that is not closed, and the link does not turn back.

    :param Dict[str, Lane] lanes: Lane.id to lane; their ``predecessors`` and ``successors`` are filled in place
    """
    # _add_link fills both lanes, so a link listed on either side joins both; ids missing from the map are skipped
    for lane in lanes.values():
        for other in (lanes.get(x) for x in lane.apollo_predecessors):
            if other is not None:
                _add_link(other, lane)
        for other in (lanes.get(x) for x in lane.apollo_successors):
            if other is not None:
                _add_link(lane, other)

    # start-to-end straight distance of each lane. A closed lane, ending within 1e-6 m (dedupe's tolerance) of its
    # start, is left out: its chord of 0 would make gap 0 and so stop these links for the whole map
    chords = [c for c in (np.hypot(*(l.center[-1, :2] - l.center[0, :2])) for l in lanes.values()) if c > 1e-6]
    gap = min(min(chords) / 2, MAX_LINK_GAP) if chords else 0.0
    # by the links resolved above, so an Apollo link listed on the other lane only also counts
    ends = [l for l in lanes.values() if not l.successors]
    starts = [l for l in lanes.values() if not l.predecessors]
    if gap <= 0 or not ends or not starts:
        return
    tree = STRtree(shapely.points([l.center[0, :2] for l in starts]))
    for lane in ends:
        hits = tree.query(shapely.Point(lane.center[-1, :2]), predicate='dwithin', distance=gap)
        for i in sorted(hits):              # tree order is arbitrary; sorted, links follow the file order of lanes
            other = starts[i]
            # dwithin also keeps a start exactly gap away; links need strictly closer
            if other is lane or np.hypot(*(lane.center[-1, :2] - other.center[0, :2])) >= gap:
                continue
            # an end touching the start of a lane heading back (a turn of more than 120 degrees) is no link
            if geometry.reverses_direction(lane.center, other.center):
                continue
            _add_link(lane, other)

def _share_boundaries(lanes: Dict[str, Lane]) -> None:
    """
    Neighbouring lanes of one Apollo road share the centre line of their two boundaries, where these coincide.

    Maps exported by LGSVL write every boundary 1.75 m beside the centre line, so a median or a lane that is metres
    longer than its neighbour leaves boundaries apart. Such lanes keep their own boundaries and end up in separate
    roads.

    The shared boundary is :func:`geometry.midline` of the two, which simplifies it at 0.02 m rather than at
    ``simplify_tolerance``. It takes the kind of the lane whose neighbour entry starts the merge, the first such lane
    in file order; each lane keeps its own marks.

    :param Dict[str, Lane] lanes: Lane.id to lane; their ``left`` and ``right`` references are changed in place
    """
    for lane in lanes.values():
        # neighbours in the same direction: my right boundary meets their left one, my left meets their right
        for other_id, side in ((lane.right_forward, 'right'), (lane.left_forward, 'left')):
            other = lanes.get(other_id) if other_id else None
            # lanes of different Apollo roads keep their own boundaries
            if other is None or other.road != lane.road:
                continue
            mine, theirs = (lane.right, other.left) if side == 'right' else (lane.left, other.right)
            # skip a pair the neighbour already merged when it was visited first, or boundaries that are apart
            if mine.boundary is theirs.boundary or not _coincide(mine.points, theirs.points):
                continue
            # both run in the driving direction, so midline needs no flip and neither reference is reversed
            shared = Boundary(geometry.midline(mine.points, theirs.points), mine.boundary.kind)
            mine.boundary = theirs.boundary = shared
            mine.reversed = theirs.reversed = False
        # neighbour in the opposite direction: the two left boundaries face each other, stored in opposite orders.
        # _coincide ignores point order, so they are compared as stored
        other = lanes.get(lane.left_reverse) if lane.left_reverse else None
        if other is not None and other.road == lane.road and lane.left.boundary is not other.left.boundary and \
                _coincide(lane.left.points, other.left.points):
            # flip theirs so midline gets both in my direction; the neighbour then reads the shared one reversed
            shared = Boundary(geometry.midline(lane.left.points, other.left.points[::-1]), lane.left.boundary.kind)
            lane.left.boundary = other.left.boundary = shared
            lane.left.reversed, other.left.reversed = False, True

def _coincide(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Whether two boundaries are within ``SHARE_TOLERANCE`` of each other.

    The test is the Hausdorff distance in x/y, so point order does not matter. Shapely measures it from the vertices
    only, which can come out low for strongly bent lines.

    :param np.ndarray a: one boundary, shape (N, 2) or (N, 3)
    :param np.ndarray b: the other boundary, shape (M, 2) or (M, 3)
    :returns: True if the distance is at most ``SHARE_TOLERANCE``; False if either has fewer than 2 points
    :rtype: bool
    """
    if len(a) < 2 or len(b) < 2:            # a LineString needs two points
        return False
    return LineString(a[:, :2]).hausdorff_distance(LineString(b[:, :2])) <= SHARE_TOLERANCE

def _read_controls(hd_map, to_local: _LocalFrame) -> List[TrafficControl]:
    """
    Traffic lights, stop signs and yield signs, with the lanes and junction that Apollo's overlaps tie them to.

    A light stands at its middle subsignal with a location, or the centre of its box, and takes its face size from
    the box. A sign gets a 0.95 x 0.95 face; :mod:`~apollo2xodr.signals` places it from its stop line. Controls
    without a stop line, and lights without a position, are skipped with a warning.

    :param hd_map: Apollo map message, as returned by :func:`~apollo2xodr.proto.load_map`
    :param _LocalFrame to_local: the frame to convert positions and stop lines into
    :returns: lights, then stop signs, then yield signs, each in file order
    :rtype: List[TrafficControl]
    """
    # control id -> (lane id, start s) of every lane sharing an overlap with it, and -> its first junction
    lanes_of: Dict[str, List[Tuple[str, float]]] = {}
    junction_of: Dict[str, str] = {}
    for overlap in hd_map.overlap:
        objects = list(getattr(overlap, 'object'))
        lanes = [(o.id.id, o.lane_overlap_info.start_s) for o in objects if o.HasField('lane_overlap_info')]
        junctions = [o.id.id for o in objects if o.HasField('junction_overlap_info')]
        # every control in this overlap takes all of its lanes, and its first junction unless one was found already
        for o in objects:
            if o.HasField('signal_overlap_info') or o.HasField('stop_sign_overlap_info') or \
                    o.HasField('yield_sign_overlap_info'):
                lanes_of.setdefault(o.id.id, []).extend(lanes)
                if junctions:
                    junction_of.setdefault(o.id.id, junctions[0])

    controls = []
    for signal in hd_map.signal:
        if not signal.stop_line:
            log.warning('Signal %s has no stop line, skipped', signal.id.id)
            continue
        corners = to_local.points(signal.boundary.point)    # the face's box, shape (K, 3)
        # a subsignal without a location reads as NaN, as PointENU's x and y default to NaN
        bulbs = to_local.points(x.location for x in signal.subsignal)
        bulbs = bulbs[np.isfinite(bulbs).all(axis=1)]
        # Apollo lists a light's subsignals in a row, so the middle one is the centre of the light
        if len(bulbs):
            position = bulbs[len(bulbs) // 2]
        elif len(corners):
            position = corners.mean(axis=0)
        else:
            log.warning('Signal %s has no position, skipped', signal.id.id)
            continue
        stop_lines = [to_local.curve(line) for line in signal.stop_line]
        control = TrafficControl(signal.id.id, 'signal', position, stop_lines[0],
                                 lanes_of.get(signal.id.id, []), junction_of.get(signal.id.id), stop_lines=stop_lines)
        # a vertical face seen from above is a segment, so the largest x/y distance between corners is its width
        if len(corners) >= 3:
            control.width = float(np.max(np.linalg.norm(corners[:, None, :2] - corners[None, :, :2], axis=2)))
            control.height = float(np.ptp(corners[:, 2]))
            control.bottom = float(corners[:, 2].min())
            # a flat or degenerate box: back to the dataclass default face
            if control.width <= 0 or control.height <= 0:
                control.width, control.height, control.bottom = 0.65, 1.5, None
        controls.append(control)
    # 'yield' is a Python keyword, so hd_map.yield cannot be written
    for kind, signs in (('stop', hd_map.stop_sign), ('yield', getattr(hd_map, 'yield'))):
        for sign in signs:
            if not sign.stop_line:
                log.warning('%s sign %s has no stop line, skipped', kind, sign.id.id)
                continue
            stop_lines = [to_local.curve(line) for line in sign.stop_line]
            # a sign's position is not used, so it stays (0, 0, 0)
            controls.append(TrafficControl(sign.id.id, kind, np.zeros(3), stop_lines[0],
                                           lanes_of.get(sign.id.id, []), junction_of.get(sign.id.id),
                                           width=0.95, height=0.95, stop_lines=stop_lines))
    return controls
