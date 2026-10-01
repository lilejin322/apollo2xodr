"""
Fit lane widths and elevation, and repair folding road surfaces.
Repair candidates change the reference line and must recompute widths and lane
sections together. Keep those operations here so their state stays consistent.
"""

import logging
from typing import Callable, Dict, Tuple, Optional, Sequence
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import LineString
from .. import geometry
from ..geometry.hermite import cubic_values
from ..model import Lane
from .model import Road

log = logging.getLogger(__name__)

WIDTH_STEP = 1.0
"""m between lane width samples"""
MIN_SECTION = 0.01
"""m, shortest lane section of a path road"""
ELEVATION_TOLERANCE = 0.02
"""m, elevation tolerance"""
WIDTH_TOLERANCE = 0.01
"""m, lane widths and offsets are written with the fewest knots that stay this close to samples"""
FOLD_STEP = 0.05
"""m between curvature samples when looking for folding lane borders"""
FOLD_LIMIT = 0.98
"""a border at offset t folds where curvature * t reaches 1; keep clear of it"""
NARROW_LIMIT = 0.9
"""a narrowed lane's half-width is at most 90 % of its turning radius"""
SMOOTHING_SPACING_RANGE = (1.0, 4.0)
"""m, B-spline control spacings searched on a sharp bend"""
END_SCALE_RANGE = (0.4, 2.0)
"""endpoint tangent lengths searched for a smoothed reference, times their default"""
REPAIR_MAX_SHIFT = 0.25
"""m, maximum reference-line movement where tangent lengths are searched to repair a lane"""
SEARCH_BUDGET = 40
"""objective evaluations per parameter search"""
SPACING_BISECTIONS = 5
"""halvings of the smoothing spacing range, to within a 4.4 % step"""
GLOBAL_POPULATION = 4
"""candidates per parameter in a search over the whole box (differential evolution)"""
GLOBAL_GENERATIONS = 5
"""generations such a search runs at most"""
INFEASIBLE = 10.0
"""added to the cost of a candidate that breaks a constraint, so any candidate keeping them all wins"""

def plain_widths(road: Road, end_widths: Dict[Tuple[Lane, bool], float]) -> None:
    """
    Lane widths from Apollo's width samples, meeting the widths of attached neighbours exactly.
    Apollo's boundaries of coarsely sampled maps are polygons; widths measured against them would make the lane
    borders follow every corner. The width samples are smooth; a lane without them is measured against its boundary.
    Each lane gets one width profile over the whole road, from s = 0 to its length. Where a lane continues on an
    attached road, a linear correction makes its width equal that road's there and fades out towards the other end.

    :param Road road: a plain road with its reference line set; road.widths[lane] is written for every lane of
                      road.right and road.left
    :param Dict[Tuple[Lane, bool], float] end_widths: (lane, at_end) to the width in m that the attached road has
                                                      where this lane meets it; at_end is True at s = length.
                                                      Lanes and ends not listed keep their own width there
    """
    length = road.reference.length
    s = np.unique(np.append(np.arange(0.0, length, WIDTH_STEP), length))  # (M,) stations, both ends included
    t = s / length if length > 0 else np.zeros_like(s)                    # (M,) fraction of the road, 0..1
    for side, sign in ((road.right, -1.0), (road.left, 1.0)):             # sign: direction away from the reference
        inner = np.zeros_like(s)  # (M,) signed offset of the current lane's inner border, from the lanes inside it
        for lane in side:
            if lane.width_samples is not None and len(lane.width_samples):
                along = t if sign < 0 else 1 - t  # left lanes run against s
                width = np.interp(along, lane.width_samples[:, 0], lane.width_samples[:, 1])
            else:
                # lane.right is the outer border on either side, since left lanes drive against s
                width = sign * (road.reference.lateral_offsets(s, lane.right.points) - inner)
            # at_end: the road's end at s = length; the gap to the neighbour's width fades out towards the other end
            for at_end, weight in ((False, 1 - t), (True, t)):
                target = end_widths.get((lane, at_end))
                if target is not None:
                    width = width + (target - width[-1 if at_end else 0]) * weight
            width = _nonnegative(lane, width)
            road.widths[lane] = geometry.cubic_segments(s, width, WIDTH_TOLERANCE)
            inner = inner + sign * width  # the next lane outward starts at this lane's outer border

def path_widths(road: Road, end_widths: Dict[bool, float], narrow: bool = False) -> None:
    """
    Fit path sections and centred offsets, optionally narrowing tight turns.
    A path road's lanes follow one another along it, one lane per lane section,
    all on the right of the reference. Each section takes its lane's Apollo widths;
    the reference runs along the lane centre, so the lane offset is half the width.

    :param Road road: a path road with its reference line set; road.section_starts, road.widths[lane] for every
                      lane of road.right, and road.lane_offset are written
    :param Dict[bool, float] end_widths: at_end to the width in m of the attached lane where the path meets it;
                                         at_end is True at s = length. An end not listed keeps its own width
    :param bool narrow: also narrow the lane where it turns tighter than its half-width allows (see _turnable)
    :returns: None; the road is updated in place
    :rtype: None
    """
    length = road.reference.length
    # Reserve room for every remaining lane, including the final section.
    # Very short paths cannot meet MIN_SECTION; share their available length instead.
    minimum = min(MIN_SECTION, length / len(road.right))
    # A path's lanes follow one another along it, one lane per section; each section starts where its lane does.
    starts = [0.0]
    for index, lane in enumerate(road.right[1:], 1):
        s, _ = road.reference.project(lane.center[0])
        latest = length - minimum * (len(road.right) - index)   # leaves `minimum` for this and every later section
        starts.append(min(max(s, starts[-1] + minimum), latest))
    road.section_starts = starts
    bounds = starts + [length]  # section i spans bounds[i] .. bounds[i + 1]

    s = np.unique(np.concatenate([np.arange(0.0, length, WIDTH_STEP), bounds]))  # (M,) stations, section ends included
    width = np.zeros_like(s)  # (M,) one width profile for the whole path, each section filled from its own lane
    for lane, lo, hi in zip(road.right, bounds[:-1], bounds[1:]):
        inside = (s >= lo) & (s <= hi)
        width[inside] = _apollo_width(road, lane, s[inside], lo, hi)
    # meet the neighbours' written widths exactly, spreading the difference linearly along the road
    t = s / length if length > 0 else np.zeros_like(s)
    width += (end_widths.get(False, width[0]) - width[0]) * (1 - t) + (end_widths.get(True, width[-1]) - width[-1]) * t
    width = _nonnegative(road.right[0], width)  # one profile for all sections, reported under the first lane
    if narrow:
        width = _turnable(road, s, width)
    for lane, lo, hi in zip(road.right, bounds[:-1], bounds[1:]):
        inside = (s >= lo) & (s <= hi)
        if hi - lo < 1e-9 or inside.sum() < 2:  # too short to fit a cubic: a constant width
            road.widths[lane] = [(0.0, float(np.interp(lo, s, width)), 0.0, 0.0, 0.0)]
        else:
            road.widths[lane] = geometry.cubic_segments(s[inside] - lo, width[inside], WIDTH_TOLERANCE)
    # The offset is half the width, so the lane is centred on the reference. Use exactly the same polynomials:
    # a second fit across section boundaries has different PCHIP slopes and can shift the lane centre by half a metre.
    road.lane_offset = [(lo + ds, a / 2, b / 2, c / 2, d / 2)
                        for lane, lo in zip(road.right, starts)
                        for ds, a, b, c, d in road.widths[lane]]

def settle_surface(road: Road, update_widths: Callable[[bool], None]) -> None:
    """
    Keep lane borders from folding back on themselves in tight turns.

    A border at a constant lateral offset t turns back where the reference line's curvature k reaches 1 / t. A road
    counts as folding once k * t reaches FOLD_LIMIT at any sampled station (every FOLD_STEP and at geometry joints);
    a road that does not fold is left unchanged. Short sharp bends of the Apollo polylines are smoothed with a
    clamped cubic B-spline that keeps both ends and their headings. Each candidate is judged with its widths
    refitted but not narrowed. In order:

    1. Spacing: a wider control spacing usually turns less but moves further. If the widest spacing keeps the
       borders apart, bisection between the narrowest and widest spacing finds one that does; it is the narrowest
       such spacing only if the fold ratio falls steadily with spacing, which is not guaranteed. From there a local
       search over the spacing and the two endpoint tangent lengths, which the adjacent roads leave free, looks for
       a curve that moves the reference less while no border folds.
    2. Nearby curves: if both the narrowest and the widest spacing fold, a global search looks among curves moving
       the reference at most REPAIR_MAX_SHIFT for one that does not; one found is refined as in 1, which never ends
       on a curve that moves further.
    3. Narrowing: if that search finds no fold-free curve, as in a U-turn, a path road is narrowed on the curve it
       found that needs the least width removed (by area, among the curves evaluated), or else on the original
       reference. Both searches are of limited size, so finding nothing does not prove that no curve exists.

    A fold that none of this removes is only logged as a warning; this function raises no exception for it.

    :param Road road: a road with its reference line and widths set. road.reference may be replaced by a smoothed
                      curve, the widths are refitted through update_widths, and road.geometry_adjustment records
                      how far the reference moved (Hausdorff distance, m) when it changes
    :param Callable[[bool], None] update_widths: refits the widths (and, for a path road, its sections and lane
                                                 offset) on the current road.reference; its argument asks to
                                                 narrow tight turns, which only path_widths honours
    """
    if not _folds(road):
        return
    original = road.reference
    original_line = LineString(original.xy)
    bounds = [np.log(SMOOTHING_SPACING_RANGE)] + 2 * [np.log(END_SCALE_RANGE)]  # (lower, upper) of each entry of p

    def shape(p):  # p: logarithms of the control spacing and of the two endpoint tangent lengths over their default
        try:
            return original.smoothed(float(np.exp(p[0])), tuple(np.exp(p[1:])))
        except ValueError:
            return None

    # side effect: a valid shape is left on the road as road.reference with its unnarrowed widths; an invalid one
    # leaves road.reference None and the widths as they were
    def fold_ratio(p):
        road.reference = shape(p)
        if road.reference is None:
            return np.inf
        update_widths(False)
        return _fold_ratio(road)

    def shift():
        return original_line.hausdorff_distance(LineString(road.reference.xy))

    def cost(p):  # fold-free: the movement; folding: above INFEASIBLE, lower the less it folds
        ratio = fold_ratio(p)
        return INFEASIBLE + min(ratio, INFEASIBLE) if ratio >= FOLD_LIMIT else shift()

    # Tiers, lowest first, all within REPAIR_MAX_SHIFT unless noted:
    #   fold-free                          -> the movement, below INFEASIBLE
    #   narrowing keeps the borders apart  -> INFEASIBLE + removed width area, the area counted up to INFEASIBLE - 1
    #   still folds after narrowing        -> 2 * INFEASIBLE + fold ratio
    #   moved more than REPAIR_MAX_SHIFT   -> 3 * INFEASIBLE + the excess
    #   no valid shape                     -> 4 * INFEASIBLE
    def nearby_cost(p):
        ratio = fold_ratio(p)
        if ratio == np.inf:
            return 4 * INFEASIBLE
        excess = shift() - REPAIR_MAX_SHIFT
        if excess > 0:
            return 3 * INFEASIBLE + min(excess, INFEASIBLE)
        if ratio < FOLD_LIMIT:
            return shift()
        full = dict(road.widths)  # the widths before narrowing, to measure how much narrowing removes
        update_widths(True)
        if _folds(road):
            return 2 * INFEASIBLE + min(_fold_ratio(road), INFEASIBLE)
        return INFEASIBLE + min(_removed_width(road, full), INFEASIBLE - 1)

    # Spacing only, at the default tangent lengths (log 1 = 0): low = narrowest spacing, high = widest.
    low, high = np.array([bounds[0][0], 0.0, 0.0]), np.array([bounds[0][1], 0.0, 0.0])
    if fold_ratio(low) < FOLD_LIMIT:            # even the narrowest spacing keeps the borders apart
        high = low
    elif fold_ratio(high) < FOLD_LIMIT:         # bisect towards a narrow spacing that does; high never folds
        for _ in range(SPACING_BISECTIONS):
            middle = (low + high) / 2
            if fold_ratio(middle) < FOLD_LIMIT:
                high = middle
            else:
                low = middle
    else:                                       # both end spacings fold: search spacing and tangents together
        high, value = _search_box(nearby_cost, low, bounds, INFEASIBLE)
        if value >= INFEASIBLE:  # the search found no fold-free curve within REPAIR_MAX_SHIFT
            # value in [INFEASIBLE, 2 * INFEASIBLE): narrowing that curve keeps its borders apart
            least, high = shape(high) if value < 2 * INFEASIBLE else None, None
    if high is not None:   # refine from the fold-free point for less movement; never ends worse than high
        attempts = [(shape(_minimize(cost, high, bounds, 1e-3)[0]), False)]
    else:
        attempts = [(least, True)] if least else []
    # The first attempt that keeps the borders apart wins; the last resort narrows the original reference.
    for reference, narrow in attempts + [(original, True)]:
        road.reference = reference
        update_widths(narrow)
        if not _folds(road):
            if reference is not original:
                road.geometry_adjustment = LineString(original.xy).hausdorff_distance(LineString(reference.xy))
            action = ' and '.join(([] if reference is original else ['smoothed a sharp bend']) +
                                  (['narrowed a lane'] if narrow else []))
            log.info('Road %d: %s so that no lane border folds (reference moved %.3f m)', road.id, action,
                     road.geometry_adjustment)
            return
    log.warning('Road %d: a lane border still folds in a tight turn', road.id)

def fit_curved_path_borders(road: Road, end_widths: Dict[bool, float]) -> None:
    """
    Fit the drawn borders when a wide, tight path cannot stay centred on its reference.

    Apollo supplies its centre, borders, and width independently. On a tight bend those
    three curves need not describe a centred normal-offset strip: keeping the lane
    centred can make the inside border stall at a junction even if it does not fold.
    This tries to improve that: move the reference only slightly and fit laneOffset towards
    both drawn borders, with the widths refitted (not narrowed, ends matching the neighbours)
    and the exact lane ends shared with neighbouring roads kept. A local search of limited
    size over the reference's two endpoint tangent lengths picks, among the candidates it
    evaluates that keep every constraint, the one with the lowest weighted border and centre
    error; it is not guaranteed to be the closest fit, and no check confirms the stall is gone.

    Only a single-lane path joined at both ends (only path roads have incoming and outgoing
    lanes), 4 to 12 m long, at least 4 m wide and turning tightly near its start, in either
    direction, whose drawn right border lies outside the centred one, is fitted; any other road
    returns at once. A fit is kept only if it moves the reference at most REPAIR_MAX_SHIFT,
    keeps the centre within 0.15 m and the right border within 0.25 m of Apollo's, and passes
    the sampled fold check; the left border error only enters the score. Otherwise the road is
    left as it came in. A kept fit replaces any narrowing settle_surface applied.

    :param Road road: a path road after settle_surface, with its reference, widths, lane offset and
                      sections set. A kept fit replaces road.reference, road.widths,
                      road.lane_offset and road.section_starts, and raises road.geometry_adjustment
                      to this fit's reference movement (m) if that is larger; it is the largest
                      single correction, not the movement summed over corrections
    :param Dict[bool, float] end_widths: at_end to the width in m of the attached lane where the path
                                         meets it, as for path_widths; at_end is True at s = length
    """
    # Only a single-lane path joined at both ends, 4 to 12 m long ...
    if len(road.right) != 1 or road.incoming is None or road.outgoing is None:
        return
    original = road.reference
    if not 4.0 < original.length < 12.0:
        return
    lane = road.right[0]
    probe = np.array([1.0, 2.0])  # stations, m from the start
    width = cubic_values(road.widths[lane], probe)  # (2,)
    # ... at least 4 m wide and turning tightly there (half-width x curvature at least 0.8) ...
    if width[0] < 4.0 or np.max(np.abs(original.curvature(probe) * width / 2)) < 0.8:
        return
    # ... whose drawn right border lies beyond the centred one (offset -width / 2): the hook this fit corrects.
    drawn_inside = original.lateral_offsets(probe, lane.right.points)  # (2,), signed, left positive
    discrepancy = drawn_inside + width / 2
    if not (-0.6 < discrepancy[0] < -0.08 and discrepancy[1] < -0.2):
        return

    original_widths = road.widths
    original_offset = road.lane_offset
    original_starts = road.section_starts
    # Candidate fits overwrite all these fields, including previously narrowed widths.
    road.widths = road.widths.copy()
    original_line = LineString(original.xy)
    source_inside = LineString(lane.right.points[:, :2])
    source_outside = LineString(lane.left.points[:, :2])
    source_center = LineString(lane.center[:, :2])
    # cost and (reference, widths, lane offset, section starts, shift) of the best fit keeping every constraint
    best = [np.inf, None]

    def cost(p):  # p: (2,) logarithms of the two endpoint tangent lengths over their default; 4 m control spacing
        # Every candidate retains the exact endpoint positions and directions already attached to the adjacent roads.
        try:
            candidate = original.smoothed(4.0, tuple(np.exp(p)))
        except ValueError:
            return 2 * INFEASIBLE
        shift = original_line.hausdorff_distance(LineString(candidate.xy))
        road.reference = candidate
        path_widths(road, end_widths)  # Apollo's widths on this candidate, not narrowed
        s = np.unique(np.r_[np.arange(0.0, candidate.length, FOLD_STEP), candidate.length])  # (S,) stations
        width = cubic_values(road.widths[lane], s)
        # (S,) signed offsets (left positive) of the drawn borders and centre; "inside" is the lane's right border
        inside = candidate.lateral_offsets(s, lane.right.points)
        outside = candidate.lateral_offsets(s, lane.left.points)
        drawn_center = candidate.lateral_offsets(s, lane.center)
        # Use the drawn borders to correct the hook, but keep the lane
        # centre close to Apollo's centre rather than shifting the whole
        # lane to make one border exact.
        border_target = 0.7 * inside + 0.3 * (outside - width)
        inside = 0.4 * border_target + 0.6 * (drawn_center - width / 2)
        curvature = candidate.curvature(s)
        # Keep the inside offset away from the local turning radius. The polynomial fit below
        # is checked again by _fold_ratio, which adds the geometry joints to these samples.
        limit = 0.95 / np.maximum(np.abs(curvature), 1e-9)
        inside = np.clip(inside, -limit, limit)
        inside[0], inside[-1] = -width[0] / 2, -width[-1] / 2  # ends stay centred, where the neighbours join
        # laneOffset is the lane's left border; its right border lies at laneOffset - width
        road.lane_offset = geometry.cubic_segments(s, inside + width, WIDTH_TOLERANCE / 2)
        offset = cubic_values(road.lane_offset, s)
        x, y, hdg = candidate.at(s)
        normal = np.column_stack([-np.sin(hdg), np.cos(hdg)])  # (S, 2) unit left normals
        center = np.column_stack([x, y])                         # (S, 2) reference points
        fitted_inside = LineString(center + (offset - width)[:, None] * normal)
        fitted_outside = LineString(center + offset[:, None] * normal)
        fitted_center = LineString(center + (offset - width / 2)[:, None] * normal)
        inside_error = fitted_inside.hausdorff_distance(source_inside)
        outside_error = fitted_outside.hausdorff_distance(source_outside)
        center_error = fitted_center.hausdorff_distance(source_center)
        ratio = _fold_ratio(road)
        # constraints: reference moved at most 0.25 m, centre within 0.15 m and right border within 0.25 m of
        # Apollo's, and no fold; together, how far they are broken
        violation = (max(0.0, shift - REPAIR_MAX_SHIFT) + max(0.0, center_error - 0.15) +
                     max(0.0, inside_error - 0.25) + max(0.0, min(ratio, 2.0) - FOLD_LIMIT))
        if violation > 0 or ratio >= FOLD_LIMIT:
            return INFEASIBLE + min(violation, INFEASIBLE - 1)
        score = inside_error + outside_error + 2 * center_error + 0.3 * shift
        if score < best[0]:
            best[:] = score, (candidate, road.widths.copy(), road.lane_offset, road.section_starts, shift)
        return score

    try:
        _minimize(cost, np.zeros(2), 2 * [np.log(END_SCALE_RANGE)], 1e-3)
    finally:
        road.reference = original
        road.widths = original_widths
        road.lane_offset = original_offset
        road.section_starts = original_starts
    if best[1] is None:
        return
    road.reference, road.widths, road.lane_offset, road.section_starts, shift = best[1]
    road.geometry_adjustment = max(road.geometry_adjustment, shift)
    log.info('Road %d: fitted drawn borders within %.3f m reference movement', road.id, shift)

def _search_box(cost, start: np.ndarray, bounds: Sequence[Tuple[float, float]],
                enough: float) -> Tuple[Optional[np.ndarray], float]:
    """
    The lowest-cost point found anywhere in a box, and its cost.

    Meant for costs whose low region is small and surrounded by a kinked landscape, where a local search from one
    point stalls. Differential evolution evolves a population spread over the whole box, including ``start``, with
    a fixed seed so that conversions are reproducible.

    With P parameters it evaluates at most (GLOBAL_GENERATIONS + 1) x GLOBAL_POPULATION x P points, 72 for P = 3.
    Once a cost below ``enough`` has been seen, the search stops at the end of the current generation, not at once.

    :param Callable[[np.ndarray], float] cost: cost of a point, shape (P,)
    :param np.ndarray start: a point placed in the first population, shape (P,)
    :param Sequence[Tuple[float, float]] bounds: (low, high) of each of the P parameters
    :param float enough: a cost below which the search stops early
    :returns: (point, cost) with the lowest cost evaluated; the point has shape (P,), or is None if every
              cost was inf
    :rtype: Tuple[Optional[np.ndarray], float]
    """
    best = [np.inf, None]  # lowest cost evaluated so far, and its point

    def tracked(p):
        value = cost(p)
        if value < best[0]:
            best[:] = value, np.array(p, dtype=float)
        return value

    # the callback runs after each generation; returning True there stops the search
    differential_evolution(tracked, bounds, x0=start, popsize=GLOBAL_POPULATION, maxiter=GLOBAL_GENERATIONS, seed=0,
                           polish=False, callback=lambda *_, **__: best[0] < enough)
    return best[1], best[0]

def _minimize(cost: Callable[[np.ndarray], float], start: np.ndarray,
              bounds: Sequence[Tuple[float, float]],
              tolerance: float) -> Tuple[np.ndarray, float]:
    """
    The lowest-cost point found in a box, from ``start``, and its cost.

    The costs here are not smooth (largest curvatures, Hausdorff distances, a step where a constraint starts to hold),
    so Nelder-Mead is used, which needs no gradients.

    Its first simplex spans a quarter of each range, so the first steps already look well away from ``start``.
    SciPy's default simplex would step only 0.00025 from a start at 0, the usual start here.

    It evaluates the cost at most SEARCH_BUDGET times, and stops earlier only once the simplex is within 0.01 in
    every parameter and its costs are within ``tolerance`` of each other. The best vertex never gets worse, so the
    result costs no more than ``start``; it is a local result, not a guaranteed minimum.

    :param Callable[[np.ndarray], float] cost: cost of a point, shape (P,)
    :param np.ndarray start: the first vertex, shape (P,); clipped into the box
    :param Sequence[Tuple[float, float]] bounds: (low, high) of each of the P parameters
    :param float tolerance: largest spread of the simplex's costs at which the search may stop
    :returns: (point, cost) of the best vertex found; the point has shape (P,)
    :rtype: Tuple[np.ndarray, float]
    """
    lo, hi = np.asarray(bounds, dtype=float).T  # (P,) each
    start = np.clip(np.asarray(start, dtype=float), lo, hi)
    simplex = [start]  # P + 1 vertices: start, then a quarter-range step along each parameter
    for k, step in enumerate((hi - lo) / 4):
        vertex = start.copy()
        vertex[k] += step if start[k] + step <= hi[k] else -step  # step backwards where forwards leaves the box
        simplex.append(vertex)
    result = minimize(cost, start, method='Nelder-Mead', bounds=list(zip(lo, hi)),
                      options={'initial_simplex': np.array(simplex), 'maxfev': SEARCH_BUDGET,
                               'xatol': 1e-2, 'fatol': tolerance})
    return result.x, float(result.fun)

def _folds(road: Road) -> bool:
    """
    Whether a lane border counts as folding: curvature times its lateral offset reaches FOLD_LIMIT.
    The limit is kept just below 1, where a border at a constant offset turns back, to leave a margin.
    Only the stations sampled by :func:`_fold_ratio` are checked, every FOLD_STEP and at geometry joints.

    :param Road road: a road with its reference line, widths, lane offset and sections set
    :returns: True if the largest curvature x offset at the sampled stations is at least FOLD_LIMIT
    :rtype: bool
    """
    return _fold_ratio(road) >= FOLD_LIMIT

def _fold_ratio(road: Road) -> float:
    """
    Largest curvature times lateral offset of a lane border, over the sampled stations of every lane section.

    A border at a constant offset turns back where this reaches 1; :func:`_folds` uses FOLD_LIMIT, just below it.
    At one station curvature x offset grows with the offset towards the inside of the turn, so only the outermost
    border on each side is checked: the left one for a left turn (k > 0), the right one for a right turn. Each
    section is evaluated with its own lanes, every FOLD_STEP and at geometry joints, including both its ends.

    :param Road road: a road with its reference line, widths, lane offset and sections set
    :returns: the largest curvature x offset, dimensionless; 0 on a straight road, -inf if it has no sections
    :rtype: float
    """
    reference = road.reference
    # every FOLD_STEP, plus the geometry joints where curvature jumps
    s = np.unique(np.r_[np.arange(0.0, reference.length, FOLD_STEP), reference.vertex_s, reference.length])
    ratio = -np.inf
    for lo, right, left in road.sections:
        _, hi = road.lane_range((right or left)[0])  # the section's end
        local = np.unique(np.r_[lo, s[(s >= lo) & (s <= hi)], hi])  # this section's stations, both ends included
        curvature = reference.curvature(local)
        offset = cubic_values(road.lane_offset, local)
        # Both sections may probe a joint, but each has its own complete borders.
        low = offset - sum(cubic_values(road.widths[lane], local - lo) for lane in right)   # outer right border
        high = offset + sum(cubic_values(road.widths[lane], local - lo) for lane in left)   # outer left border
        # a left turn (k > 0) folds the left border, a right turn the right one: k * offset is positive for both
        ratio = max(ratio, float(np.max(np.maximum(curvature * low, curvature * high))))
    return ratio

def _removed_width(road: Road, full: Dict[Lane, list]) -> float:
    """
    Area by which the road's lane widths fall short of ``full``, its lane widths before narrowing.

    Only removed width counts: where a lane is wider than in ``full``, nothing is subtracted. Each lane is
    integrated over its own range with the trapezoid rule, at most FOLD_STEP between samples. settle_surface
    uses this to pick the narrowing that removes least.

    :param Road road: the road after narrowing, with its widths and sections set
    :param Dict[Lane, list] full: lane to its (sOffset, a, b, c, d) width records before narrowing
    :returns: the removed area in m², summed over the lanes
    :rtype: float
    """
    area = 0.0
    for lane in road.lanes:
        lo, hi = road.lane_range(lane)
        s = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / FOLD_STEP)) + 1))  # (K,) at most FOLD_STEP apart
        # (K,) width removed at each station; width records are relative to the start of the lane's section
        cut = np.maximum(cubic_values(full[lane], s - lo) - cubic_values(road.widths[lane], s - lo), 0.0)
        area += float(trapezoid(cut, s))
    return area

def _turnable(road: Road, s: np.ndarray, width: np.ndarray) -> np.ndarray:
    """
    Narrow a path road's lane where it turns tighter than its half-width allows.

    The lane is centred on its reference line, so its inner border folds where the turning radius falls to half
    the width. Each interior sample is capped at 2 x NARROW_LIMIT x the tightest turning radius over its two
    neighbouring intervals, read at stations every FOLD_STEP and at geometry joints. The shape-preserving cubic
    through the capped samples then stays within the cap between interior samples (cubic_segments may move it by
    up to WIDTH_TOLERANCE). The end samples keep their width so that they meet the neighbours; the intervals next
    to the ends are therefore not limited, and a sharp bend there can still fold.

    :param Road road: a path road with its reference line set
    :param np.ndarray s: stations of the width samples in m, shape (M,), ascending
    :param np.ndarray width: lane width at each station in m, shape (M,)
    :returns: the capped widths, a new array of shape (M,); width itself is not changed
    :rtype: np.ndarray
    """
    reference = road.reference
    fine = np.unique(np.r_[np.arange(0.0, reference.length, FOLD_STEP), reference.vertex_s, reference.length])
    curvature = np.abs(reference.curvature(fine))
    # widest a centred lane may be at each fine station: twice NARROW_LIMIT times the turning radius
    limit = np.where(curvature > 1e-9, 2 * NARROW_LIMIT / np.maximum(curvature, 1e-9), np.inf)
    capped = width.copy()
    for i in range(1, len(s) - 1):  # the end samples are left alone: they meet the neighbours
        window = (fine >= s[i - 1]) & (fine <= s[i + 1])
        if window.any():
            capped[i] = min(capped[i], limit[window].min())
    return capped

def _apollo_width(road: Road, lane: Lane, s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Width of a path lane over its lane section [lo, hi], before any end matching or narrowing.

    Apollo's width samples, given over the fraction of the lane's own length, are spread over the section, so they
    stretch or shrink with it when the section is not as long as the Apollo lane; a zero-length section takes the
    first sample. A lane without samples is measured as the distance between its two boundaries along the
    reference's normals; that can be negative where they cross, which path_widths clamps to zero.

    :param Road road: the path road, with its reference line set
    :param Lane lane: the lane of this section
    :param np.ndarray s: stations in m within [lo, hi], shape (K,)
    :param float lo: start of the section along the reference, m
    :param float hi: end of the section along the reference, m
    :returns: the lane width in m at each station, shape (K,)
    :rtype: np.ndarray
    """
    if lane.width_samples is not None and len(lane.width_samples):
        fraction = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)  # (K,) position in the section, 0..1
        return np.interp(fraction, lane.width_samples[:, 0], lane.width_samples[:, 1])
    # signed offsets, left positive: left border minus right border is the width
    return (road.reference.lateral_offsets(s, lane.left.points) -
            road.reference.lateral_offsets(s, lane.right.points))

def _nonnegative(lane: Lane, samples: np.ndarray) -> np.ndarray:
    """
    Width samples with every negative value clamped to zero.

    Negative widths arise where Apollo's boundaries cross or an end correction overshoots. A warning is logged only
    below -1e-6 m, so rounding noise is clamped silently.

    :param Lane lane: the lane named in the warning; path_widths passes the path's first lane for its whole profile
    :param np.ndarray samples: widths in m, shape (M,)
    :returns: the clamped widths, a new array of shape (M,); samples itself is not changed
    :rtype: np.ndarray
    """
    if np.min(samples) < -1e-6:
        log.warning('Lane %s has a negative sampled width (minimum %.3f m); clamping those samples to zero',
                    lane.id, np.min(samples))
    return np.maximum(samples, 0.0)

def fit_elevation(road: Road, reference: np.ndarray) -> None:
    """
    Elevation records for the road: a piecewise-linear profile through the z of its source polyline.

    The source points keep their heights but are placed at their stations on the final reference line, which
    corner rounding or surface repair may have moved. Each point is projected onto it; a projection that would
    step back is held at the previous station, the first and last points are pinned to 0 and the road's length,
    and of points sharing a station only the last is kept. The knots are then thinned with Shapely's simplify in
    the (s, z) plane, which keeps every remaining point within ELEVATION_TOLERANCE measured across the profile;
    vertically that is about sqrt(1 + g^2) times as much on a local grade g. A source without any height change
    gives one constant record.

    :param Road road: a road with its final reference line set; road.elevation is replaced
    :param np.ndarray reference: the source polyline, shape (N, 3), x, y and z in m: a plain road's source
                                 polyline, or a path road's Apollo centre line
    :returns: None; road.elevation becomes a list of (s, a, b, 0, 0) records, a + b ds, one per linear piece
    :rtype: None
    """
    road.elevation = []
    if np.ptp(reference[:, 2]) <= 1e-12:
        road.elevation = [(0.0, float(reference[0, 2]), 0.0, 0.0, 0.0)]
        return
    # Source z is independent of horizontal corner rounding or surface repair. Map its
    # stations to the final reference instead of assigning a corner's z to the whole arc.
    # (N,) station of each source point on the final reference; a projection never steps back
    s = np.maximum.accumulate([road.reference.project(point)[0] for point in reference])
    s[0], s[-1] = 0.0, road.reference.length  # the profile covers the whole road
    keep = np.r_[np.diff(s) > 1e-9, True]     # of points sharing a station, keep the last
    s, reference = s[keep], reference[keep]
    # knots of a piecewise-linear profile within ELEVATION_TOLERANCE of the source heights
    kept = np.asarray(LineString(np.column_stack([s, reference[:, 2]])).simplify(ELEVATION_TOLERANCE).coords) \
        if len(s) > 1 and s[-1] > 0 else np.array([[0.0, reference[0, 2]]])
    for (s0, z0), (s1, z1) in zip(kept[:-1], kept[1:]):  # one linear record (a = z0, b = slope) per piece
        road.elevation.append((float(s0), float(z0), float((z1 - z0) / (s1 - s0)) if s1 > s0 else 0.0, 0.0, 0.0))
    if not road.elevation:
        road.elevation.append((0.0, float(kept[0, 1]), 0.0, 0.0, 0.0))
