"""
Geometry module for the reverse engineering of apollo hdmap.
"""

import numpy as np
from shapely.geometry import LineString
from typing import Optional, Tuple, List
from .polyline import dedupe, arc_lengths
from .constraint import Chords
from .plan_geometry import PlanGeometry
from .b_spline import Spline

#################################### Configuration Space ####################################

FIT_KNOT_DEVIATION = 0.2
"""m, the initial knots are the vertices a simplification to this tolerance keeps"""
FIT_KNOT_GAP = 0.05
"""m, vertices closer than this along the polyline share a knot"""
FIT_MAX_KNOTS_PER_VERTEX = 4
"""knots allowed per polyline vertex (plus a few), so refinement always ends"""
FIT_MAX_REFINE = 40
"""maximum number of refinement iterations"""
FIT_SMOOTHING = (1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
"""m^3, curvature penalties tried, smoothest first"""
FIT_MIN_SPAN = 0.05
"""m, spans are not halved below this"""
FIT_TOLERANCE = 0.05
"""m, largest distance of a polyline vertex from the fitted plan view"""

##############################################################################################

def significant(pts: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Indices of the vertices a Douglas-Peucker simplification keeps. These are the initial spline breakpoints: a vertex
    is kept where the polyline bends by more than ``tolerance``, and dropped along a straight stretch.

    :param np.ndarray pts: polyline vertices, shape (N, 2)
    :param float tolerance: max distance, in m, from a dropped vertex to the simplified polyline
    :returns: indices into ``pts``, shape (S,) with ``2 <= S <= N``, sorted, always including 0 and N - 1
    :rtype: np.ndarray
    """
    # simplify returns a subset of the original coordinates, in order
    kept = np.asarray(LineString(pts).simplify(tolerance, preserve_topology=False).coords)
    index, j = [], 0
    for i, p in enumerate(pts):
        # exact match: the kept points are original vertices, not new samples
        if j < len(kept) and p[0] == kept[j][0] and p[1] == kept[j][1]:
            index.append(i)
            j += 1
    # endpoints stay even if a match was missed; unique also sorts
    return np.unique(np.r_[0, index, len(pts) - 1])

def unit(v: np.ndarray) -> np.ndarray:
    """
    The same direction as ``v``, scaled to length 1. Used as an end heading when no angle is given.

    :param np.ndarray v: a non-zero direction, shape (2,)
    :returns: unit vector, shape (2,)
    :rtype: np.ndarray
    """
    return v / np.linalg.norm(v)

def fit_plan_view(points: np.ndarray, tolerance: float = FIT_TOLERANCE, start_heading: Optional[float] = None,
                  end_heading: Optional[float] = None) -> Tuple[List[PlanGeometry], np.ndarray]:
    """
    The smoothest cubic spline within ``tolerance`` of a polyline, as OpenDRIVE plan-view geometries.

    :param np.ndarray points: polyline vertices, shape (N, 2) or (N, 3); only x, y are used
    :param float tolerance: max distance, in m, from a vertex to the fitted curve
    :param Optional[float] start_heading: start heading in radians, or None to follow the first segment
    :param Optional[float] end_heading: end heading in radians, or None to follow the last segment
    :returns: ``(geometries, s)``. ``s`` is float64, shape ``(G + 1,)``, ``G = len(geometries)``:
                the reference-line arc length at each geometry's start, then the total length. 
                Fewer than two points gives ``G = 0`` and ``s`` of shape ``(1,)``
    :rtype: Tuple[List[PlanGeometry], np.ndarray]
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Geometry tolerance must be finite and positive')
    pts = dedupe(np.asarray(points, dtype=float)[:, :2])
    if len(pts) < 2:
        return [], np.array([0.0])
    vertex_u = arc_lengths(pts)
    length = float(vertex_u[-1])
    # a neighbour passes its heading in; otherwise the end segment's direction is the heading
    ends = tuple(np.array([np.cos(h), np.sin(h)]) if h is not None else unit(d)
                 for h, d in ((start_heading, pts[1] - pts[0]), (end_heading, pts[-1] - pts[-2])))
    chords = Chords(pts, vertex_u, ends, tolerance)
    # initial knots only where the polyline bends; endpoints are added separately
    breaks = [0.0]
    for u in vertex_u[significant(pts, FIT_KNOT_DEVIATION)][1:-1]:
        if u - breaks[-1] >= FIT_KNOT_GAP and length - u >= FIT_KNOT_GAP:
            breaks.append(float(u))
    breaks = np.array(breaks + [length])
    best = None  # (miss, spline, control) of the closest fit, used if every attempt still exceeds tolerance
    for _ in range(FIT_MAX_REFINE):
        spline = Spline(breaks, pts, vertex_u, ends, chords)
        # largest penalty first: the smoothest curve that still stays inside both bounds
        for smoothing in FIT_SMOOTHING:
            control = spline.solve(smoothing)
            vertex, excess = spline.span_errors(control, chords)
            split = (vertex > tolerance) | (excess > 0)
            if not split.any():
                return spline.geometries(control)
            miss = max(float(vertex.max()) - tolerance, float(excess.max()))
            if best is None or miss < best[0]:
                best = (miss, spline, control)
        # a span shorter than 2 * FIT_MIN_SPAN cannot be halved again
        split &= np.diff(breaks) > 2 * FIT_MIN_SPAN
        if not split.any() or len(breaks) + split.sum() > FIT_MAX_KNOTS_PER_VERTEX * len(pts) + 10:
            break
        breaks = np.sort(np.r_[breaks, (breaks[:-1][split] + breaks[1:][split]) / 2])
    return best[1].geometries(best[2])
