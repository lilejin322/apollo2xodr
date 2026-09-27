"""Polyline utilities and OpenDRIVE geometry fitting (NumPy + Shapely).

Polylines are ``(N, 3)`` float arrays; only x/y take part in plan-view geometry, z is carried along for elevation.
"""

import shapely
import numpy as np
from typing import Optional
from shapely.geometry import LineString

def dedupe(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Drop consecutive points closer than ``eps`` (x/y), always keeping the last point.
    The reason why no use of `shapely` is that it does not preserve the last point, e.g.
               A ------- B ------- C - D
                                     ↑ too close
            Shapely's dedupe:
               A ------- B ------- C
            We need to keep the last point D during dedupe:
               A ------- B ----------- D
    
    :param np.ndarray points: the input points
    :param float eps: the tolerance for deduplication
    :returns: the deduplicated points
    :rtype: np.ndarray
    """
    if len(points) < 2:    # less than 2 points, no depulication needed
        return points
    step = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    keep = np.concatenate([[True], step > eps])
    if not keep[-1]:       # keep for the last point
        keep[np.flatnonzero(keep)[-1]] = False
        keep[-1] = True
    return points[keep]

def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Simplify x/y while also bounding the vertical interpolation error at source vertices.

    :param np.ndarray points: the input points
    :param float tolerance: the tolerance for simplification
    :returns: the simplified points
    :rtype: np.ndarray
    """
    points = dedupe(points)                  # deduplicate points in x/y
    if tolerance <= 0 or len(points) < 3:
        return points
    kept = np.asarray(LineString(points[:, :2]).simplify(tolerance, preserve_topology=False).coords)
    mask = np.zeros(len(points), dtype=bool)
    j = 0
    for i, p in enumerate(points[:, :2]):
        if j < len(kept) and p[0] == kept[j][0] and p[1] == kept[j][1]:
            mask[i] = True
            j += 1
    mask[0] = mask[-1] = True
    if points.shape[1] > 2:                  # if there is elevation z, check the vertical error
        s = arc_lengths(points)
        knots = np.flatnonzero(mask)
        stack = list(zip(knots[:-1], knots[1:]))
        while stack:                         # Douglas–Peucker algrithm without recursion
            a, b = stack.pop()
            if b <= a + 1:
                continue
            z = np.interp(s[a + 1:b], s[[a, b]], points[[a, b], 2])
            error = abs(z - points[a + 1:b, 2])
            k = a + 1 + int(np.argmax(error))
            if error[k - a - 1] > tolerance:
                mask[k] = True
                stack.extend(((a, k), (k, b)))
    return points[mask]

def slice_polyline(points: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Slice at x/y arc lengths, interpolating z as well as x/y at the new ends.
    Note that the z is interpolated as well as x/y at the new ends, that's why no shapely used here.
    Original polyline, parameterized by XY arc length s:
                        P0 -------- P1 ------------- P2 -------- P3
                        s=0        s=3              s=8         s=12
                        z=0        z=1              z=3         z=4
                                    |<-- keep ------>|

                        slice_polyline(points, start=2, end=10)
                                     start                     end
                                       ↓                        ↓
                        P0 ----------- A -- P1 -------- P2 ---- B ----- P3
                        s=0           s=2 s=3          s=8    s=10    s=12
                                    ↑                        ↑
                                interpolated             interpolated
                                (x, y, z)                (x, y, z)

                        Result:
                        A -------- P1 ------------- P2 -------- B
                        s=2       s=3              s=8         s=10
    :param np.ndarray points: the input points
    :param float start: the start x/y arc length
    :param float end: the end x/y arc length
    :returns: the sliced points
    :rtype: np.ndarray
    """
    s = arc_lengths(points)
    samples = np.r_[start, s[(s > start) & (s < end)], end]
    return np.column_stack([np.interp(samples, s, points[:, k]) for k in range(points.shape[1])])

def arc_lengths(points: np.ndarray) -> np.ndarray:
    """
    Cumulative x/y arc length at every vertex.
    
    :param np.ndarray points: the input points
    :returns: the cumulative x/y arc length at every vertex
    :rtype: np.ndarray
    """
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))])

def end_direction(points: np.ndarray, at_start: bool) -> Optional[np.ndarray]:
    """
    Unit x/y direction of the first (or last) non-degenerate segment, pointing along the polyline.
    Example:
            polyline order:
                                P0 -----> P1
                                        |
                                        |
                                        v
                                        P2 -----> P3 -----> P4

            1) at_start=True:
                                P0 -----> P1
                                     d            direction = d / ||d||
            2) at_start=False: Points are traversed in reverse, so internally:
                                P3 <----- P4
                                     d            direction = -d / ||d||

    :param np.ndarray points: the input points
    :param bool at_start: whether to get the direction at the start or end
    :returns: the unit x/y direction
    :rtype: Optional[np.ndarray]
    """
    pts = points[:, :2] if at_start else points[::-1, :2]
    for q in pts[1:]:
        d = q - pts[0]
        n = np.hypot(*d)
        if n > 1e-9:
            return d / n if at_start else -d / n
    return None

def extend(points: np.ndarray, length: float) -> np.ndarray:
    """
    Extend both ends straight along their end directions (z kept constant).
           head <---- P0 ---- P1 ---- P2 ---- P3 ----> tail
                         d0 ->           d1 ->
               head = P0 - d0 * length                tail = P3 + d1 * length

    :param np.ndarray points: the input points
    :param float length: the length to extend
    :returns: the extended points
    :rtype: np.ndarray
    """
    d0, d1 = end_direction(points, True), end_direction(points, False)
    if d0 is None or d1 is None:
        return points
    head = points[0].copy()
    head[:2] -= d0 * length
    tail = points[-1].copy()
    tail[:2] += d1 * length
    return np.vstack([head, points, tail])

def resample(points: np.ndarray, step: float) -> np.ndarray:
    """
    The polyline's vertices plus points every ``step`` metres (x/y arc length) in between.
    original:                         resampled:
    P0 o-------------o P1             P0 o----x----x----o P1
                      \                                 \
                       \                                 x
                        \                                 \
                         o P2---------o P3                 o P2---x-----o P3

    :param np.ndarray points: the input points
    :param float step: the sampling step
    :returns: the resampled points
    :rtype: np.ndarray
    """
    s = arc_lengths(points)
    extra = np.arange(step, s[-1], step)
    s_all = np.unique(np.concatenate([s, extra]))
    return np.column_stack([np.interp(s_all, s, points[:, k]) for k in range(points.shape[1])])

def midline(a: np.ndarray, b: np.ndarray, step: float = 1.0, tolerance: float = 0.02) -> np.ndarray:
    """
    Centre line between two polylines running in the same direction.
    ``a`` is sampled every ``step`` metres; every sample is paired with its closest point on ``b``.
    Sampling only one line keeps the result ordered even where the two lines start or end at different places.

    Example:
                                a:  A0 o------o------o------o A3
                                        \      \      \      \
                                mid:     x------x------x------x        -> we need to find this
                                          \      \      \      \
                                b:      B0 o------o------o------o B3
    
    :param np.ndarray a: the first polyline
    :param np.ndarray b: the second polyline
    :param float step: the sampling step for the first polyline, i.e., ``a`` in the Fig.
    :param float tolerance: the tolerance for the simplification
    :returns: the middle line
    :rtype: np.ndarray
    """
    samples = resample(dedupe(a), step)
    lb = LineString(b)
    on_b = shapely.get_coordinates(
        shapely.line_interpolate_point(lb, shapely.line_locate_point(lb, shapely.points(samples[:, :2]))),
        include_z=True)
    if on_b.shape[1] == 2 or np.isnan(on_b[:, 2]).any():
        on_b = np.column_stack([on_b[:, :2], np.interp(shapely.line_locate_point(lb, shapely.points(on_b)),
                                                       arc_lengths(b), b[:, 2])])
    return simplify((samples + on_b) / 2, tolerance)

def remove_end_kinks(points: np.ndarray, max_length: float = 1.0, max_turn_deg: float = 30.0) -> np.ndarray:
    """
    Drop the second (second-to-last) vertex while the first (last) segment is short and sharply bent.
    Averaging boundary end points where lanes connect can leave such hooks at the ends of a reference line.
    Start kink:
      p0                                 p0  \
        \                                      \
        p1 ----- p2 ----- p3               p1    \  p2 ----- p3
         ↑  p1 is the start kink, default > 30 deg

    End kink:
      p0 ------- p1 ------- p2           p0 ------- p1         p2
                            /                            \
                          p3                                  p3
    
    :param np.ndarray points: the input points
    :param float max_length: the maximum length of the first segment
    :param float max_turn_deg: the maximum turn angle in degrees
    :returns: the points without the end kinks
    :rtype: np.ndarray
    """
    cos_max = np.cos(np.radians(max_turn_deg))

    def bent(p0, p1, p2):
        d0, d1 = p1[:2] - p0[:2], p2[:2] - p1[:2]
        n0, n1 = np.hypot(*d0), np.hypot(*d1)
        return n0 < max_length and n1 > 0 and (n0 == 0 or float(d0 @ d1) / (n0 * n1) < cos_max)

    while len(points) > 2 and bent(points[0], points[1], points[2]):
        points = np.delete(points, 1, axis=0)
    while len(points) > 2 and bent(points[-1], points[-2], points[-3]):
        points = np.delete(points, -2, axis=0)
    return points

def heading_of(vec) -> float:
    """
    Calculate the heading of a vector

    :param np.ndarray vec: the input vector
    :returns: the heading of the vector, in radians
    :rtype: float
    """
    return float(np.arctan2(vec[1], vec[0]))

def reverses_direction(predecessor: np.ndarray, successor: np.ndarray, max_turn_deg: float = 120.0) -> bool:
    """
    Whether driving from the end of one polyline into the start of the other turns back more than ``max_turn``.
    Judge the reverse:
                                predecessor
                        ------------------------->
                                                  *
                                                 /
                                                /
                                      <---------
                                       successor

    :param np.ndarray predecessor: the predecessor polyline
    :param np.ndarray successor: the successor polyline
    :param float max_turn_deg: the maximum turn angle in deg
    :returns: whether the reverse is true
    :rtype: bool
    """
    a, b = end_direction(predecessor, False), end_direction(successor, True)
    return a is not None and b is not None and float(a @ b) < np.cos(np.radians(max_turn_deg))
