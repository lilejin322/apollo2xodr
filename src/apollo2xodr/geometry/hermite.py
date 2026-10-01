"""
Hermite curves submodule for geometry.
"""

import numpy as np
from functools import cached_property
from typing import Tuple, List
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator, PPoly
from scipy.optimize import minimize
from shapely.geometry import LineString
from .polyline import heading_of, arc_lengths, end_direction
from .plan_geometry import _GAUSS_P, _GAUSS_W

class HermiteCurve:
    """
    One XY cubic on the normalized parameter interval ``t = 0..1``.
    The instance owns its endpoint data and one SciPy spline, expressed relative
    to p0 to keep derivatives well conditioned at large world coordinates.
    """

    def __init__(self, p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> None:
        """
        Constructor

        :param np.ndarray p0: the start point, shape (2,) or (3,)
        :param np.ndarray p1: the end point, shape (2,) or (3,)
        :param np.ndarray m0: XY derivative at t=0, shape (2,)
        :param np.ndarray m1: XY derivative at t=1, shape (2,)
        """
        self._start = np.array(p0, dtype=float, copy=True)
        self._end = np.array(p1, dtype=float, copy=True)
        if self._start.ndim != 1 or self._start.size < 2 or self._start.shape != self._end.shape:
            raise ValueError('Hermite endpoints must have matching shapes (D,), D >= 2')
        self._spline = CubicHermiteSpline(
            [0.0, 1.0], np.vstack([np.zeros(2), self._end[:2] - self._start[:2]]), np.vstack([m0, m1]))

    def at(self, t: float | np.ndarray) -> np.ndarray:
        """
        Points on the curve at parameter t, in world coordinates.
        t = 0 gives p0 and t = 1 gives p1 exactly. x, y follow the cubic;
        z (if any) goes linearly from p0 to p1.

        :param float | np.ndarray t: curve parameter in [0, 1], a scalar or an array of any shape
        :returns: the points, shape t.shape + (D,), where D = len(p0) is 2 or 3;
                  a scalar t gives one point, shape (D,)
        :rtype: np.ndarray
        """
        t = np.asarray(t, dtype=float)
        xy = self._spline(t) + self._start[:2]
        extra = (1 - t)[..., None] * self._start[2:] + t[..., None] * self._end[2:]
        points = np.concatenate([xy, extra], axis=-1)
        # Keep shared road endpoints exact despite polynomial evaluation rounding.
        points = np.where((t == 0)[..., None], self._start, points)
        return np.where((t == 1)[..., None], self._end, points)

    def derivative(self, t: float | np.ndarray, order: int = 1) -> np.ndarray:
        """
        Derivatives of the curve with respect to t (not arc length) at parameter t.
        Order 0 gives the points, as :meth:`at`. Order 1 is the velocity d(x, y)/dt in m per unit t,
        not a unit tangent. z (if any) is linear: its first derivative is p1.z - p0.z, higher ones are 0.

        :param float | np.ndarray t: curve parameter in [0, 1], a scalar or an array of any shape
        :param int order: derivative order, a non-negative integer; orders above 3 give 0
        :returns: the derivatives, shape t.shape + (D,), where D = len(p0) is 2 or 3
        :rtype: np.ndarray
        """
        if order == 0:
            return self.at(t)
        if order < 0:
            raise ValueError('Derivative order must be nonnegative')
        t = np.asarray(t, dtype=float)
        xy = self._spline(t, order)
        extra = np.broadcast_to(self._end[2:] - self._start[2:], t.shape + self._start[2:].shape)
        return np.concatenate([xy, extra if order == 1 else np.zeros_like(extra)], axis=-1)

    def curvature(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Signed XY curvature at parameter t, in 1/m: positive turning left, negative turning right. z is ignored.
        Where the XY speed is exactly 0 (a cusp, or p0 = p1 with zero derivatives) the value is 0 instead of nan;
        close to such a point it stays finite but can be very large.

        :param float | np.ndarray t: curve parameter in [0, 1], a scalar or an array of any shape
        :returns: the curvature; a float for a scalar t, otherwise an array of shape t.shape
        :rtype: float | np.ndarray
        """
        return self._scaled_curvature(t, 1.0)

    def _scaled_curvature(self, t: float | np.ndarray, scale: float) -> float | np.ndarray:
        """
        ``scale`` times :meth:`curvature`, up to rounding. :meth:`fit_directions` passes the XY chord, which makes
        the value dimensionless (curvature x chord): the same shape gives the same value at any size.

        :param float | np.ndarray t: curve parameter in [0, 1], a scalar or an array of any shape
        :param float scale: factor applied before dividing by the speed cubed; 1 gives the curvature in 1/m
        :returns: scale x curvature; a float for a scalar t, otherwise an array of shape t.shape
        :rtype: float | np.ndarray
        """
        velocity, acceleration = self.derivative(t), self.derivative(t, 2)
        speed = np.maximum(np.hypot(velocity[..., 0], velocity[..., 1]), 1e-12)
        # Multiply by scale before dividing. The order is fixed for reproducible fits, not for accuracy:
        # a rounding-level change here can send SLSQP to a different local optimum.
        values = scale * (velocity[..., 0] * acceleration[..., 1] - velocity[..., 1] * acceleration[..., 0]) / speed ** 3
        return float(values) if np.ndim(values) == 0 else values

    @cached_property
    def length(self) -> float:
        """
        XY arc length in metres, integrated with 16-point Gauss-Legendre quadrature.

        :returns: the length
        :rtype: float
        """
        velocity = self.derivative(_GAUSS_P)
        return float(np.sum(_GAUSS_W * np.hypot(velocity[:, 0], velocity[:, 1])))

    def sample(self, step: float = 0.25) -> np.ndarray:
        """
        Points at N values of t evenly spaced in [0, 1], both ends included (exactly p0 and p1).
        N = max(3, ceil(XY chord / step) + 1), so step only sets the count from the chord: neighbouring points are
        not step apart. Where the curve moves fast in t they can be about twice step apart, where it moves slowly
        much closer.

        :param float step: target spacing in m, measured against the XY chord; finite and positive
        :returns: the points, shape (N, D), where D = len(p0) is 2 or 3
        :rtype: np.ndarray
        """
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Sample step must be finite and positive')
        chord = float(np.hypot(*(self._end[:2] - self._start[:2])))
        return self.at(np.linspace(0.0, 1.0, max(3, int(np.ceil(chord / step)) + 1)))

    def to_param_poly3(self) -> Tuple[float, Tuple[float, ...], float]:
        """
        The curve as an OpenDRIVE paramPoly3 with normalized parameter p = t in [0, 1]; z is not exported.
        In the frame at p0 rotated to the start tangent, u(p) = aU + bU p + cU p^2 + dU p^3 and likewise v(p);
        world x, y = p0 + R(heading) (u, v). So aU = aV = 0, bU = |m0| and bV = 0 up to rounding.
        With m0 = 0 the start has no direction and heading is 0.

        :returns: (heading, coefficients, length): heading in rad, in [-pi, pi], the direction of m0;
                  coefficients, a tuple of 8 floats (aU, bU, cU, dU, aV, bV, cV, dV) in m;
                  length, the XY arc length in m, equal to :attr:`length`
        :rtype: Tuple[float, Tuple[float, ...], float]
        """
        heading = heading_of(self.derivative(0.0)[:2])
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, s], [-s, c]])  # world -> uv
        # Transform coefficients of the owned spline; no second fit is needed.
        coefficients = self._spline.c[::-1, 0, :] @ rotation.T
        return heading, tuple(float(v) for v in coefficients.T.ravel()), self.length

    @classmethod
    def fit_directions(cls, p0: np.ndarray, p1: np.ndarray, direction0: np.ndarray,
                       direction1: np.ndarray, scale_range: Tuple[float, float] = (0.4, 2.0)) -> 'HermiteCurve':
        """
        Fit a curve from p0 to p1 that leaves p0 along direction0 and reaches p1 along direction1,
        with low maximum curvature.

        The end derivatives are m0 = a * direction0 and m1 = b * direction1. The search variable is
        p = (log(a / chord), log(b / chord)), so p = 0 means both tangents are as long as the XY chord,
        and a / chord, b / chord stay within ``scale_range``. SLSQP minimises an upper bound on
        |curvature| x chord at 41 probes, plus 1e-3 * |p|^2 as a small preference for p = 0. Curvature
        between the probes is not checked. With an XY chord of 1e-9 m or less, no search is made.

        :param np.ndarray p0: start point, shape (2,) or (3,): x, y in m, and z if given
        :param np.ndarray p1: end point, same shape as p0
        :param np.ndarray direction0: unit XY direction at p0, shape (2,)
        :param np.ndarray direction1: unit XY direction at p1, shape (2,)
        :param Tuple[float, float] scale_range: (low, high), the tangent lengths allowed, as multiples of the
                                               XY chord; finite, positive and low <= high
        :returns: the fitted curve
        :rtype: HermiteCurve
        """
        low, high = (float(v) for v in scale_range)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low <= high):
            raise ValueError(f'scale_range must be finite, positive and ordered, got {scale_range}')
        box = (np.log(low), np.log(high))            # bounds of each entry of p, (lower, upper)
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)                  # (D,), D = 2 or 3
        direction0, direction1 = np.asarray(direction0, dtype=float), np.asarray(direction1, dtype=float)  # (2,)
        chord = float(np.hypot(*(p1[:2] - p0[:2])))  # XY chord length, m
        probe = np.linspace(0.0, 1.0, 41)            # t of the curvature probes, (41,)

        def curve(p):                                # p: (2,) -> HermiteCurve
            a, b = chord * np.exp(p)                 # tangent lengths, m
            return cls(p0, p1, a * direction0, b * direction1)

        def curvatures(p):                           # p: (2,) -> |curvature| x chord at the probes, (41,)
            return np.abs(curve(p)._scaled_curvature(probe, chord))

        def cost(p):                                 # p: (2,) -> float, used to compare candidates
            return float(np.max(curvatures(p))) + 1e-3 * float(p @ p)

        p = np.clip(np.zeros(2), *box)  # (2,): chord-length tangents, or the nearest scale the range allows
        if chord > 1e-9:
            # q = (p[0], p[1], bound), (3,): minimise the bound plus the preference for p = 0,
            # subject to bound >= the value at every probe (41 inequality constraints)
            result = minimize(lambda q: q[2] + 1e-3 * (q[0] ** 2 + q[1] ** 2), np.r_[p, np.max(curvatures(p))],
                              method='SLSQP', constraints=[{'type': 'ineq', 'fun': lambda q: q[2] - curvatures(q[:2])}],
                              bounds=[box, box, (0.0, None)], options={'ftol': 1e-9, 'maxiter': 50})
            found = np.clip(result.x[:2], *box)  # (2,)
            p = found if cost(found) < cost(p) else p
        return curve(p)

def cubic_segments(s: np.ndarray, values: np.ndarray,
                   tolerance: float = 0.0) -> List[Tuple[float, float, float, float, float]]:
    """
    Shape-preserving cubic Hermite as OpenDRIVE (sOffset, a, b, c, d) records.
    Weighted harmonic slopes (PCHIP) keep each interval within its endpoint values.
    In particular, nonnegative lane-width samples remain nonnegative between samples,
    including tapers and zero-width plateaus. With a ``tolerance``, only some samples are
    used as knots, yet the written cubic stays within ``tolerance`` of every sample.

    :param np.ndarray s: sample positions for s[i], shape (N,) in meters, strictly increasing
    :param np.ndarray values: sample values for values[i], shape (N,) in meters
    :param float tolerance: largest vertical distance, in m, from the cubic to any sample;
                            0 uses every sample as a knot
    :returns: one record per interval between knots, each (sOffset, a, b, c, d) for
              a + b ds + c ds^2 + d ds^3 with ds = s - sOffset
    :rtype: List[Tuple[float, float, float, float, float]]
    """
    s, values = np.asarray(s, dtype=float), np.asarray(values, dtype=float)
    if len(s) == 1:
        return [(float(s[0]), float(values[0]), 0.0, 0.0, 0.0)]
    if np.any(np.diff(s) <= 0):
        raise ValueError('Cubic sample positions must be strictly increasing')
    knots = np.ones(len(s), dtype=bool)
    if tolerance > 0 and len(s) > 2:
        # Start from the knots a polyline needs, then refine: the cubic between knots is not that polyline,
        # so add the worst sample of every interval where the cubic still misses one by more than tolerance.
        kept = np.asarray(LineString(np.column_stack([s, values])).simplify(tolerance).coords)
        knots = np.isin(s, kept[:, 0])
        while True:
            error = np.abs(PchipInterpolator(s[knots], values[knots])(s) - values)
            bad = np.flatnonzero(error > tolerance)
            if not len(bad):
                break
            interval = (np.cumsum(knots) - 1)[bad]        # interval between knots holding each bad sample
            order = bad[np.lexsort((-error[bad], interval))]
            first = np.r_[True, np.diff(np.sort(interval)) != 0]
            knots[order[first]] = True
    # core interpolation, using shape-preserving cubic Hermite (PCHIP)
    curve = PchipInterpolator(s[knots], values[knots])
    out = []
    for i, s0 in enumerate(curve.x[:-1]):
        if curve.x[i + 1] - s0 <= 1e-9:
            continue
        d, c, b, a = curve.c[:, i]
        out.append((float(s0), float(a), float(b), float(c), float(d)))
    return out

def cubic_values(records: List[Tuple[float, float, float, float, float]], 
                 s: float | np.ndarray) -> np.ndarray:
    """
    Evaluate OpenDRIVE ``(sOffset, a, b, c, d)`` records at stations s.
    Each station uses the last record whose sOffset <= s, so at a knot the record starting there wins
    (the last one, if several start there). Before the first record its polynomial is extrapolated,
    and past the last record the last one is. Empty records mean a profile of zero.

    :param List[Tuple[float, float, float, float, float]] records: M records with ascending sOffset,
                                                                   as from :func:`cubic_segments`
    :param float | np.ndarray s: stations in m, a scalar or an array of any shape
    :returns: the values, shape s.shape; a scalar s gives a 0-d array, not a float
    :rtype: np.ndarray
    """
    stations = np.asarray(s, dtype=float)
    if not len(records):
        return np.zeros_like(stations)
    table = np.asarray(records, dtype=float)
    # Records have no final endpoint. Its position does not affect evaluation:
    # PPoly extrapolates the final polynomial beyond this artificial interval.
    breaks = np.r_[table[:, 0], table[-1, 0] + 1.0]
    return PPoly(table[:, :0:-1].T, breaks, extrapolate=True)(stations)

def blend_length(points: np.ndarray, heading: float, at_start: bool,
                 per_radian: float = 10.0, minimum: float = 1.0) -> float:
    """
    Length, in m, over which a polyline end should turn into ``heading``: ``per_radian`` m for every
    radian between the end's own direction and heading, and at least ``minimum``, also when they agree.

    :param np.ndarray points: polyline, shape (N, 2) or (N, 3); only x, y are used
    :param float heading: wanted direction of travel at that end, in rad
    :param bool at_start: the first point's end when true, otherwise the last point's
    :param float per_radian: blend length per radian of turn, in m
    :param float minimum: shortest blend length, in m
    :returns: the blend length in m; 0 when the polyline has no direction (all points coincide in x, y)
    :rtype: float
    """
    own = end_direction(points, at_start)
    if own is None:
        return 0.0
    turn = float(np.arccos(np.clip(own @ np.array([np.cos(heading), np.sin(heading)]), -1.0, 1.0)))
    return max(minimum, per_radian * turn)

def blend_heading(points: np.ndarray, heading: float, at_start: bool, max_share: float = 0.4) -> np.ndarray:
    """
    Make a polyline end run in direction ``heading`` by replacing its end part with a curve from
    :meth:`HermiteCurve.fit_directions`.

    The replaced part is :func:`blend_length` long, but at most ``max_share`` of the XY arc length.
    The moved end keeps its position and the curve runs along heading there; at the far end of the
    replaced part the curve joins the polyline in the direction of the segment it joins. z on the curve
    is linear in t between its two ends. Vertices beyond the replaced part are kept unchanged.

    :param np.ndarray points: polyline, shape (N, 2) or (N, 3), x, y in m
    :param float heading: wanted direction of travel at that end, in rad: leaving the first point,
                          or arriving at the last
    :param bool at_start: blend the first point's end when true, otherwise the last point's
    :param float max_share: largest share of the XY arc length that may be replaced
    :returns: the blended polyline in the same order as points, shape (M, D) with D as in points;
              points itself when the polyline has no direction or no length
    :rtype: np.ndarray
    """
    pts = points if at_start else points[::-1]
    into = np.array([np.cos(heading), np.sin(heading)]) * (1.0 if at_start else -1.0)
    s = arc_lengths(pts)
    if end_direction(pts, True) is None or s[-1] <= 1e-6:
        return points
    length = min(blend_length(points, heading, at_start), max_share * s[-1])
    end = np.array([np.interp(length, s, pts[:, k]) for k in range(pts.shape[1])])
    k = int(np.clip(np.searchsorted(s, length, side='right'), 1, len(pts) - 1))
    along = end_direction(pts[k - 1:k + 1], True)
    along = into if along is None else along
    curve = HermiteCurve.fit_directions(pts[0], end, into, along)
    out = np.vstack([curve.sample(), pts[s > length + 1e-6]])
    return out if at_start else out[::-1]
