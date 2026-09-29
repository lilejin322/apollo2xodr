"""
B-spline fitting submodule.
"""

import numpy as np
from scipy.interpolate import BSpline, PPoly
from typing import Tuple, List, Optional
import shapely
from shapely import LineString
from .constraint import Chords, FIT_CHORD_WEIGHT
from .plan_geometry import PlanGeometry
from .utils import unit_span_coefficients, span_geometry

FIT_CHECK_STEP = 0.05    # m, the fitted curve is checked against the polyline at least this densely

class Spline:
    """
    A clamped cubic B-spline on given breakpoints, fitted to a polyline with fixed end points and headings.
    Control points c0 = start and c(n-1) = end are fixed; c1 = start + alpha * start heading and c(n-2) = end - beta *
    end heading keep the end headings; alpha, beta and the other control points are the unknowns.
    """
    breaks: np.ndarray                   # NOTE: must be strict increasing order
    pts: np.ndarray                      # points of the polyline, shape (N, 2)
    vertex_u: np.ndarray                 # arc length, shape (N,)
    ends: Tuple[np.ndarray, np.ndarray]  # start and end headings, (cos\theta, sin\theta), each shape (2,)
    n: int                               # number of control points
    # The unknowns are z = [alpha, beta, x2..x(n-3), y2..y(n-3)], shape (2 + 2m,) with m = n - 4.
    # Control points are affine in z: cx = fx @ z + gx, cy = fy @ z + gy.
    fx: np.ndarray                       # linear part of the control points' x in z, shape (n, 2 + 2m)
    fy: np.ndarray                       # linear part of the control points' y in z, shape (n, 2 + 2m)
    gx: np.ndarray                       # constant part of the control points' x (fixed ends), shape (n,)
    gy: np.ndarray                       # constant part of the control points' y (fixed ends), shape (n,)
    # Weighted least squares on the fit points (vertices, then chord points): minimize |ax z - rx|^2 + |ay z - ry|^2.
    ax: np.ndarray                       # weighted basis @ fx, shape (P, 2 + 2m), P = number of fit points
    ay: np.ndarray                       # weighted basis @ fy, shape (P, 2 + 2m)
    rx: np.ndarray                       # weighted target x minus the fixed part (basis @ gx), shape (P,)
    ry: np.ndarray                       # weighted target y minus the fixed part (basis @ gy), shape (P,)
    penalty: np.ndarray                  # c^T penalty c = integral of the squared 2nd derivative, shape (n, n)

    def __init__(self, breaks: np.ndarray, pts: np.ndarray, vertex_u: np.ndarray,
                 ends: Tuple[np.ndarray, np.ndarray], chords: Chords) -> None:
        """
        Constructor

        :param np.ndarray breaks: breakpoints in meters, shape (B,), B is the num of breakpoints
        :param np.ndarray pts: points of the polyline, shape (N, 2)
        :param np.ndarray vertex_u: spline parameter (arc length) of each vertex, shape (N,)
        :param Tuple[np.ndarray, np.ndarray] ends: unit start and end headings, (cos, sin), each shape (2,)
        :param Chords chords: points sampled along the polyline's segments, fitted with a small weight
        """
        self.breaks, self.pts, self.vertex_u = breaks, pts, vertex_u
        self.start, self.end = pts[0], pts[-1]
        self.t0, self.t1 = ends
        # Cubic B-spline the order is 3, so the endpoint knots should repeat 3+1 times
        # Example:     breaks = [0, 4, 10]
        #    knots =   [0, 0, 0, 0, 4, 10, 10, 10, 10]
        self.knots = np.r_[[breaks[0]] * 3, breaks, [breaks[-1]] * 3]
        # the num of control points, also the num of basis functions
        # Example:     breaks = [0, 4, 10] -> control points = [c0, c1, c2, c3, c4]
        # where u=0, c0 is breaks[0], u=10, c4 is breaks[-1]
        self.n = len(self.knots) - 4
        # fitted to the vertices, and with a small weight to the points along the segments
        fit_points = np.vstack([pts, chords.points])
        # Vertices, then chord points. sqrt because this multiplies the residual and the fit squares it,
        # so a vertex keeps weight 1 and a chord point keeps FIT_CHORD_WEIGHT.
        weight = np.sqrt(np.r_[np.ones(len(pts)), np.full(len(chords.points), FIT_CHORD_WEIGHT)])
        # get the basis function values at each vertex and chrod
        basis = bspline_basis(self.knots, np.r_[vertex_u, chords.u], 3) * weight[:, None]
        # control points as an affine function of the unknowns z = [alpha, beta, x2.., y2..]
        m = self.n - 4
        # rows of f/g are control points: c0, c(n-1) come from g only; c1 adds alpha * t0, c(n-2) adds -beta * t1;
        # c2..c(n-3) pick their x (columns 2..2+m) or y (columns 2+m..2+2m) straight from z
        self.fx, self.fy = np.zeros((self.n, 2 + 2 * m)), np.zeros((self.n, 2 + 2 * m))
        self.gx, self.gy = np.zeros(self.n), np.zeros(self.n)
        for f, g, k in ((self.fx, self.gx, 0), (self.fy, self.gy, 1)):
            g[[0, 1]] = self.start[k]
            g[[-2, -1]] = self.end[k]
            f[1, 0] = self.t0[k]
            f[-2, 1] = -self.t1[k]
            f[2:self.n - 2, 2 + k * m:2 + (k + 1) * m] = np.eye(m)
        self.ax, self.ay = basis @ self.fx, basis @ self.fy
        self.rx, self.ry = fit_points[:, 0] * weight - basis @ self.gx, fit_points[:, 1] * weight - basis @ self.gy
        # integral of the squared second derivative: basis'' is linear on each span, so Simpson's rule is exact
        penalty = np.zeros((self.n, self.n))
        for a, b in zip(breaks[:-1], breaks[1:]):
            if b > a:
                second = bspline_basis(self.knots, np.array([a, (a + b) / 2, b]), 3, 2)
                penalty += (b - a) / 6 * (second[0][:, None] * second[0] + 4 * second[1][:, None] * second[1] +
                                          second[2][:, None] * second[2])
        self.penalty = penalty

    def control_points(self, z: np.ndarray) -> np.ndarray:
        """
        the control points of B-spline

        :param np.ndarray z: the unknown vector, shape (2 + 2m,) (m = n - 4)
        :returns: the n control points, shape (n, 2)
        :rtype: np.ndarray
        """
        return np.column_stack([self.fx @ z + self.gx, self.fy @ z + self.gy])

    def solve(self, smoothing: float) -> np.ndarray:
        """
        Solve the optimization problem to find the unknown vector z.
        J(z) = ||ax * z - rx||^2 + ||ay * z - ry||^2 + smoothing * [cx^T * P * cx + cy^T * P * cy]
        where cx = fx @ z + gx, cy = fy @ z + gy

        :param float smoothing: the smoothing factor
        :returns: the n control points, shape (n, 2)
        :rtype: np.ndarray
        """
        # solve the normal equation
        fx, fy, gx, gy, pen = self.fx, self.fy, self.gx, self.gy, self.penalty
        lhs = self.ax.T @ self.ax + self.ay.T @ self.ay + smoothing * (fx.T @ pen @ fx + fy.T @ pen @ fy)
        rhs = self.ax.T @ self.rx + self.ay.T @ self.ry - smoothing * (fx.T @ pen @ gx + fy.T @ pen @ gy)
        # prevent the matrix being singular, where no matrix^(-1) exists
        lhs += np.eye(len(lhs)) * 1e-12 * max(1.0, np.trace(lhs))
        z = np.linalg.solve(lhs, rhs)
        # the end tangents must point forwards; a reversed one would make a cusp
        first, last = self.breaks[1] - self.breaks[0], self.breaks[-1] - self.breaks[-2]
        # ensure forward-pointing endpoint tangents
        z[0], z[1] = max(z[0], first / 30), max(z[1], last / 30)
        # get the points from vector z
        return self.control_points(z)

    def evaluate(self, control: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Points on the spline at arc lengths ``u``.

        :param np.ndarray control: control points, shape (n, 2)
        :param np.ndarray u: arc lengths on the same axis as ``breaks``, shape (M,)
        :returns: coordinates at those arc lengths, shape (M, 2)
        :rtype: np.ndarray
        """
        return bspline_basis(self.knots, u, 3) @ control

    def span_errors(self, control: np.ndarray, chords: Chords) -> Tuple[np.ndarray, np.ndarray]:
        """
        On each span: the largest distance of a vertex from the curve,
        and the largest excess of the curve's distance from the polyline's 
        segments over what they allow (positive where it strays too far).
        
        :param np.ndarray control: control points, shape (n, 2)
        :param Chords chords: the polyline's segments and their allowances
        :returns: (vertex, excess), each shape (B-1,), B is the num of breakpoints
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        excess = np.full(len(self.breaks) - 1, -np.inf)
        vertex = np.zeros(len(self.breaks) - 1)
        samples, owner = [], []
        # for each segment, sample at least 20 points, step about 5cm
        # owner is the tag of this segment
        for i, (a, b) in enumerate(zip(self.breaks[:-1], self.breaks[1:])):
            u = np.linspace(a, b, max(20, int(np.ceil((b - a) / FIT_CHECK_STEP)) + 1))
            samples.append(u)
            owner.append(np.full(len(u), i))
        u, owner = np.concatenate(samples), np.concatenate(owner)
        # curve to sample points
        curve = self.evaluate(control, u)
        np.maximum.at(excess, owner, chords.excess(curve))
        # sample points to polyline points
        back = shapely.distance(LineString(curve), shapely.points(self.pts))
        span = np.clip(np.searchsorted(self.breaks, self.vertex_u, side='right') - 1, 0, len(vertex) - 1)
        np.maximum.at(vertex, span, back)
        return vertex, excess

    def geometries(self, control: np.ndarray) -> Tuple[List[PlanGeometry], np.ndarray]:
        """
        convert the control points OpenDRIVE format paramPoly3

        :param np.ndarray control: control points, shape (n, 2)
        :returns: (geometries, s_values), where s_values shape(G+1,), G is the num of geometries
        :rtype: Tuple[List[PlanGeometry], np.ndarray]
        """
        geoms, s_values = [], [0.0]
        # convert the B-spline to PPoly
        pieces = [
            PPoly.from_spline((self.knots, np.ascontiguousarray(control[:, axis], dtype=float), 3))
            for axis in range(control.shape[1])
        ]
        # get the geometries and s_values
        for a, b in zip(self.breaks[:-1], self.breaks[1:]):
            if b - a <= 1e-12:
                continue
            origin = self.evaluate(control, np.array([a]))[0]
            coeffs = unit_span_coefficients(pieces, a, b, origin)  # rows: 1, p, p^2, p^3; columns: x, y
            geometry = span_geometry(s_values[-1], origin, coeffs)
            if geometry is None:
                continue
            last = geoms[-1] if geoms else None
            if last is not None and last.coeffs is None and geometry.coeffs is None and \
                    abs(np.angle(np.exp(1j * (geometry.hdg - last.hdg)))) < 1e-9:
                # consecutive straight spans in the same direction are one line
                last.length = float(np.hypot(geometry.x - last.x, geometry.y - last.y)) + geometry.length
                s_values[-1] = last.s + last.length
                continue
            geoms.append(geometry)
            s_values.append(s_values[-1] + geometry.length)
        return geoms, np.asarray(s_values)

def bspline_basis(knots: np.ndarray, x: np.ndarray, degree: int,
                  derivative: int = 0) -> np.ndarray:
    """
    Values (or derivatives) of all B-spline basis functions of ``degree`` over ``knots`` at ``x``.
    Past the right end the last span is extended. Left of the first knot the basis stays zero: a clamped
    knot vector starts with repeated knots, and the previous evaluator clamped onto that empty span.

    :param np.ndarray knots: knot vector, shape (K,). A clamped cubic from ``B`` breakpoints has ``K = B + 6``; other callers may pass a different vector
    :param np.ndarray x: spline parameters to evaluate at, in m, any shape
    :param int degree: degree of the basis; 3 for a cubic
    :param int derivative: derivative order; 0 for the values themselves
    :returns: basis values (or derivatives), shape ``x.shape + (n,)`` with ``n = K - degree - 1``. ``x`` of shape ``(M,)`` gives ``(M, n)``
    :rtype: np.ndarray
    """
    x = np.asarray(x, dtype=float)
    n = len(knots) - degree - 1
    flat = np.reshape(x, -1)
    out = np.zeros((flat.size, n))
    if flat.size:
        spline = BSpline(knots, np.eye(n), degree)
        inside = (flat >= knots[0]) & (flat < knots[-1])
        if np.any(inside):
            out[inside] = spline(flat[inside], nu=derivative, extrapolate=False)
        right = flat >= knots[-1]
        if np.any(right):
            out[right] = spline(flat[right], nu=derivative, extrapolate=True)
    return out.reshape(np.shape(x) + (n,))
