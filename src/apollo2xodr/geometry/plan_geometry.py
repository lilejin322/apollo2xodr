"""
PlanGeometry class for the reconstruction of road geometry.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
import numpy as np
from numpy.polynomial import Polynomial
from .__init__ import _GAUSS_P, _GAUSS_W

@dataclass
class PlanGeometry:
    """
    One ``<planView><geometry>`` record in the local frame.

    TODO: I think we need to separate the geometry into three different classes: Line, Arc, and ParamPoly3.
    """
    s: float                                        # arc length from the start of the road
    x: float                                        # x of the world-coordinate 
    y: float                                        # y of the world-coordinate
    hdg: float                                      # heading of the start point
    length: float                                   # length of the geometry
    coeffs: Optional[Tuple[float, ...]] = None      # None -> line; (aU, bU, cU, dU, aV, bV, cV, dV) for paramPoly3
    curvature: Optional[float] = None               # constant-curvature arc (used for tight surface repairs)

    @cached_property
    def u(self) -> Polynomial:
        """
        :returns: u(p) = aU + bU * p + cU * p^2 + dU * p^3
        :rtype: Polynomial
        """
        if self.coeffs is None:
            raise ValueError("coeffs is None, this geometry is not a paramPoly3")
        return Polynomial(self.coeffs[:4])

    @cached_property
    def v(self) -> Polynomial:
        """
        :returns: v(p) = aV + bV * p + cV * p^2 + dV * p^3
        :rtype: Polynomial
        """
        if self.coeffs is None:
            raise ValueError("coeffs is None, this geometry is not a paramPoly3")
        return Polynomial(self.coeffs[4:])

    @cached_property
    def du(self) -> Polynomial:
        """
        :returns: u'(p) = bU + 2 * cU * p + 3 * dU * p^2
        :rtype: Polynomial
        """
        return self.u.deriv()
    
    @cached_property
    def dv(self) -> Polynomial:
        """
        :returns: v'(p) = bV + 2 * cV * p + 3 * dV * p^2
        :rtype: Polynomial
        """
        return self.v.deriv()

    @cached_property
    def ddu(self) -> Polynomial:
        """
        :returns: u''(p) = 2 * cU + 6 * dU * p
        :rtype: Polynomial
        """
        return self.du.deriv()

    @cached_property
    def ddv(self) -> Polynomial:
        """
        :returns: v''(p) = 2 * cV + 6 * dV * p
        :rtype: Polynomial
        """
        return self.dv.deriv()

    def _speed(self, p: np.ndarray) -> np.ndarray:
        """
        Calculate the speed of the geometry at a given parameter

        :param np.ndarray p: the parameter
        :returns: the ds/dp
        :rtype: np.ndarray
        """
        return np.hypot(self.du(p), self.dv(p))

    def _arc_length(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        Arc length between two normalized paramPoly3 parameters.

        :param np.ndarray start: the start parameter
        :param np.ndarray end: the end parameter
        :returns: the arc length
        :rtype: np.ndarray
        """
        start, end = np.asarray(start), np.asarray(end)
        p = start[..., None] + (end - start)[..., None] * _GAUSS_P
        return (end - start) * np.sum(self._speed(p) * _GAUSS_W, axis=-1)

    @cached_property
    def _arc_table(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arc length table for paramPoly3

        :returns: the arc length table
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        p = np.linspace(0.0, 1.0, 65)
        return p, np.r_[0.0, np.cumsum(self._arc_length(p[:-1], p[1:]))]

    def at(self, ds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Position and analytic heading at distance from this geometry's start.

        :param np.ndarray ds: the distance from the start of the geometry
        :returns: the position and analytic heading
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        ds = np.clip(ds, 0.0, self.length)

        # 1. Arc
        if self.curvature is not None:
            k = self.curvature
            h = self.hdg + k * ds
            # Half-angle form avoids cancellation for small turns.
            distance = ds * np.sinc(k * ds / (2 * np.pi))
            return (self.x + distance * np.cos(self.hdg + k * ds / 2),
                    self.y + distance * np.sin(self.hdg + k * ds / 2),
                    h)

        # 2. Line
        if self.coeffs is None:
            return (*self._to_world(ds, np.zeros_like(ds)),
                    np.full_like(ds, self.hdg, dtype=float))

        # 3. paramPoly3
        p = self.parameter(ds)
        du, dv = self.du(p), self.dv(p)
        return (*self._to_world(self.u(p), self.v(p)),
                self.hdg + np.arctan2(dv, du))

    def parameter(self, ds):
        """
        Normalized polynomial parameter at arc length, using the same convention as at().
        F(p) = s_{lo} + arcLength(p_{lo}, p) - target
        we need to find p such that F(p) = 0, using Newton's method.
        
        :param np.ndarray ds: the arc length ds
        :returns: the normalized polynomial parameter
        :rtype: np.ndarray
        """
        ds = np.asarray(ds, dtype=float)
        if self.length <= 0:             # No length, return 0
            return np.zeros_like(ds)
        if self.coeffs is None:          # Line or Arc, return normalized ds 
            return np.clip(ds, 0.0, self.length) / self.length
        ps, arcs = self._arc_table       # get the arc length table
        # the target arc length
        target = np.clip(ds, 0.0, self.length) / self.length * arcs[-1]
        index = np.clip(np.searchsorted(arcs, target, side='right') - 1, 0, len(ps) - 2)

        p_lo, p_hi = ps[index], ps[index + 1]
        s_lo, s_hi = arcs[index], arcs[index + 1]

        ratio = (target - s_lo) / np.maximum(s_hi - s_lo, 1e-15)
        p = p_lo + ratio * (p_hi - p_lo)

        # Newton's method to find, iteratively
        for _ in range(3):
            residual = arcs[index] + self._arc_length(p_lo, p) - target
            p = np.clip(p - residual / np.maximum(self._speed(p), 1e-15), p_lo, p_hi)

        return p

    def curvature_at(self, ds: np.ndarray) -> np.ndarray:
        """
        Signed curvature (left positive) at distances from this geometry's start.
        
        :param np.ndarray ds: the distance from the start of the geometry
        :returns: the signed curvature
        :rtype: np.ndarray
        """
        ds = np.asarray(ds, dtype=float)

        # 1. Arc
        if self.curvature is not None:
            return np.full_like(ds, self.curvature)
        # 2. Line
        if self.coeffs is None:
            return np.zeros_like(ds)
        # 3. ParamPoly3
        p = self.parameter(ds)
        du = self.du(p)
        dv = self.dv(p)
        ddu = self.ddu(p)
        ddv = self.ddv(p)
        # $\kappa = \frac{u'v'' - v'u''}{\left(u'^2 + v'^2\right)^{3/2}}$
        return (du * ddv - dv * ddu) / np.maximum(np.hypot(du, dv), 1e-12) ** 3

    def project(self, point) -> Tuple[float, float, float]:
        """
        Closest point on this geometry: local arc length, signed offset (left positive), squared distance.
        A cubic is solved exactly. Sampling it and reading the normal is only an approximation, and for a
        signal far from the reference line that approximation can move the signal visibly.

        :param np.ndarray point: the point to project, shape np.array([x, y])
        :returns: (ds, t, distance_sq)
        :rtype: Tuple[float, float, float]
        """
        point = np.asarray(point, dtype=float)
        # 1. Arc
        if self.curvature is not None:
            # A vanishing curvature is a line; dividing by it would put the center at infinity.
            if abs(self.curvature) <= 1e-15:
                return self._project_line(point)
            return self._project_arc(point)
        # 2. Line
        if self.coeffs is None:
            return self._project_line(point)
        # 3. ParamPoly3
        return self._project_param_poly3(point)

    def _to_local(self, point: np.ndarray) -> Tuple[float, float]:
        """
        Query point in the frame of the start: along the heading, and to its left.
        
        TODO: need to use np.ndarray for iteration
        :param np.ndarray point: the point to convert to local coordinate, shape np.array([x, y])
        :returns: (local_x, local_y)
        :rtype: Tuple[float, float]
        """
        c, s = np.cos(self.hdg), np.sin(self.hdg)
        dx, dy = point[0] - self.x, point[1] - self.y
        return float(c * dx + s * dy), float(-s * dx + c * dy)
    
    def _to_world(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert local coordinate to world coordinate
        $x_{world} = x_0 + ucos\theta - vsin\theta$
        $y_{world} = y_0 + usin\theta + vcos\theta$

        :param np.ndarray u: the u coordinate, shape np.array([u1, u2, ...])
        :param np.ndarray v: the v coordinate, shape np.array([v1, v2, ...])
        :returns: (world_x, world_y), shape np.array([x1, x2, ...], [y1, y2, ...])
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        c, s = np.cos(self.hdg), np.sin(self.hdg)
        return self.x + c * u - s * v, self.y + s * u + c * v

    def _project_line(self, point: np.ndarray) -> Tuple[float, float, float]:
        """
        Project onto a straight geometry.
        
        :param np.ndarray point: the point to project, shape np.array([x, y])
        :returns: (ds, t, distance_sq)
        :rtype: Tuple[float, float, float]
        """
        along, left = self._to_local(point)
        ds = float(np.clip(along, 0.0, self.length))
        return ds, left, float((along - ds) ** 2 + left ** 2)

    def _project_arc(self, point: np.ndarray) -> Tuple[float, float, float]:
        """
        Project onto a circular arc of curvature ``k`` (positive is a left turn).
        The center is one radius, ``1/k``, to the left of the start. The foot of the query lies on a
        ray from that center. ``atan2`` reports the turn in (-pi, pi], so the turns one revolution
        away are checked as well. Both endpoints are always candidates.

        :param np.ndarray point: the point to project, shape np.array([x, y])
        :returns: (ds, t, distance_sq)
        :rtype: Tuple[float, float, float]
        """
        k = self.curvature       # get the cur of arc, so R = 1 / |k|
        # get the center of arc
        center = np.array([self.x - np.sin(self.hdg) / k, self.y + np.cos(self.hdg) / k])
        # from center to start vector
        to_start = np.array([self.x, self.y]) - center
        # from center to the point vector
        to_point = point[:2] - center
        # get the turn from to_start to to_point, signed in rad, turn in (-pi, pi]
        turn = np.arctan2(to_start[0] * to_point[1] - to_start[1] * to_point[0], to_start @ to_point)
        # start the searching
        candidates = [0.0, self.length]   # in s
        for angle in (turn - 2 * np.pi, turn, turn + 2 * np.pi):
            ds = float(angle / k)
            if 0.0 < ds < self.length:
                candidates.append(ds)
        x, y, heading = self.at(np.asarray(candidates, dtype=float))
        dx, dy = point[0] - x, point[1] - y
        # get all candidates distance_sq
        squared = dx ** 2 + dy ** 2
        # find the min_distance
        i = int(np.argmin(squared))
        lateral = -np.sin(heading[i]) * dx[i] + np.cos(heading[i]) * dy[i]
        return float(candidates[i]), float(lateral), float(squared[i])

    def _project_param_poly3(self, point: np.ndarray) -> Tuple[float, float, float]:
        """
        Project onto a paramPoly3 by solving where the squared distance is stationary.
        In the start frame, D(p) = (u(p) - qu)^2 + (v(p) - qv)^2, so D'(p) = 0 is
        (u - qu) u' + (v - qv) v' = 0. Real roots in (0, 1), plus both endpoints, are scored by D.

        :param np.ndarray point: the point to project, shape np.array([x, y])
        :returns: (ds, t, distance_sq)
        :rtype: Tuple[float, float, float]
        """
        qu, qv = self._to_local(point)
        u_offset, v_offset = self.u - qu, self.v - qv
        roots = (u_offset * self.du + v_offset * self.dv).roots()
        interior = roots.real[(np.abs(roots.imag) < 1e-9) & (roots.real > 0.0) & (roots.real < 1.0)]
        # Endpoints first, so an exact tie keeps the same foot as before.
        candidates = np.r_[0.0, 1.0, interior]
        at_u, at_v = u_offset(candidates), v_offset(candidates)
        squared = at_u ** 2 + at_v ** 2
        # get the min_distance
        i = int(np.argmin(squared))
        p = float(candidates[i])
        tangent_u, tangent_v = float(self.du(p)), float(self.dv(p))
        speed = float(np.hypot(tangent_u, tangent_v))
        # u_offset, v_offset are curve - point. Left of (u', v') is (-v', u'), dotted with point - curve.
        lateral = 0.0 if speed <= 1e-15 else (tangent_v * at_u[i] - tangent_u * at_v[i]) / speed
        return self._distance_at_parameter(p), float(lateral), float(squared[i])

    def _distance_at_parameter(self, p: float) -> float:
        """
        Local arc length of a paramPoly3 parameter ``p``, scaled to the declared geometry length.
                            p -> s
        
        :param float p: the parameter, in [0, 1]
        :returns: the local arc length, in [0, length] meters
        :rtype: float
        """
        p = float(np.clip(p, 0.0, 1.0))   # guarantee the param in [0, 1]
        ps, arcs = self._arc_table        # get the arc table
        if arcs[-1] <= 1e-15:             # if the length is 0, return 0.0
            return 0.0
        # search p in the arc table, get the index i
        i = int(np.clip(np.searchsorted(ps, p, side='right') - 1, 0, len(ps) - 2))
        # calculate the local arc length, add from i to p
        return float((arcs[i] + self._arc_length(ps[i], p)) / arcs[-1] * self.length)

    def sample(self, n: int) -> np.ndarray:
        """
        ``n`` points (x, y) along the geometry, including both ends.
        
        :param int n: the number of points to sample
        :returns: the sampled points, shape (n, 2) as np.array([[x1, y1], [x2, y2], ...])
        :rtype: np.ndarray
        """
        p = np.linspace(0.0, 1.0, n)
        # 1. Curve
        if self.curvature is not None:
            return np.column_stack(self.at(p * self.length)[:2])
        # 2. Line
        if self.coeffs is None:
            u, v = p * self.length, np.zeros_like(p)
        # 3. ParamPoly3
        else:
            u, v = self.u(p), self.v(p)
        return np.column_stack(self._to_world(u, v))

    def extrema(self) -> np.ndarray:
        """
        Endpoints and all x/y extrema in inertial coordinates (normalized cubic parameter).
        
        :returns: the endpoints and all x/y extrema, shape (m, 2) as np.array([[x1, y1], [x2, y2], ...])
        :rtype: np.ndarray
        """
        # 1. Curve
        if self.curvature is not None:
            k = self.curvature
            lo, hi = sorted((self.hdg, self.hdg + k * self.length))
            headings = np.arange(np.ceil(lo / (np.pi / 2)), np.floor(hi / (np.pi / 2)) + 1) * (np.pi / 2)
            s = np.r_[0.0, (headings - self.hdg) / k, self.length]
            return np.column_stack(self.at(s)[:2])
        # 2. Line
        if self.coeffs is None:
            return self.sample(2)
        # 3. ParamPoly3
        c, s = np.cos(self.hdg), np.sin(self.hdg)
        dx = c * self.du - s * self.dv
        dy = s * self.du + c * self.dv
        ps = [0.0, 1.0]
        for deriv in (dx, dy):
            roots = deriv.roots()
            ps.extend(r.real for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < 1)
        return np.column_stack(self._to_world(self.u(ps), self.v(ps)))
