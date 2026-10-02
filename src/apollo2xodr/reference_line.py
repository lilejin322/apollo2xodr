"""
Reference Line module for OpenDRIVE format.
"""

import shapely
import numpy as np
from dataclasses import dataclass
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline
from shapely.geometry import LineString
from typing import List, Optional, Tuple
from .geometry.polyline import extend
from .geometry.utils import LENGTH_MARGIN
from .geometry.hermite import HermiteCurve
from .geometry import PlanGeometry, fit_plan_view, FIT_TOLERANCE

@dataclass
class ReferenceLine:
    """
    A densely sampled plan view, used to evaluate positions, headings and lateral offsets.
    """
    geometries: List[PlanGeometry]  # plan-view spans in order, len(geometries) = G
    vertex_s: np.ndarray            # s where each geometry starts, and [-1] the total length, shape (G+1,)
    s: np.ndarray                   # arc length at each sample point, shape (M,)
    xy: np.ndarray                  # x, y coordinates of each sample point, shape (M, 2)
    hdg: np.ndarray                 # headings of each sample point, shape (M,)

    @property
    def length(self) -> float:
        """
        The total length of the reference line.

        :returns: the total length in meters
        :rtype: float
        """
        return float(self.s[-1])

    @classmethod
    def from_polyline(cls, points: np.ndarray, tolerance: float = FIT_TOLERANCE,
                      step: float = 0.25, start_heading: Optional[float] = None,
                      end_heading: Optional[float] = None) -> 'ReferenceLine':
        """
        Fit a reference line through a polyline and sample it.

        :param np.ndarray points: polyline vertices, shape (N, 2) or (N, 3); only x, y are used
        :param float tolerance: max distance, in m, from a vertex to the fitted curve
        :param float step: sample spacing along the fitted line, in m
        :param Optional[float] start_heading: start heading in radians, or None to follow the first segment
        :param Optional[float] end_heading: end heading in radians, or None to follow the last segment
        :returns: the fitted, sampled reference line
        :rtype: ReferenceLine
        """
        geoms, vertex_s = fit_plan_view(points, tolerance, start_heading, end_heading)
        if not geoms:
            raise ValueError('A reference line needs at least two distinct points')
        return cls.from_geometries(geoms, step)

    @classmethod
    def from_geometries(cls, geoms: List[PlanGeometry], step: float = 0.25) -> 'ReferenceLine':
        """
        Sample plan-view geometries into a reference line.

        :param List[PlanGeometry] geoms: spans in order, length G. ``geoms[i].s`` is where span i starts
        :param float step: sample spacing along the line, in m
        :returns: the reference line. ``vertex_s`` has shape (G+1,); 
                  ``s`` and ``hdg`` have shape (M,), ``xy`` has shape (M, 2).
                  The point where two spans meet is stored once
        :rtype: ReferenceLine
        """
        # geometry boundaries, plus the total length
        vertex_s = np.array([g.s for g in geoms] + [geoms[-1].s + geoms[-1].length])
        s_parts, xy_parts, heading_parts = [], [], []
        for g in geoms:
            # at least two samples, and no farther apart than step
            n = max(2, int(np.ceil(g.length / step)) + 1)
            local = np.linspace(0.0, g.length, n)
            x, y, hdg = g.at(local)
            # drop this span's end; the next span starts at the same point
            s_parts.append(g.s + local[:-1])
            xy_parts.append(np.column_stack([x, y])[:-1])
            heading_parts.append(hdg[:-1])
        # the last span's end is the end of the line, so it is kept once
        s_parts.append([geoms[-1].s + geoms[-1].length])
        x, y, hdg = geoms[-1].at(geoms[-1].length)
        xy_parts.append([[x, y]])
        heading_parts.append([hdg])
        s = np.concatenate(s_parts)
        xy = np.vstack(xy_parts)
        return cls(geoms, vertex_s, s, xy, np.concatenate(heading_parts))

    def smoothed(self, spacing: float, end_scale: float = 1.0) -> 'ReferenceLine':
        """
        An exact clamped cubic B-spline, preserving endpoint positions and headings.
        Used only to repair road surfaces that fold under lateral offsets. The B-spline
        is emitted as exact paramPoly3 spans, avoiding new curvature spikes from fitting
        another polyline. Larger control spacing smooths a wider region.

        :param float spacing: desired distance, in m, between control points. 
                              The line is sampled into ``n = max(4, ceil(length / spacing) + 1)`` points
        :param end_scale: length of the end tangents, as a multiple of the control spacing.
                          A float applies to both ends; a pair ``(start, end)`` sets them separately
        :returns: the smoothed reference line, resampled at the default spacing of ``from_geometries``
        :rtype: ReferenceLine
        """
        # at least four samples: a clamped cubic needs four control points
        n = max(4, int(np.ceil(self.length / spacing)) + 1)
        x, y, h = self.at(np.linspace(0.0, self.length, n))
        controls = np.column_stack([x, y])
        step: float = self.length / (n - 1)
        start_scale, stop_scale = (end_scale, end_scale) if np.isscalar(end_scale) else end_scale
        # c1 and c(n-2) set the end tangents; the ends themselves stay on the sampled points
        controls[1] = controls[0] + start_scale * step * np.array([np.cos(h[0]), np.sin(h[0])])
        controls[-2] = controls[-1] - stop_scale * step * np.array([np.cos(h[-1]), np.sin(h[-1])])
        # parameter is the span index, 0 .. n-3, with the ends repeated four times
        knots = np.r_[np.zeros(4), np.arange(1, n - 3), np.full(4, n - 3)]
        curve: BSpline = BSpline(knots, controls, 3)

        geoms, s = [], 0.0
        p = np.linspace(0.0, 1.0, 4)
        matrix = np.vander(p, 4, increasing=True)
        for start in range(n - 3):
            # four samples fix the cubic on this span; m0, m1 are its end derivatives
            points = curve(start + p)
            # Subtract the origin before solving, to keep small derivatives well conditioned.
            coeffs = np.linalg.solve(matrix, points - points[0])
            m0, m1 = coeffs[1], coeffs[1] + 2 * coeffs[2] + 3 * coeffs[3]
            segment = HermiteCurve(points[0], points[-1], m0, m1)
            hdg, coeffs, length = segment.to_param_poly3()
            # a hair short, so a reader bracketing p in [0, 1] does not step past the span
            length *= 1 - LENGTH_MARGIN
            if length <= 1e-9:
                raise ValueError('A smoothed reference contains a degenerate span')
            geoms.append(PlanGeometry(s, *points[0], hdg, length, coeffs))
            s += length
        return ReferenceLine.from_geometries(geoms)

    def curvature(self, s: np.ndarray) -> np.ndarray:
        """
        Signed curvature (left positive) at arc length s.

        :param np.ndarray s: arc length in m, shape (M,)
        :returns: signed curvature, shape (M,)
        :rtype: np.ndarray
        """
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        # which span holds each s. side='right' puts a joint on the span that starts there;
        # clip pulls the line's end, which sits on vertex_s[-1], back onto the last span
        index = np.clip(np.searchsorted(self.vertex_s, s, side='right') - 1, 0, len(self.geometries) - 1)
        result = np.empty(s.size)
        for i in np.unique(index):
            mask = index.ravel() == i
            # curvature_at measures from this span's own start
            result[mask] = self.geometries[i].curvature_at(s.ravel()[mask] - self.geometries[i].s)
        return result.reshape(s.shape) if s.ndim else float(result[0])

    def at(self, s: float | np.ndarray) -> Tuple[float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Position and heading at arc length s.

        :param float | np.ndarray s: arc length in m. shape (M,) if array-like
        :returns: (x, y, hdg)
        :rtype: Tuple[float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray] for each array shape (M,)
        """
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        # which span holds each s. side='right' puts a joint on the span that starts there;
        # clip pulls the line's end, which sits on vertex_s[-1], back onto the last span
        index = np.clip(np.searchsorted(self.vertex_s, s, side='right') - 1, 0, len(self.geometries) - 1)
        result = np.empty((3, s.size))  # rows are x, y, hdg
        for i in np.unique(index):
            mask = index.ravel() == i
            g = self.geometries[i]
            # g.at measures from this span's own start
            result[:, mask] = g.at(s.ravel()[mask] - g.s)
        # a float s comes back as three Python floats; an array keeps shape (M,)
        return tuple(v.reshape(s.shape) if s.ndim else float(v[0]) for v in result)

    def project(self, point: np.ndarray, s_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Closest point of a query point on the reference line.

        :param np.ndarray point: one point, shape (2,) or (3,). Only x, y are used
        :param Optional[Tuple[float, float]] s_range: finite, ordered bounds for the search, in m.
                  The bounds are clipped to the reference line; a single station is allowed
        :returns: (s, t). s is the arc length of the closest point in the search interval, in m.
                  t is the signed lateral offset, in m, left positive
        :rtype: Tuple[float, float]
        """
        if s_range is not None:
            return self._project_range(point, s_range)
        # The closest point lies within one sample spacing of a sample no farther than the nearest sample plus that
        # spacing; only the geometries holding such samples need the exact projection.
        distance = np.hypot(self.xy[:, 0] - point[0], self.xy[:, 1] - point[1])  # to each dense sample
        spacing = float(np.max(np.diff(self.s))) if len(self.s) > 1 else 0.0  # widest gap between samples
        # samples within two spacings of the nearest one; the closest point is among their spans
        near = self.s[distance <= distance.min() + 2 * spacing]
        # span that holds each of those samples
        index = np.searchsorted(self.vertex_s, near, side='right') - 1
        # also the span before each sample: the closest point can fall just before it
        index = np.unique(np.clip(np.r_[index, index - 1], 0, len(self.geometries) - 1))
        # g.project returns ds from the span start, its own t, and squared distance
        candidates = [(g.s + ds, t, d) for g in (self.geometries[i] for i in index)
                      for ds, t, d in [g.project(point)]]
        # d is squared distance; the nearest exact projection wins
        s, _, _ = min(candidates, key=lambda p: p[2])
        # t from g.project can belong to the other span at a joint; recompute it with at()
        x, y, hdg = self.at(s)
        # left of the heading is positive
        return s, float(-np.sin(hdg) * (point[0] - x) + np.cos(hdg) * (point[1] - y))

    def _project_range(self, point: np.ndarray, s_range: Tuple[float, float]) -> Tuple[float, float]:
        """Project onto the requested interval itself, including portions of its boundary spans."""
        lo, hi = s_range
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError(f's_range must be finite and ordered, got {s_range}')
        lo, hi = np.clip([lo, hi], 0.0, self.length)
        if lo == 0.0 and hi == self.length:
            return self.project(point)
        if lo == hi:
            station = float(lo)
        else:
            # Include the interval endpoints: a short interval may contain no stored samples.
            mask = (self.s >= lo) & (self.s <= hi)
            sample_s = np.r_[lo, self.s[mask], hi]
            x, y, _ = self.at(np.array([lo, hi]))
            sample_xy = np.vstack(([x[0], y[0]], self.xy[mask], [x[1], y[1]]))
            distance = np.hypot(sample_xy[:, 0] - point[0], sample_xy[:, 1] - point[1])
            spacing = float(np.max(np.diff(sample_s)))
            near = sample_s[distance <= distance.min() + 2 * spacing]
            index = np.searchsorted(self.vertex_s, near, side='right') - 1
            index = np.unique(np.clip(np.r_[index, index - 1], 0, len(self.geometries) - 1))
            candidates = []
            for i in index:
                g = self.geometries[i]
                start, end = max(0.0, lo - g.s), min(g.length, hi - g.s)
                if start > end:
                    continue
                if start == end:
                    gx, gy, _ = g.at(start)
                    ds, squared = start, float((point[0] - gx) ** 2 + (point[1] - gy) ** 2)
                elif start == 0.0 and end == g.length:
                    ds, _, squared = g.project(point)
                elif g.coeffs is None or g.curvature is not None:
                    # A line or arc clipped at either end is still a line or arc.
                    gx, gy, heading = g.at(start)
                    piece = PlanGeometry(0.0, gx, gy, heading, end - start, curvature=g.curvature)
                    local, _, squared = piece.project(point)
                    ds = start + local
                else:
                    # Compose the polynomial with its restricted parameter interval. The existing
                    # cubic projector then considers every stationary point inside that interval.
                    p0, p1 = g.parameter(np.array([start, end]))
                    parameter = Polynomial([p0, p1 - p0])
                    coefficients = tuple(value for poly in (g.u(parameter), g.v(parameter))
                                         for value in np.pad(poly.coef, (0, 4 - len(poly.coef))))
                    piece = PlanGeometry(0.0, g.x, g.y, g.hdg, end - start, coefficients)
                    local, _, squared = piece.project(point)
                    # Keep the original span's arc-length convention rather than the clipped
                    # polynomial's independently integrated table, and preserve exact endpoints.
                    if local == 0.0:
                        ds = start
                    elif local == piece.length:
                        ds = end
                    else:
                        p = p0 + (p1 - p0) * piece.parameter(local)
                        ds = float(np.clip(g._distance_at_parameter(p), start, end))
                candidates.append((g.s + ds, squared))
            station = float(np.clip(min(candidates, key=lambda candidate: candidate[1])[0], lo, hi))
        x, y, heading = self.at(station)
        lateral = -np.sin(heading) * (point[0] - x) + np.cos(heading) * (point[1] - y)
        return station, float(lateral)

    def lateral_offsets(self, s: np.ndarray, boundary: np.ndarray, reach: float = 60.0) -> np.ndarray:
        """
        Signed lateral offset (left positive) of a boundary along the normal at every s.
        The boundary is extended straight at both ends so that every normal meets it. When a normal crosses it more
        than once, the crossing closest to the boundary's nearest point is used.

        :param np.ndarray s: arc length in m, shape (M,)
        :param np.ndarray boundary: boundary polyline, shape (N, 3). Only x, y are used
        :param float reach: length, in m, of each normal and of the straight extension at both ends. Callers leave 60
        :returns: signed lateral offset, shape (M,)
        :rtype: np.ndarray
        """
        x, y, hdg = self.at(s)
        nx, ny = -np.sin(hdg), np.cos(hdg)  # left normal
        # one segment per s, reach metres to each side of the reference point
        segments = np.stack([np.column_stack([x - reach * nx, y - reach * ny]),
                             np.column_stack([x + reach * nx, y + reach * ny])], axis=1)
        normals = shapely.linestrings(segments)
        target = LineString(extend(boundary, reach)[:, :2])  # straight past both ends, xy only
        # estimate: offset of the closest boundary point, used to pick among multiple crossings
        nearest = shapely.get_coordinates(shapely.line_interpolate_point(
            target, shapely.line_locate_point(target, shapely.points(np.column_stack([x, y])))))
        estimate = (nearest[:, 0] - x) * nx + (nearest[:, 1] - y) * ny
        hits = shapely.intersection(normals, target)
        # a normal can meet the boundary more than once; index is which normal each piece came from
        parts, index = shapely.get_parts(hits, return_index=True)
        nonempty = ~shapely.is_empty(parts)
        parts, index = parts[nonempty], index[nonempty]
        t = estimate.copy()  # a normal that misses the boundary keeps this
        best = np.full(len(s), np.inf)
        if len(parts):
            coords = shapely.get_coordinates(shapely.centroid(parts))
            # signed offset of each crossing, left positive
            tt = (coords[:, 0] - x[index]) * nx[index] + (coords[:, 1] - y[index]) * ny[index]
            for k, i in enumerate(index):
                d = abs(tt[k] - estimate[i])
                if d < best[i]:
                    best[i] = d
                    t[i] = tt[k]  # crossing closest to the estimate
        return t
