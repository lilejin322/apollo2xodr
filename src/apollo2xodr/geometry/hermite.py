"""
Hermite curve related functions.
"""

import numpy as np
from typing import Tuple, List
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from shapely.geometry import LineString
from .polyline import heading_of
from .__init__ import _GAUSS_P, _GAUSS_W

def hermite_poly3(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, 
                  m1: np.ndarray) -> Tuple[float, Tuple[float, float, 
                  float, float], float]:
    """
    Cubic Hermite curve p0 -> p1 with end derivatives m0, m1 as paramPoly3 in the frame of its start tangent.
    
    :param np.ndarray p0: Start point in world frame, shape (2,) in meters
    :param np.ndarray p1: End point in world frame, shape (2,) in meters
    :param np.ndarray m0: (dx/dp, dy/dp) | p=0, shape (2,)
    :param np.ndarray m1: (dx/dp, dy/dp) | p=1, shape (2,)
    """
    hdg = heading_of(m0)
    c, s = np.cos(hdg), np.sin(hdg)
    rot = np.array([[c, s], [-s, c]])  # world -> uv
    chord = rot @ (p1 - p0)
    tangents = np.vstack([rot @ m0, rot @ m1])
    curve = CubicHermiteSpline([0.0, 1.0], np.vstack([np.zeros(2), chord]), tangents)
    # PPoly stores the cubic term first; OpenDRIVE wants a + b p + c p^2 + d p^3.
    coeffs = tuple(float(v) for v in curve.c[::-1, 0, :].T.ravel())
    speed = curve.derivative()(_GAUSS_P)
    length = float(np.sum(_GAUSS_W * np.hypot(speed[:, 0], speed[:, 1])))
    return hdg, coeffs, length

def cubic_segments(s: np.ndarray, values: np.ndarray,
                   tolerance: float = 0.0) -> List[Tuple[float, float, float, float, float]]:
    """
    Shape-preserving cubic Hermite as OpenDRIVE (sOffset, a, b, c, d) records.
    Weighted harmonic slopes (PCHIP) keep each interval within its endpoint values.
    In particular, nonnegative lane-width samples remain nonnegative between samples,
    including tapers and zero-width plateaus. With a ``tolerance``, only the samples
    a piecewise-linear curve needs to stay that close to all of them are used as knots.

    :param np.ndarray s: sample positions for s[i], shape (N,) in meters
    :param np.ndarray values: sample values for values[i], shape (N,) in meters
    :returns: OpenDRIVE's standard (sOffset, a, b, c, d)
    """
    s, values = np.asarray(s, dtype=float), np.asarray(values, dtype=float)
    if tolerance > 0 and len(s) > 2:
        kept = np.asarray(LineString(np.column_stack([s, values])).simplify(tolerance).coords)
        s, values = kept[:, 0], kept[:, 1]
    if len(s) == 1:
        return [(float(s[0]), float(values[0]), 0.0, 0.0, 0.0)]
    if np.any(np.diff(s) <= 0):
        raise ValueError('Cubic sample positions must be strictly increasing')
    # core interpolation, using shape-preserving cubic Hermite (PCHIP)
    curve = PchipInterpolator(s, values)
    out = []
    for i, s0 in enumerate(curve.x[:-1]):
        if curve.x[i + 1] - s0 <= 1e-9:
            continue
        d, c, b, a = curve.c[:, i]
        out.append((float(s0), float(a), float(b), float(c), float(d)))
    return out
