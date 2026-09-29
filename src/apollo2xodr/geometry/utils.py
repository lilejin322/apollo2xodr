"""
Utility functions for the geometry submodule.
"""

from typing import List, Optional
import numpy as np
from scipy.interpolate import PPoly
from .plan_geometry import PlanGeometry
from .__init__ import _GAUSS_P, _GAUSS_W

LENGTH_MARGIN = 1e-7     # relative, see span_geometry()

def unit_span_coefficients(pieces: List[PPoly], a: float, b: float, origin: np.ndarray) -> np.ndarray:
    """
    Power basis of one cubic span in p = (u - a) / (b - a), constant term relative to ``origin``.
    
    :param List[PPoly] pieces: the spline as piecewise polynomials, one per axis (x, y)
    :param float a: the start of the span
    :param float b: the end of the span
    :param np.ndarray origin: the curve's point at u = a, shape (2,)
    :returns: coefficients, shape (4, 2); rows: ``1, p, p^2, p^3``; columns: x, y
    :rtype: np.ndarray
    """
    h = b - a
    columns = []
    for piece in pieces:
        interval = int(np.searchsorted(piece.x, a + 0.5 * h) - 1)
        cubic, quad, linear, const = piece.c[:, interval]
        columns.append((const, linear * h, quad * h * h, cubic * h ** 3))
    coeffs = np.column_stack(columns)
    coeffs[0] -= origin
    return coeffs

def span_geometry(s: float, origin: np.ndarray, coeffs: np.ndarray) -> Optional[PlanGeometry]:
    """
    One cubic span, in power basis ``p = 0..1`` relative to its start, as a line or paramPoly3.

    :param float s: arc length along the reference line where this geometry starts, in m
    :param np.ndarray origin: curve point at the start of the span, shape (2,)
    :param np.ndarray coeffs: coefficients, shape (4, 2); rows ``1, p, p^2, p^3``, columns ``x, y``
    :returns: a line (``coeffs`` is ``None``) or a paramPoly3; ``None`` if the span is shorter than 1e-9 m
    :rtype: Optional[PlanGeometry]
    """
    hdg = float(np.arctan2(coeffs[1, 1], coeffs[1, 0]))
    c, sn = np.cos(hdg), np.sin(hdg)
    u = c * coeffs[:, 0] + sn * coeffs[:, 1]
    v = -sn * coeffs[:, 0] + c * coeffs[:, 1]
    du = u[1] + 2 * u[2] * _GAUSS_P + 3 * u[3] * _GAUSS_P ** 2
    dv = v[1] + 2 * v[2] * _GAUSS_P + 3 * v[3] * _GAUSS_P ** 2
    length = float(np.sum(_GAUSS_W * np.hypot(du, dv)))
    if length <= 1e-9:
        return None
    if max(abs(u[2]), abs(u[3]), abs(v[1]), abs(v[2]), abs(v[3])) < 1e-9 * max(length, 1.0):
        return PlanGeometry(s, origin[0], origin[1], hdg, float(u[1]))  # a uniform straight span
    # Readers that find p for a given s by bracketing it in [0, 1] need the length not to exceed their own integral
    # of the curve; declaring it a hair short keeps them clear of rounding.
    return PlanGeometry(s, origin[0], origin[1], hdg, length * (1 - LENGTH_MARGIN),
                        (0.0, float(u[1]), float(u[2]), float(u[3]), 0.0, float(v[1]), float(v[2]), float(v[3])))
