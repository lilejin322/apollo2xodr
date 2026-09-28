"""
Constraint submodule for spline fitting.
"""

import numpy as np
import shapely
from typing import Tuple

FIT_CHORD_GUARD = 2.0    # m, but never farther than this
FIT_CHORD_STEP = 0.5     # m between the points along the segments that keep straight stretches straight
FIT_CHORD_WEIGHT = 0.05  # their weight in the fit, relative to a vertex
FIT_SAGITTA = 1.5        # a curve may leave a polyline segment by this many times (plus the tolerance) 
                         # the sagitta of a circular arc that turns as much as the polyline does along the segment

class Chords:
    """
    The polyline's segments: how far a fitted curve may leave each, and points along them to fit.
    TODO: rename the class to PolylineConstraints
    """
    allowance: np.ndarray  # shape (M,) where M is the number of segments
    segments: np.ndarray   # shape (M,)
    tree: shapely.STRtree  # R-tree for fast nearest neighbor search
    u: np.ndarray          # shape (K,) where K is the sum num of segments by step
    points: np.ndarray     # shape (K, 2) each (x, y)

    def __init__(self, pts: np.ndarray, vertex_u: np.ndarray,
                 ends: Tuple[np.ndarray, np.ndarray], tolerance: float) -> None:
        """
        Constructor, save the chords for constraint check

        :param np.ndarray pts: the points of the polyline, shape (N, 2)
        :param np.ndarray vertex_u: the arc length from start, shape (N,)
        :param Tuple[np.ndarray, np.ndarray] ends: the heading at the start and end of the polyline, each shape (2,)
        :param float tolerance: the tolerance for fit
        """
        direction = np.diff(pts, axis=0)                      # get segment vectors
        length = np.hypot(direction[:, 0], direction[:, 1])   # get segment lengths
        direction = direction / length[:, None]               # normalize segment vectors
        headings = np.vstack([ends[0], direction, ends[1]])   # heading at each segment
        turn = np.arccos(np.clip(np.sum(headings[:-1] * headings[1:], axis=1), -1.0, 1.0))  # at every vertex
        # A vertex's turn is shared by its two segments in inverse proportion to their lengths: equally along a
        # uniformly sampled arc, and almost all to the short one where a long straight segment meets a short one
        # (a curve through the vertex turns within the short one). The end vertices' turns belong to their segment.
        ratio = length[1:] / (length[:-1] + length[1:])       # the ratio of the point
        # get how much bend each segment makes
        bend = turn[:-1] * np.r_[1.0, 1 - ratio] + turn[1:] * np.r_[ratio, 1.0]
        # get the sagitta of the chord (suppose an arc)
        sagitta = length / 2 * np.tan(np.minimum(bend, np.pi) / 4)
        # the allowance where spline may leave original segment
        self.allowance = np.minimum(tolerance + FIT_SAGITTA * sagitta, FIT_CHORD_GUARD)
        self.segments = shapely.linestrings(np.stack([pts[:-1], pts[1:]], axis=1))
        # construct the R-tree for fast nearest neighbor search, using shapely
        self.tree = shapely.STRtree(self.segments)
        u, points = [], []
        for i in range(len(length)):
            t = np.arange(FIT_CHORD_STEP, length[i], FIT_CHORD_STEP)
            u.append(vertex_u[i] + t)
            points.append(pts[i] + t[:, None] * direction[i])
        self.u = np.concatenate(u)
        self.points = np.vstack(points)

    def excess(self, curve: np.ndarray) -> np.ndarray:
        """
        How much farther each curve point is from its nearest segment than that segment allows.
        
        :param np.ndarray curve: the curve to check/judge, shape(P, 2), P is the num of points on the curve
        :returns: the allowance for each point on the curve, shape(P,)
                  Positive indicate out of scope, negative indicate in the allowance.
        :rtype: np.ndarray
        """
        index, distance = self.tree.query_nearest(shapely.points(curve), return_distance=True, all_matches=False)
        out = np.full(len(curve), -np.inf)
        out[index[0]] = distance - self.allowance[index[1]]
        return out
