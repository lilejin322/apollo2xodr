"""
OpenDRIVE road data and queries on its written lane sections.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ..geometry.hermite import cubic_values
from ..model import Lane
from ..reference_line import ReferenceLine

@dataclass(eq=False)
class Road:
    """
    One OpenDRIVE road: Apollo lanes on a shared reference line, with the polynomials written for them.

    Unless ``path`` is set, the road has a single lane section holding all its lanes side by side. A path road chains
    single Apollo lanes one after another along s, each in a lane section of its own as lane -1. ``path`` and
    ``junction`` are independent: a path road may lie outside any junction. Record lists hold OpenDRIVE cubic records
    ``(s, a, b, c, d)``, evaluated with :func:`cubic_values`.

    ``sections`` and the lane queries (``lane_range``, ``lane_end``, ``lane_heading``) expect a built road:
    ``reference`` set, ``lane_end`` also needs ``widths``, and on a path road ``section_starts`` holds one start per
    lane.
    """
    id: int
    """OpenDRIVE road id"""
    right: List[Lane]
    """driving along s, from the reference line outwards (ids -1, -2, ...); on a path road its lanes
    one after another along s, each in a lane section of its own (id -1)"""
    left: List[Lane]
    """driving against s, from the reference line outwards (ids 1, 2, ...)"""
    junction: Optional[str]
    """name of the junction region the road connects in, None outside junctions. A path road is left outside
    (None) when no path of its region is entered from another road, or when it has neither incoming nor outgoing"""
    reference: Optional[ReferenceLine] = None
    """plan view the lanes are offset from; None until the road is built"""
    elevation: list = field(default_factory=list)
    """elevation records (s, a, b, c, d) along the reference line"""
    widths: Dict[Lane, list] = field(default_factory=dict)
    """lane -> width records [(sOffset in its lane section, a, b, c, d)]"""
    lane_offset: list = field(default_factory=list)
    """lateral shift (s, a, b, c, d) of the centre lane (lane 0) from the reference line; empty: none"""
    section_starts: List[float] = field(default_factory=lambda: [0.0])
    """s where each lane section starts, ascending. On a path road one per lane of ``right`` once
    ``path_widths`` has run; until then the default only covers the first lane"""
    path: bool = False
    """the road is a chain of lanes (see ``right``) rather than lanes side by side"""
    incoming: Optional[Lane] = None
    """path roads: the lane the path continues from, on a road that is not a path; None if there is none"""
    outgoing: Optional[Lane] = None
    """path roads: the lane the path leads into, on a road that is not a path; None if there is none"""
    geometry_adjustment: float = 0.0
    """m, the largest single move of the reference line: the Hausdorff distance from the line a step started with,
    for ``settle_surface`` smoothing a bend and ``fit_curved_path_borders`` refitting a path. Not the total move
    from the first reference to the final one; 0 when the reference was never moved"""

    @property
    def lanes(self) -> List[Lane]:
        """
        All lanes of the road, right then left.

        :returns: list of lanes
        :rtype: List[Lane]
        """
        return self.right + self.left

    @property
    def sections(self) -> List[Tuple[float, List[Lane], List[Lane]]]:
        """
        Lane sections of the road, in order along s.

        :returns: (s where the section starts, right lanes, left lanes) of every lane section
        :rtype: List[Tuple[float, List[Lane], List[Lane]]]
        """
        if self.path:
            return [(s, [lane], []) for s, lane in zip(self.section_starts, self.right)]
        return [(0.0, self.right, self.left)]

    def lane_id(self, lane: Lane) -> int:
        """
        OpenDRIVE id of a lane in its lane section.

        :param Lane lane: a lane of this road
        :returns: -1, -2, ... for right lanes (always -1 on a path road), 1, 2, ... for left lanes
        :rtype: int
        """
        if lane in self.right:
            return -1 if self.path else -(self.right.index(lane) + 1)
        return self.left.index(lane) + 1

    def lane_range(self, lane: Lane) -> Tuple[float, float]:
        """
        Extent along s of the lane section holding ``lane``.

        :param Lane lane: a lane of this road
        :returns: s where that lane section starts and ends
        :rtype: Tuple[float, float]
        """
        if not self.path:
            return 0.0, self.reference.length
        i = self.right.index(lane)
        return self.section_starts[i], (self.section_starts + [self.reference.length])[i + 1]

    def end_lanes(self, at_end: bool) -> List[Tuple[Lane, bool]]:
        """
        Lanes that start or end at one end of the road. Right lanes start at s = 0, left lanes at the far end; on a
        path road only the first lane touches s = 0 and only the last one the far end.

        :param bool at_end: True for the end at s = length, False for s = 0
        :returns: (lane, whether the lane starts there) for each lane touching that end, right lanes first
        :rtype: List[Tuple[Lane, bool]]
        """
        if self.path:
            return [(self.right[-1], False)] if at_end else [(self.right[0], True)]
        return [(lane, not at_end) for lane in self.right] + [(lane, at_end) for lane in self.left]

    def lane_end_s(self, lane: Lane, at_start: bool) -> float:
        """
        Where the lane starts (or ends) along the reference line.

        :param Lane lane: a lane of this road
        :param bool at_start: the end where the lane starts in its driving direction when true, otherwise where it ends
        :returns: s in m, its lane section's start where a right lane starts or a left lane ends, otherwise its end
        :rtype: float
        """
        lo, hi = self.lane_range(lane)
        return lo if at_start == (lane in self.right) else hi

    def lane_end(self, lane: Lane, at_start: bool) -> np.ndarray:
        """
        x/y of the lane's (left, right) borders where it starts (or ends), as the written OpenDRIVE puts them.

        Left and right are seen in the lane's driving direction, so the left border is the inner one, on the side of
        lane 0. With a lane offset it need not be the one nearer the reference line (a path lane is centred on it).

        :param Lane lane: a lane of this road, whose widths (and the road's lane offset) are set
        :param bool at_start: the end where the lane starts in its driving direction when true, otherwise where it ends
        :returns: the border points, shape (2, 2): rows are the left and right border, columns x and y
        :rtype: np.ndarray
        """
        along = lane in self.right
        lo, _ = self.lane_range(lane)
        s = self.lane_end_s(lane, at_start)
        x, y, hdg = self.reference.at(s)
        inner = 0.0
        if not self.path:
            side = self.right if along else self.left
            inner = sum(float(cubic_values(self.widths[l], s)) for l in side[:side.index(lane)])
        outer = inner + float(cubic_values(self.widths[lane], s - lo))
        offset = float(cubic_values(self.lane_offset, s)) if self.lane_offset else 0.0
        sign = -1.0 if along else 1.0
        normal = np.array([-np.sin(hdg), np.cos(hdg)])
        return np.array([[x, y] + (offset + sign * inner) * normal, [x, y] + (offset + sign * outer) * normal])

    def lane_heading(self, lane: Lane, at_start: bool) -> float:
        """
        Reference-line heading where the lane starts (or ends), turned to the lane's driving direction.
        Changes in width and lane offset are not taken into account, so where they vary this is not the tangent of the
        written lane.

        :param Lane lane: a lane of this road
        :param bool at_start: the end where the lane starts in its driving direction when true, otherwise where it ends
        :returns: heading in radians: the reference heading for a right lane, turned by pi for a left lane
        :rtype: float
        """
        _, _, hdg = self.reference.at(self.lane_end_s(lane, at_start))
        return float(hdg if lane in self.right else hdg + np.pi)

    def elevation_at(self, s: float) -> float:
        """
        Road elevation at a position along the reference line.

        :param float s: arc length along the reference line, in m; not clamped to the road
        :returns: z in m, from the elevation records: 0 when there are none, and outside the records the first or
                  last polynomial extrapolated
        :rtype: float
        """
        return float(cubic_values(self.elevation, s))

    def __repr__(self):
        """
        :returns: string representation of the road
        :rtype: str
        """
        return f'Road({self.id}, right={self.right}, left={self.left})'
