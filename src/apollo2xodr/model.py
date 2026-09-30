"""
model module is the intermediate representation of Apollo HD Map.
Map data read from Apollo, in a local metric frame ``(e, n, z) = (easting - E0, northing - N0, z)``.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass(eq=False)
class Boundary:
    """
    A lane boundary polyline, possibly shared by two neighbouring lanes.
    """
    points: np.ndarray        # shape (N, 3), easting, northing, z in the local frame
    kind: str = 'UNKNOWN'     # Apollo marking, or 'VIRTUAL' / 'UNKNOWN'

@dataclass(eq=False)
class BoundaryRef:
    """
    A boundary as seen from one lane: its points run in that lane's driving direction.
    """
    boundary: Boundary        # shared polyline; neighbours may point at the same one
    reversed: bool = False    # this lane drives against the stored point order
    # Marking intervals in this lane's driving direction: (point, type).
    # point is float64, shape (3,) as read. A cycle slice inserts shape (2,) at the window start;
    # slicing a boundary of one point or zero length keeps one mark, shape (2,), with the first type
    # (none when there are no marks).
    # Kept off the shared geometry so sharing and attachment cannot erase transitions.
    marks: List[Tuple[np.ndarray, str]] = field(default_factory=list)

    @property
    def points(self) -> np.ndarray:
        """
        Points in the driving direction.

        :returns: the points, shape (N, 3)
        :rtype: np.ndarray
        """
        return self.boundary.points[::-1] if self.reversed else self.boundary.points

    def endpoint(self, at_start: bool) -> np.ndarray:
        """
        One end of the boundary in the driving direction.

        :param bool at_start: the first point when true, otherwise the last
        :returns: a view of that point, shape (3,). Writing it changes the shared boundary
        :rtype: np.ndarray
        """
        pts = self.points
        return pts[0] if at_start else pts[-1]

    def set_endpoint(self, at_start: bool, value: np.ndarray) -> None:
        """
        Replace one end of the shared polyline, in this lane's driving direction.

        :param bool at_start: the first point when true, otherwise the last
        :param np.ndarray value: the new point, shape (3,)
        """
        # reversed stores the opposite direction, so this lane's start is the array's end
        index = 0 if at_start != self.reversed else -1
        self.boundary.points[index] = value

@dataclass(eq=False)
class Lane:
    """
    One Apollo lane, with its centre line and the two boundaries beside it.
    """
    # Apollo lane id. A cycle split names its pieces '{id}~cycle-head', '{id}' (the core) and '{id}~cycle-tail',
    # appending '~' while a name is taken.
    id: str
    center: np.ndarray                               # (N, 3), in driving direction
    left: BoundaryRef                                # left boundary in the driving direction
    right: BoundaryRef                               # right boundary in the driving direction
    speed_limit: float                               # m/s
    # Apollo road id; 'lane:{id}' if no Apollo road lists the lane. A cycle split gives each piece
    # '~cycle-road:{piece id}'.
    road: str
    # Apollo junction id, or None outside a junction. A cycle split sets the core's to None, and a head's or tail's
    # to '~cycle:{id}' when the lane had none.
    junction: Optional[str]
    apollo_predecessors: List[str] = field(default_factory=list)  # Apollo predecessor lane ids
    apollo_successors: List[str] = field(default_factory=list)    # Apollo successor lane ids
    left_forward: Optional[str] = None               # left neighbour lane id, same direction
    right_forward: Optional[str] = None              # right neighbour lane id, same direction
    left_reverse: Optional[str] = None               # left neighbour lane id, opposite direction
    predecessors: List['Lane'] = field(default_factory=list)  # resolved incoming lanes
    successors: List['Lane'] = field(default_factory=list)    # resolved outgoing lanes
    kind: str = 'driving'                            # OpenDRIVE lane type
    width_samples: Optional[np.ndarray] = None       # (M, 2): fraction of the lane's length, width (Apollo samples)
    # original Apollo id of a piece cut from a cyclic lane, or from a lane adjoining the cycle; None if not cut
    source_id: Optional[str] = None
    sketched_roundabout: bool = False                # centre was smoothed as part of a nearly circular hand-drawn loop
    # on a second ring found through the left_forward or right_forward neighbours of a roundabout's lanes:
    # centres under 1.5 m apart, radii 0.5–6 m apart, so it may be the inner or the outer ring
    paired_roundabout: bool = False

    def __repr__(self):
        """
        String representation of the lane.

        :returns: ``Lane(<id>)``
        :rtype: str
        """
        return f'Lane({self.id})'

@dataclass(eq=False)
class TrafficControl:
    """
    A traffic light, stop sign or yield sign.
    """
    id: str                                          # Apollo signal or sign id
    kind: str                                        # 'signal', 'stop' or 'yield'
    position: np.ndarray                             # (3,), local frame; a sign's stored position is unused
    # (N, 3), a single stop line, read only to fill an empty stop_lines. The reader passes stop_lines[0];
    # nothing keeps the two in sync
    stop_line: np.ndarray
    # (lane id, start s in m) from Apollo overlaps. A cycle split replaces a cut lane's entry with the piece holding
    # s and s from that piece's start, clamped to the piece
    overlap_lanes: List[Tuple[str, float]]
    junction: Optional[str] = None                   # Apollo junction id from overlaps, if given
    # m, face size. A light's is measured from its Apollo box when that has 3 corners or more; it falls back to
    # 0.65 x 1.5, with bottom None, when there are fewer or the measured width or height is not positive.
    # A sign's is 0.95 x 0.95
    width: float = 0.65
    height: float = 1.5
    stop_lines: List[np.ndarray] = field(default_factory=list)  # every stop line, each shape (N, 3)
    bottom: Optional[float] = None  # z of the lower edge (Apollo bounding box); None: position z - height / 2

    def __post_init__(self):
        """
        Fill an empty stop_lines with stop_line.
        """
        if not self.stop_lines:
            self.stop_lines = [self.stop_line]

    def __repr__(self):
        """
        String representation of the traffic control.

        :returns: ``TrafficControl(<id>)``
        :rtype: str
        """
        return f'TrafficControl({self.id})'

@dataclass
class MapData:
    """
    One Apollo map after reading: lanes, signals, and the local-frame origin.
    """
    lanes: Dict[str, Lane]                           # Lane.id to lane, in file order; cycle pieces replace their lane
    road_order: List[str]                            # Apollo road ids in file order
    junction_order: List[str]                        # Apollo junction ids in file order
    controls: List[TrafficControl]                   # signals, stop signs, and yield signs
    origin: Tuple[float, float]                      # (E0, N0), UTM metres subtracted from every point
    utm_zone: int                                    # UTM zone of the projection, or of the header lat/lon
    projection: str                                  # Apollo header projection string
    # whether any point of Apollo's raw lane centre lines has z != 0 (boundaries and signals are not checked);
    # else signal heights are in a datum of their own
    has_heights: bool = True
