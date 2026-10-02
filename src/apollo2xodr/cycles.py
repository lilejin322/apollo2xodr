"""
Break cycles in connecting material by retaining ordinary road cores inside the ring.

Each cyclic source lane becomes head -> ordinary core -> tail. Junction paths can then
connect those cores without unrolling a loop or dropping its closing edge. Internal IDs
are unique; OpenDRIVE userData continues to name the original Apollo lane.
The same slicing and remapping machinery also gives unentered branching junctions ordinary
entry segments followed by connecting tails.
"""

import networkx as nx
import numpy as np
from dataclasses import replace
from typing import Dict, Set, List, Tuple
from shapely.geometry import LineString, Point
from shapely.ops import substring
from .model import Boundary, BoundaryRef, MapData, Lane

__all__ = ['split_cycles', 'split_connecting_entries']

def _slice_marks(ref: BoundaryRef, start: float, end: float) -> List[Tuple[np.ndarray, str]]:
    """
    Return marking states covering the arc-length window ``[start, end)``.
    
    :param BoundaryRef ref: boundary reference
    :param float start: start of the arc-length window
    :param float end: end of the arc-length window
    :returns: marking states covering the window: first the state in force at ``start``, at the window start as an
              (x, y) point of shape (2,), then every mark strictly inside the window with its point as stored, shape
              (3,) as read. A boundary of one point or of zero length gives only its first mark, at that point's x, y;
              one without marks gives none
    :rtype: List[Tuple[np.ndarray, str]]
    """
    if not ref.marks:        # no marks to slice, return []
        return []

    points = ref.points

    if len(points) < 2:      # a single point keeps the first marking state, at its x, y
        return [(np.asarray(points[0, :2], dtype=float), ref.marks[0][1])]

    line = LineString(points[:, :2])

    if line.length <= 0.0:   # degenerate boundary
        return [(np.asarray(points[0, :2], dtype=float), ref.marks[0][1])]

    located = sorted(
        (
            (line.project(Point(point[:2])), point, kind)
            for point, kind in ref.marks
        ),
        key=lambda item: item[0],
    )

    active = located[0][2]
    for station, _, kind in located:
        if station <= start:
            active = kind
        else:
            break

    start_point = np.asarray(line.interpolate(start).coords[0], dtype=float)
    kept = [(start_point, active)]

    for station, point, kind in located:
        if start < station < end:
            kept.append((point, kind))

    return kept

def _length(points: np.ndarray) -> float:
    """
    Return the x/y arc length of a polyline.
    
    :param np.ndarray points: polyline points
    :returns: x/y arc length of the polyline
    :rtype: float
    """
    if len(points) < 2:
        return 0.0
    return float(LineString(points[:, :2]).length)

def _slice(points: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Slice a polyline at x/y arc-length stations.
    
    :param np.ndarray points: polyline points
    :param float start: start of the slice
    :param float end: end of the slice
    :returns: sliced polyline
    :rtype: np.ndarray
    """
    assert 0.0 <= start <= end, f"slice needs 0 <= start <= end, got {start}..{end}"
    if len(points) < 2:
        return np.array(points, dtype=float, copy=True)
    part = substring(LineString(points), start, end)
    coords = np.asarray(part.coords, dtype=float)
    if len(coords) == 1:
        coords = np.repeat(coords, 2, axis=0)
    return coords

def _on_cycle(material: Set[Lane]) -> Set[Lane]:
    """
    Lanes that lie on a directed cycle in the subgraph induced by ``material``.

    :param Set[Lane] material: lanes to check for cycles
    :returns: lanes that lie on a directed cycle in the subgraph induced by ``material``
    :rtype: Set[Lane]
    """
    graph = nx.DiGraph(
        (lane, other)
        for lane in material
        for other in lane.successors
        if other in material
    )
    cyclic: Set[Lane] = set()
    for component in nx.strongly_connected_components(graph):
        # One lane is cyclic only when it links to itself; a larger component always contains a cycle.
        if len(component) > 1 or any(graph.has_edge(lane, lane) for lane in component):
            cyclic.update(component)
    return cyclic

def split_cycles(data: MapData, material: Set[Lane]) -> bool:
    """
    Split cyclic SCCs in-place; return whether road plans need to be rebuilt.

    :param MapData data: whole map, mutated in place
    :param Set[Lane] material: connecting lanes searched for cycles. The lanes on a cycle are cut, and so are lanes
                               outside ``material`` directly before or after one, at their end towards it; a lane of
                               ``material`` next to a cycle but not on one stays whole
    :returns: True if road plans need to be rebuilt, False otherwise
    :rtype: bool
    """
    cyclic = _on_cycle(material)
    if not cyclic:
        return False  # no cycles to split

    # A sharp turn can lie exactly on the boundary with an ordinary incoming/outgoing
    # lane. Include its adjoining end in the connector, so the transition has physical
    # room rather than requiring a finite-width lane to turn at a single point.
    adjoining = {other for lane in cyclic for other in lane.predecessors + lane.successors
                 if other not in material}

    cuts = {}
    for lane in data.lanes.values():
        if lane not in cyclic | adjoining:
            continue
        length = _length(lane.center)
        if length <= 1e-9:
            if lane in cyclic:
                raise ValueError(f"Cannot split cyclic lane {lane.id!r}: centerline has zero length")
            continue
        # Keep the core short so the connecting pieces have room to make a finite-width turn.
        half_core = min(5.0, length / 10) / (2 * length)
        has_head = lane in cyclic or any(p in cyclic for p in lane.predecessors)
        has_tail = lane in cyclic or any(p in cyclic for p in lane.successors)
        a = 0.5 - half_core if has_head else 0.0
        b = 0.5 + half_core if has_tail else 1.0
        cuts[lane] = ([(0.0, a, 'head')] if has_head else []) + [(a, b, 'core')] + \
                     ([(b, 1.0, 'tail')] if has_tail else [])
    _split_lanes(data, cuts, 'cycle')
    return True

def split_connecting_entries(data: MapData, lanes: Set[Lane]) -> None:
    """
    Give a junction at the map edge ordinary incoming roads.

    Each selected source lane is cut halfway into an ordinary core and a connecting tail. The core retains its id;
    the tail takes ``{id}~entry-tail`` (with extra ``~`` if needed). Geometry, widths, markings, connectivity and
    signal overlaps are sliced and remapped as for a cycle split. Callers must rebuild road plans afterwards.

    :param MapData data: map changed in place
    :param Set[Lane] lanes: connecting lanes without predecessors that need ordinary entry segments
    """
    if not lanes:
        return
    for lane in lanes:
        if _length(lane.center) <= 1e-9:
            raise ValueError(f"Cannot split entry lane {lane.id!r}: centerline has zero length")
    _split_lanes(data, {lane: [(0.0, 0.5, 'core'), (0.5, 1.0, 'tail')] for lane in lanes}, 'entry')

def _split_lanes(data: MapData, cuts: Dict[Lane, List[Tuple[float, float, str]]], tag: str) -> None:
    """
    Replace selected lanes with fractional intervals, retaining the core id and remapping links and overlaps.

    Each piece is a new lane. Its centre line, both boundaries and width samples are cut at the piece's fractions of
    each line's own length; the boundaries become unshared copies with their marks. It keeps the other fields of the
    lane it is cut from, names the original Apollo lane in ``source_id``, lies on a road of its own and has no
    neighbour links. Apollo's link id lists are emptied; only predecessors and successors are rebuilt. Pieces of one
    lane follow one another, and a link between two lanes now runs from the last piece of the first to the first
    piece of the second. A signal overlap on a cut lane moves to the first piece that ends beyond its station: one
    at a cut goes to the later piece, one at or past the lane's end to the last piece. The station is then measured
    from that piece's start and clamped to the piece; on a 100 m lane cut in half, 50 m becomes 0 m on the tail and
    150 m becomes 50 m.

    :param MapData data: map changed in place; ``data.lanes`` holds each cut lane's pieces in its place, in order
    :param Dict[Lane, List[Tuple[float, float, str]]] cuts: lane to its pieces as ``(a, b, role)``: fractions of
        the lane's length, in driving order and covering 0 to 1 without gaps, and a role of ``'head'``, ``'core'``
        or ``'tail'``. Exactly one piece is the core: it keeps the lane's id and is outside any junction. The others
        take ``{id}~{tag}-{role}`` (``~`` appended while that is taken) and keep the lane's junction, or get
        ``~{tag}:{id}`` when it has none
    :param str tag: what the cut is for, ``'cycle'`` or ``'entry'``; it names the pieces, their roads
                    (``~{tag}-road:{piece id}``) and their junction when the lane has none
    """
    occupied = set(data.lanes)
    pieces, ranges = {}, {}
    for lane in data.lanes.values():
        if lane not in cuts:
            pieces[lane] = [lane]
            continue
        length = _length(lane.center)
        intervals = cuts[lane]
        split = []
        # Create a new lane for each interval
        for a, b, role in intervals:
            name = lane.id
            if role != 'core':
                name = f'{lane.id}~{tag}-{role}'
                while name in occupied:
                    name += '~'
                occupied.add(name)
            boundaries = []
            # Cut each boundary at the same fractions of its own length; the piece gets its own copy, no longer shared
            for ref in (lane.left, lane.right):
                size = _length(ref.points)
                boundaries.append(BoundaryRef(Boundary(_slice(ref.points, a * size, b * size), ref.boundary.kind),
                                              marks=_slice_marks(ref, a * size, b * size)))
            widths = lane.width_samples
            if widths is not None and len(widths):
                samples = np.r_[a, widths[(widths[:, 0] > a) & (widths[:, 0] < b), 0], b]
                widths = np.column_stack([(samples - a) / (b - a),
                                          np.interp(samples, widths[:, 0], widths[:, 1])])
            part = replace(lane, id=name, source_id=lane.source_id or lane.id,
                           center=_slice(lane.center, a * length, b * length),
                           left=boundaries[0], right=boundaries[1], width_samples=widths,
                           road=f'~{tag}-road:{name}',
                           junction=None if role == 'core' else lane.junction or f'~{tag}:{lane.id}',
                           left_forward=None, right_forward=None, left_reverse=None,
                           predecessors=[], successors=[], apollo_predecessors=[], apollo_successors=[])
            split.append(part)
        pieces[lane] = split
        ranges[lane.id] = [(p, a * length, b * length) for p, (a, b, _) in zip(split, intervals)]

    edges = [(lane, other) for lane in data.lanes.values() for other in lane.successors]
    # Update predecessors and successors for each lane in the piece
    for parts in pieces.values():
        for lane in parts:
            lane.predecessors, lane.successors = [], []
        for a, b in zip(parts, parts[1:]):
            a.successors.append(b)
            b.predecessors.append(a)
    # Update successors and predecessors for each edge
    for a, b in edges:
        start, end = pieces[a][-1], pieces[b][0]
        start.successors.append(end)
        end.predecessors.append(start)
    # Update overlap lanes for each control
    for control in data.controls:
        overlaps = []
        for lane_id, s in control.overlap_lanes:
            if lane_id not in ranges:
                overlaps.append((lane_id, s))
                continue
            span = next((span for span in ranges[lane_id] if s < span[2]), ranges[lane_id][-1])
            lane, a, b = span
            overlaps.append((lane.id, min(max(s - a, 0.0), b - a)))
        control.overlap_lanes = overlaps
    data.lanes = {part.id: part for parts in pieces.values() for part in parts}
