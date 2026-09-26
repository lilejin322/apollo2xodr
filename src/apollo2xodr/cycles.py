"""
Break cycles in connecting material by retaining ordinary road cores inside the ring.

Each cyclic source lane becomes head -> ordinary core -> tail. Junction paths can then
connect those cores without unrolling a loop or dropping its closing edge. Internal IDs
are unique; OpenDRIVE userData continues to name the original Apollo lane.
"""

import networkx as nx
import numpy as np
from dataclasses import replace
from typing import Set, List, Tuple
from shapely.geometry import LineString, Point
from shapely.ops import substring
from .model import Boundary, BoundaryRef, MapData, Lane

def _slice_marks(ref: BoundaryRef, start: float, end: float) -> List[Tuple[np.ndarray, str]]:
    """
    Return marking states covering the arc-length window ``[start, end)``.
    
    :param BoundaryRef ref: boundary reference
    :param float start: start of the arc-length window
    :param float end: end of the arc-length window
    :returns: marking states covering the arc-length window
    :rtype: List[Tuple[np.ndarray, str]]
    """
    if not ref.marks:        # no marks to slice, return []
        return []

    points = ref.points

    if len(points) < 2:      # a single point cannot have a marking
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
    :param Set[Lane] material: connecting lanes searched for cycles; only a cyclic component and adjoining ends outside it are cut
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

    occupied = set(data.lanes)
    pieces, ranges = {}, {}
    for lane in data.lanes.values():
        if lane not in cyclic | adjoining:
            pieces[lane] = [lane]
            continue
        length = _length(lane.center)
        if length <= 1e-9:
            if lane in cyclic:
                raise ValueError(f"Cannot split cyclic lane {lane.id!r}: centerline has zero length")
            pieces[lane] = [lane]
            continue
        # Keep the core short so the connecting pieces have room to make a finite-width turn.
        half_core = min(5.0, length / 10) / (2 * length)
        has_head = lane in cyclic or any(p in cyclic for p in lane.predecessors)
        has_tail = lane in cyclic or any(p in cyclic for p in lane.successors)
        a = 0.5 - half_core if has_head else 0.0
        b = 0.5 + half_core if has_tail else 1.0
        intervals = ([(0.0, a, 'head')] if has_head else []) + [(a, b, 'core')] + \
                    ([(b, 1.0, 'tail')] if has_tail else [])
        split = []
        # Create a new lane for each interval
        for a, b, role in intervals:
            name = lane.id
            if role != 'core':
                name = f'{lane.id}~cycle-{role}'
                while name in occupied:
                    name += '~'
                occupied.add(name)
            boundaries = []
            for ref in (lane.left, lane.right):
                size = _length(ref.points)
                boundaries.append(BoundaryRef(Boundary(_slice(ref.points, a * size, b * size), ref.boundary.kind),
                                              marks=_slice_marks(ref, a * size, b * size)))
            widths = lane.width_samples
            if widths is not None:
                samples = np.r_[a, widths[(widths[:, 0] > a) & (widths[:, 0] < b), 0], b]
                widths = np.column_stack([(samples - a) / (b - a),
                                          np.interp(samples, widths[:, 0], widths[:, 1])])
            part = replace(lane, id=name, source_id=lane.source_id or lane.id,
                           center=_slice(lane.center, a * length, b * length),
                           left=boundaries[0], right=boundaries[1], width_samples=widths,
                           road=f'~cycle-road:{name}',
                           junction=None if role == 'core' else lane.junction or f'~cycle:{lane.id}',
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
    return True
