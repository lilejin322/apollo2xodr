"""
Group Apollo lanes and enumerate the paths through junction regions.
Planning operates on lane connectivity and shared boundaries, before reference
lines or lane-width polynomials are constructed.
"""

import numpy as np
import networkx as nx
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple
from .. import geometry
from ..model import Lane, MapData

PARALLEL_TURN = np.radians(12.0)
"""neighbours whose directions differ more than this at an end are not side by side"""
SHORT_ISLAND = 8.0
"""m, largest multi-lane island absorbed into surrounding junction paths"""

@dataclass(eq=False)
class RoadPlan:
    """
    Lanes that would form one road: side by side, sharing boundaries.
    """
    right: List[Lane]
    """driven along the future road's s, left to right"""
    left: List[Lane]
    """driven against s; empty for a one-way road"""

    @property
    def lanes(self) -> List[Lane]:
        """
        All lanes of the road, right then left.

        :returns: list of lanes
        :rtype: List[Lane]
        """
        return self.right + self.left

    def beyond(self, at_end: bool) -> List[Lane]:
        """
        Lanes linked to this road's lanes across its start (or end).
        Right lanes run along s, so their successors lie beyond the end; left lanes run against s, so their
        predecessors do.

        :param at_end: True for the lanes beyond the road's end, False for those beyond its start
        :returns: linked lanes, those of the right lanes first; a lane linked to several of this road's lanes repeats
        :rtype: List[Lane]
        """
        if at_end:
            return [n for l in self.right for n in l.successors] + [n for l in self.left for n in l.predecessors]
        return [n for l in self.right for n in l.predecessors] + [n for l in self.left for n in l.successors]

def lane_order(data: MapData) -> List[Lane]:
    """
    Lanes of single-lane roads, then of multi-lane roads, then of junction roads (by junction), Apollo order.
    Each Apollo road is classified as a whole by its first lane's junction; the reader does not guarantee that all
    lanes of a road share one (a road without a junction id takes each lane's own). Roads follow Apollo's road list,
    then roads only lanes name; junctions follow Apollo's junction list, then the remaining ids sorted; lanes of one
    road keep the map's lane order. Lane grouping and the order of road plans start from lanes in this order.

    :param data: the Apollo map
    :returns: every lane of ``data`` exactly once
    :rtype: List[Lane]
    """
    by_road: Dict[str, List[Lane]] = {}
    for lane in data.lanes.values():
        by_road.setdefault(lane.road, []).append(lane)
    # roads Apollo lists first, then road ids only lanes carry ('lane:{id}', cycle pieces)
    roads = data.road_order + [r for r in by_road if r not in data.road_order]
    # classify each Apollo road by its first lane's junction, assuming uniform membership (the reader does not
    # guarantee it: a road without a junction id takes each lane's own)
    plain = [by_road[r] for r in roads if r in by_road and by_road[r][0].junction is None]
    ordered = [l for lanes in plain if len(lanes) == 1 for l in lanes]
    ordered += [l for lanes in plain if len(lanes) > 1 for l in lanes]
    # junctions Apollo lists first, then ids only lanes carry (e.g. '~cycle:{id}' from a cycle split), sorted
    junctions = data.junction_order + sorted({l.junction for l in data.lanes.values() if l.junction} -
                                             set(data.junction_order))
    for junction in junctions:
        ordered += [l for r in roads if r in by_road and by_road[r][0].junction == junction for l in by_road[r]]
    return ordered

def lane_groups(data: MapData, order: List[Lane],
                parting: Set[frozenset] = frozenset()) -> List[List[Lane]]:
    """
    Runs of same-direction neighbours of one Apollo road that share a boundary and are not in ``parting``, left to
    right. Junction lanes stay alone. Neighbours that part (Apollo sometimes puts a lane leaving a roundabout beside
    the ring lane it leaves) cannot be one road, whose lanes keep their widths; they become roads of their own. Their
    headings are not checked here: only ``parting`` keeps them apart, so pass ``parting_pairs(data)``. A one-sided
    neighbour link that leaves a lane out of the run found from it raises ValueError.

    :param MapData data: the Apollo map, whose lanes resolve neighbour ids
    :param List[Lane] order: lanes in the order runs are started (see ``lane_order``); groups come out in the order
        their first lane is met
    :param Set[frozenset] parting: neighbour pairs never put in one group, normally ``parting_pairs(data)``; the
        default keeps no pair apart, so neighbours of any heading are merged
    :returns: groups, each from left to right in the driving direction, holding every lane of ``order`` exactly once;
        a group may also hold neighbours outside ``order``, and no lane is in two groups
    :rtype: List[List[Lane]]
    """
    lanes = data.lanes
    grouped = set()
    groups = []

    def neighbour(lane: Lane, side: Literal['left', 'right']) -> Optional[Lane]:
        """
        Same-direction neighbour on ``side`` that may join ``lane``'s group: one of the same plain Apollo road, sharing
        the boundary between them and not parting from ``lane``.

        :param lane: the lane whose neighbour is looked up
        :param side: 'left' or 'right', in ``lane``'s driving direction
        :returns: the neighbour, or None
        :rtype: Optional[Lane]
        """
        other = lanes.get(lane.left_forward if side == 'left' else lane.right_forward or '')
        if other is None or other.road != lane.road or other.junction is not None:
            return None
        # the very same Boundary object, not merely a nearby polyline
        shared = lane.left.boundary is other.right.boundary if side == 'left' else \
            lane.right.boundary is other.left.boundary
        return other if shared and frozenset((lane, other)) not in parting else None

    for lane in order:
        if lane in grouped:
            continue
        if lane.junction is not None:
            group = [lane]
        else:
            # walk left to the leftmost lane; `seen` stops on neighbour links that loop
            leftmost, seen = lane, {lane}
            while neighbour(leftmost, 'left') not in (None, *seen):
                leftmost = neighbour(leftmost, 'left')
                seen.add(leftmost)
            # then collect the run from there to the right
            group, cur = [leftmost], leftmost
            while neighbour(cur, 'right') not in (None, *group):
                cur = neighbour(cur, 'right')
                group.append(cur)
            # a one-sided neighbour link leaves the lane out of the run found from it, unassigned for good
            if lane not in group:
                raise ValueError(f'Lane {lane.id} is not in the run of neighbours found from it '
                                 f'({", ".join(l.id for l in group)}); neighbour links must be mutual')
        # remove lanes already assigned to a previous group
        group = [l for l in group if l not in grouped]
        grouped.update(group)
        groups.append(group)
    return groups

def parting_pairs(data: MapData) -> Set[frozenset]:
    """
    Pairs of same-direction neighbours that part (as Apollo draws them, before any smoothing).
    Their centre lines head more than ``PARALLEL_TURN`` apart at the start or at the end. The pairs are not checked
    for a shared boundary or a common road; ``lane_groups`` consults them only for neighbours that pass both checks.
    Call it before ``smooth_sketched_lanes`` changes the centre lines.

    :param data: the Apollo map
    :returns: unordered pairs of lanes, each found from either lane's ``left_forward`` or ``right_forward``
    :rtype: Set[frozenset]
    """
    pairs = set()
    for lane in data.lanes.values():
        for other_id in (lane.left_forward, lane.right_forward):
            other = data.lanes.get(other_id or '')
            if other is not None and not _parallel(lane, other):
                pairs.add(frozenset((lane, other)))  # unordered: found from either lane
    return pairs

def _parallel(a: Lane, b: Lane) -> bool:
    """
    Whether the centre lines of ``a`` and ``b`` head within ``PARALLEL_TURN`` of each other at both ends.
    Start is compared with start and end with end, using the unit direction of each centre line's first (or last)
    non-degenerate segment, so their dot product is the cosine of the angle between them.

    :param a: one lane
    :param b: the other lane, driven in the same direction
    :returns: False if the headings differ by more than ``PARALLEL_TURN`` at either end; True otherwise, including at
              an end where either centre line has no direction
    :rtype: bool
    """
    for at_start in (True, False):
        da, db = geometry.end_direction(a.center, at_start), geometry.end_direction(b.center, at_start)
        # an end without a direction (degenerate centre line) does not count against the pair
        if da is not None and db is not None and float(da @ db) < np.cos(PARALLEL_TURN):
            return False
    return True

def plan_roads(data: MapData, parting: Set[frozenset] = frozenset()) -> List[RoadPlan]:
    """
    Lane groups, two-way pairs merged, returned in breadth-first order along the driving direction from each start.
    Road ids are not assigned here; ``build_roads`` numbers the plain plans in this order. Two groups pair up when
    their leftmost lanes are reverse neighbours of one Apollo road sharing their left boundary (the centre line); a
    group pairs at most once. Traversal starts at lanes without predecessors, then at lanes left unreached (on or
    behind a loop), in ``lane_order``. A group reached first becomes the right side of its plan, its partner (if any)
    the left side; the successors of both are queued next. The ValueError of ``lane_groups`` passes through.

    :param MapData data: the Apollo map
    :param Set[frozenset] parting: neighbour pairs never put in one group, normally ``parting_pairs(data)``; the
        default keeps no pair apart (see ``lane_groups``)
    :returns: one plan per group or pair of groups, every lane in exactly one plan, in traversal order
    :rtype: List[RoadPlan]
    """
    order = lane_order(data)
    groups = lane_groups(data, order, parting)
    group_of = {lane: i for i, group in enumerate(groups) for lane in group}

    # two groups whose leftmost lanes are reverse neighbours of one Apollo road and share their left boundary
    partner: Dict[int, int] = {}
    for i, group in enumerate(groups):
        other = data.lanes.get(group[0].left_reverse) if group[0].left_reverse else None
        if other is None or other.road != group[0].road or group_of.get(other) is None or \
                other.left.boundary is not group[0].left.boundary:
            continue
        j = group_of[other]
        # `other` must lead its own group too, and each group takes one partner only
        if j != i and groups[j][0] is other and i not in partner and j not in partner:
            partner[i], partner[j] = j, i

    # a two-way road follows the side reached first
    # traversal starts: lanes nothing leads into, then (below) lanes only reachable from a loop
    starts = [l for l in order if not l.predecessors]
    graph = nx.DiGraph((lane, other) for lane in order for other in lane.successors)
    graph.add_nodes_from(order)
    reached = {lane for layer in nx.bfs_layers(graph, starts) for lane in layer}
    starts += [l for l in order if l not in reached]

    plans: List[RoadPlan] = []
    done = set()
    for start in starts:
        queue = deque([start])
        while queue:
            lane = queue.popleft()
            i = group_of[lane]
            if i in done:
                continue
            j = partner.get(i)
            done.update({i, j} - {None})
            # the group reached first becomes the right side, so its lanes run along s
            plans.append(RoadPlan(groups[i], groups[j] if j is not None else []))
            # expand both sides in group order, so the queue does not depend on which side was reached first
            considered = groups[min(i, j)] + groups[max(i, j)] if j is not None else groups[i]
            queue.extend(l for c in considered for l in c.successors if group_of[l] not in done)
    return plans

def connecting_plans(plans: List[RoadPlan], plan_of: Dict[Lane, RoadPlan]) -> Set[RoadPlan]:
    """
    Apollo junction lanes, short islands between junctions, and split/merge roads.
    A plain road end links to one road or to one junction, so beyond it lies a single plain road or connecting
    material only. A short two-way, multi-lane island enclosed by junctions is traversed by junction paths; otherwise
    it pins both ends of several very short turns and leaves no room for their lane borders.

    :param List[RoadPlan] plans: road plans from ``plan_roads``
    :param Dict[Lane, RoadPlan] plan_of: the plan holding each lane of ``plans``
    :returns: plans to write as connecting material: those whose first right lane is in an Apollo junction, those
              beyond a plain road end that meets several plans, has a lane with several neighbours at that end,
              or meets connecting material (grown until nothing changes), and short islands (two-way, at least
              three lanes, none longer than ``SHORT_ISLAND``, every lane entered from and leaving into connecting
              material only)
    :rtype: Set[RoadPlan]
    """
    connecting = {plan for plan in plans if plan.right[0].junction is not None}
    # Grow to a fixed point. A lane may split/merge with several lanes of just one neighbouring plan; counting plans
    # alone would miss that and ordinary lane links could not represent all its neighbours.
    changed = True
    while changed:
        changed = False
        for plan in plans:
            if plan in connecting:
                continue
            for at_end in (False, True):
                beyond = {plan_of[n] for n in plan.beyond(at_end)}
                branching = any(len(lane.successors if at_end == (lane in plan.right) else lane.predecessors) > 1
                                for lane in plan.lanes)
                if len(beyond) > 1 or branching or beyond & connecting:
                    added = beyond - connecting
                    if added:
                        connecting |= added
                        changed = True
    # A tiny multi-lane island between two junctions is not a useful plain road: making it
    # one fixes each connector's heading at both ends only a few metres apart. Let paths
    # traverse the island so their turns can be fitted over the whole crossing instead.
    for plan in plans:
        # two-way with at least three lanes
        if plan in connecting or not plan.right or not plan.left or len(plan.lanes) < 3:
            continue
        if max(geometry.arc_lengths(lane.center)[-1] for lane in plan.lanes) > SHORT_ISLAND:
            continue
        # enclosed: every lane is entered from and leaves into connecting material only
        if all(lane.predecessors and lane.successors and
               all(plan_of[n] in connecting for n in lane.predecessors + lane.successors)
               for lane in plan.lanes):
            connecting.add(plan)
    return connecting

def junction_regions(plans: List[RoadPlan], plan_of: Dict[Lane, RoadPlan],
                     connecting: Set[RoadPlan]) -> Dict[RoadPlan, str]:
    """
    Junction region of every connecting plan, named after its first Apollo junction.
    Linked connecting material is one region, and so is everything met at one plain road end (it links to a single
    junction). Plans of one Apollo junction end up in one region, a plan counting by its first right lane's junction
    only: a paired lane of another junction (possible where an Apollo road without a junction id mixes junctions)
    goes to the region of its plan, so one Apollo junction's lanes can be split across regions. One region may hold
    several Apollo junctions. Regions are the connected components of an undirected graph over the connecting plans.
    A region takes the junction id of its first plan in ``plans`` order, or ``'~N'`` without one; a name starting
    with ``'~'`` (``'~N'``, or ``'~cycle:{id}'`` from a cycle split) gives way to the id of a later plan.

    :param List[RoadPlan] plans: road plans from ``plan_roads``
    :param Dict[Lane, RoadPlan] plan_of: the plan holding each lane of ``plans``
    :param Set[RoadPlan] connecting: connecting plans from ``connecting_plans``
    :returns: region name of every connecting plan; plans with one name form one junction
    :rtype: Dict[RoadPlan, str]
    """
    # undirected graph over connecting plans; each connected component is one region
    graph = nx.Graph()
    graph.add_nodes_from(connecting)
    by_apollo: Dict[str, List[RoadPlan]] = {}
    for plan in plans:
        if plan in connecting:
            # connecting plans linked lane to lane
            neighbours = {plan_of[n] for lane in plan.lanes for n in lane.successors if plan_of[n] in connecting}
            graph.add_edges_from((plan, other) for other in neighbours if other is not plan)
            if plan.right[0].junction is not None:
                by_apollo.setdefault(plan.right[0].junction, []).append(plan)
        else:
            # connecting plans met at one plain road end; a path through them is enough to join them
            for at_end in (False, True):
                neighbours = [plan_of[n] for n in plan.beyond(at_end) if plan_of[n] in connecting]
                if len(neighbours) > 1:
                    nx.add_path(graph, neighbours)
    # plans of one Apollo junction (by their first right lane), even where no lane links them
    for group in by_apollo.values():
        if len(group) > 1:
            nx.add_path(graph, group)

    component_of = {}
    for component in nx.connected_components(graph):
        frozen = frozenset(component)
        for plan in frozen:
            component_of[plan] = frozen
    # Name each region after the first Apollo junction met in plan order. A region without one gets a
    # synthetic '~N' name, which a later plan carrying an Apollo junction id replaces.
    names: Dict[frozenset, str] = {}
    for plan in plans:
        if plan in connecting:
            root = component_of[plan]
            if root not in names or (names[root].startswith('~') and plan.right[0].junction is not None):
                names[root] = plan.right[0].junction or f'~{len(names)}'
    return {plan: names[component_of[plan]] for plan in connecting}

def connecting_paths(plans: List[RoadPlan], connecting: Set[RoadPlan]) -> List[Tuple[Optional[Lane], 
                                                                                     List[Lane], Optional[Lane]]]:
    """
    (incoming plain lane, connecting lanes, outgoing plain lane) of every path through connecting material.
    The connecting lanes must form no cycle (``split_cycles`` removes them); a remaining cycle raises ValueError before
    any path is enumerated. Paths start at every lane a plain lane leads into and at lanes without predecessors; any
    lane not reached that way starts a path of its own, so every connecting lane is written at least once. From each
    start, every route through the material is followed depth first; a route yields one path per plain lane it leaves
    into, and may go on inside as well. Lanes where routes fork or merge therefore appear in several paths.

    :param List[RoadPlan] plans: road plans from ``plan_roads``; their order (and each plan's right lanes, then left)
        sets the order of the starts
    :param Set[RoadPlan] connecting: connecting plans from ``connecting_plans``
    :returns: paths as (incoming, lanes, outgoing); incoming is None for a path started at a lane without
        predecessors (or by the fallback), outgoing None for one ending at a lane without successors
    :rtype: List[Tuple[Optional[Lane], List[Lane], Optional[Lane]]]
    """
    order = [lane for plan in plans if plan in connecting for lane in plan.lanes]
    material = set(order)
    graph = nx.DiGraph(
        (lane, other) for lane in material for other in lane.successors if other in material
    )
    # cycles should have been split by split_cycles; enumerating paths through one would never end
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('A cycle remains in connecting material; cannot enumerate junction paths')
    paths, covered = [], set()

    def walk(first: Lane, incoming: Optional[Lane]) -> None:
        """
        Record every path from ``first`` until it leaves connecting material or dead-ends (depth first).
        Paths are appended to ``paths`` and their lanes to ``covered``.

        :param first: connecting lane the paths start at
        :param incoming: plain lane leading into ``first``, or None
        """
        stack = [[first]]  # partial paths still to extend
        while stack:
            lanes = stack.pop()
            covered.update(lanes)
            inner = [n for n in lanes[-1].successors if n in material]
            outer = [n for n in lanes[-1].successors if n not in material]
            # one path per plain lane this one leaves into; it may still continue inside as well
            paths.extend((incoming, lanes, n) for n in outer)
            if not inner and not outer:
                paths.append((incoming, lanes, None))  # dead end, e.g. at the edge of the map
            # reversed so the first successor is popped, and its paths recorded, first
            stack.extend(lanes + [n] for n in reversed(inner))

    for lane in order:
        # one walk per plain lane leading in, so each entry gets its own paths
        for pred in lane.predecessors:
            if pred not in material:
                walk(lane, pred)
        if not lane.predecessors:
            walk(lane, None)
    # fallback: start paths at any connecting lanes not covered by the entry walks (with links kept mutual and the
    # material acyclic, the entry walks already cover every lane)
    for lane in order:
        if lane not in covered:
            walk(lane, None)
    return paths
