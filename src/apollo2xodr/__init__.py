"""
Convert Apollo HD maps (``base_map.bin`` / ``.txt``) to OpenDRIVE 1.4 (``.xodr``).

>>> import apollo2xodr
>>> apollo2xodr.convert('base_map.bin', 'map.xodr')               # LGSVL / Scenic frame
>>> apollo2xodr.convert('base_map.bin', 'map.xodr', frame='utm')  # plain UTM coordinates
"""

from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree
from .frames import FRAMES, Frame
from .geometry import FIT_TOLERANCE
from .opendrive import build_document, write_document
from .reader import read_map
from .roads import build_roads
from .signals import place_signals, SignalLayout
from .topology import build_topology
from .model import MapData
from .roads import Road
from .topology import Topology

__version__ = '0.1.0'
__all__ = ['convert', 'build', 'FRAMES']

def build(map_path: Path, frame: str = 'lgsvl', tolerance: float = FIT_TOLERANCE) -> ElementTree.ElementTree:
    """
    Convert an Apollo map and return the OpenDRIVE document as an ``xml.etree.ElementTree.ElementTree``.
    ``tolerance`` (m) bounds how far the written reference lines may leave the Apollo polylines.

    :param Path path: Path to the Apollo map file
    :param str frame: Frame to use for the output, default is 'lgsvl' (LGSVL/Scenic frame)
    :param float tolerance: Tolerance for the output
    :returns: OpenDRIVE document as an ``xml.etree.ElementTree.ElementTree``
    :rtype: ElementTree.ElementTree
    """
    if frame not in FRAMES:
        raise ValueError(f'frame must be one of {FRAMES}, not {frame!r}')
    data: MapData = read_map(map_path, simplify_tolerance=tolerance / 2)
    roads: List[Road] = build_roads(data, tolerance)
    topo: Topology = build_topology(data, roads)
    layout: SignalLayout = place_signals(data, roads, topo, first_controller_id=len(roads) + len(topo.junctions))
    return build_document(roads, topo, layout, Frame(frame, data.origin, data.utm_zone, data.projection))

def convert(map_path, output: Optional[str] = None, frame: str = 'lgsvl', tolerance: float = FIT_TOLERANCE) -> Path:
    """
    Convert an Apollo map to an ``.xodr`` file (default: next to the map, named after its folder).
    
    :param str map_path: path to the Apollo map file, will be converted to a Path object
    :param Optional[str] output: path to the output file, will be converted to a Path object
    :param str frame: frame to use for the output, default is 'lgsvl' (LGSVL/Scenic frame)
    :param float tolerance: tolerance for the output, default is 0.05 m
    :returns: path to the output file
    :rtype: Path
    """
    map_path: Path = Path(map_path)
    output: Path = Path(output) if output else map_path.parent / f'{map_path.resolve().parent.name}.xodr'
    write_document(build(map_path, frame, tolerance), output)
    return output
