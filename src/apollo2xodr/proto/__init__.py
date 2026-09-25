"""Apollo HD map protobuf messages, built at runtime from a bundled descriptor set.

``apollo_map.desc`` is a serialized ``FileDescriptorSet`` of ``modules/map/proto/map.proto``
and its dependencies. The message classes are created in a private descriptor pool, so they 
work with any protobuf runtime (pure Python, C++ or upb) and never clash with other Apollo
``*_pb2`` modules loaded in the same process.
"""

import functools
from pathlib import Path
from importlib import resources
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format

_DESCRIPTOR_SET = 'apollo_map.desc'
MAP_MESSAGE = 'apollo.hdmap.Map'

@functools.lru_cache(maxsize=None)
def _pool() -> descriptor_pool.DescriptorPool:
    """
    Get the descriptor pool.

    :returns: Descriptor pool
    :rtype: descriptor_pool.DescriptorPool
    """
    data = resources.files(__package__).joinpath(_DESCRIPTOR_SET).read_bytes()
    files = descriptor_pb2.FileDescriptorSet.FromString(data)
    pool = descriptor_pool.DescriptorPool()
    for file_proto in files.file:  # stored in dependency order
        pool.Add(file_proto)
    return pool

def message_class(full_name: str) -> message_factory.MessageFactory:
    """
    Message class for a fully qualified Apollo type name, e.g. ``apollo.hdmap.Map``.

    :param str full_name: the fully qualified Apollo type name
    :returns: the message class
    :rtype: message_factory.MessageFactory
    """
    descriptor = _pool().FindMessageTypeByName(full_name)
    if hasattr(message_factory, 'GetMessageClass'):  # protobuf >= 4.21
        return message_factory.GetMessageClass(descriptor)
    return message_factory.MessageFactory(_pool()).GetPrototype(descriptor)

def load_map(path: Path):
    """
    Read an Apollo map (binary ``base_map.bin`` or text format ``.txt``).

    :param Path path: Path to the map file
    """
    Map = message_class(MAP_MESSAGE)
    hd_map = Map()
    with open(path, 'rb') as f:
        data = f.read()
    if str(path).endswith('.txt'):
        text_format.Parse(data.decode('utf-8'), hd_map)
    else:
        hd_map.ParseFromString(data)
    return hd_map
