"""
UTM conversion and the output coordinate frames.

All geometry is computed in the local frame ``(e, n) = (easting - E0, northing - N0)``. A frame maps local points and
headings to the coordinates written to the xodr file:

* ``lgsvl`` (default): the LGSVL Simulator scene frame used by Scenic & LGSVL maps,
  ``x = N0 - northing, y = easting - E0`` (the local frame turned by +90 degrees).
* ``utm``: plain UTM coordinates, ``x = easting, y = northing``.
"""

import math
import utm
from dataclasses import dataclass
from typing import Tuple

def utm_zone(lat: float, lon: float) -> int:
    """
    UTM zone number for a WGS84 latitude and longitude, including the Norway and Svalbard exceptions.
    
    :param float lat: latitude in degrees
    :param float lon: longitude in degrees
    :returns: UTM zone number
    :rtype: int
    """
    return utm.latlon_to_zone_number(lat, lon)

def latlon_to_utm(lat: float, lon: float, zone: int) -> Tuple[float, float]:
    """
    (easting, northing) of a WGS84 point forced into ``zone``. Southern points use the 10 000 km false northing.
    
    :param float lat: latitude in degrees
    :param float lon: longitude in degrees
    :param int zone: UTM zone number
    :returns: (easting, northing)
    :rtype: Tuple[float, float]
    """
    easting, northing, _, _ = utm.from_latlon(lat, lon, force_zone_number=zone)
    return float(easting), float(northing)

def utm_to_latlon(easting: float, northing: float, zone: int) -> Tuple[float, float]:
    """
    (lat, lon) of a northern-hemisphere UTM point. Map origins passed here are north of the equator.
    
    :param float easting: easting in meters
    :param float northing: northing in meters
    :param int zone: UTM zone number
    :returns: (latitude, longitude)
    :rtype: Tuple[float, float]
    """
    lat, lon = utm.to_latlon(easting, northing, zone, northern=True, strict=False)
    return float(lat), float(lon)

@dataclass(frozen=True)
class Frame:
    """
    Maps local (e, n) coordinates and headings to output coordinates.
    """
    name: str
    """Name of the frame"""
    origin: Tuple[float, float]
    """UTM easting and northing of the local map origin (E0, N0), in meters"""
    zone: int
    """UTM zone number of the local map origin"""
    projection: str
    """Apollo header projection string"""

    def point(self, e: float, n: float) -> Tuple[float, float]:
        """
        Convert local (e, n) coordinates to output coordinates.

        :param float e: local easting coordinate in meters
        :param float n: local northing coordinate in meters
        :returns: (output easting, output northing)
        :rtype: Tuple[float, float]
        """
        if self.name == 'lgsvl':   # LGSVL/Scenic frame, counter-clockwise rotation by 90 degrees
            return -n, e
        return e + self.origin[0], n + self.origin[1]  # UTM frame, translation by (E0, N0)

    def heading(self, hdg: float) -> float:
        """
        Convert local heading to output heading.

        :param float hdg: local heading in radians
        :returns: output heading in radians
        :rtype: float
        """
        if self.name == 'lgsvl':  # LGSVL/Scenic frame, counter-clockwise rotation by 90 degrees
            hdg += math.pi / 2
        return math.atan2(math.sin(hdg), math.cos(hdg))  # covert into [-pi, pi] range

    def geo_reference(self) -> str:
        """
        PROJ geo-reference string for the output OpenDRIVE map.

        :returns: PROJ geo-reference string
        :rtype: str
        """
        if self.name == 'utm':
            return f' {self.projection.strip()} '
        lat, lon = utm_to_latlon(self.origin[0], self.origin[1], self.zone)
        return (f' +proj=tmerc +lat_0={lat:.15g} +lon_0={lon:.15g} +k=1 +x_0={self.origin[0]:.15g} '
                f'+y_0={self.origin[1]:.15g} +datum=WGS84 +units=m +no_defs ')

FRAMES: Tuple[str, str] = ('lgsvl', 'utm')
"""Supported frames"""
