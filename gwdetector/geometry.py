# ============================================================
# gwdetectors.geometry — detector siting, arm vectors, response tensor
#
# Conventions follow bilby (and therefore LALSuite), so a Detector built here
# has the same `detector_tensor` as the equivalent bilby Interferometer:
#
#   * latitude / longitude in degrees (geodetic, WGS84), elevation in metres
#   * xarm_azimuth / yarm_azimuth in degrees, measured COUNTER-CLOCKWISE
#     FROM LOCAL EAST (LAL stores the same angle clockwise from North; the
#     two differ by az_bilby = 90 - az_LAL)
#   * arm tilts in radians above the local horizontal plane
#
# The response tensor of an L-shaped interferometer with arm unit vectors
# u (x-arm) and v (y-arm) is
#
#     D^{ab} = (1/2) (u^a u^b - v^a v^b)
#
# so that h(t) = D^{ab} h_{ab}(t).
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# WGS84 reference ellipsoid
WGS84_A = 6_378_137.0                  # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563          # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)   # first eccentricity squared

C_LIGHT = 299_792_458.0                # [m/s]
R_EARTH_MEAN = 6_371_000.0             # mean radius, used for triangle layout


def geodetic_to_cartesian(lat_deg: float, lon_deg: float,
                          elevation_m: float = 0.0) -> np.ndarray:
    """Geodetic (lat, lon, h) -> geocentric Cartesian [m], WGS84."""
    lat = np.radians(float(lat_deg))
    lon = np.radians(float(lon_deg))
    h = float(elevation_m)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([
        (N + h) * np.cos(lat) * np.cos(lon),
        (N + h) * np.cos(lat) * np.sin(lon),
        (N * (1.0 - WGS84_E2) + h) * np.sin(lat),
    ], dtype=float)


def local_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local (East, North, Up) unit vectors in the geocentric frame."""
    lat = np.radians(float(lat_deg))
    lon = np.radians(float(lon_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon),
                      -np.sin(lat) * np.sin(lon),
                      np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)])
    return east, north, up


def arm_direction(lat_deg: float, lon_deg: float, azimuth_deg: float,
                  tilt_rad: float = 0.0) -> np.ndarray:
    """Unit vector along an arm, from azimuth (CCW from East) and tilt."""
    east, north, up = local_basis(lat_deg, lon_deg)
    az = np.radians(float(azimuth_deg))
    tilt = float(tilt_rad)
    return (np.cos(tilt) * (np.cos(az) * east + np.sin(az) * north)
            + np.sin(tilt) * up)


@dataclass
class Geometry:
    """Where a detector sits and how its arms point."""

    latitude: float                 # [deg]
    longitude: float                # [deg]
    elevation: float = 0.0          # [m]
    xarm_azimuth: float = 0.0       # [deg], CCW from East
    yarm_azimuth: float = 90.0      # [deg], CCW from East
    xarm_tilt: float = 0.0          # [rad]
    yarm_tilt: float = 0.0          # [rad]
    length_km: float = 4.0          # arm length, for reference only

    # cached derived quantities
    _vertex: np.ndarray = field(default=None, repr=False, compare=False)
    _tensor: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def vertex(self) -> np.ndarray:
        """Geocentric position of the corner station [m]."""
        if self._vertex is None:
            self._vertex = geodetic_to_cartesian(
                self.latitude, self.longitude, self.elevation)
        return self._vertex

    @property
    def x_arm(self) -> np.ndarray:
        return arm_direction(self.latitude, self.longitude,
                             self.xarm_azimuth, self.xarm_tilt)

    @property
    def y_arm(self) -> np.ndarray:
        return arm_direction(self.latitude, self.longitude,
                             self.yarm_azimuth, self.yarm_tilt)

    @property
    def detector_tensor(self) -> np.ndarray:
        """D^{ab} = (u^a u^b - v^a v^b)/2, shape (3, 3)."""
        if self._tensor is None:
            u, v = self.x_arm, self.y_arm
            self._tensor = 0.5 * (np.outer(u, u) - np.outer(v, v))
        return self._tensor

    @property
    def opening_angle_deg(self) -> float:
        """Angle between the two arms [deg] (90 for L-shaped, 60 for ET)."""
        return float(np.degrees(np.arccos(
            np.clip(np.dot(self.x_arm, self.y_arm), -1.0, 1.0))))


def geometry_from_vectors(vertex, x_arm, y_arm) -> Geometry:
    """A Geometry built straight from Cartesian vectors.

    For detectors that arrive as vertex + arm directions rather than as
    (latitude, longitude, azimuth) — bilby interferometers, or the synthetic
    spherical pairs `orf.selftest` compares against the literature. Only the
    vertex and the detector tensor are used downstream, so those are set
    directly and the geodetic fields are left at their defaults rather than
    back-solved into angles that nothing reads.
    """
    v = np.asarray(vertex, float)
    u = np.asarray(x_arm, float)
    w = np.asarray(y_arm, float)
    u = u / np.linalg.norm(u)
    w = w / np.linalg.norm(w)
    geo = Geometry(latitude=0.0, longitude=0.0)
    geo._vertex = v
    geo._tensor = 0.5 * (np.outer(u, u) - np.outer(w, w))
    return geo


def separation(geo_a: Geometry, geo_b: Geometry) -> Tuple[np.ndarray, float]:
    """Baseline vector (a - b) [m] and its length [m]."""
    d = geo_a.vertex - geo_b.vertex
    return d, float(np.linalg.norm(d))


def light_travel_time(geo_a: Geometry, geo_b: Geometry) -> float:
    """Light travel time across the baseline [s]."""
    return separation(geo_a, geo_b)[1] / C_LIGHT


def triangle_geometries(latitude, longitude, elevation, xarm_azimuth,
                        length_km=10.0, opening_deg=60.0):
    """Three nested interferometers of a triangular observatory (e.g. ET).

    Reproduces bilby's TriangularInterferometer layout: each successive
    interferometer is rotated by 240 deg in azimuth and its vertex is moved to
    the next corner of the triangle.
    """
    geos = []
    lat, lon, az = float(latitude), float(longitude), float(xarm_azimuth)
    for _ in range(3):
        geos.append(Geometry(
            latitude=lat, longitude=lon, elevation=float(elevation),
            xarm_azimuth=az, yarm_azimuth=az + float(opening_deg),
            length_km=float(length_km),
        ))
        az += 240.0
        lat += np.degrees(np.arctan(
            length_km * np.sin(np.radians(az)) * 1e3 / R_EARTH_MEAN))
        lon += np.degrees(np.arctan(
            length_km * np.cos(np.radians(az)) * 1e3 / R_EARTH_MEAN))
    return geos
