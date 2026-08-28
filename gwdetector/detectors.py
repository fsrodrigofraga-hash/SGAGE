# ============================================================
# gwdetectors.detectors — registry of current and planned observatories
#
# Same pattern as mgtheories: one registry, one interface.
#
#     import gwdetectors as gwd
#     det = gwd.get("H1")
#     det.asd(freqs)                       # strain ASD [1/sqrt(Hz)]
#     det.geometry.detector_tensor         # D^{ab}
#
# Siting and arm azimuths follow bilby's detector files (which follow LAL), so
# `Detector.geometry.detector_tensor` matches bilby's `detector_tensor` to
# machine precision for H1/L1/V1/K1.
#
# HONESTY NOTES on the future detectors
# -------------------------------------
# * COSMIC EXPLORER: no site has been chosen. Following bilby we place CE at
#   the LIGO Hanford site. Any CE baseline, light travel time or ORF computed
#   from it is ILLUSTRATIVE — it will change once sites are fixed. The
#   sensitivity curve itself does not depend on siting.
# * EINSTEIN TELESCOPE: two candidate sites are under study (Sos Enattos in
#   Sardinia and the Euregio Meuse-Rhine). Both are provided. bilby's own "ET"
#   entry instead sits at the Virgo site, kept here as `ET_virgo*` for
#   cross-checking against bilby.
# * LIGO-INDIA (A1): site fixed (Aundha), instrument under construction.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .curves import SensitivityCurve, get_curve
from .geometry import Geometry, light_travel_time, separation, triangle_geometries


@dataclass
class Detector:
    """One interferometer: where it is, how it responds, how loud it is."""

    key: str
    label: str
    geometry: Geometry
    curve_key: str
    generation: str = "2G"           # "2G", "2G+", "3G", "space"
    status: str = "operating"        # operating | construction | proposed
    f_min: float = 10.0
    f_max: float = 2048.0
    reference: str = ""
    notes: str = ""
    color: str = "#333333"
    linestyle: str = "-"

    _curve: Optional[SensitivityCurve] = field(default=None, repr=False,
                                               compare=False)

    # ── sensitivity ────────────────────────────────────────────────────
    @property
    def curve(self) -> SensitivityCurve:
        if self._curve is None:
            self._curve = get_curve(self.curve_key)
        return self._curve

    def psd(self, frequencies) -> np.ndarray:
        """One-sided strain PSD S_n(f) [1/Hz]; NaN outside the usable band."""
        f = np.asarray(frequencies, float)
        out = self.curve(f)
        return np.where((f >= self.f_min) & (f <= self.f_max), out, np.nan)

    def asd(self, frequencies) -> np.ndarray:
        """Strain ASD sqrt(S_n(f)) [1/sqrt(Hz)]."""
        return np.sqrt(self.psd(frequencies))

    # ── geometry shortcuts ─────────────────────────────────────────────
    @property
    def detector_tensor(self) -> np.ndarray:
        return self.geometry.detector_tensor

    def baseline_length(self, other: "Detector") -> float:
        """Distance between the two corner stations [m]."""
        return separation(self.geometry, other.geometry)[1]

    def light_travel_time(self, other: "Detector") -> float:
        return light_travel_time(self.geometry, other.geometry)

    def __repr__(self) -> str:
        return f"<Detector {self.key!r} {self.generation} curve={self.curve_key!r}>"


# ── Registry ───────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, Detector] = {}


def register(det: Detector) -> Detector:
    if det.key in _REGISTRY:
        raise KeyError(f"detector already registered: {det.key!r}")
    _REGISTRY[det.key] = det
    return det


def get(key: str) -> Detector:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"detector {key!r} is not registered. Available: {sorted(_REGISTRY)}"
        ) from None


def all_detectors(generation: Optional[str] = None) -> List[Detector]:
    dets = list(_REGISTRY.values())
    if generation is None:
        return dets
    return [d for d in dets if d.generation == generation]


def keys(generation: Optional[str] = None) -> List[str]:
    return [d.key for d in all_detectors(generation)]


def catalog() -> str:
    """Human-readable table of the registry."""
    rows = []
    for gen in ("2G", "2G+", "3G", "space"):
        dets = all_detectors(gen)
        if not dets:
            continue
        rows.append(f"\n[{gen}]")
        for d in dets:
            rows.append(f"  {d.key:<12} {d.label:<34} {d.curve_key:<14} "
                        f"{d.f_min:>6.1f}-{d.f_max:<7.0f} Hz  {d.status}")
            if d.notes:
                rows.append(f"  {'':<12} note: {d.notes}")
    return "\n".join(rows)


def _dms(d, m, s):
    """Degrees-minutes-seconds -> decimal degrees."""
    return d + m / 60.0 + s / 3600.0


# ── L-shaped ground-based detectors (geometry from bilby/LAL) ──────────────
register(Detector(
    "H1", "LIGO Hanford", Geometry(
        latitude=_dms(46, 27, 18.528), longitude=-_dms(119, 24, 27.5657),
        elevation=142.554, xarm_azimuth=125.9994, yarm_azimuth=215.9994,
        xarm_tilt=-6.195e-4, yarm_tilt=1.25e-5, length_km=4.0),
    "aLIGO_O4", generation="2G", status="operating", f_min=10.0, f_max=2048.0,
    reference="LIGO-T980044-10", color="#EE6677", linestyle="-"))

register(Detector(
    "L1", "LIGO Livingston", Geometry(
        latitude=_dms(30, 33, 46.4196), longitude=-_dms(90, 46, 27.2654),
        elevation=-6.574, xarm_azimuth=197.7165, yarm_azimuth=287.7165,
        xarm_tilt=-3.121e-4, yarm_tilt=-6.107e-4, length_km=4.0),
    "aLIGO_O4", generation="2G", status="operating", f_min=10.0, f_max=2048.0,
    reference="LIGO-T980044-10", color="#CC6677", linestyle="--"))

register(Detector(
    "V1", "Virgo", Geometry(
        latitude=_dms(43, 37, 53.0921), longitude=_dms(10, 30, 16.1878),
        elevation=51.884, xarm_azimuth=70.5674, yarm_azimuth=160.5674,
        length_km=3.0),
    "AdV", generation="2G", status="operating", f_min=10.0, f_max=2048.0,
    color="#4477AA", linestyle="-"))

register(Detector(
    "K1", "KAGRA", Geometry(
        latitude=_dms(36, 24, 42.69722), longitude=_dms(137, 18, 21.44171),
        elevation=414.181, xarm_azimuth=90.0 - 60.39623489157727,
        yarm_azimuth=90.0 + 29.60357629670688,
        xarm_tilt=0.0031414, yarm_tilt=-0.0036270, length_km=3.0),
    "KAGRA_design", generation="2G", status="operating", f_min=10.0,
    f_max=2048.0, color="#228833", linestyle="-"))

register(Detector(
    "GEO600", "GEO600", Geometry(
        latitude=52.246944, longitude=9.808333, elevation=114.425,
        xarm_azimuth=68.7749, yarm_azimuth=291.6117, length_km=0.6),
    "GEO600", generation="2G", status="operating", f_min=40.0, f_max=2048.0,
    color="#999933", linestyle=":"))

# ── upgrades / under construction ──────────────────────────────────────────
register(Detector(
    "H1_Aplus", "LIGO Hanford (A+)", get("H1").geometry, "Aplus",
    generation="2G+", status="proposed", f_min=10.0, f_max=2048.0,
    reference="LIGO-T1800042", color="#EE6677", linestyle="--"))

register(Detector(
    "L1_Aplus", "LIGO Livingston (A+)", get("L1").geometry, "Aplus",
    generation="2G+", status="proposed", f_min=10.0, f_max=2048.0,
    reference="LIGO-T1800042", color="#CC6677", linestyle=":"))

register(Detector(
    "A1", "LIGO-India (Aundha, A+)", Geometry(
        latitude=_dms(19, 36, 47.9017), longitude=_dms(77, 1, 51.0997),
        elevation=440.0, xarm_azimuth=117.6157, yarm_azimuth=207.6165,
        length_km=4.0),
    "Aplus", generation="2G+", status="construction", f_min=10.0, f_max=2048.0,
    color="#AA3377", linestyle="-"))

register(Detector(
    "NEMO", "NEMO (Australia)", Geometry(
        latitude=-31.34, longitude=115.91, elevation=0.0,
        xarm_azimuth=2.0, yarm_azimuth=125.0, length_km=4.0),
    "NEMO", generation="2G+", status="proposed", f_min=10.0, f_max=4096.0,
    reference="arXiv:2007.03128",
    notes="high-frequency-optimised; arms are NOT orthogonal (123 deg)",
    color="#66CCEE", linestyle="--"))

# ── third generation ───────────────────────────────────────────────────────
register(Detector(
    "CE", "Cosmic Explorer (40 km)", Geometry(
        latitude=_dms(46, 27, 18.528), longitude=-_dms(119, 24, 27.5657),
        elevation=142.554, xarm_azimuth=125.9994, yarm_azimuth=215.994,
        xarm_tilt=-6.195e-4, yarm_tilt=1.25e-5, length_km=40.0),
    "CE", generation="3G", status="proposed", f_min=5.0, f_max=4096.0,
    reference="LIGO-P1600143",
    notes="SITE NOT CHOSEN — placed at LIGO Hanford as a placeholder (bilby "
          "convention). Baselines/ORFs involving CE are illustrative.",
    color="#0077BB", linestyle="-"))

# Second Cosmic Explorer observatory. The CE Horizon Study (arXiv:2109.09882)
# proposes TWO observatories (a 40 km and a 20 km) at separate US sites, which
# is what makes a CE-only stochastic search possible at all: a single detector
# has nothing to cross-correlate against. Neither site is chosen, so we use the
# LIGO Livingston site as a placeholder, exactly as many forecast papers do.
# bilby ships no dedicated 20 km curve, so CE2 reuses the 40 km CE curve; that
# is OPTIMISTIC for the 20 km option.
register(Detector(
    "CE2", "Cosmic Explorer #2 (placeholder site)", Geometry(
        latitude=_dms(30, 33, 46.4196), longitude=-_dms(90, 46, 27.2654),
        elevation=-6.574, xarm_azimuth=197.7165, yarm_azimuth=287.7165,
        length_km=40.0),
    "CE", generation="3G", status="proposed", f_min=5.0, f_max=4096.0,
    reference="CE Horizon Study, arXiv:2109.09882",
    notes="SITE NOT CHOSEN — placed at LIGO Livingston as a placeholder so "
          "that a CE-CE baseline exists. Uses the 40 km curve.",
    color="#33BBEE", linestyle="--"))

register(Detector(
    "CE_wb", "Cosmic Explorer (wideband)", get("CE").geometry, "CE_wb",
    generation="3G", status="proposed", f_min=5.0, f_max=4096.0,
    reference="LIGO-P1600143", notes="same placeholder siting as CE",
    color="#0077BB", linestyle=":"))


def _register_triangle(prefix, label, lat, lon, elev, az, curve_key,
                       length_km=10.0, color="#EE7733", notes="",
                       reference="", f_min=5.0, f_max=4096.0):
    """Register the three nested interferometers of a triangular observatory."""
    geos = triangle_geometries(lat, lon, elev, az, length_km=length_km,
                               opening_deg=60.0)
    styles = ["-", "--", ":"]
    out = []
    for i, geo in enumerate(geos, start=1):
        out.append(register(Detector(
            f"{prefix}{i}", f"{label} (V{i})", geo, curve_key,
            generation="3G", status="proposed", f_min=f_min, f_max=f_max,
            reference=reference, notes=notes, color=color,
            linestyle=styles[i - 1])))
    return out


# Sardinia (Sos Enattos) and Euregio Meuse-Rhine candidate sites — same
# coordinates already used by SimulatedSignal._ET_SITE_PARAMS.
_register_triangle("ET", "Einstein Telescope (Sardinia)",
                   40.5209, 9.4263, 0.0, 70.5674, "ET_D",
                   reference="ET-D Design Report 2020, site A (Sos Enattos)",
                   notes="candidate site; 60 deg opening angle, 3 nested V's",
                   color="#EE7733")

_register_triangle("ETM", "Einstein Telescope (Meuse-Rhine)",
                   50.7259, 6.0080, 0.0, 75.0000, "ET_D",
                   reference="ET-D Design Report 2020, site B (Euregio)",
                   notes="candidate site; 60 deg opening angle, 3 nested V's",
                   color="#CC3311")

# bilby's own ET placement (at the Virgo site) — for cross-checks only.
_register_triangle("ETV", "Einstein Telescope (bilby, Virgo site)",
                   _dms(43, 37, 53.0921), _dms(10, 30, 16.1878), 51.884,
                   70.5674, "ET_D",
                   reference="bilby ET.interferometer",
                   notes="bilby's placeholder siting, kept for cross-checking",
                   color="#BBBBBB")

# ── space ──────────────────────────────────────────────────────────────────
register(Detector(
    "LISA", "LISA", Geometry(latitude=0.0, longitude=0.0, length_km=2.5e6),
    "LISA", generation="space", status="proposed", f_min=1e-5, f_max=1.0,
    reference="arXiv:1702.00786",
    notes="heliocentric constellation — the Geometry entry is a PLACEHOLDER "
          "and must NOT be used for ORFs with ground-based detectors",
    color="#000000", linestyle="-"))
