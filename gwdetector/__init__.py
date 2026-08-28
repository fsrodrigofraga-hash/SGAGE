# ============================================================
# gwdetectors — sensitivity curves, geometry and ORFs for GW observatories
#
# Companion package to ppEMG / mgtheories. It answers three questions:
#
#   1. how loud is each detector?        -> Detector.asd(f) / .psd(f)
#   2. how are two detectors related?    -> orf(det_a, det_b, f, polarization)
#   3. what Omega_GW can a network see?  -> omega_effective / PI curve
#
#     import gwdetectors as gwd
#
#     print(gwd.catalog())                       # everything registered
#     et1, ce = gwd.get("ET1"), gwd.get("CE")
#
#     f = np.logspace(0, 3, 500)
#     ce.asd(f)                                  # strain ASD
#     gwd.orf(et1, ce, f, "tensor")              # overlap reduction function
#     gwd.power_law_integrated_curve([et1, ce], f, T_obs_yr=1.0)
#
# Sensitivity curves are the official tabulated files bundled with bilby; no
# analytic fits are invented here. Detector siting follows bilby/LAL, so
# detector tensors agree with bilby to machine precision.
#
# Read the module docstrings before publishing: the Cosmic Explorer site is a
# placeholder, the Einstein Telescope sites are candidates, and the vector /
# scalar ORF normalization convention is stated explicitly in orf.py.
# ============================================================
from __future__ import annotations

from .curves import (
    SensitivityCurve,
    curve_keys,
    get_curve,
    register_curve,
)
from .detectors import (
    Detector,
    all_detectors,
    catalog,
    get,
    keys,
    register,
)
from .geometry import (
    geometry_from_vectors,
    C_LIGHT,
    Geometry,
    arm_direction,
    geodetic_to_cartesian,
    light_travel_time,
    separation,
    triangle_geometries,
)
from .orf import POLARIZATIONS, orf, overlap_reduction_function
from .stochastic import (
    H0_SI,
    effective_strain_psd,
    omega_effective,
    omega_prefactor,
    power_law_integrated_curve,
    sigma_omega,
)

__version__ = "1.0.0"

__all__ = [
    "geometry_from_vectors",
    # geometry
    "Geometry", "geodetic_to_cartesian", "arm_direction", "separation",
    "light_travel_time", "triangle_geometries", "C_LIGHT",
    # curves
    "SensitivityCurve", "get_curve", "register_curve", "curve_keys",
    # detectors
    "Detector", "get", "keys", "all_detectors", "catalog", "register",
    # orf
    "orf", "overlap_reduction_function", "POLARIZATIONS",
    # stochastic
    "omega_prefactor", "effective_strain_psd", "omega_effective",
    "power_law_integrated_curve", "sigma_omega", "H0_SI",
    "__version__",
]
