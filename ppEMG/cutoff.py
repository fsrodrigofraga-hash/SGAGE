# ============================================================
# ppEMG.cutoff — inspiral-end frequencies (ISCO / peak)
# ============================================================
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from astropy.constants import G, c, M_sun


# ============================================================
# Inspiral cutoff helpers
# ============================================================
def f_isco_source_hz(dataset: Dict[str, np.ndarray]) -> np.ndarray:
    m1 = np.asarray(dataset["mass_1"], dtype=float)
    m2 = np.asarray(dataset["mass_2"], dtype=float)
    Mtot_kg = (m1 + m2) * M_sun.value
    return c.value**3 / ((6.0**1.5) * np.pi * G.value * Mtot_kg)


def robust_inspiral_end_frequency_obs(
    dataset: Dict[str, np.ndarray],
    f_peak_obs: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_isco_src = f_isco_source_hz(dataset)
    z = np.asarray(dataset["redshift"], dtype=float)
    f_isco_obs = f_isco_src / (1.0 + z)

    if f_peak_obs is None:
        return f_isco_obs.copy(), f_isco_obs, f_isco_src

    f_peak_obs = np.asarray(f_peak_obs, dtype=float)
    f_insp_end_obs = f_isco_obs.copy()

    good = np.isfinite(f_peak_obs) & (f_peak_obs > 0)
    f_insp_end_obs[good] = np.minimum(f_isco_obs[good], f_peak_obs[good])

    return f_insp_end_obs, f_isco_obs, f_isco_src
