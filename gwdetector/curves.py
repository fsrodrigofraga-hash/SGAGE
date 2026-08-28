# ============================================================
# gwdetectors.curves — detector sensitivity curves
#
# All curves shipped here are the OFFICIAL tabulated files distributed with
# bilby (bilby/gw/detector/noise_curves). Nothing is an analytic fit invented
# here: if bilby is not installed, loading raises instead of silently
# substituting an approximation.
#
# A curve is stored as a one-sided power spectral density S_n(f) [1/Hz], with
# ASD = sqrt(S_n) [1/sqrt(Hz)]. Files whose name ends in `_asd` are squared on
# load; files ending in `_psd` are used directly.
#
# CAVEAT on ET: bilby applies ET_D_psd.txt to EACH of the three nested
# interferometers of the triangle. Published ET-D curves are sometimes quoted
# for the full triangle instead. Check which convention your reference uses
# before combining sensitivities.
#
# CAVEAT on CE: the Cosmic Explorer site has not been chosen. bilby places CE
# at the LIGO Hanford site as a placeholder, and so do we (see detectors.py).
# Any CE baseline / ORF is therefore illustrative, not a prediction.
# ============================================================
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def bilby_noise_curve_dir() -> str:
    """Absolute path of bilby's bundled noise_curves directory."""
    try:
        import bilby
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "gwdetectors.curves needs bilby for the tabulated sensitivity "
            "curves. Install bilby, or pass an explicit file with "
            "SensitivityCurve.from_file()."
        ) from exc
    path = os.path.join(os.path.dirname(bilby.__file__),
                        "gw", "detector", "noise_curves")
    if not os.path.isdir(path):  # pragma: no cover
        raise FileNotFoundError(f"bilby noise_curves not found at {path}")
    return path


@dataclass
class SensitivityCurve:
    """Tabulated one-sided PSD S_n(f) of a detector."""

    key: str
    label: str
    frequencies: np.ndarray          # [Hz], strictly increasing
    psd: np.ndarray                  # [1/Hz]
    reference: str = ""
    notes: str = ""
    source_file: str = ""

    # ── constructors ───────────────────────────────────────────────────
    @classmethod
    def from_file(cls, path: str, *, key: str = "", label: str = "",
                  kind: Optional[str] = None, reference: str = "",
                  notes: str = "") -> "SensitivityCurve":
        """Load a two-column (frequency, ASD|PSD) text file.

        `kind` is "asd" or "psd"; when None it is inferred from the filename.
        """
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{path}: expected two columns (f, asd|psd)")
        f, y = np.asarray(data[:, 0], float), np.asarray(data[:, 1], float)

        if kind is None:
            base = os.path.basename(path).lower()
            if "_asd" in base:
                kind = "asd"
            elif "_psd" in base:
                kind = "psd"
            else:
                raise ValueError(
                    f"{path}: cannot infer asd/psd from the filename; "
                    f"pass kind='asd' or kind='psd' explicitly")
        psd = y ** 2 if kind == "asd" else y

        order = np.argsort(f)
        f, psd = f[order], psd[order]
        good = np.isfinite(f) & np.isfinite(psd) & (f > 0) & (psd > 0)
        return cls(key=key or os.path.basename(path), label=label or key,
                   frequencies=f[good], psd=psd[good], reference=reference,
                   notes=notes, source_file=path)

    @classmethod
    def from_bilby(cls, filename: str, **kw) -> "SensitivityCurve":
        return cls.from_file(os.path.join(bilby_noise_curve_dir(), filename), **kw)

    # ── evaluation ─────────────────────────────────────────────────────
    @property
    def asd(self) -> np.ndarray:
        return np.sqrt(self.psd)

    @property
    def f_min(self) -> float:
        return float(self.frequencies[0])

    @property
    def f_max(self) -> float:
        return float(self.frequencies[-1])

    def __call__(self, frequencies) -> np.ndarray:
        """Interpolate S_n(f). Outside the tabulated band returns NaN.

        Interpolation is linear in log-log, which is the right thing for
        power-law-like noise budgets and avoids the large errors that linear
        interpolation makes across the steep seismic wall.
        """
        f = np.asarray(frequencies, float)
        out = np.full(f.shape, np.nan)
        inside = (f >= self.f_min) & (f <= self.f_max) & np.isfinite(f)
        if np.any(inside):
            out[inside] = np.exp(np.interp(
                np.log(f[inside]), np.log(self.frequencies), np.log(self.psd)))
        return out

    def asd_at(self, frequencies) -> np.ndarray:
        return np.sqrt(self(frequencies))

    def __repr__(self) -> str:
        return (f"SensitivityCurve({self.key!r}, "
                f"{self.f_min:g}-{self.f_max:g} Hz)")


# ── Curve registry ─────────────────────────────────────────────────────────
# key -> (bilby filename, label, reference, notes)
_CURVE_FILES: Dict[str, tuple] = {
    # ── current generation ────────────────────────────────────────────
    "aLIGO_O4": ("aLIGO_O4_high_asd.txt", "aLIGO O4 (high)",
                 "LIGO-T2000012", ""),
    "aLIGO_design": ("aLIGO_ZERO_DET_high_P_psd.txt", "aLIGO design (ZDHP)",
                     "LIGO-T0900288", ""),
    "aLIGO_early": ("aLIGO_early_psd.txt", "aLIGO early", "LIGO-T2000012", ""),
    "aLIGO_mid": ("aLIGO_mid_psd.txt", "aLIGO mid", "LIGO-T2000012", ""),
    "aLIGO_late": ("aLIGO_late_psd.txt", "aLIGO late", "LIGO-T2000012", ""),
    "LIGO_srd": ("LIGO_srd_psd.txt", "Initial LIGO SRD", "LIGO-T990080", ""),
    "AdV": ("AdV_psd.txt", "Advanced Virgo", "LIGO-T2000012", ""),
    "KAGRA_design": ("KAGRA_design_psd.txt", "KAGRA design",
                     "LIGO-T2000012", ""),
    "GEO600": ("GEO600_S6e_asd.txt", "GEO600 (S6e)", "", ""),
    # ── upgrades ──────────────────────────────────────────────────────
    "Aplus": ("Aplus_asd.txt", "LIGO A+", "LIGO-T1800042", ""),
    "NEMO": ("NEMO_psd.txt", "NEMO", "arXiv:2007.03128",
             "high-frequency-optimised, neutron-star-merger focused"),
    # ── third generation ──────────────────────────────────────────────
    "ET_D": ("ET_D_psd.txt", "Einstein Telescope (ET-D)",
             "Hild et al. 2011, CQG 28, 094013",
             "bilby applies this curve to EACH nested interferometer"),
    "ET_B": ("ET_B_psd.txt", "Einstein Telescope (ET-B)",
             "Hild et al. 2008, CQG 25, 095005", ""),
    "CE": ("CE_psd.txt", "Cosmic Explorer (40 km)",
           "LIGO-P1600143", ""),
    "CE_wb": ("CE_wb_psd.txt", "Cosmic Explorer (wideband)",
              "LIGO-P1600143", ""),
    # ── space ─────────────────────────────────────────────────────────
    "LISA": ("lisa_psd.txt", "LISA", "arXiv:1702.00786",
             "millihertz band; no ORF with ground-based detectors"),
}

_CACHE: Dict[str, SensitivityCurve] = {}


def get_curve(key: str) -> SensitivityCurve:
    """Load (and memoize) a registered sensitivity curve."""
    if key in _CACHE:
        return _CACHE[key]
    if key not in _CURVE_FILES:
        raise KeyError(f"unknown curve {key!r}. Available: {sorted(_CURVE_FILES)}")
    filename, label, ref, notes = _CURVE_FILES[key]
    curve = SensitivityCurve.from_bilby(filename, key=key, label=label,
                                        reference=ref, notes=notes)
    _CACHE[key] = curve
    return curve


def register_curve(key: str, path: str, **kw) -> SensitivityCurve:
    """Register a user-supplied curve file under `key`."""
    curve = SensitivityCurve.from_file(path, key=key, **kw)
    _CACHE[key] = curve
    return curve


def curve_keys() -> List[str]:
    return sorted(set(_CURVE_FILES) | set(_CACHE))
