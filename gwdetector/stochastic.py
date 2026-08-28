# ============================================================
# gwdetectors.stochastic — turning detector noise into Omega_GW sensitivity
#
# This is the bridge between this package and ppEMG/mgtheories: it converts
# strain PSDs plus ORFs into the quantity the rest of the code speaks in,
# Omega_GW(f).
#
# Effective strain sensitivity of a network (Thrane & Romano 2013,
# PRD 88, 124032, arXiv:1310.5300, Eqs. 28-30):
#
#     S_eff(f) = [ sum_{I<J} gamma_IJ(f)^2 / (P_I(f) P_J(f)) ]^{-1/2}
#     Omega_eff(f) = (10 pi^2 / (3 H0^2)) f^3 S_eff(f)
#
# and the power-law integrated (PI) curve, which is what should actually be
# drawn on an Omega_GW plot when claiming detectability:
#
#     Omega_beta = (SNR_thr / sqrt(2 T)) [ Int df (f^beta / Omega_eff)^2 ]^{-1/2}
#     Omega_PI(f) = max_beta [ Omega_beta (f / f_ref)^beta ]
#
# A PI curve is NOT the same as Omega_eff: a power law tangent to the PI curve
# is detectable at the stated SNR, whereas Omega_eff is a per-bin sensitivity
# and systematically looks more pessimistic.
# ============================================================
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18

from .orf import orf

#: Hubble constant in SI [1/s], Planck18 (same cosmology as mgtheories).
H0_SI = float(Planck18.H0.to(1.0 / u.s).value)


def omega_prefactor(frequencies) -> np.ndarray:
    """(10 pi^2 / (3 H0^2)) f^3 — converts a strain PSD into Omega_GW."""
    f = np.asarray(frequencies, float)
    return (10.0 * np.pi ** 2 / (3.0 * H0_SI ** 2)) * f ** 3


def effective_strain_psd(detectors: Sequence, frequencies, *,
                         polarization: str = "tensor",
                         pairs: Optional[Iterable[Tuple]] = None) -> np.ndarray:
    """S_eff(f) for a network, combining all independent baselines."""
    f = np.asarray(frequencies, float)
    dets = list(detectors)
    if pairs is None:
        pairs = list(combinations(range(len(dets)), 2))
    if not pairs:
        raise ValueError("need at least two detectors to cross-correlate")

    total = np.zeros_like(f)
    for i, j in pairs:
        a, b = dets[i], dets[j]
        gamma = orf(a, b, f, polarization)
        Pa, Pb = a.psd(f), b.psd(f)
        term = gamma ** 2 / (Pa * Pb)
        total += np.where(np.isfinite(term), term, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, total ** -0.5, np.inf)


def omega_effective(detectors: Sequence, frequencies, *,
                    polarization: str = "tensor",
                    pairs: Optional[Iterable[Tuple]] = None) -> np.ndarray:
    """Omega_eff(f): per-bin effective sensitivity of a network."""
    f = np.asarray(frequencies, float)
    return omega_prefactor(f) * effective_strain_psd(
        detectors, f, polarization=polarization, pairs=pairs)


def power_law_integrated_curve(
    detectors: Sequence,
    frequencies,
    *,
    T_obs_yr: float = 1.0,
    snr_threshold: float = 1.0,
    betas=None,
    f_ref: float = 25.0,
    polarization: str = "tensor",
    pairs: Optional[Iterable[Tuple]] = None,
):
    """Power-law integrated sensitivity curve (Thrane & Romano 2013).

    Returns
    -------
    omega_pi : ndarray
        The PI curve on `frequencies`.
    info : dict
        `omega_eff`, the beta grid, and Omega_beta for each beta.
    """
    f = np.asarray(frequencies, float)
    if betas is None:
        betas = np.arange(-8.0, 8.01, 0.5)
    betas = np.asarray(betas, float)

    T = float(T_obs_yr) * 365.25 * 24.0 * 3600.0
    omega_eff = omega_effective(detectors, f, polarization=polarization,
                                pairs=pairs)

    good = np.isfinite(omega_eff) & (omega_eff > 0) & np.isfinite(f) & (f > 0)
    if not np.any(good):
        raise ValueError("no usable band: check f_min/f_max of the detectors")
    fg, oe = f[good], omega_eff[good]

    # NOTE: (f/f_ref)^beta must be used in BOTH the integral and the tangent
    # power law below. Mixing a raw f^beta here with (f/f_ref)^beta there
    # leaves a stray f_ref^beta and makes the curve diverge at the band edges.
    # With this form the PI curve is independent of f_ref, as it must be.
    omega_betas = []
    for beta in betas:
        integrand = ((fg / float(f_ref)) ** beta / oe) ** 2
        integral = np.trapezoid(integrand, fg) if hasattr(np, "trapezoid") \
            else np.trapz(integrand, fg)
        if integral <= 0:
            omega_betas.append(np.inf)
            continue
        omega_betas.append(
            float(snr_threshold) / np.sqrt(2.0 * T) * integral ** -0.5)
    omega_betas = np.asarray(omega_betas, float)

    # Omega_PI(f) = max over beta of the tangent power laws
    with np.errstate(over="ignore", invalid="ignore"):
        lines = omega_betas[:, None] * (f[None, :] / float(f_ref)) ** betas[:, None]
    omega_pi = np.nanmax(np.where(np.isfinite(lines), lines, -np.inf), axis=0)
    omega_pi = np.where(np.isfinite(omega_pi) & (omega_pi > 0), omega_pi, np.nan)

    return omega_pi, dict(omega_eff=omega_eff, betas=betas,
                          omega_betas=omega_betas, f_ref=float(f_ref),
                          T_obs_yr=float(T_obs_yr),
                          snr_threshold=float(snr_threshold))


def sigma_omega(det_a, det_b, frequencies, *, T_obs_yr: float = 1.0,
                delta_f: Optional[float] = None,
                polarization: str = "tensor") -> np.ndarray:
    """Per-bin 1-sigma uncertainty on Omega_GW for one baseline.

    sigma(f) = (1 / sqrt(2 T df)) (10 pi^2 / 3 H0^2) f^3 sqrt(P1 P2) / |gamma|
    """
    f = np.asarray(frequencies, float)
    T = float(T_obs_yr) * 365.25 * 24.0 * 3600.0
    if delta_f is None:
        delta_f = float(np.median(np.diff(f))) if f.size > 1 else 1.0

    gamma = orf(det_a, det_b, f, polarization)
    Pa, Pb = det_a.psd(f), det_b.psd(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (omega_prefactor(f) * np.sqrt(Pa * Pb)
                / (np.abs(gamma) * np.sqrt(2.0 * T * float(delta_f))))
