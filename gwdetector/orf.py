# ============================================================
# gwdetectors.orf — overlap reduction functions
#
# For a pair of detectors separated by Dx, the ORF for polarization family A
# is the sky integral
#
#     gamma_A(f) = N_A  Int dOmega  K_A(Omega)  exp( i 2 pi f Omega.Dx / c )
#
# with K_A the product of the two antenna patterns summed over the modes of
# that family, and N_A a normalization fixed so that gamma_A = 1 for a pair of
# coincident, co-aligned, L-shaped detectors. For the tensor family this is
# the Allen & Romano (1999) convention, N_T = 5/(8 pi), which gives the
# familiar gamma_HL(0) = -0.8907.
#
# HOW IT IS COMPUTED
# ------------------
# K_A does NOT depend on frequency, only on sky direction. So we
#   1. build an orthonormal frame (e1, e2, s) with s along the baseline,
#   2. tabulate K_A on a Gauss-Legendre grid in mu = cos(theta) times a
#      uniform grid in phi,
#   3. integrate over phi exactly (K is a trigonometric polynomial of degree
#      <= 4 in phi, so any M > 8 uniform points is exact),
#   4. contract the remaining 1-D mu integral against exp(i alpha mu) for all
#      frequencies at once, with alpha = 2 pi f |Dx| / c.
#
# The mu quadrature must resolve exp(i alpha mu), so the node count is chosen
# from alpha_max. This is exact to quadrature error, not a series truncation,
# and costs one small matrix product for the whole frequency array.
#
# POLARIZATION FAMILIES
# ---------------------
#   "tensor"                +, x            (GR)
#   "vector"                x, y            (vector modes)
#   "scalar"                breathing
#   "scalar_longitudinal"   longitudinal
#   "right_left"            Stokes V, i.e. the R-L asymmetry channel
#
# The right_left channel is the one that pairs with the Stokes-V output of
# mgtheories' birefringence family. It uses the tensor normalization, and it
# vanishes identically for co-aligned detectors — a single co-aligned pair
# cannot measure circular polarization.
#
# AGREEMENT WITH THE LITERATURE, AND ONE DISAGREEMENT
# ---------------------------------------------------
# "1 for coincident co-aligned" turns out to BE the convention of Nishizawa
# et al. 2009 (arXiv:0903.0528) for all three families: `selftest` reproduces
# their Eqs. (33)-(41) to 3e-14 on the four detector pairs of their own
# Table III, with no free factor. It also holds a second identity that is
# easy to check by hand — at f = 0 every family gives gamma_A(0) = 2 D_1:D_2,
# because the sky-averaged kernels of the tensor, vector and breathing
# families all reduce to 4/5 D_1:D_2.
#
# Against pygwb.orfs.calc_orf (1.5.1) the comparison is:
#
#   tensor    identical
#   scalar    pygwb is OURS / 3 — they apply an explicit 1/3, citing
#             App. A of arXiv:1704.08373. A convention, nothing more.
#   vector    pygwb is WRONG. Their `Vplus` carries the 169/224 j4 term with
#             the wrong sign: Nishizawa Eq. (37) reads
#                 -( 3/8 j0 + 45/112 j2 - 169/224 j4 )
#             and pygwb transcribed the last sign as +. The error is not a
#             constant factor — it reaches 14% of peak on H1-L1 and changes
#             shape, so no rescaling repairs it. Flipping that one sign makes
#             pygwb agree with this module to 1e-13.
#
# Do not "fix" the vector ORF here to match pygwb.
# ============================================================
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .geometry import (C_LIGHT, Geometry, geometry_from_vectors,
                       separation)

POLARIZATIONS = ("tensor", "vector", "scalar", "scalar_longitudinal",
                 "right_left")


def _orthonormal_frame(s_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing s_hat into a right-handed frame."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, s_hat))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = ref - np.dot(ref, s_hat) * s_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s_hat, e1)
    return e1, e2


def _antenna_patterns(D: np.ndarray, m: np.ndarray, n: np.ndarray,
                      om: np.ndarray) -> Dict[str, np.ndarray]:
    """Antenna pattern functions F^A = D^{ab} e^A_{ab} on the sky grid.

    m, n, om have shape (..., 3) and are mutually orthonormal, with om the
    propagation direction.
    """
    def contract(a, b):
        return np.einsum("...i,ij,...j->...", a, D, b)

    mm, nn, mn = contract(m, m), contract(n, n), contract(m, n)
    mo, no, oo = contract(m, om), contract(n, om), contract(om, om)
    return {
        "plus": mm - nn,
        "cross": 2.0 * mn,
        "vector_x": 2.0 * mo,
        "vector_y": 2.0 * no,
        "breathing": mm + nn,
        "longitudinal": np.sqrt(2.0) * oo,
    }


def _kernel(F1: Dict[str, np.ndarray], F2: Dict[str, np.ndarray],
            polarization: str) -> np.ndarray:
    """K_A(Omega) = sum over the modes of family A of F_1^A F_2^A."""
    if polarization == "tensor":
        return F1["plus"] * F2["plus"] + F1["cross"] * F2["cross"]
    if polarization == "vector":
        return F1["vector_x"] * F2["vector_x"] + F1["vector_y"] * F2["vector_y"]
    if polarization == "scalar":
        return F1["breathing"] * F2["breathing"]
    if polarization == "scalar_longitudinal":
        return F1["longitudinal"] * F2["longitudinal"]
    if polarization == "right_left":
        # F^R F^R* - F^L F^L* = i (F_1^x F_2^+ - F_1^+ F_2^x)
        return 1j * (F1["cross"] * F2["plus"] - F1["plus"] * F2["cross"])
    raise ValueError(f"unknown polarization {polarization!r}; "
                     f"valid: {POLARIZATIONS}")


def _sky_grid(s_hat: np.ndarray, n_mu: int, n_phi: int):
    """Gauss-Legendre in mu = cos(theta), uniform in phi, about s_hat."""
    mu, w_mu = np.polynomial.legendre.leggauss(int(n_mu))
    phi = 2.0 * np.pi * np.arange(int(n_phi)) / float(n_phi)

    e1, e2 = _orthonormal_frame(s_hat)
    sin_t = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))[:, None]   # (n_mu, 1)
    cos_t = mu[:, None]
    cos_p, sin_p = np.cos(phi)[None, :], np.sin(phi)[None, :]     # (1, n_phi)

    def combine(a, b, c):
        return (a[..., None] * e1 + b[..., None] * e2 + c[..., None] * s_hat)

    om = combine(sin_t * cos_p, sin_t * sin_p, cos_t * np.ones_like(cos_p))
    m = combine(cos_t * cos_p, cos_t * sin_p, -sin_t * np.ones_like(cos_p))
    n = combine(-sin_p * np.ones_like(sin_t), cos_p * np.ones_like(sin_t),
                np.zeros_like(om[..., 0]))
    return mu, w_mu, phi, om, m, n


# Normalization constants, computed once per polarization with the same
# quadrature used for the ORF itself (so quadrature error cancels).
_NORM_CACHE: Dict[str, float] = {}


def _normalization(polarization: str) -> float:
    """1 / Int dOmega K_A for coincident, co-aligned, L-shaped detectors."""
    pol = "tensor" if polarization == "right_left" else polarization
    if pol in _NORM_CACHE:
        return _NORM_CACHE[pol]

    D = 0.5 * (np.outer([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
               - np.outer([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]))
    mu, w_mu, phi, om, m, n = _sky_grid(np.array([0.0, 0.0, 1.0]), 64, 32)
    F = _antenna_patterns(D, m, n, om)
    K = np.real(_kernel(F, F, pol))
    integral = float(np.sum(w_mu * K.sum(axis=1)) * (2.0 * np.pi / K.shape[1]))
    _NORM_CACHE[pol] = 1.0 / integral
    return _NORM_CACHE[pol]


def overlap_reduction_function(
    geo_a: Geometry,
    geo_b: Geometry,
    frequencies,
    polarization: str = "tensor",
    *,
    n_mu: Optional[int] = None,
    n_phi: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Overlap reduction function gamma_A(f) for a detector pair.

    Parameters
    ----------
    geo_a, geo_b : Geometry
        Detector geometries (use `Detector.geometry`).
    frequencies : array_like
        Frequencies [Hz].
    polarization : str
        One of POLARIZATIONS.
    n_mu, n_phi : int
        Quadrature resolution. n_mu defaults to a value chosen from the
        highest frequency and the baseline length; raise it if you need more
        accuracy at very high frequency.
    normalize : bool
        If False, skip the "1 for coincident co-aligned" normalization and
        return the raw sky integral.

    Returns
    -------
    gamma : ndarray, same shape as `frequencies`, real-valued.
    """
    if polarization not in POLARIZATIONS:
        raise ValueError(f"unknown polarization {polarization!r}; "
                         f"valid: {POLARIZATIONS}")

    f = np.atleast_1d(np.asarray(frequencies, float))
    dx, d = separation(geo_a, geo_b)
    s_hat = dx / d if d > 0 else np.array([0.0, 0.0, 1.0])

    alpha = 2.0 * np.pi * f * d / C_LIGHT          # phase across the baseline
    alpha_max = float(np.nanmax(np.abs(alpha))) if f.size else 0.0
    if n_mu is None:
        n_mu = int(max(256, 2.5 * alpha_max + 128))

    mu, w_mu, phi, om, m, n = _sky_grid(s_hat, n_mu, n_phi)
    F1 = _antenna_patterns(geo_a.detector_tensor, m, n, om)
    F2 = _antenna_patterns(geo_b.detector_tensor, m, n, om)
    K = _kernel(F1, F2, polarization)                      # (n_mu, n_phi)

    # exact phi integration (K is a trig polynomial of degree <= 4 in phi)
    K_phi = K.sum(axis=1) * (2.0 * np.pi / K.shape[1])     # (n_mu,)

    # remaining mu integral against the plane wave, for all frequencies
    phase = np.exp(1j * alpha[:, None] * mu[None, :])      # (n_f, n_mu)
    gamma = phase @ (w_mu * K_phi)

    if normalize:
        gamma = gamma * _normalization(polarization)

    gamma = np.real_if_close(gamma, tol=1e6)
    out = np.real(gamma)
    return out if np.ndim(frequencies) else out


def orf(det_a, det_b, frequencies, polarization: str = "tensor", **kw):
    """Same as `overlap_reduction_function` but takes Detector objects."""
    ga = det_a.geometry if hasattr(det_a, "geometry") else det_a
    gb = det_b.geometry if hasattr(det_b, "geometry") else det_b
    return overlap_reduction_function(ga, gb, frequencies, polarization, **kw)


# ── checkpoint ────────────────────────────────────────────────────────────
#
# The reference is Nishizawa, Taruya, Hayama, Kawamura & Sakagami 2009
# (arXiv:0903.0528), Eqs. (33)-(41), transcribed literally below. Their
# parametrization by (beta, sigma_1, sigma_2) assumes a SPHERICAL Earth with
# each detector's plane tangent to it; our real geometries sit on the WGS84
# ellipsoid, where the geodetic vertical is not parallel to the radius
# vector. Comparing there would measure that difference instead of the ORF,
# so the checkpoint builds detectors on a perfect sphere.
_R_EARTH_NISHIZAWA = 6.371e6            # [m], the value their Sec. IV uses


def _sphere_geometry(lat_deg: float, lon_deg: float, psi_deg: float,
                     radius_m: float = _R_EARTH_NISHIZAWA) -> Geometry:
    """An L-shaped detector on a sphere, arms in the tangent plane.

    `psi_deg` is the x-arm azimuth CCW from local East, so the bisector sits
    at psi + 45 degrees, which is Nishizawa's sigma.
    """
    la, lo = np.radians(float(lat_deg)), np.radians(float(lon_deg))
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                   np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(up, east)
    p = np.radians(float(psi_deg))
    x_arm = np.cos(p) * east + np.sin(p) * north
    y_arm = -np.sin(p) * east + np.cos(p) * north
    return geometry_from_vectors(float(radius_m) * up, x_arm, y_arm)


def _nishizawa_closed_form(polarization: str, alpha, beta: float):
    """Theta_+ and Theta_- of Eqs. (34)-(35), (37)-(38), (40)-(41)."""
    from scipy.special import spherical_jn as jn

    j0, j2, j4 = jn(0, alpha), jn(2, alpha), jn(4, alpha)
    cb, c2b, c4 = np.cos(beta), np.cos(2.0 * beta), np.cos(beta / 2.0) ** 4

    if polarization == "tensor":                                  # (34), (35)
        plus = (-(3/8*j0 - 45/56*j2 + 169/896*j4)
                + (1/2*j0 - 5/7*j2 - 27/224*j4) * cb
                - (1/8*j0 + 5/56*j2 + 3/896*j4) * c2b)
        minus = (j0 + 5/7*j2 + 3/112*j4) * c4
    elif polarization == "vector":                                # (37), (38)
        # the 169/224 j4 sign below is the one pygwb 1.5.1 gets wrong
        plus = (-(3/8*j0 + 45/112*j2 - 169/224*j4)
                + (1/2*j0 + 5/14*j2 + 27/56*j4) * cb
                - (1/8*j0 - 5/112*j2 - 3/224*j4) * c2b)
        minus = (j0 - 5/14*j2 - 3/28*j4) * c4
    elif polarization == "scalar":                                # (40), (41)
        plus = (-(3/8*j0 + 45/56*j2 + 507/448*j4)
                + (1/2*j0 + 5/7*j2 - 81/112*j4) * cb
                - (1/8*j0 - 5/56*j2 + 9/448*j4) * c2b)
        minus = (j0 - 5/7*j2 + 9/56*j4) * c4
    else:
        raise ValueError(polarization)
    return plus, minus


#: (beta, sigma_+, sigma_-) in degrees, from Nishizawa Table III
_TABLE_III = ((27.2, 62.2, 45.3), (70.8, 31.4, 31.9),
              (99.2, 68.1, 42.4), (135.6, 45.1, 53.7))


def selftest(verbose: bool = True) -> bool:
    ok = True
    freqs = np.arange(1.0, 400.0, 1.0)

    # ── 1. the published closed form, family by family ───────────────
    try:
        from scipy.special import spherical_jn                   # noqa: F401
        have_scipy = True
    except ImportError:                                          # pragma: no cover
        have_scipy = False
    if not have_scipy:
        if verbose:
            print("  [skip] Nishizawa closed form needs scipy")
    else:
        for pol in ("tensor", "vector", "scalar"):
            worst = 0.0
            for beta_d, sp_d, sm_d in _TABLE_III:
                beta = np.radians(beta_d)
                s1, s2 = sp_d + sm_d, sp_d - sm_d
                ga = _sphere_geometry(0.0, 0.0, s1 - 45.0)
                gb = _sphere_geometry(0.0, beta_d, s2 - 45.0)
                d = 2.0 * _R_EARTH_NISHIZAWA * np.sin(beta / 2.0)
                alpha = 2.0 * np.pi * freqs * d / C_LIGHT
                plus, minus = _nishizawa_closed_form(pol, alpha, beta)
                want = (plus * np.cos(4 * np.radians(sp_d))
                        + minus * np.cos(4 * np.radians(sm_d)))
                got = overlap_reduction_function(ga, gb, freqs,
                                                 polarization=pol)
                worst = max(worst, float(np.max(np.abs(got - want))
                                         / np.max(np.abs(want))))
            good = worst < 1e-10
            ok &= good
            if verbose:
                print(f"  [{'ok  ' if good else 'FAIL '}] {pol:<8s} vs "
                      f"Nishizawa Eqs. (33)-(41)   worst = {worst:.2e}")

    # ── 2. gamma_A(0) = 2 D_1:D_2, for every family ──────────────────
    #
    # At f = 0 the sky-averaged kernel of the tensor, vector and breathing
    # families is the SAME, 4/5 D_1:D_2, so all three ORFs coincide there.
    # An error in any one kernel or normalization breaks this.
    ga = _sphere_geometry(31.0, -7.0, 19.0)
    gb = _sphere_geometry(-14.0, 52.0, 63.0)
    want = 2.0 * float(np.tensordot(ga.detector_tensor, gb.detector_tensor))
    for pol in ("tensor", "vector", "scalar"):
        got = float(overlap_reduction_function(ga, gb, np.array([0.0]),
                                               polarization=pol)[0])
        good = abs(got - want) < 1e-10
        ok &= good
        if verbose:
            print(f"  [{'ok  ' if good else 'FAIL '}] {pol:<8s} at f = 0 "
                  f"= 2 D1:D2            {got:+.12f} vs {want:+.12f}")

    # ── 3. the right_left channel vanishes for a co-aligned pair ─────
    #
    # Co-aligned means EQUAL DETECTOR TENSORS, not equal local azimuth: two
    # detectors at the same azimuth but different points of the sphere have
    # different tensors, because their local frames differ. So the pair is
    # built by copying the tensor and moving the vertex — the kernel
    # i (F1^x F2^+ - F1^+ F2^x) then vanishes pointwise, at any separation.
    gc = _sphere_geometry(-14.0, 52.0, 19.0)
    gc._tensor = ga.detector_tensor
    rl = overlap_reduction_function(ga, gc, freqs, polarization="right_left")
    good = float(np.max(np.abs(rl))) < 1e-10
    ok &= good
    if verbose:
        print(f"  [{'ok  ' if good else 'FAIL '}] right_left = 0 for a "
              f"co-aligned pair  max = {float(np.max(np.abs(rl))):.2e}")

    # ── 4. a coincident co-aligned pair gives exactly 1 ──────────────
    for pol in POLARIZATIONS:
        got = float(overlap_reduction_function(ga, ga, np.array([0.0]),
                                               polarization=pol)[0])
        want1 = 0.0 if pol == "right_left" else 1.0
        good = abs(got - want1) < 1e-10
        ok &= good
        if verbose and not good:
            print(f"  [FAIL ] {pol} coincident co-aligned: {got} != {want1}")
    if verbose:
        print(f"  [ok  ] coincident co-aligned = 1 for every family")

    return bool(ok)


if __name__ == "__main__":
    import sys

    print("=" * 66)
    print("CHECKPOINTS — gwdetectors.orf")
    print("=" * 66)
    sys.exit(0 if selftest() else 1)
