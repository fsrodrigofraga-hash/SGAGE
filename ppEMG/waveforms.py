# ============================================================
# ppEMG.waveforms — waveform generator, context and |h(f)|^2 cache
# (exact worker + "fast binning" approximation)
# ============================================================
from __future__ import annotations

import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Optional, Tuple

import numpy as np
import bilby

from .config import (
    CorrectionContext,
    FreqGridConfig,
    WaveformConfig,
    build_frequency_grid,
    build_models,
)


# ============================================================
# Compact-binary type helpers (BBH / BNS / BHNS / NSBH)
# ============================================================
def _normalize_binary_type(binary_type: Any) -> str:
    """Normalize user input into one of: BBH, BNS, BHNS, NSBH."""
    if binary_type is None:
        return "BBH"
    s = str(binary_type).strip().upper()
    s = s.replace("-", "").replace("_", "")

    if s in ("BBH", "BHBH", "BINARYBLACKHOLE"):
        return "BBH"
    if s in ("BNS", "NSNS", "BINARYNEUTRONSTAR"):
        return "BNS"
    if s in ("BHNS", "NSBH", "NEUTRONSTARBLACKHOLE", "BLACKHOLENEUTRONSTAR"):
        # preserve explicit ordering if provided
        return "BHNS" if s == "BHNS" else "NSBH"
    return s


def _resolve_bilby_source_and_conversion(binary_type: Any):
    """Return (frequency_domain_source_model, parameter_conversion, normalized_type)."""
    bt = _normalize_binary_type(binary_type)

    # BBH
    if bt == "BBH":
        return (
            bilby.gw.source.lal_binary_black_hole,
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            bt,
        )

    # BNS
    if bt == "BNS":
        src = getattr(bilby.gw.source, "lal_binary_neutron_star", None)
        conv = getattr(bilby.gw.conversion, "convert_to_lal_binary_neutron_star_parameters", None)
        if (src is None) or (conv is None):
            warnings.warn(
                "bilby does not expose BNS waveform helpers "
                "(lal_binary_neutron_star / convert_to_lal_binary_neutron_star_parameters) "
                "in this installation. Falling back to BBH source model/conversion.",
                RuntimeWarning,
            )
            return (
                bilby.gw.source.lal_binary_black_hole,
                bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                bt,
            )
        return (src, conv, bt)

    # NSBH / BHNS (mixed)
    if bt in ("NSBH", "BHNS"):
        # bilby exposes BBH and BNS source models; BHNS/NSBH systems are typically handled
        # by using the BNS source model with one object's tidal deformability set to 0.
        # (i.e., set lambda_1/lambda_2 appropriately in the dataset).
        src = getattr(bilby.gw.source, "lal_binary_neutron_star", None)
        conv = getattr(bilby.gw.conversion, "convert_to_lal_binary_neutron_star_parameters", None)

        if (src is None) or (conv is None):
            warnings.warn(
                "bilby does not expose BNS waveform helpers "
                "(lal_binary_neutron_star / convert_to_lal_binary_neutron_star_parameters) "
                "in this installation. Falling back to BBH source model/conversion (no tidal effects).",
                RuntimeWarning,
            )
            return (
                bilby.gw.source.lal_binary_black_hole,
                bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                bt,
            )

        return (src, conv, bt)

    raise ValueError(f"Unsupported binary_type={binary_type!r}. Use one of: 'BBH', 'BNS', 'BHNS', 'NSBH'.")


def build_waveform_generator(cfg: WaveformConfig) -> bilby.gw.WaveformGenerator:
    """Create a Bilby waveform generator for frequency-domain compact-binary waveforms."""
    approximant = cfg.waveform_approximant

    wargs = {
        "waveform_approximant": approximant,
        "reference_frequency": cfg.reference_frequency,
        "minimum_frequency": cfg.minimum_frequency,
    }
    if cfg.maximum_frequency is not None:
        # Some bilby/LAL paths accept this; if not, it will be ignored downstream.
        wargs["maximum_frequency"] = float(cfg.maximum_frequency)

    src_model, conv, _ = _resolve_bilby_source_and_conversion(getattr(cfg, "binary_type", "BBH"))

    return bilby.gw.WaveformGenerator(
        duration=cfg.duration,
        sampling_frequency=cfg.sampling_frequency,
        frequency_domain_source_model=src_model,
        parameter_conversion=conv,
        waveform_arguments=wargs,
    )


def prepare_context(
    wf_cfg: WaveformConfig = WaveformConfig(),
    fgrid_cfg: FreqGridConfig = FreqGridConfig(),
    z_max_models: float = 10.0,
) -> CorrectionContext:
    """
    Prepare reusable context once:
    - Build models
    - Build waveform generator (only to get its full frequency array)
    - Build reduced frequency grid + mapping indices
    """
    models = build_models(z_max=z_max_models)

    wg = build_waveform_generator(wf_cfg)
    full_f = wg.frequency_array

    frequencies, idx = build_frequency_grid(full_f, fgrid_cfg)

    return CorrectionContext(
        models=models,
        wf_cfg=wf_cfg,
        fgrid_cfg=fgrid_cfg,
        frequencies=frequencies,
        idx_freq=idx,
    )


# ============================================================
# Waveform |h(f)|^2 cache (EXACT worker)
# ============================================================
_WORK_DATASET = None
_WORK_IDX = None
_WORK_WFGEN = None
_WORK_FULL_F = None
_WORK_FMIN = None
_WORK_INSPIRAL_ONLY = False

_WF_PARAM_KEYS = (
    "mass_1", "mass_2",
    "luminosity_distance",
    "theta_jn", "phase",
    "a_1", "a_2",
    "tilt_1", "tilt_2",
    "phi_12", "phi_jl",
    "lambda_1", "lambda_2",
    "geocent_time",
)

def _worker_init(dataset: Dict[str, np.ndarray], idx: np.ndarray, wf_cfg: WaveformConfig) -> None:
    global _WORK_DATASET, _WORK_IDX, _WORK_WFGEN, _WORK_FULL_F, _WORK_FMIN, _WORK_INSPIRAL_ONLY
    _WORK_DATASET = dataset
    _WORK_IDX = idx
    _WORK_WFGEN = build_waveform_generator(wf_cfg)
    _WORK_FULL_F = _WORK_WFGEN.frequency_array
    _WORK_FMIN = float(wf_cfg.minimum_frequency)
    _WORK_INSPIRAL_ONLY = bool(getattr(wf_cfg, "inspiral_only", False))


def _compute_h2_and_fpeak_worker(i: int) -> Tuple[np.ndarray, float]:
    ds = _WORK_DATASET
    idx = _WORK_IDX
    wg = _WORK_WFGEN
    full_f = _WORK_FULL_F
    fmin = _WORK_FMIN

    params = {k: float(ds[k][i]) for k in _WF_PARAM_KEYS}
    pol = wg.frequency_domain_strain(params)

    h2_full = np.abs(pol["plus"])**2 + np.abs(pol["cross"])**2

    # Inspiral-only: we do NOT compute f_peak
    if _WORK_INSPIRAL_ONLY:
        return h2_full[idx], float("nan")

    # IMR-based f_peak estimate
    score = h2_full * np.power(full_f, 7.0 / 3.0, where=(full_f > 0), out=np.zeros_like(full_f))

    band = (
        (full_f >= fmin) &
        (full_f > 1.05 * fmin) &
        np.isfinite(score)
    )

    if not np.any(band):
        f_peak = np.nan
    else:
        j = int(np.argmax(score[band]))
        f_peak = float(full_f[band][j])

    return h2_full[idx], f_peak


def _compute_exact_h2_cache_and_fpeak_parallel(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    nproc: Optional[int],
    chunksize: int,
) -> Tuple[np.ndarray, np.ndarray]:
    N_sys = len(dataset["mass_1"])
    nproc = cpu_count() if nproc is None else int(nproc)

    with Pool(
        processes=nproc,
        initializer=_worker_init,
        initargs=(dataset, idx, wf_cfg),
    ) as pool:
        out = pool.map(_compute_h2_and_fpeak_worker, range(N_sys), chunksize=int(chunksize))

    h2_list, fpeak_list = zip(*out)
    return np.asarray(h2_list), np.asarray(fpeak_list, dtype=float)


# ============================================================
# FAST BINNED approximation for h2_cache + f_peak_obs
# ============================================================
def _chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def _safe_interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Simple robust linear interpolation with endpoint clipping (x must be increasing)."""
    return np.interp(xq, x, y, left=y[0], right=y[-1])

def _choose_bins_counts(max_bins: int) -> Tuple[int, int]:
    nm = int(np.sqrt(max_bins))
    nq = max(2, int(max_bins / max(2, nm)))
    nm = max(2, nm)
    return nm, nq

def _assign_quantile_bins(x: np.ndarray, nbin: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (bin_id, edges). bin_id in [0, nbin-1].
    Uses quantile edges to balance counts.
    """
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0.0, 1.0, int(nbin) + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.zeros_like(x, dtype=int), edges
    internal = edges[1:-1]
    bid = np.digitize(x, internal, right=False)
    return bid.astype(int), edges

def _compute_fast_binned_h2_cache_and_fpeak(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    frequencies: np.ndarray,
    nproc: Optional[int],
    chunksize: int,
    max_bins: int = 300,
    use_frequency_warp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate h2_cache and f_peak_obs by computing one waveform per bin in (log Mchirp_z, q).
    If wf_cfg.inspiral_only=True, f_peak_obs will be NaN (by design).
    """
    m1 = np.asarray(dataset["mass_1"], dtype=float)
    m2 = np.asarray(dataset["mass_2"], dtype=float)
    q  = np.asarray(dataset["mass_ratio"], dtype=float)
    z  = np.asarray(dataset["redshift"], dtype=float)
    dl = np.asarray(dataset["luminosity_distance"], dtype=float)

    Mc = _chirp_mass(m1, m2)
    Mcz = Mc * (1.0 + z)
    Mtotz = (m1 + m2) * (1.0 + z)

    nm, nq = _choose_bins_counts(int(max_bins))

    b_m, _ = _assign_quantile_bins(np.log10(Mcz + 1e-30), nm)
    b_q, _ = _assign_quantile_bins(q, nq)

    gid = b_m * nq + b_q
    uniq = np.unique(gid)

    reps = []
    groups = []
    for g in uniq:
        idxs = np.where(gid == g)[0]
        if idxs.size == 0:
            continue
        reps.append(int(idxs[idxs.size // 2]))
        groups.append(idxs)

    reps = np.asarray(reps, dtype=int)
    if reps.size == 0:
        raise RuntimeError("No groups built for fast_binning (unexpected).")

    nproc_eff = cpu_count() if nproc is None else int(nproc)

    with Pool(
        processes=nproc_eff,
        initializer=_worker_init,
        initargs=(dataset, idx, wf_cfg),
    ) as pool:
        out = pool.map(_compute_h2_and_fpeak_worker, reps.tolist(), chunksize=max(1, int(chunksize)))

    rep_h2_list, rep_fpeak_list = zip(*out)
    rep_h2 = np.asarray(rep_h2_list)               # (n_groups, Nf)
    rep_fpeak = np.asarray(rep_fpeak_list, float)  # (n_groups,)

    N = len(m1)
    Nf = len(frequencies)
    h2_cache = np.empty((N, Nf), dtype=float)
    fpeak_obs = np.empty(N, dtype=float)

    f = np.asarray(frequencies, dtype=float)

    for k, idxs in enumerate(groups):
        irep = reps[k]

        Mcz_rep = Mcz[irep]
        Mtotz_rep = Mtotz[irep]
        dl_rep = dl[irep]

        h2_rep = rep_h2[k]
        fp_rep = rep_fpeak[k]

        x = f
        y = h2_rep

        for i in idxs:
            amp = (Mcz[i] / Mcz_rep)**(5.0/3.0) * (dl_rep / dl[i])**2

            if use_frequency_warp:
                shift = (Mtotz[i] / Mtotz_rep)  # f' = f * shift
                fq = f * shift
                h2_i = amp * _safe_interp_1d(x, y, fq)
                fpeak_obs[i] = fp_rep / shift if np.isfinite(fp_rep) else np.nan
            else:
                h2_i = amp * y
                fpeak_obs[i] = fp_rep

            h2_cache[i, :] = h2_i

    return h2_cache, fpeak_obs


def compute_h2_cache_and_fpeak_parallel(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    frequencies: Optional[np.ndarray] = None,
    nproc: Optional[int] = None,
    chunksize: int = 20,
    *,
    fast_binning: bool = True,
    max_bins: int = 300,
    use_frequency_warp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute:
      - h2_cache with shape (N_sys, N_freq_eff)
      - f_peak_obs with shape (N_sys,)

    If wf_cfg.inspiral_only=True, f_peak_obs will be NaN, and inspiral cutoff reduces to ISCO-only.
    """
    if frequencies is None:
        wg = build_waveform_generator(wf_cfg)
        frequencies = wg.frequency_array[idx]

    if fast_binning:
        return _compute_fast_binned_h2_cache_and_fpeak(
            dataset=dataset,
            idx=idx,
            wf_cfg=wf_cfg,
            frequencies=np.asarray(frequencies, float),
            nproc=nproc,
            chunksize=int(chunksize),
            max_bins=int(max_bins),
            use_frequency_warp=bool(use_frequency_warp),
        )

    return _compute_exact_h2_cache_and_fpeak_parallel(
        dataset=dataset,
        idx=idx,
        wf_cfg=wf_cfg,
        nproc=nproc,
        chunksize=int(chunksize),
    )


def compute_h2_cache_parallel(
    dataset: Dict[str, np.ndarray],
    idx: np.ndarray,
    wf_cfg: WaveformConfig,
    frequencies: Optional[np.ndarray] = None,
    nproc: Optional[int] = None,
    chunksize: int = 20,
    *,
    fast_binning: bool = True,
    max_bins: int = 300,
    use_frequency_warp: bool = True,
) -> np.ndarray:
    h2_cache, _ = compute_h2_cache_and_fpeak_parallel(
        dataset=dataset,
        idx=idx,
        wf_cfg=wf_cfg,
        frequencies=frequencies,
        nproc=nproc,
        chunksize=chunksize,
        fast_binning=fast_binning,
        max_bins=max_bins,
        use_frequency_warp=use_frequency_warp,
    )
    return h2_cache
