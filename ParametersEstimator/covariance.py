from __future__ import annotations

# ============================================================
# ParametersEstimator.covariance — Ledoit-Wolf shrinkage [FIX-3] and
# Monte Carlo estimation of C_pop [FIX-1,2,3,6,7]
# ============================================================
import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

import ppEMG as corr

from .config import EstimatorConfig
from .models import omega_full_from_corrections


# Ledoit-Wolf shrinkage  [FIX-3]
# ============================================================
def _ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf OAS shrinkage (via sklearn if available, analytic fallback).
    Optimal for K << n_bins — no manual tuning required.
    Returns (C_shrunk, shrinkage_coeff).
    """
    try:
        from sklearn.covariance import OAS
        oas = OAS()
        oas.fit(X)
        return np.asarray(oas.covariance_, dtype=np.float64), float(oas.shrinkage_)
    except ImportError:
        pass

    # Analytic fallback (Oracle Approximating Shrinkage)
    K, p = X.shape
    S    = np.cov(X, rowvar=False, ddof=1)
    mu   = np.trace(S) / p
    S2   = S @ S
    rho_n = ((K - 2) / K) * np.trace(S2) + np.trace(S) ** 2
    rho_d = (K + 2) * (np.trace(S2) - np.trace(S) ** 2 / p)
    rho   = float(np.clip(rho_n / max(rho_d, 1e-300), 0.0, 1.0))
    C     = (1.0 - rho) * S + rho * mu * np.eye(p)
    return np.asarray(C, dtype=np.float64), rho


# ============================================================
# C_pop Monte Carlo estimation  [FIX-1,2,3,6,7]
# ============================================================
def _cpop_worker(
    k_idx: int,
    seed_base: int,
    freq_settings: Any,
    det_settings: Any,
    td_settings: Any,
    welch_settings: Any,
    pop_settings: Any,
    a_ref: float,
    alpha_ref_callable: Optional[Callable],
    alpha_ref_scalar: Optional[float],
    f_bin: np.ndarray,
    # Bootstrap reweighting — evita reconstruir PopStock por worker
    precomputed_dataset: Optional[Dict[str, Any]] = None,
    precomputed_h2_cache: Optional[np.ndarray] = None,
    precomputed_probabilities: Optional[np.ndarray] = None,
    precomputed_omega_fid: Optional[np.ndarray] = None,
    precomputed_freqs: Optional[np.ndarray] = None,
    precomputed_ctx: Optional[Any] = None,
    precomputed_f_peak_obs: Optional[np.ndarray] = None,
    base_rate: float = 1.0,
    Lambda_base: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    """
    Compute Omega_MODEL(f; theta_true) for one realization of the population.

    Fast mode (bootstrap reweighting):
      If precomputed_* are supplied, bootstrap the probabilities to emulate
      the population variance without rebuilding PopStock.
      Cost: O(N_sys x N_freq) per worker — far faster.

    Legacy mode (fallback):
      Rebuilds PopStock from scratch if precomputed_* are not supplied.
    """
    seed_k = int(seed_base + k_idx * 1_000_003) % (2 ** 31 - 1)
    rng    = np.random.default_rng(seed_k)

    # ------------------------------------------------------------------
    # Fast mode: bootstrap reweighting
    # ------------------------------------------------------------------
    if (precomputed_dataset is not None
            and precomputed_h2_cache is not None
            and precomputed_probabilities is not None
            and precomputed_omega_fid is not None
            and precomputed_freqs is not None
            and precomputed_ctx is not None):
        try:
            N_sys = len(precomputed_probabilities)
            idx_boot = rng.integers(0, N_sys, size=N_sys)

            # COHERENT bootstrap: dataset, probabilities and h2 resampled
            # with the SAME indices. (Previously the dataset was not resampled,
            # which misaligned ppe_correction's Q_i from the bootstrapped
            # weights/h2.)
            dataset_boot = {
                k: np.asarray(v)[idx_boot] for k, v in precomputed_dataset.items()
            }
            probs_boot = precomputed_probabilities[idx_boot]
            h2_boot    = precomputed_h2_cache[idx_boot]

            # popstock-consistent weights (w = p/pdraw via dataset["pdraw"])
            w_base = corr.weights_from_probabilities(
                precomputed_dataset, precomputed_probabilities)
            w_boot = corr.weights_from_probabilities(dataset_boot, probs_boot)

            s_base = float(np.sum(w_base))
            s_boot = float(np.sum(w_boot))
            if s_base <= 0 or s_boot <= 0:
                return None
            # Preserve the total normalization (shape bootstrap, as before)
            w_boot = w_boot * (s_base / s_boot)

            den_boot  = np.sum(w_boot[:, None] * h2_boot, axis=0)
            den_base  = np.sum(w_base[:, None] * precomputed_h2_cache, axis=0)
            den_ratio = np.where(
                (den_base > 0) & (den_boot > 0),
                den_boot / den_base, 1.0,
            )

            alpha_arg = alpha_ref_callable if alpha_ref_callable is not None                         else float(alpha_ref_scalar or 1e-6)

            omega_fid_boot = precomputed_omega_fid * den_ratio

            # Use the base Lambda — the reweighting already captured the
            # population variance via the bootstrap of the probabilities
            Lambda_boot = dict(Lambda_base) if Lambda_base is not None else {}

            # f_peak bootstrapped (ou NaN): evita que compute_correction
            # descarte h2_boot e recompute o cache do dataset inteiro.
            if precomputed_f_peak_obs is not None:
                f_peak_boot = np.asarray(
                    precomputed_f_peak_obs, dtype=float)[idx_boot]
            else:
                f_peak_boot = np.full(N_sys, np.nan, dtype=float)

            omega_k = omega_full_from_corrections(
                precomputed_freqs, omega_fid_boot,
                dataset_boot, Lambda_boot,
                precomputed_ctx, probs_boot,
                a=float(a_ref),
                alpha_ppE=alpha_arg,
                precomputed_h2_cache=h2_boot,
                precomputed_f_peak_obs=f_peak_boot,
                nproc=1,
            )

            omega_k = np.asarray(omega_k, dtype=float)
            omega_k = np.where(np.isfinite(omega_k) & (omega_k >= 0), omega_k, 0.0)
            return np.interp(np.asarray(f_bin, dtype=float),
                             precomputed_freqs, omega_k, left=0.0, right=0.0)
        except Exception as exc:
            import warnings
            warnings.warn(f"[C_pop worker {k_idx}] bootstrap failed ({exc}), "
                          f"falling back to full rebuild.", RuntimeWarning)

    # ------------------------------------------------------------------
    # Legacy mode: full rebuild
    # ------------------------------------------------------------------
    from SimulatedSignal import SimulatedSignal, InjectionSettings, TimeDomainSettings

    try:
        inj_k = InjectionSettings(a_true=float(a_ref), alpha_true=float(alpha_ref_scalar or 1e-6))
        td_k  = TimeDomainSettings(
            duration=td_settings.duration,
            n_segs=td_settings.n_segs,
            fs=td_settings.fs,
            seed_noise=int(seed_k + 1),
            seed_signal=int(seed_k),
        )
        sim_k = SimulatedSignal(
            freq=freq_settings,
            inj=inj_k,
            det=det_settings,
            td=td_k,
            welch=welch_settings,
            popset=pop_settings,
        )
        np.random.seed(seed_k)
        sim_k.precompute_population()

        alpha_arg = alpha_ref_callable if alpha_ref_callable is not None                     else float(alpha_ref_scalar or 1e-6)

        omega_k, _ = sim_k.build_injected_omega(a=float(a_ref), alpha_ppE=alpha_arg)

        f_model = np.asarray(sim_k._ctx.frequencies, dtype=float)
        omega_k = np.asarray(omega_k, dtype=float)
        omega_k = np.where(np.isfinite(omega_k) & (omega_k >= 0), omega_k, 0.0)
        return np.interp(np.asarray(f_bin, dtype=float), f_model, omega_k,
                         left=0.0, right=0.0)
    except Exception as exc:
        import warnings
        warnings.warn(f"[C_pop worker {k_idx}] failed: {exc}", RuntimeWarning)
        return None


def estimate_C_pop(
    f_bin: np.ndarray,
    a_ref: float,
    *,
    # alpha at injection truth — either scalar or callable(dataset)->array
    alpha_ref_callable: Optional[Callable] = None,
    alpha_ref_scalar: Optional[float] = None,
    # SimulatedSignal settings objects (from SimulatedSignal)
    freq_settings: Any,
    det_settings: Any,
    td_settings: Any,
    welch_settings: Any,
    pop_settings: Any,
    # MC parameters  [FIX-2]
    k_pop: int = 50,
    seed_base: int = 90000,
    jitter: float = 1e-10,
    n_jobs: int = 2,
    # Lambda used to precompute the base PopStock (bootstrap reweighting)
    Lambda: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the population covariance matrix C_pop via K Monte Carlo
    realisations of Omega_MODEL at the injection truth parameters.

    Fast mode (bootstrap reweighting):
      Precomputes the base PopStock once, then each worker bootstraps the
      probabilities to emulate the population variance.
      Cost: 1 PopStock + K_POP x O(N_sys x N_freq) — ~K_POP times faster.

    Returns (C_pop, omega_samples) where:
      - C_pop has shape (n_bins, n_bins), Ledoit-Wolf shrunk  [FIX-3]
      - omega_samples has shape (K_ok, n_bins)
    """
    f_bin  = np.asarray(f_bin, dtype=float)
    n_bins = int(f_bin.size)

    # ------------------------------------------------------------------
    # Precompute the base PopStock once for the bootstrap mode
    # ------------------------------------------------------------------
    precomputed_dataset       = None
    precomputed_h2_cache      = None
    precomputed_probabilities = None
    precomputed_omega_fid     = None
    precomputed_freqs         = None
    precomputed_ctx           = None
    precomputed_f_peak_obs    = None
    base_rate                 = 1.0

    try:
        from SimulatedSignal import SimulatedSignal, InjectionSettings
        alpha_arg = alpha_ref_callable if alpha_ref_callable is not None                     else float(alpha_ref_scalar or 1e-6)

        sim_base = SimulatedSignal(
            freq=freq_settings,
            inj=InjectionSettings(a_true=float(a_ref),
                                  alpha_true=float(alpha_ref_scalar or 1e-6)),
            det=det_settings,
            td=td_settings,
            welch=welch_settings,
            popset=pop_settings,
            Lambda=Lambda,
        )
        sim_base.precompute_population()

        h2_cache, f_peak_base = corr.compute_h2_cache_and_fpeak_parallel(
            dataset=sim_base._dataset,
            idx=sim_base._ctx.idx_freq,
            wf_cfg=sim_base._ctx.wf_cfg,
            frequencies=sim_base._ctx.frequencies,
            nproc=1, chunksize=20,
            fast_binning=True, max_bins=100, use_frequency_warp=True,
        )

        omega_base, _ = sim_base.build_injected_omega(
            a=float(a_ref), alpha_ppE=alpha_arg,
        )

        precomputed_dataset       = sim_base._dataset
        precomputed_h2_cache      = h2_cache
        precomputed_probabilities = np.asarray(sim_base._probabilities, dtype=np.float64)
        precomputed_omega_fid     = np.asarray(sim_base._omega_fid, dtype=np.float64)
        precomputed_freqs         = np.asarray(sim_base._ctx.frequencies, dtype=np.float64)
        precomputed_ctx           = sim_base._ctx
        precomputed_f_peak_obs    = (
            np.asarray(f_peak_base, dtype=np.float64)
            if f_peak_base is not None else None
        )
        base_rate                 = float((Lambda or {}).get("rate", 1.0))

        print(f"[estimate_C_pop] base PopStock precomputed — "
              f"bootstrap reweighting para {k_pop} workers.", flush=True)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"[estimate_C_pop] Precomputation failed ({exc}). "
            f"Falling back to legacy mode (K_POP full rebuilds).",
            RuntimeWarning,
        )

    results_raw = Parallel(n_jobs=int(n_jobs), backend="loky", verbose=0)(
        delayed(_cpop_worker)(
            k_idx=k,
            seed_base=int(seed_base),
            freq_settings=freq_settings,
            det_settings=det_settings,
            td_settings=td_settings,
            welch_settings=welch_settings,
            pop_settings=pop_settings,
            a_ref=float(a_ref),
            alpha_ref_callable=alpha_ref_callable,
            alpha_ref_scalar=alpha_ref_scalar,
            f_bin=f_bin,
            precomputed_dataset=precomputed_dataset,
            precomputed_h2_cache=precomputed_h2_cache,
            precomputed_probabilities=precomputed_probabilities,
            precomputed_omega_fid=precomputed_omega_fid,
            precomputed_freqs=precomputed_freqs,
            precomputed_ctx=precomputed_ctx,
            precomputed_f_peak_obs=precomputed_f_peak_obs,
            base_rate=base_rate,
            Lambda_base=Lambda,
        )
        for k in range(int(k_pop))
    )

    omega_list = [r for r in results_raw if r is not None]
    if not omega_list:
        import warnings
        warnings.warn("[estimate_C_pop] All workers failed — returning jitter*I.", RuntimeWarning)
        return np.eye(n_bins, dtype=np.float64) * jitter, np.empty((0, n_bins))

    omega_arr = np.asarray(omega_list, dtype=float)
    omega_ok  = omega_arr[np.all(np.isfinite(omega_arr), axis=1)]
    k_ok      = omega_ok.shape[0]

    if k_ok < 3:
        import warnings
        warnings.warn(f"[estimate_C_pop] Only {k_ok}/{k_pop} valid workers — returning jitter*I.", RuntimeWarning)
        return np.eye(n_bins, dtype=np.float64) * jitter, omega_arr

    C_pop, alpha_lw = _ledoit_wolf_shrinkage(omega_ok)

    diag_mean = max(float(np.mean(np.abs(np.diag(C_pop)))), 1e-300)
    C_pop    += np.eye(C_pop.shape[0]) * (jitter * diag_mean)
    C_pop     = np.asarray(C_pop, dtype=np.float64)

    return C_pop, omega_ok
