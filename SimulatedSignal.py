# SimulatedSignal.py
# ============================================================
# SGWB simulator (cross-correlation) with ppE corrections, built on ppEMG
# and following the noncommutative_sweep.py example closely.
#
# - Precompute: context + PopStock omega(fid) + dataset + probabilities
# - build_injected_omega: applies compute_correction (ppE + G)
# - simulate_and_estimate: injects into the pygwb Simulator and estimates
#   Omega_hat(f)
#     * 2-detector mode: uses pygwb.baseline.Baseline (backward compatible)
#     * 3+ detector mode: uses pygwb.network.Network with optimal combination
# - save_npz: saves the arrays and basic metadata
#
# Default detectors: CE (Cosmic Explorer 40 km) + ET1/ET2/ET3 (Einstein
# Telescope). det3/det4 are optional — if absent, only a CE-ET1 Baseline is
# used, keeping compatibility with earlier code.
# ============================================================

from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, Tuple, Union

import numpy as np

import bilby
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

from pygwb.simulator import Simulator
from pygwb.constants import H0
from pygwb.orfs import calc_orf

import ppEMG as corr

from popstock.PopulationOmegaGW import PopulationOmegaGW
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift


# ============================================================
# Types
# ============================================================
GWeightFn = Callable[[Dict[str, Any]], np.ndarray]
AlphaPPE = Union[float, Callable[[Dict[str, Any]], np.ndarray]]


# ============================================================
# Configs
# ============================================================
@dataclass(frozen=True)
class FrequencySettings:
    fmin: float = 10.0
    fmax: float = 2048.0
    n_freq_eff: int = 800          # N_freq_eff
    n_proposal_samples: int = 10000
    rebin_df_hz: float = 0.5
    gamma_min: float = 0.10
    floor: float = 1e-60


@dataclass(frozen=True)
class InjectionSettings:
    a_true: float = 2.0
    # Used if alpha_ppE is not supplied. A callable(dataset) -> (N_sys,) is as
    # valid here as a float: build_injected_omega accepts both, and a per-event
    # theory (noncommutative gravity) has no scalar alpha to put in this slot.
    alpha_true: AlphaPPE = 4e-3


@dataclass(frozen=True)
class DetectorSettings:
    # Primary and secondary detectors (required — they set the main baseline)
    det1_name: str = "CE"       # Cosmic Explorer 40 km
    det2_name: str = "ET1"      # Einstein Telescope, arm 1
    # Detectores adicionais (opcionais — quando presentes ativam o Network)
    det3_name: Optional[str] = None   # ex.: "ET2"
    det4_name: Optional[str] = None   # ex.: "ET3"
    # Networks with MORE than four detectors: set det_names and the det1..det4
    # fields are ignored. CE+ET needs five (CE, CE2, ET1, ET2, ET3), which the
    # numbered fields cannot express — the fifth used to be dropped silently.
    det_names: Optional[Tuple[str, ...]] = None
    # NOTE: the default is False on purpose. It used to be True, and an
    # unknown name (e.g. "CE2", which bilby does not ship) was silently
    # replaced by L1 WITH THE aLIGO PSD and merely renamed — a detector two
    # orders of magnitude noisier than intended, invisible in the output.
    placeholder_if_missing: bool = False

    # IDEALISED sensitivity: force gamma = 1 on every baseline, i.e. pretend
    # all detectors are co-located, co-aligned, and have independent noise.
    # This is an UPPER BOUND, not a forecast — no such network can be built,
    # and it must be labelled as a limit wherever it is reported.
    #
    # It exists because a run named "CE_CE" builds the same interferometer
    # twice and therefore already has gamma == 1 by accident. Making the
    # assumption an explicit flag means a network of five detectors can be
    # compared with that run under the SAME hypothesis, instead of one side
    # silently getting the idealisation and the other paying real geometry.
    idealized_orf: bool = False
    # WARNING: psd_scale != 1.0 multiplies the detector PSD and changes the
    # noise artificially. Default 1.0 = real physical detector.
    psd_scale: float = 1.0

    # Einstein Telescope site — the only field the user needs to touch.
    # "sardinia" (default) or "meuse". The physical parameters (coordinates,
    # arm length, PSD) are managed internally.
    et_site: str = "sardinia"    # "sardinia" | "meuse"


@dataclass(frozen=True)
class TimeDomainSettings:
    duration: int = 64
    n_segs: int = 5
    fs: int = 4096
    seed_noise: int = 123
    seed_signal: int = 999


@dataclass(frozen=True)
class WelchSettings:
    fft_length: int = 32
    overlap: int = 16


@dataclass(frozen=True)
class PopulationSettings:
    binary_type: str = "BBH"
    waveform_approximant: str = "IMRPhenomD"
    inspiral_only: bool = True
    disable_inspiral_cutoff: bool = True
    minimum_frequency: float = 10.0
    sampling_frequency: float = 4096.0
    duration: float = 4.0

    # gwpopulation models
    mmin_internal: float = 2.0
    mmax_internal: float = 100.0
    z_max: float = 10.0

    # compute_correction controls
    chunksize: int = 20
    fast_binning: bool = True
    max_bins: int = 250
    use_frequency_warp: bool = True


# ============================================================
# PopStock helpers
# ============================================================
def _infer_required_kwargs(callable_obj) -> list[str]:
    try:
        sig = inspect.signature(callable_obj)
    except TypeError:
        sig = inspect.signature(callable_obj.__call__)

    params = list(sig.parameters.values())
    nonself = [p for p in params if p.name != "self"]
    if len(nonself) >= 1:
        nonself = nonself[1:]  # remove dataset

    names = []
    for p in nonself:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        names.append(p.name)
    return names


def _ensure_popstock_args_and_fiducial(pop: PopulationOmegaGW, Lambda: Dict[str, float]) -> Dict[str, float]:
    if not hasattr(pop, "model_args") or pop.model_args is None:
        pop.model_args = {}

    mass_obj = pop.models["mass"]
    red_obj = pop.models["redshift"]

    mass_keys = _infer_required_kwargs(mass_obj)
    red_keys = _infer_required_kwargs(red_obj)

    if len(mass_keys) == 0:
        mass_keys = ["alpha", "mmin", "mmax", "lam", "mpp", "sigpp", "beta", "delta_m"]
    if len(red_keys) == 0:
        red_keys = ["gamma", "kappa", "z_peak"]

    pop.model_args["mass"] = mass_keys
    pop.model_args["redshift"] = red_keys

    fid = {}
    for k in mass_keys:
        if k in Lambda:
            fid[k] = Lambda[k]
    for k in red_keys:
        if k in Lambda:
            fid[k] = Lambda[k]
    if "rate" in Lambda:
        fid["rate"] = Lambda["rate"]

    for attr in ("fiducial_parameters", "fiducial_params", "fiducial_parameter", "_fiducial_parameters"):
        if hasattr(pop, attr):
            try:
                setattr(pop, attr, fid)
            except Exception:
                pass

    return fid


def pop_draw_samples(pop: PopulationOmegaGW, Lambda: Dict[str, float], N: int, seed: int) -> Any:
    np.random.seed(int(seed))
    fid = _ensure_popstock_args_and_fiducial(pop, Lambda)

    sig = inspect.signature(pop.draw_and_set_proposal_samples)

    for method in ("direct", "grid"):
        try:
            kwargs = dict(N_proposal_samples=int(N), mass=method, redshift=method)
            if "seed" in sig.parameters:
                kwargs["seed"] = int(seed)
            return pop.draw_and_set_proposal_samples(fid, **kwargs)
        except UnboundLocalError:
            continue
        except TypeError:
            try:
                kwargs = dict(N_proposal_samples=int(N))
                if "seed" in sig.parameters:
                    kwargs["seed"] = int(seed)
                return pop.draw_and_set_proposal_samples(fid, **kwargs)
            except Exception:
                continue

    raise RuntimeError("Failed to draw proposal samples from PopStock with methods ('direct', 'grid').")


def _call_with_supported_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **supported)


def pop_calculate_omega(
    pop: PopulationOmegaGW,
    Lambda: Dict[str, float],
    waveform_approximant: str,
    sampling_frequency: float,
    waveform_minimum_frequency: float,
    multiprocess: bool = False,
) -> Any:
    sig = inspect.signature(pop.calculate_omega_gw)
    kw = dict(
        waveform_approximant=waveform_approximant,
        sampling_frequency=sampling_frequency,
        waveform_minimum_frequency=waveform_minimum_frequency,
        minimum_frequency=waveform_minimum_frequency,
        multiprocess=multiprocess,
    )
    if "Lambda" in sig.parameters:
        return _call_with_supported_kwargs(pop.calculate_omega_gw, Lambda=Lambda, **kw)
    return _call_with_supported_kwargs(pop.calculate_omega_gw, Lambda, **kw)


# ============================================================
# Main class
# ============================================================
class SimulatedSignal:
    """
    API principal:

      models = {"mass": ..., "redshift": ...}
      ctx = build_context(...)
      pop = PopulationOmegaGW(models=models, frequency_array=ctx.frequencies)
      pop_draw_samples(...)
      pop_calculate_omega(...)
      dataset = corr.dataset_from_popstock_samples(pop, set_extrinsics=True)
      probabilities = pop.calculate_probabilities(dataset, Lambda)
      corr_ppE, corr_G, extras = corr.compute_correction(...)

    """

    def __init__(
        self,
        freq: FrequencySettings = FrequencySettings(),
        inj: InjectionSettings = InjectionSettings(),
        det: DetectorSettings = DetectorSettings(),
        td: TimeDomainSettings = TimeDomainSettings(),
        welch: WelchSettings = WelchSettings(),
        popset: PopulationSettings = PopulationSettings(),
        Lambda: Optional[Dict[str, float]] = None,
    ):
        self.freq = freq
        self.inj = inj
        self.det = det
        self.td = td
        self.welch = welch
        self.popset = popset

        if Lambda is None:
            Lambda = {
                "alpha": 2.5, "beta": 1.0, "delta_m": 3.0, "lam": 0.04,
                "mmin": 5.0, "mmax": 100.0, "mpp": 33.0, "sigpp": 5.0,
                "gamma": 2.7, "kappa": 5.0, "z_peak": 1.9,
                "rate": 15.0,
            }
        self.Lambda = dict(Lambda)

        self._models: Optional[Dict[str, Any]] = None
        self._ctx: Optional[Any] = None
        self._pop: Optional[PopulationOmegaGW] = None
        self._dataset: Optional[Dict[str, Any]] = None
        self._omega_fid: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None
        self._pdraws: Optional[np.ndarray] = None

    # -------------------------
    # Context construction
    # -------------------------
    def make_models(self) -> Dict[str, Any]:
        return {
            "mass": SinglePeakSmoothedMassDistribution(
                mmin=float(self.popset.mmin_internal),
                mmax=float(self.popset.mmax_internal),
            ),
            "redshift": MadauDickinsonRedshift(z_max=float(self.popset.z_max)),
        }

    def prepare_context_with_models(self, models, wf_cfg, fgrid_cfg):
        wg = corr.build_waveform_generator(wf_cfg)
        full_f = wg.frequency_array
        frequencies, idx = corr.build_frequency_grid(full_f, fgrid_cfg)
        return corr.CorrectionContext(
            models=models,
            wf_cfg=wf_cfg,
            fgrid_cfg=fgrid_cfg,
            frequencies=frequencies,
            idx_freq=idx,
        )

    def build_context(self) -> Any:
        wf_cfg = corr.WaveformConfig(
            binary_type=str(self.popset.binary_type),
            waveform_approximant=str(self.popset.waveform_approximant),
            inspiral_only=bool(self.popset.inspiral_only),
            disable_inspiral_cutoff=bool(self.popset.disable_inspiral_cutoff),
            minimum_frequency=float(self.freq.fmin),
            sampling_frequency=float(self.popset.sampling_frequency),
            duration=float(self.popset.duration),
        )
        fgrid_cfg = corr.FreqGridConfig(
            fmin=float(self.freq.fmin),
            fmax=float(self.freq.fmax),
            N_freq_eff=int(self.freq.n_freq_eff),
        )
        models = self.make_models()
        return self.prepare_context_with_models(models, wf_cfg, fgrid_cfg)

    # -------------------------
    # Precompute pipeline
    # -------------------------
    def precompute_population(self) -> None:
        ctx = self.build_context()
        models = ctx.models

        pop = PopulationOmegaGW(models=models, frequency_array=ctx.frequencies)
        pop_draw_samples(pop, self.Lambda, N=int(self.freq.n_proposal_samples), seed=int(self.td.seed_signal))

        pop_calculate_omega(
            pop,
            Lambda=self.Lambda,
            waveform_approximant=str(self.popset.waveform_approximant),
            sampling_frequency=float(self.popset.sampling_frequency),
            waveform_minimum_frequency=float(self.freq.fmin),
            multiprocess=False,
        )

        dataset = corr.dataset_from_popstock_samples(pop, set_extrinsics=True)
        omega = np.asarray(getattr(pop, "omega_gw"), dtype=float)
        omega = np.where(np.isfinite(omega) & (omega >= 0), omega, 0.0)

        # pdraw (p(theta|Lambda_draw)) lives inside the dataset — the ppEMG
        # corrections automatically use weights w = p/pdraw, identical to
        # popstock's importance weights.
        self._pdraws = dataset.get("pdraw")
        if self._pdraws is None:
            print("[SimulatedSignal] WARNING: pdraw missing — corrections "
                  "will use the legacy p(theta|Lambda) weighting.", flush=True)
        else:
            print("[SimulatedSignal] popstock-consistent weights active "
                  "(w = p/pdraw).", flush=True)

        # Probabilities: same as in the reference example (pop.calculate_probabilities)
        pop_tmp = PopulationOmegaGW(models=models)
        _ensure_popstock_args_and_fiducial(pop_tmp, self.Lambda)
        probabilities = pop_tmp.calculate_probabilities(dataset, self.Lambda)
        probabilities = np.asarray(probabilities, dtype=float)

        self._ctx = ctx
        self._models = models
        self._pop = pop
        self._dataset = dataset
        self._omega_fid = omega
        self._probabilities = probabilities

    def _ensure_precomputed(self):
        if self._ctx is None:
            self.precompute_population()

    # -------------------------
    # Correction application
    # -------------------------
    @staticmethod
    def _unity_G(dataset: Dict[str, Any]) -> np.ndarray:
        return np.ones_like(np.asarray(dataset["redshift"], dtype=float), dtype=float)

    def build_injected_omega(
        self,
        a: Optional[float] = None,
        alpha_ppE: Optional[AlphaPPE] = None,
        *,
        G_event_weight: Optional[GWeightFn] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Retorna omega_inj(f) e extras do compute_correction.
        - alpha_ppE pode ser float OU callable(dataset)->array(N_sys,)
        - G_event_weight deve ser callable(dataset)->array(N_sys,)
        """
        self._ensure_precomputed()

        a_use = float(self.inj.a_true if a is None else a)
        alpha_use = self.inj.alpha_true if alpha_ppE is None else alpha_ppE

        if G_event_weight is None:
            G_event_weight = self._unity_G

        # Garantir shapes corretas (N_sys,)
        def _wrap_G(ds):
            w = np.asarray(G_event_weight(ds), dtype=float)
            n = len(np.asarray(ds["redshift"]))
            if w.shape != (n,):
                raise ValueError(f"G_event_weight must return shape (N_sys,), got {w.shape}, expected ({n},)")
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            return w

        if callable(alpha_use):
            def _alpha_callable(ds):
                v = np.asarray(alpha_use(ds), dtype=float)
                n = len(np.asarray(ds["redshift"]))
                if v.shape != (n,):
                    raise ValueError(f"alpha_ppE callable must return shape (N_sys,), got {v.shape}, expected ({n},)")
                v = np.where(np.isfinite(v), v, 0.0)
                return v
            alpha_arg = _alpha_callable
        else:
            alpha_arg = float(alpha_use)

        corr_ppE, corr_G, extras = corr.compute_correction(
            dataset=self._dataset,
            Lambda=self.Lambda,
            ctx=self._ctx,
            a=a_use,
            alpha_ppE=alpha_arg,
            G_event_weight=_wrap_G,
            precomputed_probabilities=self._probabilities,
            nproc=None,
            chunksize=int(self.popset.chunksize),
            fast_binning=bool(self.popset.fast_binning),
            max_bins=int(self.popset.max_bins),
            use_frequency_warp=bool(self.popset.use_frequency_warp),
        )

        corr_ppE = np.asarray(corr_ppE, dtype=float)
        corr_G = np.asarray(corr_G, dtype=float)

        omega_inj = self._omega_fid * corr_ppE * corr_G
        omega_inj = np.where(np.isfinite(omega_inj) & (omega_inj >= 0), omega_inj, 0.0)

        return omega_inj, extras

    # -------------------------
    # Simulation + estimation
    # -------------------------
    def omega_to_sh(self, freqs_hz: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Omega(f) -> Sh(f):
          Sh = [3 H0^2 / (10 pi^2)] * Omega / f^3
        """
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        omega = np.asarray(omega, dtype=float)

        hfac = (3.0 * (H0.si.value ** 2)) / (10.0 * np.pi**2)
        sh = np.full_like(freqs_hz, self.freq.floor, dtype=float)

        ok = np.isfinite(freqs_hz) & (freqs_hz > 0) & np.isfinite(omega) & (omega >= 0)
        sh[ok] = hfac * omega[ok] / (freqs_hz[ok] ** 3)
        if sh.size > 1:
            sh[0] = sh[1]
        sh = np.where(np.isfinite(sh) & (sh > 0), sh, self.freq.floor)
        return sh

    # ------------------------------------------------------------------
    # Einstein Telescope name mapping
    # ------------------------------------------------------------------
    # bilby.gw.detector.get_empty_interferometer does NOT know "ET1"/"ET2"/"ET3".
    # ET is modelled in bilby via TriangularInterferometer, which creates the
    # 3 arms internally. The physical parameters of each site are fixed and
    # managed here — the user only picks et_site in DetectorSettings.
    #
    # Sources:
    #   Sardinia  — ET-D Design Report 2020, site A (Sos Enattos)
    #   Meuse     — ET-D Design Report 2020, site B (Euregio Meuse-Rhine)
    _ET_ALIASES: dict = {"ET1": 0, "ET2": 1, "ET3": 2}

    _ET_SITE_PARAMS: dict = {
        # (latitude_N_deg, longitude_E_deg, length_km, xarm_azimuth_deg)
        "sardinia": (40.5209, 9.4263, 10.0, 70.5674),
        "meuse":    (50.7259, 6.0080, 10.0, 75.0000),
    }

    def _et_params(self) -> tuple:
        """Return (lat, lon, length_km, xarm_az) for the configured ET site."""
        site = str(self.det.et_site).lower()
        if site not in self._ET_SITE_PARAMS:
            raise ValueError(
                f"et_site='{site}' is unknown. "
                f"Opcoes validas: {list(self._ET_SITE_PARAMS)}"
            )
        return self._ET_SITE_PARAMS[site]

    @staticmethod
    def _bilby_psd_from_noise_curves(filename: str):
        """
        Load a PSD/ASD from bilby's noise_curves directory.
        Tries psd_file and asd_file automatically based on the file extension.
        Returns a PowerSpectralDensity, or None if the file does not exist.
        """
        import os
        bilby_dir = os.path.dirname(bilby.gw.detector.__file__)
        curves_dir = os.path.join(bilby_dir, "noise_curves")
        path = os.path.join(curves_dir, filename)
        if not os.path.exists(path):
            return None
        if "asd" in filename.lower():
            return bilby.gw.detector.PowerSpectralDensity(asd_file=path)
        return bilby.gw.detector.PowerSpectralDensity(psd_file=path)

    def _et_psd(self):
        """
        PSD do ET-D: tenta carregar ET_D_psd.txt do bilby.
        Falls back to aLIGO if the file is not available.
        """
        psd = self._bilby_psd_from_noise_curves("ET_D_psd.txt")
        if psd is None:
            # Fallback — warn but do not abort
            import warnings
            warnings.warn(
                "ET_D_psd.txt not found in bilby. "
                "Usando aLIGO como proxy (menos preciso).",
                RuntimeWarning,
            )
            psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        return psd

    def _get_or_make_ifo(self, name: str, allow_placeholder: bool):
        # A+ upgrade: "H1_Aplus", "L1_Aplus", "A1_Aplus". bilby knows the SITES
        # (H1, L1, ...) and ships the A+ curve as a file, but has no
        # interferometer that puts the two together — so we do it here, taking
        # the geometry from bilby and the PSD from Aplus_asd.txt. These are the
        # keys gwdetectors registers for the O5 network (LIGO-T1800042), and
        # the two packages must name the same thing the same way.
        if name.endswith("_Aplus"):
            base = name[: -len("_Aplus")]
            ifo = bilby.gw.detector.get_empty_interferometer(base)
            psd = self._bilby_psd_from_noise_curves("Aplus_asd.txt")
            if psd is None:
                raise ValueError(
                    f"detector {name!r}: bilby ships no Aplus_asd.txt, so the "
                    f"A+ sensitivity cannot be built. Keeping the {base} design "
                    f"curve under an A+ name would misreport the network."
                )
            ifo.power_spectral_density = psd
            ifo.name = name
            return ifo, False

        # Caso especial: bracos do Einstein Telescope (ET1 / ET2 / ET3)
        if name in self._ET_ALIASES:
            idx = self._ET_ALIASES[name]
            lat, lon, length, xarm_az = self._et_params()
            try:
                tri = bilby.gw.detector.TriangularInterferometer(
                    name="ET",
                    power_spectral_density=self._et_psd(),   # PSD ET-D correta
                    minimum_frequency=float(self.freq.fmin),
                    maximum_frequency=float(self.freq.fmax),
                    length=length,
                    latitude=lat,
                    longitude=lon,
                    elevation=0.0,
                    xarm_azimuth=xarm_az,
                    yarm_azimuth=xarm_az + 60.0,
                )
                ifo = tri[idx]
                ifo.name = name
                return ifo, False
            except Exception:
                if not allow_placeholder:
                    raise
                ifo = bilby.gw.detector.get_empty_interferometer("H1")
                ifo.name = name
                return ifo, True

        # Second Cosmic Explorer. bilby ships only "CE", so CE2 is built by
        # taking that interferometer — hence the CE PSD, NOT aLIGO's — and
        # moving it to the placeholder site used by gwdetectors (LIGO
        # Livingston). Both packages must agree, otherwise the simulated
        # baseline and the PI curves drawn against it describe different
        # observatories. The site is not chosen; this is a placeholder, and
        # the 40 km curve is optimistic for a 20 km CE2.
        if name in ("CE2", "CE_2"):
            ifo = bilby.gw.detector.get_empty_interferometer("CE")
            ifo.name = "CE2"
            ce2 = None
            try:
                import gwdetectors as gwd
                ce2 = gwd.get("CE2").geometry
            except Exception:
                pass
            if ce2 is not None:
                ifo.latitude = float(np.degrees(ce2.latitude)
                                     if abs(ce2.latitude) < 1.6 else ce2.latitude)
                ifo.longitude = float(np.degrees(ce2.longitude)
                                      if abs(ce2.longitude) < 3.2 else ce2.longitude)
                ifo.elevation = float(ce2.elevation)
                ifo.xarm_azimuth = float(ce2.xarm_azimuth)
                ifo.yarm_azimuth = float(ce2.yarm_azimuth)
            else:
                # fall back to the Livingston site, the same placeholder
                l1 = bilby.gw.detector.get_empty_interferometer("L1")
                ifo.latitude, ifo.longitude = l1.latitude, l1.longitude
                ifo.elevation = l1.elevation
                ifo.xarm_azimuth, ifo.yarm_azimuth = (l1.xarm_azimuth,
                                                      l1.yarm_azimuth)
            return ifo, False

        # General case: detectors recognized by bilby (CE, H1, L1, V1, ...)
        # get_empty_interferometer("CE") already loads the CE PSD natively.
        try:
            ifo = bilby.gw.detector.get_empty_interferometer(name)
            return ifo, False
        except Exception:
            if not allow_placeholder:
                raise ValueError(
                    f"unknown detector {name!r}: bilby does not ship it and "
                    f"SimulatedSignal has no builder for it. Substituting L1 "
                    f"(aLIGO noise) would silently change the network, so this "
                    f"is an error. Set placeholder_if_missing=True only if you "
                    f"really want that substitution."
                )
            print(f"[WARN] detector {name!r} unknown — SUBSTITUTING L1 with the "
                  f"aLIGO PSD, merely renamed. The network is NOT what the name "
                  f"says.", flush=True)
            ifo = bilby.gw.detector.get_empty_interferometer("L1")
            ifo.name = name
            return ifo, True

    def _make_ifos(self):
        # det_names, when given, is the whole network and wins over det1..det4;
        # otherwise build from the numbered fields (det1/det2 required).
        if self.det.det_names:
            names = [n for n in self.det.det_names if n]
            if len(names) < 2:
                raise ValueError("det_names needs at least two detectors: a "
                                 "stochastic background is measured by "
                                 "cross-correlation")
        else:
            names = [self.det.det1_name, self.det.det2_name]
            if self.det.det3_name:
                names.append(self.det.det3_name)
            if self.det.det4_name:
                names.append(self.det.det4_name)

        ifos = []
        for name in names:
            ifo, _ = self._get_or_make_ifo(name, self.det.placeholder_if_missing)
            ifos.append(ifo)

        for ifo in ifos:
            ifo.duration = int(self.td.duration)
            ifo.sampling_frequency = int(self.td.fs)

            f = ifo.frequency_array.astype(float)
            psd = ifo.power_spectral_density_array.astype(float).copy()

            bad = (~np.isfinite(psd)) | (psd <= 0)
            if bad.any() and (~bad).any():
                psd[bad] = np.interp(f[bad], f[~bad], psd[~bad])

            psd *= float(self.det.psd_scale)
            psd = np.where(np.isfinite(psd) & (psd > 0), psd, self.freq.floor)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(f, psd)

        return ifos

    def _inject_timeseries(self, ifo_list, intensity_GW: FrequencySeries, *, seed: int, no_noise: bool):
        sim = Simulator(
            interferometers=ifo_list,
            N_segments=int(self.td.n_segs),
            duration=int(self.td.duration),
            sampling_frequency=int(self.td.fs),
            intensity_GW=intensity_GW,
            no_noise=bool(no_noise),
            seed=int(seed),
        )
        return sim.generate_data()

    # ------------------------------------------------------------------
    # Robust manual estimator (gwpy Welch + pygwb ORF)
    # ------------------------------------------------------------------
    # pygwb.Baseline/Network require native pygwb interferometers and do not
    # accept bilby objects directly. The manual estimator below uses:
    #   - PSD/CSD via gwpy (Welch) — same method as before, already validated
    #   - ORF gamma(f) via pygwb.orfs.calc_orf — correct for any pair
    #   - Multi-baseline combination by inverse-variance weighting
    # ------------------------------------------------------------------

    def _compute_orf(self, ifo_a, ifo_b, freqs: np.ndarray) -> np.ndarray:
        """Compute the ORF gamma(f) for a pair of bilby interferometers."""
        from pygwb.orfs import calc_orf
        import inspect

        def _vertex(ifo):
            if hasattr(ifo, "geometry") and hasattr(ifo.geometry, "vertex"):
                return np.asarray(ifo.geometry.vertex, dtype=float)
            return np.asarray(ifo.vertex, dtype=float)

        def _arm(ifo, which):
            if hasattr(ifo, "geometry"):
                for nm in (which, f"{which}_arm", f"{which}arm"):
                    if hasattr(ifo.geometry, nm):
                        return np.asarray(getattr(ifo.geometry, nm), dtype=float)
            for nm in (which, f"{which}_arm", f"{which}arm"):
                if hasattr(ifo, nm):
                    return np.asarray(getattr(ifo, nm), dtype=float)
            raise AttributeError(f"Arm {which} not found in {ifo.name}")

        pos1, pos2 = _vertex(ifo_a), _vertex(ifo_b)
        x1, x2 = _arm(ifo_a, "x"), _arm(ifo_b, "x")
        y1, y2 = _arm(ifo_a, "y"), _arm(ifo_b, "y")
        freqs = np.asarray(freqs, dtype=float)

        sig = inspect.signature(calc_orf)
        params = list(sig.parameters.keys())
        try:
            if params and "frequencies" in params[0].lower():
                orf = calc_orf(freqs, pos1, pos2, x1, x2, y1, y2)
            else:
                orf = calc_orf(pos1, pos2, x1, x2, y1, y2, freqs)
        except Exception:
            orf = calc_orf(freqs, pos1, pos2, x1, x2, y1, y2)

        orf = np.asarray(orf, dtype=float)
        return np.where(np.isfinite(orf), orf, 0.0)

    def _estimate_pair(
        self,
        ts_a: TimeSeries,
        ts_b: TimeSeries,
        ifo_a,
        ifo_b,
    ) -> Tuple[np.ndarray, ...]:
        """
        Estima Ω̂(f) e σ_Ω(f) para um par de detectores via Welch + ORF.
        Retorna (f, Omega_hat, sigma_Omega, good_gamma, P1, P2, gamma).
        P1, P2 and gamma are exported to the .npz and used by the
        optimal-filter likelihood (likelihood_mode="optimal").
        """
        fft_length = int(self.welch.fft_length)
        overlap    = int(self.welch.overlap)

        a64 = TimeSeries(ts_a.value.astype(np.float64),
                         t0=ts_a.t0, sample_rate=ts_a.sample_rate)
        b64 = TimeSeries(ts_b.value.astype(np.float64),
                         t0=ts_b.t0, sample_rate=ts_b.sample_rate)

        psd_a = a64.psd(fftlength=fft_length, overlap=overlap)
        psd_b = b64.psd(fftlength=fft_length, overlap=overlap)
        csd   = a64.csd(b64, fftlength=fft_length, overlap=overlap)

        f  = np.asarray(csd.frequencies.value, dtype=float)
        P1 = np.asarray(psd_a.value, dtype=float)
        P2 = np.asarray(psd_b.value, dtype=float)
        C  = np.asarray(csd.value,   dtype=complex)

        P1 = np.where(np.isfinite(P1) & (P1 > 0), P1, self.freq.floor)
        P2 = np.where(np.isfinite(P2) & (P2 > 0), P2, self.freq.floor)

        gamma = self._compute_orf(ifo_a, ifo_b, f)
        good  = np.abs(gamma) >= float(self.freq.gamma_min)
        g_safe = np.where(good, gamma, np.nan)

        H0_SI = float(H0.si.value)
        conv  = (10.0 * np.pi**2) / (3.0 * H0_SI**2)
        f3    = np.where((f > 0) & np.isfinite(f), f**3, np.nan)

        Sh_hat   = np.real(C) / g_safe

        # analytic sigma for T_obs = 1 year
        fs     = ifo_a.sampling_frequency   # Hz
        df     = fs / fft_length            # actual spectral resolution
        T_obs  = float(self.td.duration) * int(self.td.n_segs)  # total simulated duration
        sig2_Sh = (P1 * P2) / (2.0 * T_obs * df * g_safe**2)

        Om_hat  = conv * f3 * np.where(np.isfinite(Sh_hat), Sh_hat, 0.0)
        sig_Om  = conv * f3 * np.sqrt(
            np.where(np.isfinite(sig2_Sh) & (sig2_Sh > 0), sig2_Sh, np.inf)
        )

        return f, Om_hat, sig_Om, good, P1, P2, gamma

    def _estimate_omega_baseline(
        self,
        data_list: list,
        ifo_list: list,
    ) -> Tuple[np.ndarray, ...]:
        """
        Two-detector mode: Welch estimator + ORF for the CE-ET1 baseline.
        Returns (f, Omega_hat, sigma_Omega, P1, P2, gamma) filtered by band.
        """
        f, Om, sOm, good, P1, P2, gamma = self._estimate_pair(
            data_list[0], data_list[1],
            ifo_list[0],  ifo_list[1],
        )
        band = (
            (f >= float(self.freq.fmin)) & (f <= float(self.freq.fmax))
            & np.isfinite(Om) & np.isfinite(sOm)
            & (sOm > 0) & (sOm < np.inf) & good
        )
        return f[band], Om[band], sOm[band], P1[band], P2[band], gamma[band]

    def _estimate_omega_network(
        self,
        data_list: list,
        ifo_list: list,
    ) -> Tuple[np.ndarray, ...]:
        """
        N-detector mode: combines all baselines by inverse-variance
        weighting. Returns (f, Omega_hat, sigma_Omega, P1_eff, P2_eff, gamma_eff)
        where P1_eff, P2_eff, gamma_eff are weighted averages across baselines.
        """
        from itertools import combinations

        # ── every baseline lives on the SAME Welch grid ──────────────────
        # A frequency is kept when AT LEAST ONE baseline is usable there. The
        # previous version intersected the per-baseline |gamma| >= gamma_min
        # masks, so a single bad baseline deleted the frequency for the whole
        # network — which made adding detectors REMOVE information. For
        # CE+CE2+ET1+ET2+ET3 the three co-located ET pairs are usable
        # everywhere while the six transcontinental CE-ET pairs are usable in
        # 1-5% of the band, so the intersection was 8% of the bins at
        # gamma_min = 0.05 and EMPTY at 0.10. Inverse-variance weighting is a
        # sum over baselines: it can only ever decrease sigma.
        f_grid = None
        Om_num = w_total = None
        S_opt = None          # sum_b gamma_b^2 / (P1_b P2_b), see below
        n_used = None

        for (i, ifo_a), (j, ifo_b) in combinations(enumerate(ifo_list), 2):
            f, Om, sOm, good, P1, P2, gamma = self._estimate_pair(
                data_list[i], data_list[j], ifo_a, ifo_b,
            )
            if f_grid is None:
                f_grid = f
                Om_num  = np.zeros_like(f_grid)
                w_total = np.zeros_like(f_grid)
                S_opt   = np.zeros_like(f_grid)
                n_used  = np.zeros_like(f_grid, dtype=int)
            elif f.shape != f_grid.shape or not np.allclose(f, f_grid):
                raise ValueError(
                    "baselines returned different frequency grids; the "
                    "inverse-variance combination assumes a common Welch grid"
                )

            usable = (
                (f >= float(self.freq.fmin)) & (f <= float(self.freq.fmax))
                & np.isfinite(Om) & np.isfinite(sOm)
                & (sOm > 0) & (sOm < np.inf) & good
            )
            w = np.where(usable, 1.0 / np.where(sOm > 0, sOm, np.inf) ** 2, 0.0)
            Om_num  += w * np.where(usable, Om, 0.0)
            w_total += w
            n_used  += usable.astype(int)

            # optimal-filter weight of the NETWORK, Allen & Romano:
            # w(f) ∝ sum_b gamma_b^2 / (P1_b P2_b). Averaging gamma across
            # baselines is meaningless — gamma changes sign from pair to pair.
            with np.errstate(divide="ignore", invalid="ignore"):
                s_b = gamma ** 2 / (P1 * P2)
            S_opt += np.where(usable & np.isfinite(s_b), s_b, 0.0)

        keep = (w_total > 0) & (S_opt > 0)
        if not np.any(keep):
            empty = np.array([])
            return empty, empty, empty, empty, empty, empty

        f_out   = f_grid[keep]
        Om_out  = Om_num[keep] / w_total[keep]
        sOm_out = 1.0 / np.sqrt(w_total[keep])

        # The .npz schema carries (P1, P2, gamma) so that the optimal-filter
        # likelihood can rebuild gamma^2/(P1 P2). For a network that product
        # is S_opt, so export the equivalent single-baseline triple:
        # gamma_eff = 1 and P1_eff = P2_eff = 1/sqrt(S_opt).
        g_out  = np.ones_like(f_out)
        p_out  = 1.0 / np.sqrt(S_opt[keep])

        print(f"[network] {len(list(combinations(range(len(ifo_list)), 2)))} "
              f"baselines, {keep.sum()} of {f_grid.size} bins kept "
              f"(median {np.median(n_used[keep]):.0f} baselines per bin)")
        return f_out, Om_out, sOm_out, p_out, p_out, g_out

    def _estimate_omega_from_timeseries(
        self,
        data_list: list,
        ifo_list: list,
    ) -> Tuple[np.ndarray, ...]:
        """
        Main dispatcher: uses Baseline (2 detectors) or
        Network (3+ detectors) depending on how many IFOs are configured.
        Retorna (f, Omega_hat, sigma_Omega, P1, P2, gamma).
        """
        if len(ifo_list) == 2:
            return self._estimate_omega_baseline(data_list, ifo_list)
        return self._estimate_omega_network(data_list, ifo_list)

    def rebin_inverse_variance(self, f, y, sigma, df_hz):
        f = np.asarray(f, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        good = np.isfinite(f) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0) & (sigma < np.inf)
        f, y, sigma = f[good], y[good], sigma[good]
        if f.size == 0:
            return np.array([]), np.array([]), np.array([])

        edges = np.arange(np.min(f), np.max(f) + df_hz, df_hz)
        idx = np.digitize(f, edges) - 1
        nb = edges.size - 1

        fb = np.full(nb, np.nan, dtype=float)
        yb = np.full(nb, np.nan, dtype=float)
        sb = np.full(nb, np.nan, dtype=float)

        for b in range(nb):
            m = idx == b
            if not np.any(m):
                continue
            w = 1.0 / sigma[m] ** 2
            wsum = np.sum(w)
            if wsum <= 0:
                continue
            yb[b] = np.sum(w * y[m]) / wsum
            sb[b] = 1.0 / np.sqrt(wsum)
            fb[b] = 0.5 * (edges[b] + edges[b + 1])

        ok = np.isfinite(fb) & np.isfinite(yb) & np.isfinite(sb) & (sb > 0)
        return fb[ok], yb[ok], sb[ok]

    def simulate_and_estimate(
        self,
        omega_inj: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        Simula o sinal SGWB e estima Ω̂(f).

        The number of active detectors is set by DetectorSettings:
          - det1 + det2 only    -> a single Baseline (backward-compatible mode)
          - det1 + det2 + det3  → Network com 3 baselines
          - det1 + det2 + det3 + det4 → Network com 6 baselines (CE + ET completo)

        Retorna
        -------
        f_raw, Om_raw, sOm_raw  : arrays espectrais brutos da Baseline/Network
        f_bin, Om_bin, sOm_bin  : rebinados por inverse-variance
        meta                    : dictionary of simulation metadata
        """
        self._ensure_precomputed()

        # _make_ifos() already returns N IFOs according to DetectorSettings
        ifos = self._make_ifos()

        # Use the frequency array of the first IFO as the reference
        f_ifo = ifos[0].frequency_array.astype(float)

        Sh_pop = self.omega_to_sh(
            np.asarray(self._ctx.frequencies, dtype=float),
            np.asarray(omega_inj, dtype=float),
        )
        Sh_ifo = np.interp(
            f_ifo,
            np.asarray(self._ctx.frequencies, dtype=float),
            Sh_pop,
            left=self.freq.floor,
            right=self.freq.floor,
        )
        if Sh_ifo.size > 0:
            Sh_ifo[0] = self.freq.floor
        Sh_ifo = np.where(np.isfinite(Sh_ifo) & (Sh_ifo > 0), Sh_ifo, self.freq.floor)

        I_inj  = FrequencySeries(Sh_ifo.astype(float),             frequencies=f_ifo)
        I_zero = FrequencySeries(np.zeros_like(f_ifo, dtype=float), frequencies=f_ifo)

        # Generate noise and signal separately so they can be summed deterministically
        data_noise = self._inject_timeseries(ifos, I_zero, seed=self.td.seed_noise, no_noise=False)
        data_sig   = self._inject_timeseries(ifos, I_inj,  seed=self.td.seed_signal, no_noise=True)

        # Combine noise + signal into lists parallel to ifos
        data_inj = []
        for tsn, tss in zip(data_noise, data_sig):
            v = tsn.value.astype(np.float64) + tss.value.astype(np.float64)
            data_inj.append(TimeSeries(v, t0=tsn.t0, sample_rate=tsn.sample_rate))

        # Despachante: Baseline (2 IFOs) ou Network (3+ IFOs)
        f_raw, Om_raw, sOm_raw, P1_raw, P2_raw, gamma_raw = \
            self._estimate_omega_from_timeseries(data_inj, ifos)
        f_bin, Om_bin, sOm_bin = self.rebin_inverse_variance(
            f_raw, Om_raw, sOm_raw, float(self.freq.rebin_df_hz)
        )
        # Rebin P1, P2, gamma with the same bins for export
        _, P1_bin,    _ = self.rebin_inverse_variance(f_raw, P1_raw,    sOm_raw, float(self.freq.rebin_df_hz))
        _, P2_bin,    _ = self.rebin_inverse_variance(f_raw, P2_raw,    sOm_raw, float(self.freq.rebin_df_hz))
        _, gamma_bin, _ = self.rebin_inverse_variance(f_raw, gamma_raw, sOm_raw, float(self.freq.rebin_df_hz))

        # Record the estimation mode used in the metadata
        n_ifos = len(ifos)
        n_baselines = n_ifos * (n_ifos - 1) // 2
        estimation_mode = "baseline" if n_ifos == 2 else f"network_{n_baselines}bl"

        meta = {
            "freq": asdict(self.freq),
            "inj": asdict(self.inj),
            "det": asdict(self.det),
            "td": asdict(self.td),
            "welch": asdict(self.welch),
            "popset": asdict(self.popset),
            "Lambda": dict(self.Lambda),
            "binary_type": str(self.popset.binary_type),
            "waveform_approximant": str(self.popset.waveform_approximant),
            "corr_module": "ppEMG",
            "pdraw_weights": bool(self._pdraws is not None),
            "estimation_mode": estimation_mode,
            "n_detectors": n_ifos,
            "n_baselines": n_baselines,
            "detector_names": [ifo.name for ifo in ifos],
        }
        return f_raw, Om_raw, sOm_raw, f_bin, Om_bin, sOm_bin, P1_bin, P2_bin, gamma_bin, meta

    def simulate_analytical(
        self,
        omega_inj: np.ndarray,
        T_obs: float = 365.25 * 24 * 3600,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """
        Analytic version of simulate_and_estimate.

        Instead of simulating time series, it computes Omega_hat directly as:
            Omega_hat(f) = Omega_true(f) + N(0, sigma(f))   [if add_noise=True]
            Omega_hat(f) = Omega_true(f)                     [if add_noise=False]

        IMPORTANT — choice of frequency grid (Bug 1 fix):
        -------------------------------------------------
        The previous version used ifo.frequency_array (spacing 1/td.duration,
        e.g. 1/512 Hz) together with the formula sigma ~ 1/sqrt(2T*df) where
        df = 1/welch.fft_length (e.g. 1/64 Hz). That was inconsistent: noise was
        injected at ~512 points per Hz and then inverse-variance-rebinned
        treating them all as independent — which underestimated sigma_bin by
        sqrt(td.duration/welch.fft_length) ~ sqrt(8) ~ 2.83x.

        Fix: compute everything directly on the final grid (uniform in
        rebin_df_hz), using df = rebin_df_hz in the formula. No rebinning is
        needed any more, and the resulting sigma follows Allen-Romano for the
        chosen bin width.

        Parameters
        ----------
        omega_inj : array
            Espectro de energia injetado Omega_GW(f), no grid de self._ctx.frequencies.
        T_obs : float
            Observation time in seconds (default: 1 year).
        add_noise : bool
            If True (default), adds Gaussian noise to the signal.
            If False, returns the pure signal — useful for validating emulator bias.
        """
        self._ensure_precomputed()

        ifos  = self._make_ifos()                       # only to get PSDs and ORF
        f_ifo = ifos[0].frequency_array.astype(float)

        # === GRID FINAL DIRETO — sem rebin ===
        df_bin = float(self.freq.rebin_df_hz)
        fmin   = float(self.freq.fmin)
        fmax   = float(self.freq.fmax)
        f_bin_target = np.arange(fmin, fmax + 0.5 * df_bin, df_bin)

        # ── ALL baselines, not just ifos[0]-ifos[1] ─────────────────────────
        # This function used to read P1/P2 from ifos[0] and ifos[1] and take a
        # single ORF, so detectors 3, 4, 5... were built and then ignored. A
        # run named ET1_ET2_ET3 measured only the ET1-ET2 pair (one of three
        # baselines) and one named CE_CE2_ET1_ET2_ET3 would have measured only
        # CE-CE2. Allen & Romano for a network is a sum over pairs:
        #
        #     1/sigma_net^2 = 2 T df * sum_b gamma_b^2 / (P1_b P2_b)
        #
        # which reduces to the old expression for a single pair.
        from itertools import combinations

        def _psd_on_grid(ifo):
            p = np.asarray(ifo.power_spectral_density_array, float)
            p = np.where(np.isfinite(p) & (p > 0), p, self.freq.floor)
            p = np.interp(f_bin_target, f_ifo, p,
                          left=self.freq.floor, right=self.freq.floor)
            return np.where(np.isfinite(p) & (p > 0), p, self.freq.floor)

        psds = [_psd_on_grid(ifo) for ifo in ifos]
        S_net = np.zeros_like(f_bin_target)      # sum_b gamma_b^2/(P1_b P2_b)
        good = np.zeros_like(f_bin_target, dtype=bool)
        n_pairs = 0
        idealized = bool(getattr(self.det, "idealized_orf", False))
        for i, j in combinations(range(len(ifos)), 2):
            if idealized:
                # gamma = 1 on every baseline; the gamma_min cut is meaningless
                # here, so the whole band is kept.
                g_ij = np.ones_like(f_bin_target)
                ok_ij = np.ones_like(f_bin_target, dtype=bool)
            else:
                g_ij = self._compute_orf(ifos[i], ifos[j], f_bin_target)
                ok_ij = np.abs(g_ij) >= float(self.freq.gamma_min)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_ij = g_ij ** 2 / (psds[i] * psds[j])
            S_net += np.where(ok_ij & np.isfinite(s_ij), s_ij, 0.0)
            # a frequency survives when AT LEAST ONE baseline is usable there
            good |= ok_ij
            n_pairs += 1

        good &= S_net > 0
        S_safe = np.where(good, S_net, np.nan)

        # analytic sigma (Allen-Romano) with df = final bin width
        H0_SI = float(H0.si.value)
        conv  = (10.0 * np.pi**2) / (3.0 * H0_SI**2)
        f3    = np.where((f_bin_target > 0) & np.isfinite(f_bin_target),
                         f_bin_target**3, np.nan)

        sig_Om = conv * f3 / np.sqrt(2.0 * T_obs * df_bin * S_safe)

        # The .npz carries (P1, P2, gamma) so the optimal-filter likelihood can
        # rebuild gamma^2/(P1 P2). For a network that product IS S_net, so
        # export the equivalent single-baseline triple.
        P1 = np.where(good, 1.0 / np.sqrt(S_safe), self.freq.floor)
        P2 = P1.copy()
        gamma = np.where(good, 1.0, 0.0)
        mode = ("IDEALISED gamma=1 (upper bound, not a forecast)" if idealized
                else f"real ORF, gamma_min={self.freq.gamma_min:g}")
        print(f"[simulate_analytical] {len(ifos)} detectors, {n_pairs} "
              f"baselines, {int(good.sum())} of {good.size} bins usable "
              f"— {mode}", flush=True)

        # Omega_true interpolated onto the final grid
        omega_true = np.interp(
            f_bin_target,
            np.asarray(self._ctx.frequencies, dtype=float),
            np.asarray(omega_inj, dtype=float),
            left=0.0, right=0.0,
        )

        # Omega_hat = true signal + Gaussian noise (optional)
        if add_noise:
            rng   = np.random.default_rng(int(self.td.seed_signal))
            noise = rng.normal(0.0, np.where(np.isfinite(sig_Om), sig_Om, 0.0))
            Om_full = omega_true + noise
        else:
            Om_full = omega_true.copy()

        # Valid band
        band = (
            np.isfinite(Om_full) & np.isfinite(sig_Om)
            & (sig_Om > 0) & (sig_Om < np.inf) & good
        )
        f_bin    = f_bin_target[band]
        Om_bin   = Om_full[band]
        sOm_bin  = sig_Om[band]
        P1_bin   = P1[band]
        P2_bin   = P2[band]
        gamma_bin = gamma[band]

        # raw == bin (no rebinning any more; both outputs kept for compatibility)
        f_raw, Om_raw, sOm_raw = f_bin.copy(), Om_bin.copy(), sOm_bin.copy()

        meta = {
            "freq":                  asdict(self.freq),
            "inj":                   asdict(self.inj),
            "det":                   asdict(self.det),
            "td":                    asdict(self.td),
            "welch":                 asdict(self.welch),
            "popset":                asdict(self.popset),
            "Lambda":                dict(self.Lambda),
            "binary_type":           str(self.popset.binary_type),
            "waveform_approximant":  str(self.popset.waveform_approximant),
            "corr_module":           "ppEMG",
            "pdraw_weights":         bool(self._pdraws is not None),
            "estimation_mode":       "analytical",
            "T_obs_yr":              float(T_obs / (365.25 * 24 * 3600)),
            "add_noise":             bool(add_noise),
            "df_bin_hz":             float(df_bin),
            "n_detectors":           len(ifos),
            "n_baselines":           1,
            "detector_names":        [ifo.name for ifo in ifos],
        }
        return f_raw, Om_raw, sOm_raw, f_bin, Om_bin, sOm_bin, P1_bin, P2_bin, gamma_bin, meta

    def save_npz(
        self,
        outpath: str,
        *,
        f_raw: np.ndarray,
        Om_raw: np.ndarray,
        sOm_raw: np.ndarray,
        f_bin: np.ndarray,
        Om_bin: np.ndarray,
        sOm_bin: np.ndarray,
        P1_bin: Optional[np.ndarray] = None,
        P2_bin: Optional[np.ndarray] = None,
        gamma_bin: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save the spectral arrays in .npz format.

        P1_bin, P2_bin, gamma_bin are optional — when present they enable the
        optimal-filter likelihood (likelihood_mode="optimal") in
        ParametersEstimator.
        """
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        payload = dict(
            f_raw=np.asarray(f_raw, dtype=float),
            Omega_hat_raw=np.asarray(Om_raw, dtype=float),
            sigma_Omega_raw=np.asarray(sOm_raw, dtype=float),
            f_bin=np.asarray(f_bin, dtype=float),
            Omega_hat_bin=np.asarray(Om_bin, dtype=float),
            sigma_Omega_bin=np.asarray(sOm_bin, dtype=float),
        )
        if P1_bin is not None:
            payload["P1_bin"]    = np.asarray(P1_bin,    dtype=float)
        if P2_bin is not None:
            payload["P2_bin"]    = np.asarray(P2_bin,    dtype=float)
        if gamma_bin is not None:
            payload["gamma_bin"] = np.asarray(gamma_bin, dtype=float)
        if meta is not None:
            payload["meta"] = np.array([meta], dtype=object)
        np.savez(outpath, **payload)
