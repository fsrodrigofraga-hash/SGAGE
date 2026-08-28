from __future__ import annotations

# ============================================================
# ParametersEstimator.plotting — figure mixin
#
# Split out of estimator.py because this is the only part that depends on
# matplotlib and corner; inference runs without them. The method body is the
# same as in the monolithic module, unchanged.
# ============================================================
import os
from typing import Any, Dict, List, Optional, Sequence

import bilby

import numpy as np


class CornerPlotMixin:
    """ParametersEstimator figures (corner plot with truth markers)."""

    # -------------------------
    # Corner plot with truth markers
    # -------------------------
    def plot_corner_with_truths(
        self,
        result: "bilby.core.result.Result",
        *,
        # Truth values for ppE/alpha params — pass only what you know
        truth_a: Optional[float] = None,
        truth_alpha_ppE: Optional[float] = None,
        truth_alpha_params: Optional[Dict[str, float]] = None,
        # Extra per-parameter truth overrides (takes precedence over everything else)
        extra_truths: Optional[Dict[str, float]] = None,
        # Params to show in the corner plot (None = all inferred params except k_sigma)
        parameters: Optional[List[str]] = None,
        # Corner plot cosmetics
        bins: int = 40,
        smooth: float = 0.9,
        title_quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        fontsize: int = 12,
        truth_color: str = "orange",
        save: bool = False,
    ) -> "matplotlib.figure.Figure":
        """
        Build a corner plot from a bilby Result, overlaying orange truth markers
        for every parameter whose true value is known.

        Truth values are resolved in this priority order (highest wins):
          1. ``extra_truths`` dict
          2. ``truth_a`` / ``truth_alpha_ppE`` / ``truth_alpha_params`` keyword args
          3. ``self.lambda_true_values`` (set via set_lambda_true_values())

        Parameters
        ----------
        result : bilby.core.result.Result
            Output of run_inference().
        truth_a : float, optional
            True value of the ppE exponent ``a``.
        truth_alpha_ppE : float, optional
            True value of scalar alpha_ppE (used when alpha_event_model is None).
        truth_alpha_params : dict, optional
            True values of alpha_event_model parameters keyed by spec name.
        extra_truths : dict, optional
            Any additional {param_name: true_value} pairs (highest priority).
        parameters : list of str, optional
            Which parameters to include in the corner plot.
            Defaults to all inferred params minus ``k_sigma``.
        bins, smooth, title_quantiles : corner plot style kwargs.
        fontsize : int
            Font size for axis labels and diagonal titles.
        truth_color : str
            Color for truth lines and markers (default "orange").
        save : bool
            If True, save a PDF to ``{cfg.outdir}/corner_with_truths_{cache_hash}.pdf``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if self.specs is None:
            raise RuntimeError("Call prepare_model() before plot_corner_with_truths().")

        # ---- Resolve parameter list ----
        all_inferred = [s.name for s in self.specs]
        if parameters is None:
            params_plot = [p for p in all_inferred if p in result.posterior.columns
                          and p != "k_sigma"]
        else:
            params_plot = [p for p in parameters if p in result.posterior.columns]

        if not params_plot:
            raise ValueError("No valid parameters found in result.posterior for plotting.")

        # ---- Build truth dict (lowest to highest priority) ----
        truths: Dict[str, float] = {}

        # Lambda free params (lowest priority)
        truths.update(self.lambda_true_values)

        # alpha event-model params
        if truth_alpha_params:
            truths.update({k: float(v) for k, v in truth_alpha_params.items()})

        # scalar alpha_ppE
        if truth_alpha_ppE is not None:
            truths["alpha_ppE"] = float(truth_alpha_ppE)

        # ppE exponent a
        if truth_a is not None:
            truths["a"] = float(truth_a)

        # extra overrides (highest priority)
        if extra_truths:
            truths.update({k: float(v) for k, v in extra_truths.items()})

        # ---- Build labels from spec map ----
        spec_map = {s.name: s for s in self.specs}
        labels_plot = [
            (spec_map[p].latex_label if p in spec_map and spec_map[p].latex_label else p)
            for p in params_plot
        ]

        # ---- Draw corner plot ----
        post_plot = result.posterior[params_plot].copy()
        original_posterior = result.posterior
        try:
            result.posterior = post_plot
            fig = result.plot_corner(
                parameters=params_plot,
                labels=labels_plot,
                bins=int(bins),
                smooth=float(smooth),
                show_titles=True,
                title_quantiles=list(title_quantiles),
                save=False,
                use_math_text=True,
            )
        finally:
            result.posterior = original_posterior

        n_params = len(params_plot)

        # ---- Helper: get axes by position ----
        def _diag_ax(i):
            idx = i * n_params + i
            return fig.axes[idx] if idx < len(fig.axes) else None

        def _off_ax(row, col):
            idx = row * n_params + col
            return fig.axes[idx] if idx < len(fig.axes) else None

        # ---- Overlay truth markers ----
        for i, p in enumerate(params_plot):
            if p not in truths:
                continue
            ax = _diag_ax(i)
            if ax is not None:
                ax.axvline(truths[p], linewidth=1.4, alpha=0.95,
                           color=truth_color, zorder=9)

        for i_row in range(n_params):
            for i_col in range(i_row):
                ax = _off_ax(i_row, i_col)
                if ax is None:
                    continue
                p_x = params_plot[i_col]
                p_y = params_plot[i_row]
                tx = truths.get(p_x)
                ty = truths.get(p_y)
                if tx is not None:
                    ax.axvline(tx, linewidth=1.2, alpha=0.9,
                               color=truth_color, zorder=9)
                if ty is not None:
                    ax.axhline(ty, linewidth=1.2, alpha=0.9,
                               color=truth_color, zorder=9)
                if tx is not None and ty is not None:
                    ax.scatter(tx, ty, marker="s", s=90,
                               c=truth_color, linewidths=0, zorder=10)

        # ---- Font sizes ----
        for i in range(n_params):
            ax = _diag_ax(i)
            if ax is not None:
                ax.set_title(ax.get_title(), fontsize=fontsize)
        for ax in fig.axes:
            ax.xaxis.label.set_size(fontsize)
            ax.yaxis.label.set_size(fontsize)

        # ---- Optionally save ----
        if save:
            pdf_path = os.path.join(
                self.cfg.outdir,
                f"corner_with_truths_{self.cache_hash or 'nohash'}.pdf",
            )
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"[plot_corner_with_truths] saved: {pdf_path}", flush=True)

        return fig
