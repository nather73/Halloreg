"""
experiments.common
==================

Execution configuration, result registry and plotting utilities
shared by every hypothesis experiment.

[Confirmatory vs exploratory]
Only the **pre-specified primary tests** of each hypothesis are
registered as confirmatory; everything else is exploratory. Holm
(FWER control) applies to the confirmatory family, BH-FDR to the
exploratory one — structurally preventing "run many metrics, report
the significant ones".

[Verification culture]
A `verdict` records "supported" only when both the p value and the
**direction** hold. Unsupported results stay unsupported — no forced
positive narratives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")                   # headless: file output only
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.core.logging_utils import get_logger, set_korean_font
from AIF_IPD.ipd.metrics import bh_fdr, holm

LOGGER = get_logger("HalloReg.exp")

set_korean_font(plt, font_manager)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
})

#: Fixed per-type colours, consistent across all figures.
TYPE_COLORS = {
    "tft": "#4C72B0", "gtft": "#55A868", "wsls": "#C44E52",
    "allc": "#8172B2", "alld": "#937860", "halloreg": "#DA8BC3",
    "empathic_lo": "#8C8C8C", "empathic_hi": "#CCB974",
}


# ==================================================================== Config
@dataclass
class Config:
    """Global experiment configuration (populated by the CLI)."""
    seeds: int = 120                 # default seed count (per spec)
    rounds: int = 120
    eval_from: int = 0               # evaluation-window start (0 = full span)
    jobs: int = -1                   # parallel workers (-1 -> cores-1)
    results: Path = Path("results")
    env_error: float = 0.05          # env execution error (symmetric)
    n_particles: int = 400
    horizon: int = 6                     # trait-space rollout horizon
    policy_particles: int = 64
    prop_sd: float = 0.40
    policy_gamma: float = 8.0
    w_epi_j: float = 10.0
    w_epi_r: float = 1.0
    w_cplx: float = 0.15
    payoff_access: str = "naive"
    qrtd_gamma: float = 0.9
    qrtd_lr: float = 0.20
    w_cd: float = 0.5                # Empathy affect-context weight
    lam_gain: float = 0.05           # lambda integrator gain eta
    w_tonic: float = 0.10            # retired v1.8.0 (signature compat)
    lam_gain_down: float = 0.45      # threat-direction relaxation
    aff_gain: float = 0.30           # affect gain (round-level jumps)
    tom_es_mode: str = "mirror"      # ToM es: 'mirror' | 'analytic' (legacy)
    w_ig_r: float = 5.0              # epistemic weight w_R of -G_social
    w_ig_j: float = 5.0              # epistemic weight w_theta of -G_social
    r_surv_fixed: object = None      # fixed survival ref (None = maximin)
    beta_g: float = 3.0              # social-EFE precision beta
    # ── Social-EFE weights (v3.8.2 sweep evidence) ────────────────
    # Effective weights beta*(w_U, w_R, w_theta). An 800R sweep
    # (halve/double/zero/untie each axis; verdict criteria fixed in
    # advance: ALLD defence intact, HR-HR CC within +/-0.02, mixed
    # payoff maximal) passed all 7 candidates within a 0.030 band —
    # a **robust plateau**; the nominal best (+0.009) is within seed
    # noise, so the configuration stands. w_R = w_theta is
    # unit-consistent (path-A expected KL and IG_theta share nats,
    # v3.7.2) and untying showed no gain. Epistemic weights stay > 0
    # not for steady-state payoff (zero is equivalent there) but for
    # measured **early function**: directed probing, 3x faster WSLS
    # state coverage, H1 probe discrimination — functions invisible
    # to dyad payoff. The w_U magnitude balances steady-state
    # pragmatic dominance (~1-3% epistemic share) against early
    # epistemic engagement (~15% in R1-5).
    w_u: float = 40.0                # v3.9.0: effective beta*w_U = 120
    #   [v3.9.0 battery] w_U=40, w_R=w_theta=5 (effective 15):
    #   HR-HR 0.893(.017), ALLD 0.039/1.225, WSLS 0.741 (in band),
    #   mixed 2.449, lambda-hat_j ladder preserved.
    plan_sweeps: int = 1             # model-based planning sweeps/round
    plan_update: str = "conf"        # 'conf' (model-confidence Dyna) | 'replace' | 'dyna' | 'td'
    n_step: int = 1                  # n of on-policy n-step SARSA
    plan_lr: float = 0.5             # fixed lr for plan_update='td'
    bootstrap: str = "sarsa"         # bootstrap: 'sarsa' | 'greedy'
    alpha_kappa: float = 0.0         # history bias alpha ~ N(0, kappa^2)
    sp_disposition: float = 0.50     # dispositional part m of the setpoint
    quick: bool = False              # smoke mode
    max_compositions: int = 0        # 0 = full enumeration

    def halloreg_kwargs(self) -> dict:
        """HalloRegAgent constructor kwargs (identical across all
        experiments)."""
        return {"n_particles": self.n_particles, "planning_horizon": self.horizon,
                "policy_particles": self.policy_particles,
                "prop_sd": self.prop_sd, "policy_gamma": self.policy_gamma,
                "w_epi_j": self.w_epi_j, "w_epi_r": self.w_epi_r,
                "w_cplx": self.w_cplx,
                "payoff_access": self.payoff_access,
                "qrtd_gamma": self.qrtd_gamma, "qrtd_lr": self.qrtd_lr,
                "w_cd": self.w_cd, "lam_gain": self.lam_gain,
                "w_tonic": self.w_tonic,
                "aff_gain": self.aff_gain,
                "lam_gain_down": self.lam_gain_down,
                "tom_es_mode": self.tom_es_mode,
                "w_ig_r": self.w_ig_r,
                "w_ig_j": self.w_ig_j,
                "r_surv_fixed": self.r_surv_fixed,
                "beta_g": self.beta_g,
                "w_u": self.w_u,
                "plan_sweeps": self.plan_sweeps,
                "plan_update": self.plan_update,
                "plan_lr": self.plan_lr,
                "n_step": self.n_step,
                "bootstrap": self.bootstrap,
                "alpha_kappa": self.alpha_kappa,
                "sp_disposition": self.sp_disposition}

    def empathic_kwargs(self, lam: float) -> dict:
        """Fixed-lambda control constructor kwargs."""
        return {"lam": lam, "n_particles": self.n_particles,
                "planning_horizon": self.horizon}

    @property
    def figdir(self) -> Path:
        d = self.results / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ==================================================================== Registry
@dataclass
class Registry:
    """Accumulator of confirmatory/exploratory test results."""
    primary: List[dict] = field(default_factory=list)
    exploratory: List[dict] = field(default_factory=list)

    def confirm(self, hyp: str, label: str, p: float, direction_ok: bool,
                effect: str = "", detail: Optional[dict] = None) -> None:
        """Register a confirmatory test; Holm is applied once after
        all hypotheses finish."""
        self.primary.append({"hypothesis": hyp, "label": label,
                             "p_raw": float(p),
                             "direction_ok": bool(direction_ok),
                             "effect": effect, "detail": detail or {}})
        LOGGER.info("  [confirmatory][%s] %s — %s, direction_ok=%s (p_raw=%.4g)",
                    hyp, label, effect, direction_ok, p)

    def explore(self, hyp: str, label: str, p: float,
                effect: str = "", detail: Optional[dict] = None) -> None:
        """Register an exploratory test; BH-FDR is applied once after
        all hypotheses finish."""
        self.exploratory.append({"hypothesis": hyp, "label": label,
                                 "p_raw": float(p), "effect": effect,
                                 "detail": detail or {}})
        LOGGER.info("  [exploratory][%s] %s — %s (p_raw=%.4g)", hyp, label, effect, p)

    def finalize(self, alpha: float = 0.05) -> dict:
        """
        Build the final verdict table after multiple-comparison
        correction.

        Confirmatory: Holm-adjusted p < alpha **and** direction holds
        -> supported. Exploratory: BH-FDR q < alpha -> significant
        (no separate direction verdict).
        """
        if self.primary:
            adj = holm([r["p_raw"] for r in self.primary])
            for r, a in zip(self.primary, adj):
                r["p_holm"] = a
                r["supported"] = bool(a < alpha and r["direction_ok"])
        if self.exploratory:
            adj = bh_fdr([r["p_raw"] for r in self.exploratory])
            for r, a in zip(self.exploratory, adj):
                r["q_bh"] = a
                r["significant"] = bool(a < alpha)

        # Per-hypothesis verdict: supported iff **all** its
        # confirmatory tests are supported.
        by_hyp: Dict[str, dict] = {}
        for r in self.primary:
            h = by_hyp.setdefault(r["hypothesis"], {"n": 0, "n_ok": 0})
            h["n"] += 1
            h["n_ok"] += int(r["supported"])
        for h, d in by_hyp.items():
            d["verdict"] = ("supported" if d["n_ok"] == d["n"]
                            else ("partial" if d["n_ok"] > 0
                                  else "not supported"))
        return {"alpha": alpha, "primary": self.primary,
                "exploratory": self.exploratory, "by_hypothesis": by_hyp}


# ==================================================================== Serialisation
def _jsonable(o):
    """Make objects (including numpy types) JSON-serialisable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, path: Path) -> None:
    """Save a result dict as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


# ==================================================================== Plotting
def save_fig(fig, cfg: Config, name: str) -> Path:
    """Save a figure as PNG and close it."""
    path = cfg.figdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("  figure saved: %s", path)
    return path


def bar_with_ci(ax, labels, means, cis, colors=None, ylabel: str = "",
                title: str = "", rotate: int = 0) -> None:
    """Bar chart with bootstrap 95% CI error bars."""
    x = np.arange(len(labels))
    lo = np.array([m - c[0] for m, c in zip(means, cis)])
    hi = np.array([c[1] - m for m, c in zip(means, cis)])
    ax.bar(x, means, yerr=[lo, hi], capsize=3,
           color=colors if colors else "#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate,
                       ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def band_plot(ax, traces: np.ndarray, color: str = "#4C72B0",
              label: str = "", q: tuple = (25, 75)) -> None:
    """
    Median line + interquartile band over a set of seed
    trajectories. Percentile bands are used instead of mean +/- SD:
    for bounded, skewed metrics (lambda, cooperation rates) SD bands
    leave the domain and mislead.
    """
    T = traces.shape[1]
    t = np.arange(T)
    med = np.nanmedian(traces, axis=0)
    lo = np.nanpercentile(traces, q[0], axis=0)
    hi = np.nanpercentile(traces, q[1], axis=0)
    ax.plot(t, med, color=color, lw=1.6, label=label)
    ax.fill_between(t, lo, hi, color=color, alpha=0.18, lw=0)


def annotate_n(ax, n: int) -> None:
    """Annotate the sample size inside the figure (reproducibility
    aid)."""
    ax.text(0.99, 0.01, f"n={n}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#555555")
