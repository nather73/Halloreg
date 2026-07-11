#!/usr/bin/env python
"""
validate_tom_recovery.py — ToM 파라미터 복원 연구 (보완안 §2, H3 를 판별→복원 격상).

입자필터가 상정하는 것과 **동일한 생성 모형**으로 알려진 θ=(α, ρ, β)의 합성
상대를 만들고, N 라운드 상호작용 후 사후평균의 편향·RMSE, 90% 신용구간의
경험적 커버리지, 라운드에 따른 사후 수축을 보고한다. 낮은 β 에서 α·ρ 가
혼동되는 식별 불가능 영역을 히트맵으로 드러낸다.

사용:
    python scripts/validate_tom_recovery.py                 # 4×4×4 격자, rounds=60,120
    python scripts/validate_tom_recovery.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.core.logging_utils import get_logger, set_korean_font
from AIF_IPD.ipd.tom.inversion import OpponentInversion, ObservationContext

LOGGER = get_logger("HalloReg.recovery")
RESULTS = _PKG_ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class SyntheticOpponent:
    """입자필터의 우도와 동일한 협력확률 모형으로 행동하는 합성 상대."""

    def __init__(self, alpha, rho, beta, lambda_j=0.3, seed=0):
        self.alpha, self.rho, self.beta = alpha, rho, beta
        self.lambda_j = lambda_j
        self.rng = np.random.default_rng(seed)
        self.my_coop_rate = 0.5

    def coop_prob(self, focal_last):
        f = 0.0 if focal_last is None else (1.0 - 2.0 * focal_last)
        shift = 5.0 * self.lambda_j - self.my_coop_rate - 1.0
        return float(_logistic(self.beta * (self.alpha + self.rho * f + shift)))

    def act(self, focal_last):
        return 0 if self.rng.random() < self.coop_prob(focal_last) else 1


def recover_once(theta, rounds, seed):
    """한 합성 상대에 대해 rounds 라운드 관측 후 사후 궤적을 반환."""
    alpha, rho, beta = theta
    opp = SyntheticOpponent(alpha, rho, beta, seed=seed)
    inv = OpponentInversion(n_particles=400, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    focal_last = None
    means_trace, stds_trace = [], []
    for t in range(rounds):
        opp_a = opp.act(focal_last)
        ctx = ObservationContext(my_last_action=focal_last,
                                 their_last_action=None,
                                 joint_outcome=None, round_number=t)
        inv.update(opp_a, ctx)
        means_trace.append(dict(inv.posterior_means()))
        stds_trace.append(dict(inv.posterior_stds()))
        focal_last = int(rng.random() < 0.5)      # 탐색적 무작위 focal 행동
    return means_trace, stds_trace


def run_grid(levels, rounds_list, reps, seed0=0):
    axes = {"alpha": np.linspace(0.2, 1.6, levels),
            "rho": np.linspace(0.2, 1.6, levels),
            "beta": np.linspace(0.5, 8.0, levels)}
    records = []
    sid = seed0
    for a in axes["alpha"]:
        for r in axes["rho"]:
            for b in axes["beta"]:
                for R in rounds_list:
                    biases, rmses, covers, shrink = [], [], [], []
                    for rep in range(reps):
                        mt, st = recover_once((a, r, b), R, sid); sid += 1
                        final = mt[-1]
                        true = {"alpha": a, "rho": r, "beta": b}
                        for ax_ in ("alpha", "rho", "beta"):
                            biases.append((ax_, final[ax_] - true[ax_]))
                            rmses.append((ax_, (final[ax_] - true[ax_]) ** 2))
                            lo = final[ax_] - 1.645 * st[-1][ax_]
                            hi = final[ax_] + 1.645 * st[-1][ax_]
                            covers.append((ax_, lo <= true[ax_] <= hi))
                            shrink.append((ax_, st[0][ax_] - st[-1][ax_]))
                    records.append({"alpha": a, "rho": r, "beta": b, "rounds": R,
                                    "bias": biases, "sq": rmses,
                                    "cover": covers, "shrink": shrink})
    return records, axes


def summarize(records):
    def agg(key, ax_, fn):
        vals = [v for rec in records for (a, v) in rec[key] if a == ax_]
        return float(fn(vals))
    out = {}
    for ax_ in ("alpha", "rho", "beta"):
        out[ax_] = {"bias": agg("bias", ax_, np.mean),
                    "rmse": float(np.sqrt(agg("sq", ax_, np.mean))),
                    "coverage_90": agg("cover", ax_, np.mean),
                    "shrinkage": agg("shrink", ax_, np.mean)}
    return out


def identifiability_map(records, axes):
    """낮은 β 에서 α·ρ 혼동: β 수준별 α RMSE 를 (α, ρ) 격자로."""
    betas = sorted({rec["beta"] for rec in records})
    lo_b, hi_b = betas[0], betas[-1]
    alphas = sorted({rec["alpha"] for rec in records})
    rhos = sorted({rec["rho"] for rec in records})
    grids = {}
    for tag, bsel in (("low_beta", lo_b), ("high_beta", hi_b)):
        G = np.full((len(alphas), len(rhos)), np.nan)
        for i, a in enumerate(alphas):
            for j, r in enumerate(rhos):
                sq = [v for rec in records if abs(rec["beta"] - bsel) < 1e-9
                      and abs(rec["alpha"] - a) < 1e-9
                      and abs(rec["rho"] - r) < 1e-9
                      for (ax_, v) in rec["sq"] if ax_ == "alpha"]
                if sq:
                    G[i, j] = np.sqrt(np.mean(sq))
        grids[tag] = (G, alphas, rhos, bsel)
    return grids


def make_figure(summary, grids, rounds_list):
    try:
        set_korean_font(plt, font_manager)
    except Exception:
        pass
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    axes_names = ["alpha", "rho", "beta"]
    x = np.arange(3)
    ax[0].bar(x - 0.2, [summary[a]["bias"] for a in axes_names], 0.4, label="편향")
    ax[0].bar(x + 0.2, [summary[a]["rmse"] for a in axes_names], 0.4, label="RMSE")
    ax[0].axhline(0, color="0.5", lw=1); ax[0].set_xticks(x)
    ax[0].set_xticklabels(axes_names); ax[0].legend(fontsize=8)
    ax[0].set_title("사후 편향 / RMSE")
    ax[1].bar(x, [summary[a]["coverage_90"] for a in axes_names],
              color=["C0", "C1", "C2"])
    ax[1].axhline(0.90, color="r", ls="--", label="목표 0.90")
    ax[1].set_xticks(x); ax[1].set_xticklabels(axes_names)
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].set_title("90% 신용구간 경험적 커버리지")
    G, alphas, rhos, bsel = grids["low_beta"]
    im = ax[2].imshow(G, cmap="magma", origin="lower", aspect="auto")
    ax[2].set_xticks(range(len(rhos)))
    ax[2].set_xticklabels([f"{r:.1f}" for r in rhos], fontsize=7)
    ax[2].set_yticks(range(len(alphas)))
    ax[2].set_yticklabels([f"{a:.1f}" for a in alphas], fontsize=7)
    ax[2].set_xlabel("ρ (true)"); ax[2].set_ylabel("α (true)")
    ax[2].set_title(f"α RMSE @ 낮은 β={bsel:.1f}\n(식별 불가능 영역)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.savefig(RESULTS / "recovery_tom.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "recovery_tom.pdf", bbox_inches="tight")
    with open(RESULTS / "recovery_tom.caption.json", "w", encoding="utf-8") as f:
        json.dump({"figure": "recovery_tom",
                   "caption": "ToM 파라미터 복원: 사후 편향·RMSE, 90% 커버리지, "
                              "낮은 β 에서 α·ρ 식별 불가능 영역. H3 를 판별에서 "
                              "복원으로 격상 (보완안 §2).",
                   "rounds": rounds_list}, f, ensure_ascii=False, indent=2)
    plt.close(fig)
    LOGGER.info("그림 저장: %s", RESULTS / "recovery_tom.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--rounds", nargs="*", type=int, default=[60, 120])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.levels, args.rounds, args.reps = 3, [60], 1
    LOGGER.info("복원 연구: levels=%d rounds=%s reps=%d",
                args.levels, args.rounds, args.reps)
    records, axes = run_grid(args.levels, args.rounds, args.reps)
    summary = summarize(records)
    grids = identifiability_map(records, axes)
    for ax_, s in summary.items():
        LOGGER.info("[%s] 편향=%.3f RMSE=%.3f 커버리지=%.2f 수축=%.3f",
                    ax_, s["bias"], s["rmse"], s["coverage_90"], s["shrinkage"])
    make_figure(summary, grids, args.rounds)
    with open(RESULTS / "recovery_tom.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary,
                   "identifiability": {k: {"alpha_rmse_grid": g[0].tolist(),
                                           "beta": g[3]}
                                       for k, g in grids.items()},
                   "config": vars(args)}, f, ensure_ascii=False, indent=2)
    LOGGER.info("결과 저장: %s", RESULTS / "recovery_tom.json")


if __name__ == "__main__":
    main()
