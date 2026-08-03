"""
experiments.h1a_tracking
========================

**H1A — OpponentInversion 은 타인의 *변동하는* 의도를 잘 추적하며, λ 값도
적절히 복원되는가?**

H1 이 '정적 유형의 분류' 라면, H1A 는 '동적 형질의 추적' 이다. 두 개의 독립적인
하위 과제로 나눈다.

────────────────────────────────────────────────────────────────────────
과제 A — 형질 전환 추적 (intent tracking)
────────────────────────────────────────────────────────────────────────
상대는 30 라운드마다 기질이 바뀌는 `StrategyAgent` 다(예: TFT → ALLD → GTFT →
TFT). 실행잡음도 함께 부과해, '잡음' 과 '의도 변화' 가 공존하는 조건을 만든다.

지표 — **전환 정렬 추적 지수(switch-aligned tracking index)**:
각 전환 시점 τ 를 기준으로, 전환 직전 창(τ−W, τ) 과 전환 후 창(τ+L, τ+L+W) 에서
θ̂ 의 협력성향 사영을 비교한다. 사영은

    c(θ̂) = tanh(α̂ / 2) + tanh(ρ̂ / 2)·0      → α̂ 축의 부호 있는 협력편향

가 아니라, **국면의 참 협력률과 θ̂ 가 예측하는 협력률의 상관** 으로 정의한다.
즉 각 라운드에서 추론기가 내놓는 예측 협력확률 `pred_coop` 을, 그 라운드에
실제로 활성인 국면의 참 협력률과 대응시킨다. 이것이 "추적" 의 가장 직접적인
조작화다: 상대가 협력적 국면이면 추론기도 높은 협력확률을 예측해야 한다.

  · 확증 A : 시드별 상관 r > 0 (부호뒤집기 순열검정).
  · 탐색 A : 전환 후 회복 지연(latency) — |Δpred_coop| 이 전환 전후 차이의
             절반을 넘기까지 걸린 라운드 수.

────────────────────────────────────────────────────────────────────────
과제 B — λ 복원 (lambda recovery)
────────────────────────────────────────────────────────────────────────
상대를 **EmpathicAgent(고정 λ)** 로 두면 참 λ 를 우리가 안다. λ 를 세션 중간에
전환시키면(λ_schedule) '변동하는 공감' 을 만들 수 있다.

  조건 1 (정적 λ)  : λ_true ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} 각각 고정.
                     → 최종 λ̂_j 와 λ_true 의 상관 = 복원 정확도.
  조건 2 (동적 λ)  : λ 가 30 라운드마다 저↔고로 전환.
                     → 라운드별 λ̂_j 와 라운드별 λ_true 의 상관 = 추적 정확도.

  · 확증 B1 : 정적 조건 — corr(λ̂_j, λ_true) > 0.
  · 확증 B2 : 동적 조건 — 시드 내 corr(λ̂_j(t), λ_true(t)) > 0.

[해석상의 주의 — 반드시 명시]
λ_j 는 상대의 행동에 **s(λ_j, p) = (T−S)·λ_j + …** 라는 절편 형태로만 들어간다.
따라서 α 와 λ_j 는 고정 보수 하에서 부분적으로 공선이다(둘 다 협력확률의
절편을 올린다). 이 공선성은 본 설계의 한계이며, 복원 상관이 완전하지 않을 것을
**사전에 예측**한다. H6 에서는 보수(T, S)를 블록마다 변조해 이 공선성을 깨는
설계를 별도로 수행한다. 여기서는 실험 조건(고정 PD)에서의 복원 가능성을
있는 그대로 보고한다.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.ipd.env import SWITCH_PERIOD, SWITCH_SCENARIOS, switch_rounds
from AIF_IPD.ipd.metrics import (
    boot_mean_ci, corr_boot, one_sample_perm,
)
from AIF_IPD.ipd.sim import run_many
from .common import Config, Registry, save_fig, save_json

LOGGER = get_logger("HalloReg.H1A")

#: 정적 λ 복원 조건에서 사용할 참 λ 격자
LAM_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: 동적 λ 전환의 저/고 수준
LAM_LO, LAM_HI = 0.1, 0.9

#: 각 고정전략 국면의 '참 협력 성향' (전환 추적의 참조 신호).
#: 상대가 그 국면에서 focal 의 협력에 대해 보일 협력확률의 대표값이다.
#: ALLD 는 0, ALLC 는 1, 상호성 계열은 focal 이 협력적일 때 높다.
PHASE_COOP = {"tft": 0.9, "gtft": 0.95, "wsls": 0.7,
              "allc": 1.0, "alld": 0.0, "random": 0.5}


def _safe_mean(x) -> float:
    """
    NaN 을 제외한 평균. 유효값이 하나도 없으면 NaN 을 반환한다.

    시드별 상관 배열은 그 시드에서 신호가 상수였을 때 NaN 이 된다(예: 지평이
    짧아 국면 전환이 한 번도 없으면 참조 신호가 상수). np.nanmean 은 이때
    RuntimeWarning 을 내므로, 명시적으로 처리해 로그를 깨끗하게 유지한다.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


# ==================================================================== 과제 A
def _phase_signal(scenario: str, n_rounds: int) -> np.ndarray:
    """라운드별 활성 국면의 참 협력성향 (n_rounds,)."""
    cycle = SWITCH_SCENARIOS[scenario]
    return np.array([PHASE_COOP[cycle[(t // SWITCH_PERIOD) % len(cycle)]]
                     for t in range(n_rounds)])


def run_tracking(cfg: Config, reg: Registry) -> dict:
    """과제 A — 형질 전환 추적."""
    scenarios = list(SWITCH_SCENARIOS)
    hk = cfg.halloreg_kwargs()

    specs, registry = [], {}
    for si, sc in enumerate(scenarios):
        for sd in range(cfg.seeds):
            registry[(si, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 3000 + sd * 19 + si, **hk},
                # 형질 전환 상대는 schedule 을 가진 StrategyAgent 로 만든다.
                "opponent": {"type": "strategy",
                             "kind": SWITCH_SCENARIOS[sc][0],
                             "seed": 4000 + sd * 19 + si,
                             "error": 0.05,
                             "schedule": [
                                 (r, SWITCH_SCENARIOS[sc][
                                     (r // SWITCH_PERIOD) % len(SWITCH_SCENARIOS[sc])])
                                 for r in range(0, cfg.rounds, SWITCH_PERIOD)]},
                "env_err_agent": cfg.env_error, "env_err_opponent": 0.0,
                "noise_seed": 610_000 + sd * 71 + si,
            })

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H1A 형질전환 다이애드")

    corr_by_scenario: Dict[str, np.ndarray] = {}
    traces: Dict[str, np.ndarray] = {}
    for si, sc in enumerate(scenarios):
        truth = _phase_signal(sc, cfg.rounds)
        rs = np.full(cfg.seeds, np.nan)
        tr = np.zeros((cfg.seeds, cfg.rounds))
        for sd in range(cfg.seeds):
            pc = np.asarray(res[registry[(si, sd)]]["agent_log"]["pred_coop"],
                            dtype=float)
            tr[sd] = pc
            if pc.std() > 0 and truth.std() > 0:
                rs[sd] = float(np.corrcoef(pc, truth)[0, 1])
        corr_by_scenario[sc] = rs
        traces[sc] = tr

    all_r = np.concatenate([corr_by_scenario[s] for s in scenarios])
    t = one_sample_perm(all_r, 0.0, alternative="greater")
    ci = boot_mean_ci(all_r)
    reg.confirm("H1A", "형질전환 추적: corr(예측협력확률, 참 국면) > 0",
                t["p"], direction_ok=bool(_safe_mean(all_r) > 0),
                effect=f"r̄={ci['mean']:.3f} [{ci['ci'][0]:.3f}, {ci['ci'][1]:.3f}]",
                detail={"mean_r": ci["mean"], "ci": ci["ci"]})

    # 탐색: 시나리오별 추적 상관
    for sc in scenarios:
        rs = corr_by_scenario[sc]
        tt = one_sample_perm(rs, 0.0, alternative="greater")
        c = boot_mean_ci(rs)
        reg.explore("H1A", f"시나리오 {sc} 추적상관", tt["p"],
                    effect=f"r̄={c['mean']:.3f}")

    return {"scenarios": scenarios,
            "corr": {s: corr_by_scenario[s] for s in scenarios},
            "mean_corr": float(_safe_mean(all_r)),
            "traces": traces,
            "truth": {s: _phase_signal(s, cfg.rounds) for s in scenarios},
            "switch_rounds": switch_rounds(cfg.rounds)}


# ==================================================================== 과제 B
def run_lambda_recovery(cfg: Config, reg: Registry) -> dict:
    """과제 B — λ 복원 (정적 + 동적)."""
    hk = cfg.halloreg_kwargs()
    ek = {"n_particles": cfg.n_particles, "planning_horizon": cfg.horizon}

    # ---- 조건 1: 정적 λ ----
    specs, reg_static = [], {}
    for li, lam in enumerate(LAM_GRID):
        for sd in range(cfg.seeds):
            reg_static[(li, sd)] = len(specs)
            specs.append({
                "agent": {"type": "halloreg", "seed": 5000 + sd * 23 + li, **hk},
                "opponent": {"type": "empathic", "lam": float(lam),
                             "seed": 6000 + sd * 23 + li, **ek},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 620_000 + sd * 83 + li})
    n_static = len(specs)

    # ---- 조건 2: 동적 λ (30 라운드마다 저↔고 전환) ----
    lam_sched = [(r, LAM_LO if (r // SWITCH_PERIOD) % 2 == 0 else LAM_HI)
                 for r in range(0, cfg.rounds, SWITCH_PERIOD)]
    reg_dyn = {}
    for sd in range(cfg.seeds):
        reg_dyn[sd] = len(specs)
        specs.append({
            "agent": {"type": "halloreg", "seed": 7000 + sd * 29, **hk},
            "opponent": {"type": "empathic", "lam": LAM_LO,
                         "lam_schedule": lam_sched,
                         "seed": 8000 + sd * 29, **ek},
            "env_err_agent": cfg.env_error, "env_err_opponent": cfg.env_error,
            "noise_seed": 630_000 + sd * 89})

    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="H1A λ 복원 다이애드")

    # ---- 정적 조건 분석 ----
    lam_hat = np.zeros((len(LAM_GRID), cfg.seeds))
    for li in range(len(LAM_GRID)):
        for sd in range(cfg.seeds):
            lam_hat[li, sd] = float(
                res[reg_static[(li, sd)]]["agent_log"]["E_lambda_j"][-1])

    x = np.repeat(np.array(LAM_GRID), cfg.seeds)
    y = lam_hat.reshape(-1)
    cs = corr_boot(x, y)
    reg.confirm("H1A", "정적 λ 복원: corr(λ̂ⱼ, λ_true) > 0", cs["p"],
                direction_ok=bool(cs["r"] > 0),
                effect=f"r={cs['r']:.3f} [{cs['ci'][0]:.3f}, {cs['ci'][1]:.3f}]",
                detail=cs)

    # ---- 동적 조건 분석: 시드 내 라운드별 상관 ----
    lam_true_t = np.array([LAM_LO if (t // SWITCH_PERIOD) % 2 == 0 else LAM_HI
                           for t in range(cfg.rounds)])
    dyn_r = np.full(cfg.seeds, np.nan)
    dyn_traces = np.zeros((cfg.seeds, cfg.rounds))
    for sd in range(cfg.seeds):
        lj = np.asarray(res[reg_dyn[sd]]["agent_log"]["E_lambda_j"], dtype=float)
        dyn_traces[sd] = lj
        if lj.std() > 0:
            dyn_r[sd] = float(np.corrcoef(lj, lam_true_t)[0, 1])
    td = one_sample_perm(dyn_r, 0.0, alternative="greater")
    cd = boot_mean_ci(dyn_r)
    reg.confirm("H1A", "동적 λ 추적: 시드 내 corr(λ̂ⱼ(t), λ_true(t)) > 0",
                td["p"], direction_ok=bool(cd["mean"] > 0),
                effect=f"r̄={cd['mean']:.3f} [{cd['ci'][0]:.3f}, {cd['ci'][1]:.3f}]",
                detail={"mean_r": cd["mean"], "ci": cd["ci"]})

    return {"lam_grid": list(LAM_GRID), "lam_hat": lam_hat,
            "static_corr": cs, "dyn_corr": dyn_r,
            "dyn_traces": dyn_traces, "lam_true_t": lam_true_t,
            "n_static_specs": n_static}


# ==================================================================== 실행
def run(cfg: Config, reg: Registry) -> dict:
    LOGGER.info("[H1A] 변동 의도 추적 및 λ 복원")
    a = run_tracking(cfg, reg)
    b = run_lambda_recovery(cfg, reg)
    out = {"tracking": a, "lambda_recovery": b}
    _plot(cfg, a, b)
    save_json(out, cfg.results / "H1A.json")
    return out


# ==================================================================== 시각화
def _plot(cfg: Config, a: dict, b: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import band_plot, annotate_n

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.30)
    scenarios = a["scenarios"]

    # (a~c) 시나리오별 예측 협력확률 궤적 vs 참 국면
    for i, sc in enumerate(scenarios[:3]):
        ax = fig.add_subplot(gs[0, i])
        band_plot(ax, a["traces"][sc], color="#4C72B0", label="추론 예측 협력확률")
        ax.plot(a["truth"][sc], color="crimson", lw=1.4, ls="--",
                label="참 국면 협력성향")
        for r in a["switch_rounds"]:
            ax.axvline(r, color="#999999", lw=0.7, ls=":")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("라운드"); ax.set_ylabel("협력확률")
        ax.set_title(f"({'abc'[i]}) 형질전환 추적 — {sc}\n"
                     f"r̄={_safe_mean(a['corr'][sc]):.3f}")
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)
        annotate_n(ax, cfg.seeds)

    # (d) 정적 λ 복원 산점
    ax = fig.add_subplot(gs[1, 0])
    grid = b["lam_grid"]
    m = b["lam_hat"].mean(axis=1)
    s = b["lam_hat"].std(axis=1, ddof=1)
    for li, lam in enumerate(grid):
        ax.scatter(np.full(b["lam_hat"].shape[1], lam), b["lam_hat"][li],
                   s=5, alpha=0.18, color="#4C72B0")
    ax.errorbar(grid, m, yerr=s, fmt="o-", color="crimson", capsize=3, lw=1.5,
                label="시드 평균 ± SD")
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1, label="완전복원")
    ax.set_xlabel("참 λ (상대의 고정 공감)"); ax.set_ylabel("추정 λ̂ⱼ (최종)")
    ax.set_title(f"(d) 정적 λ 복원 — r={b['static_corr']['r']:.3f}")
    ax.legend(fontsize=7)

    # (e) 동적 λ 추적 궤적
    ax = fig.add_subplot(gs[1, 1])
    band_plot(ax, b["dyn_traces"], color="#55A868", label="추정 λ̂ⱼ")
    ax2 = ax.twinx()
    ax2.plot(b["lam_true_t"], color="crimson", ls="--", lw=1.4, label="참 λ(t)")
    ax2.set_ylim(-0.05, 1.05); ax2.set_ylabel("참 λ", color="crimson")
    ax2.grid(False)
    ax.set_xlabel("라운드"); ax.set_ylabel("추정 λ̂ⱼ")
    ax.set_title(f"(e) 동적 λ 추적 — r̄={_safe_mean(b['dyn_corr']):.3f}")
    ax.legend(loc="upper left", fontsize=7)
    annotate_n(ax, cfg.seeds)

    # (f) 시드별 추적 상관 분포
    ax = fig.add_subplot(gs[1, 2])
    data = [a["corr"][s] for s in scenarios] + [b["dyn_corr"]]
    labels = list(scenarios) + ["λ 추적"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#8172B2"); patch.set_alpha(0.7)
    ax.axhline(0, color="crimson", ls="--", lw=1.2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("시드별 상관계수")
    ax.set_title("(f) 추적 상관의 시드 분포\n(0 = 추적 실패)")

    fig.suptitle("H1A — 변동하는 의도의 추적과 λ 복원", fontsize=12, y=0.98)
    save_fig(fig, cfg, "H1A_tracking_recovery")
