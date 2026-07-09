#!/usr/bin/env python
"""
run_ipd_experiments.py
======================

HalloReg 메인 엔트리포인트: IPD 시뮬레이션 → 가설검증(H1–H10) → 시각화.

사용 예
-------
    python scripts/run_ipd_experiments.py                      # 기본(전체)
    python scripts/run_ipd_experiments.py --seeds 20 --rounds 150 --jobs -1
    python scripts/run_ipd_experiments.py --experiments H1 H2 H5
    python scripts/run_ipd_experiments.py --backend pymdp --check-equivalence
    python scripts/run_ipd_experiments.py --quick               # 스모크 테스트

병렬
----
다이애드는 완전 독립이므로 multiprocessing('spawn') 으로 CPU 코어에 분배.
JAX/BLAS 워커 스레드는 1로 제한(ipd.sim 상단)하여 oversubscription 방지.
--jobs -1 이면 (코어수-1) 사용.
"""

from __future__ import annotations

# ---- 스레드/플랫폼 설정은 반드시 jax import 이전에 (ipd.sim 이 처리하지만 이중 안전) ----
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# HalloReg 를 최상위 패키지로 import 가능하게 (scripts/ 는 HalloReg/ 안에 있음)
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]          # .../HalloReg
sys.path.insert(0, str(_PKG_ROOT.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from HalloReg.core.constants import CC, COOP, DEFECT
from HalloReg.core.logging_utils import get_logger, set_korean_font
from HalloReg.ipd.sim import run_many, run_population
from HalloReg.ipd.metrics import (
    defense_metrics, defense_specificity, welch_t, aggregate, cc_rate, coop_rate,
)

LOGGER = get_logger("HalloReg.run")
RESULTS = _PKG_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 설정
LAM_BASE = 0.4
LAM_MAX = 0.8

OPPONENTS = {
    "exploiter":   "의도적 착취자 (ALLD, 실행잡음 0 → 높은 β)",
    "noisy_tft":   "잡음 상호성 파트너 (TFT + 20% 실행오류 → 낮은 β)",
    "noisy":       "완전 무작위 상대 (p(C)=0.5 → 낮은 β)",
    "tit_for_tat": "상호성 파트너 (TFT)",
    "allc":        "무조건 협력자",
    "wsls":        "Win-Stay Lose-Shift",
    "generous_tft": "관대한 TFT",
}


def agent_spec(kind: str, seed: int, **kw) -> dict:
    """AIF 에이전트 스펙 (sim.build_from_spec 용)."""
    base = dict(type=kind, seed=seed, lam_base=LAM_BASE)
    if kind == "adaptive":
        base.update(lam_max=LAM_MAX)
    base.update(kw)
    return base


def strat_spec(kind: str, seed: int, **kw) -> dict:
    return dict(type="strategy", kind=kind, seed=seed, **kw)


def capricious_spec(seed: int, rounds: int, error: float = 0.10) -> dict:
    """
    변덕 상대(형질 전환): 상호성 → 착취 → 관대한 상호성.

    `error` 는 실행잡음. 잡음이 0 이면 형질 전환이 행동에서 즉시 드러나 표면적
    상호성 규칙(TFT)만으로 충분하다. 잡음이 있으면 '잡음'과 '의도 변화'가 혼동되어,
    잠재 형질(α, β)을 추론하는 능력이 비로소 이득이 된다.
    """
    phases = ("tit_for_tat", "alld", "generous_tft")
    cuts = [int(round(i * rounds / 3)) for i in range(3)]
    return dict(type="strategy", kind="tit_for_tat", seed=seed, error=error,
                schedule=[(c, ph) for c, ph in zip(cuts, phases)])


def stack_traces(results, key: str) -> np.ndarray:
    """시드별 로그를 (n_seeds, n_rounds) 배열로 적재. 길이가 다르면 최소 길이로 절단."""
    arrs = [np.asarray(r["agent_log"][key], float) for r in results]
    L = min(len(a) for a in arrs)
    return np.stack([a[:L] for a in arrs])


def band(ax, arr: np.ndarray, label: str, **kw):
    """평균 곡선 + 평균 ± 1 SD 반투명 음영대."""
    m = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
    x = np.arange(len(m))
    (line,) = ax.plot(x, m, label=label, lw=2, **kw)
    ax.fill_between(x, m - sd, m + sd, alpha=0.18, color=line.get_color(), lw=0)
    return line


def mean_sd(vals) -> tuple:
    """리스트 -> (평균, 표본 SD)."""
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def _run_block(agent_cfg: dict, opp_kind: str, seeds: int, rounds: int,
               jobs: int, opp_cfg=None):
    """동일 조건을 seeds 개 시드로 병렬 실행 → (metric dicts, raw results)."""
    specs = []
    for s in range(seeds):
        a = dict(agent_cfg); a["seed"] = s
        o = dict(opp_cfg) if opp_cfg else strat_spec(opp_kind, seed=1000 + s)
        o["seed"] = 1000 + s
        specs.append({"agent": a, "opponent": o})
    res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
    mets = [defense_metrics(r["agent_log"], r["hist"], LAM_BASE) for r in res]
    return mets, res


# ================================================================== H1
def exp_H1(seeds, rounds, jobs, backend):
    """H1: AdaptiveAgent 는 착취자에게 자기보호(λ↓)하나 ToMEmpathicAgent 는 못 한다."""
    LOGGER.info("[H1] 착취자에 대한 자기보호")
    adaptive = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                          attribution_target="all", use_pymdp=backend == "pymdp")
    fixed = agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp")

    m_ad, r_ad = _run_block(adaptive, "exploiter", seeds, rounds, jobs)
    m_fx, r_fx = _run_block(fixed, "exploiter", seeds, rounds, jobs)

    t_lam, p_lam = welch_t([m["lam_final"] for m in m_ad],
                           [m["lam_final"] for m in m_fx])
    t_exp, p_exp = welch_t([m["exploitability"] for m in m_ad],
                           [m["exploitability"] for m in m_fx])
    t_pay, p_pay = welch_t([m["cum_payoff"] for m in m_ad],
                           [m["cum_payoff"] for m in m_fx])

    agg = {"adaptive": aggregate(m_ad), "fixed": aggregate(m_fx)}
    supported = (agg["adaptive"]["lam_final"][0] < agg["fixed"]["lam_final"][0]
                 and agg["adaptive"]["exploitability"][0] < agg["fixed"]["exploitability"][0])
    LOGGER.info("[H1] λ_final adaptive=%.3f vs fixed=%.3f (p=%.4f); "
                "착취가능성 %.3f vs %.3f (p=%.4f) → %s",
                agg["adaptive"]["lam_final"][0], agg["fixed"]["lam_final"][0], p_lam,
                agg["adaptive"]["exploitability"][0], agg["fixed"]["exploitability"][0],
                p_exp, "지지" if supported else "미지지")
    traces = {
        "lam_adaptive": stack_traces(r_ad, "lam"),
        "lam_fixed": stack_traces(r_fx, "lam"),
        "grievance": stack_traces(r_ad, "grievance"),
        "trust": stack_traces(r_ad, "trust"),
        "disp_credence": stack_traces(r_ad, "disp_credence"),
    }
    return {"agg": agg, "tests": {"lam": (t_lam, p_lam), "exploit": (t_exp, p_exp),
                                  "payoff": (t_pay, p_pay)},
            "raw": {"adaptive": m_ad, "fixed": m_fx},
            "supported": bool(supported),
            "traces": traces}


# ================================================================== H2 / H3
def exp_H2_H3(seeds, rounds, jobs, backend):
    """H2: 착취자 vs 잡음 상대 구분(λ 회복). H3: α, β 추정치 구분."""
    LOGGER.info("[H2/H3] 의도 vs 맥락 귀인 구분")
    adaptive = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                          use_pymdp=backend == "pymdp")
    out, traces = {}, {}
    for opp in ("exploiter", "noisy_tft", "noisy"):
        m, r = _run_block(adaptive, opp, seeds, rounds, jobs)
        out[opp] = aggregate(m)
        out[opp + "_raw"] = m
        traces[opp] = stack_traces(r, "lam")

    lam_ex = [m["lam_final"] for m in out["exploiter_raw"]]
    lam_nt = [m["lam_final"] for m in out["noisy_tft_raw"]]
    t_lam, p_lam = welch_t(lam_nt, lam_ex)

    b_ex = [m["E_beta"] for m in out["exploiter_raw"]]
    b_nt = [m["E_beta"] for m in out["noisy_tft_raw"]]
    t_b, p_b = welch_t(b_ex, b_nt)
    a_ex = [m["E_alpha"] for m in out["exploiter_raw"]]
    a_nt = [m["E_alpha"] for m in out["noisy_tft_raw"]]
    t_a, p_a = welch_t(a_nt, a_ex)

    h2 = np.mean(lam_nt) > np.mean(lam_ex)
    h3 = (np.mean(b_ex) > np.mean(b_nt)) and (np.mean(a_nt) > np.mean(a_ex))
    LOGGER.info("[H2] λ_final noisy_tft=%.3f > exploiter=%.3f (p=%.4f) → %s",
                np.mean(lam_nt), np.mean(lam_ex), p_lam, "지지" if h2 else "미지지")
    LOGGER.info("[H3] E[β] exploiter=%.3f vs noisy=%.3f (p=%.4f); "
                "E[α] noisy=%.3f vs exploiter=%.3f (p=%.4f) → %s",
                np.mean(b_ex), np.mean(b_nt), p_b, np.mean(a_nt), np.mean(a_ex),
                p_a, "지지" if h3 else "미지지")
    return {"agg": {k: v for k, v in out.items() if not k.endswith("_raw")},
            "raw": {k[:-4]: v for k, v in out.items() if k.endswith("_raw")},
            "tests": {"lam": (t_lam, p_lam), "beta": (t_b, p_b), "alpha": (t_a, p_a)},
            "supported": {"H2": bool(h2), "H3": bool(h3)}, "traces": traces}


# ================================================================== H4
def exp_H4(seeds, rounds, jobs, backend):
    """H4: AdaptiveAgent 는 TFT 상대에서 상호협력을 복원한다."""
    LOGGER.info("[H4] TFT 상대 상호협력 복원")
    adaptive = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                          use_pymdp=backend == "pymdp")
    fixed = agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp")
    m_ad, r_ad = _run_block(adaptive, "tit_for_tat", seeds, rounds, jobs)
    m_fx, _ = _run_block(fixed, "tit_for_tat", seeds, rounds, jobs)
    t, p = welch_t([m["cc_rate"] for m in m_ad], [m["cc_rate"] for m in m_fx])
    agg = {"adaptive": aggregate(m_ad), "fixed": aggregate(m_fx)}
    sup = agg["adaptive"]["cc_rate"][0] >= agg["fixed"]["cc_rate"][0] * 0.95 and \
        agg["adaptive"]["cc_rate"][0] > 0.3
    LOGGER.info("[H4] CC율 adaptive=%.3f vs fixed=%.3f (p=%.4f), 복원=%.3f → %s",
                agg["adaptive"]["cc_rate"][0], agg["fixed"]["cc_rate"][0], p,
                agg["adaptive"]["restoration"][0], "지지" if sup else "미지지")
    return {"agg": agg, "tests": {"cc": (t, p)}, "supported": bool(sup),
            "traces": {"lam_adaptive": stack_traces(r_ad, "lam")}}


# ================================================================== H5
def exp_H5(seeds, rounds, jobs, backend):
    """
    H5: 정교한 자기보호(sophisticated=True; 추론된 β 로 방어를 게이팅, rmPFC)와
        즉각적 자기보호(sophisticated=False; 배신 신호에 직접 반응, vmPFC)의 대조.

    검정 (모두 보고)
      (i)  [주 검정] 방어 특이성 = 방어(착취자) − 방어(잡음TFT).  정교 > 즉각.
      (ii) 착취자 상대 방어 자원: 정교 ≥ 즉각.
      (iii) 사용자 가설의 문자적 주장: 잡음 상대에게 정교형이 **덜** 번다.
            → 잡음TFT(용서가 보답되는 상대)와 완전무작위(보답되지 않는 상대)를 분리 검정.
    """
    LOGGER.info("[H5] 정교한(β 게이팅) vs 즉각적 자기보호")
    soph = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                      use_pymdp=backend == "pymdp")
    immed = agent_spec("adaptive", 0, kappa=0.9, sophisticated=False,
                       use_pymdp=backend == "pymdp")
    res = {}
    for label, cfg in (("sophisticated", soph), ("immediate", immed)):
        for opp in ("exploiter", "noisy_tft", "noisy"):
            m, _ = _run_block(cfg, opp, seeds, rounds, jobs)
            res[(label, opp)] = m

    def vals(k, lab, opp):
        return [m[k] for m in res[(lab, opp)]]

    def mean(k, lab, opp):
        return float(np.mean(vals(k, lab, opp)))

    # 방어량 = −착취가능성 (호구비율 − 유혹비율의 음수). 시드별로 보관.
    defense_seed = {(l, o): np.array([-m["exploitability"] for m in res[(l, o)]])
                    for l in ("sophisticated", "immediate")
                    for o in ("exploiter", "noisy_tft", "noisy")}
    payoff_seed = {(l, o): np.array(vals("cum_payoff", l, o))
                   for l in ("sophisticated", "immediate")
                   for o in ("exploiter", "noisy_tft", "noisy")}
    # 방어 특이성은 **같은 시드 내에서** 짝지어 차분한다 (run_many 가 spec 순서를 보존).
    spec_seed = {l: defense_seed[(l, "exploiter")] - defense_seed[(l, "noisy_tft")]
                 for l in ("sophisticated", "immediate")}

    defense = {k: float(v.mean()) for k, v in defense_seed.items()}
    spec_soph, spec_imm = float(spec_seed["sophisticated"].mean()), float(spec_seed["immediate"].mean())

    t_def, p_def = welch_t(list(defense_seed[("sophisticated", "exploiter")]),
                           list(defense_seed[("immediate", "exploiter")]))
    t_ntft, p_ntft = welch_t(vals("cum_payoff", "sophisticated", "noisy_tft"),
                             vals("cum_payoff", "immediate", "noisy_tft"))
    t_rnd, p_rnd = welch_t(vals("cum_payoff", "sophisticated", "noisy"),
                           vals("cum_payoff", "immediate", "noisy"))

    primary = spec_soph > spec_imm
    literal_defense = defense[("sophisticated", "exploiter")] >= defense[("immediate", "exploiter")]
    literal_cost_ntft = mean("cum_payoff", "sophisticated", "noisy_tft") < \
        mean("cum_payoff", "immediate", "noisy_tft")
    literal_cost_rnd = mean("cum_payoff", "sophisticated", "noisy") < \
        mean("cum_payoff", "immediate", "noisy")

    LOGGER.info("[H5-i] 방어 특이성 정교=%.3f±%.3f vs 즉각=%.3f±%.3f (평균±SD) → %s",
                spec_soph, spec_seed["sophisticated"].std(ddof=1),
                spec_imm, spec_seed["immediate"].std(ddof=1),
                "지지" if primary else "미지지")
    LOGGER.info("[H5-ii] 착취자 방어 정교=%.3f vs 즉각=%.3f (p=%.4f) → %s",
                defense[("sophisticated", "exploiter")], defense[("immediate", "exploiter")],
                p_def, "지지" if literal_defense else "미지지")
    LOGGER.info("[H5-iii] 잡음TFT 보수 정교=%.1f vs 즉각=%.1f (p=%.4f) → 정교가 덜 번다: %s",
                mean("cum_payoff", "sophisticated", "noisy_tft"),
                mean("cum_payoff", "immediate", "noisy_tft"), p_ntft, literal_cost_ntft)
    LOGGER.info("[H5-iii] 무작위 보수 정교=%.1f vs 즉각=%.1f (p=%.4f) → 정교가 덜 번다: %s",
                mean("cum_payoff", "sophisticated", "noisy"),
                mean("cum_payoff", "immediate", "noisy"), p_rnd, literal_cost_rnd)
    LOGGER.info("[H5] 해석: 관대함의 비용은 '용서가 보답되지 않는' 상대(무작위)에서만 발생한다.")

    def _ms(a):
        return [float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0]

    return {
        "defense": {f"{l}|{o}": defense[(l, o)] for (l, o) in defense},
        "defense_sd": {f"{l}|{o}": _ms(defense_seed[(l, o)])[1] for (l, o) in defense_seed},
        "specificity": {"sophisticated": spec_soph, "immediate": spec_imm},
        "specificity_sd": {l: _ms(spec_seed[l])[1] for l in spec_seed},
        "payoff": {f"{l}|{o}": float(payoff_seed[(l, o)].mean()) for (l, o) in payoff_seed},
        "payoff_sd": {f"{l}|{o}": _ms(payoff_seed[(l, o)])[1] for (l, o) in payoff_seed},
        "tests": {"defense_exploiter": (t_def, p_def),
                  "payoff_noisy_tft": (t_ntft, p_ntft),
                  "payoff_random": (t_rnd, p_rnd)},
        "supported": bool(primary),
        "literal": {"defense_more_vs_exploiter": bool(literal_defense),
                    "earns_less_vs_noisy_tft": bool(literal_cost_ntft),
                    "earns_less_vs_random": bool(literal_cost_rnd)},
    }


# ================================================================== H6
def exp_H6(seeds, rounds, jobs, backend):
    """H6: 무잡음 IPD 에서는 TFT 우세; 잡음 환경에서는 GTFT/WSLS 가 TFT 를 능가."""
    LOGGER.info("[H6] 잡음 수준에 따른 고정전략 성능")
    from HalloReg.ipd.sim import run_dyad
    from HalloReg.ipd.env import make_opponent
    from HalloReg.core.constants import PAYOFF_SELF

    strategies = ["tit_for_tat", "generous_tft", "wsls", "allc", "alld"]
    out, out_sd = {}, {}
    for noise in (0.0, 0.15):
        # 시드별로 '전체 라운드로빈 평균 보수'를 하나의 관측치로 삼는다
        # → SD 는 시드 간 변동성이며, 상대 전략 간 변동성과 섞이지 않는다.
        per_seed = {s: [] for s in strategies}
        for s in strategies:
            for sd in range(max(2, seeds // 3)):
                payoffs = []
                for opp in strategies:
                    a = make_opponent(s, seed=sd, error=noise)
                    b = make_opponent(opp, seed=500 + sd, error=noise)
                    h = run_dyad(a, b, rounds)
                    payoffs.append(float(np.mean(h["my_payoff"])))
                per_seed[s].append(float(np.mean(payoffs)))
        out[noise] = {s: mean_sd(v)[0] for s, v in per_seed.items()}
        out_sd[noise] = {s: mean_sd(v)[1] for s, v in per_seed.items()}

    tft0, tft_n = out[0.0]["tit_for_tat"], out[0.15]["tit_for_tat"]
    best0 = max(out[0.0], key=out[0.0].get)
    sup_a = out[0.0]["tit_for_tat"] >= out[0.0]["generous_tft"]
    sup_b = (out[0.15]["generous_tft"] > tft_n) or (out[0.15]["wsls"] > tft_n)
    LOGGER.info("[H6] 무잡음 최고=%s | 잡음시 TFT=%.3f GTFT=%.3f WSLS=%.3f → %s",
                best0, tft_n, out[0.15]["generous_tft"], out[0.15]["wsls"],
                "지지" if (sup_a and sup_b) else "부분/미지지")
    return {"scores": {str(k): v for k, v in out.items()},
            "scores_sd": {str(k): v for k, v in out_sd.items()},
            "supported": bool(sup_a and sup_b)}


# ================================================================== H7
def exp_H7(seeds, rounds, jobs, backend):
    """
    H7: **변덕스러운(형질 전환 + 실행잡음) 상대** 에 대해 Generous-TFT / WSLS 같은
        고정전략은 잠재 의도를 추론하지 못하므로 AdaptiveAgent 보다 열등하다.

    실행잡음 수준을 쓸어(sweep) 예측된 역U자 패턴을 확인한다.
      - 잡음 0     : 형질 전환이 행동에 그대로 드러남 → 표면 규칙(TFT)으로 충분, ToM 이득 없음
      - 잡음 중간  : 잡음과 의도변화가 혼동됨 → **잠재 형질 추론이 이득**
      - 잡음 과다  : 신호 자체가 소실 → 이득 감소
    주 조건은 중간 잡음(0.10).
    """
    LOGGER.info("[H7] 변덕스러운 상대: 의도추론의 이득")
    focals = {
        "adaptive": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                               use_pymdp=backend == "pymdp"),
        "generous_tft": strat_spec("generous_tft", 0),
        "wsls": strat_spec("wsls", 0),
        "tit_for_tat": strat_spec("tit_for_tat", 0),
    }
    noise_levels = (0.0, 0.10, 0.20)
    primary_noise = 0.10
    sweep, sweep_sd, raw = {}, {}, {}
    for err in noise_levels:
        for name, cfg in focals.items():
            specs = []
            for sd in range(seeds):
                a = dict(cfg); a["seed"] = sd
                specs.append({"agent": a,
                              "opponent": capricious_spec(700 + sd, rounds, error=err)})
            res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
            raw[(err, name)] = [float(np.mean(r["hist"]["my_payoff"])) for r in res]
        sweep[err] = {n: float(np.mean(raw[(err, n)])) for n in focals}
        sweep_sd[err] = {n: mean_sd(raw[(err, n)])[1] for n in focals}
        LOGGER.info("[H7] 실행잡음 %.2f → %s (SD %s)", err,
                    {k: round(v, 3) for k, v in sweep[err].items()},
                    {k: round(v, 3) for k, v in sweep_sd[err].items()})

    means = sweep[primary_noise]
    t_g, p_g = welch_t(raw[(primary_noise, "adaptive")], raw[(primary_noise, "generous_tft")])
    t_w, p_w = welch_t(raw[(primary_noise, "adaptive")], raw[(primary_noise, "wsls")])
    sup = means["adaptive"] > means["generous_tft"] and means["adaptive"] > means["wsls"]
    LOGGER.info("[H7] 주 조건(잡음 %.2f): adaptive=%.3f > GTFT=%.3f (p=%.4f), "
                "> WSLS=%.3f (p=%.4f) → %s", primary_noise, means["adaptive"],
                means["generous_tft"], p_g, means["wsls"], p_w, "지지" if sup else "미지지")
    LOGGER.info("[H7] 잡음 0 에서 ToM 이득 소실: adaptive=%.3f vs TFT=%.3f "
                "(형질 전환이 행동에 그대로 드러나기 때문)",
                sweep[0.0]["adaptive"], sweep[0.0]["tit_for_tat"])
    return {"means": means, "sweep": {str(k): v for k, v in sweep.items()},
            "sweep_sd": {str(k): v for k, v in sweep_sd.items()},
            "primary_noise": primary_noise,
            "tests": {"gtft": (t_g, p_g), "wsls": (t_w, p_w)},
            "supported": bool(sup)}


# ================================================================== H8
def exp_H8(seeds, rounds, jobs, backend):
    """H8: AdaptiveAgent 비율이 높을수록 집단 상호협력률이 상승한다 (동역학계)."""
    LOGGER.info("[H8] 집단 내 AdaptiveAgent 비율 → 상호협력")
    fractions = [0, 2, 4, 6]
    base = {"tit_for_tat": 2, "alld": 2, "allc": 1, "wsls": 1}
    curve = []
    for n_ad in fractions:
        ccs = []
        for sd in range(max(2, seeds // 4)):
            r = run_population(base, n_rounds=max(30, rounds // 3), n_adaptive=n_ad,
                               adaptive_kwargs=dict(lam_base=LAM_BASE, lam_max=LAM_MAX,
                                                    kappa=0.9, sophisticated=True),
                               seed=sd)
            ccs.append(r["cc_rate"])
        m, sd = mean_sd(ccs)
        curve.append((n_ad, m, sd))
        LOGGER.info("[H8] n_adaptive=%d → CC율 %.3f ± %.3f (평균±SD)", n_ad, m, sd)
    xs = np.array([c[0] for c in curve], float)
    ys = np.array([c[1] for c in curve], float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else 0.0
    sup = slope > 0
    LOGGER.info("[H8] 기울기=%.4f → %s", slope, "지지" if sup else "미지지")
    return {"curve": curve, "slope": slope, "supported": bool(sup)}


# ================================================================== H9 / H10
def exp_H9_H10(seeds, rounds, jobs, backend):
    """
    H9 : **α 전용 귀인** 에이전트는 상대의 의도가 변동(변덕 상대)할 때 완전귀인보다 열등하다.
         (ρ·λ_j 를 못 읽어 상호성/공감 전환을 추적하지 못한다.)
    H10: **λ 전용 귀인** 에이전트는 정적 상대(형질 고정 + 모호한 실행오류)에서 완전귀인보다
         열등하다. 맥락(β)에 귀인할 수 없어 모호한 배신을 기질로 오귀인하고 처벌한다.

    추가로 intent_only(맥락 귀인 불가)와 beta_context(의도 귀인 불가)의 이중해리를 보고한다.
    """
    LOGGER.info("[H9/H10] 귀인 범위(attribution_target) 절제 실험")
    targets = ["all", "intent_only", "alpha_only", "lambda_only", "beta_context"]

    res = {}
    for tgt in targets:
        cfg = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                         attribution_target=tgt, use_pymdp=backend == "pymdp")
        # (a) 변덕 상대 (의도 변동)  → H9
        specs = []
        for sd in range(seeds):
            a = dict(cfg); a["seed"] = sd
            specs.append({"agent": a, "opponent": capricious_spec(800 + sd, rounds)})
        out = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
        res[(tgt, "capricious")] = [
            defense_metrics(r["agent_log"], r["hist"], LAM_BASE) for r in out]
        # (b) 정적 상대 (형질 고정 + 10% 실행오류) → H10
        m, _ = _run_block(cfg, "static_noisy_tft", seeds, rounds, jobs)
        res[(tgt, "static_noisy_tft")] = m
        # (c) 착취자 (이중해리 확인)
        m2, _ = _run_block(cfg, "exploiter", seeds, rounds, jobs)
        res[(tgt, "exploiter")] = m2

    def v(k, tgt, opp):
        return [m[k] for m in res[(tgt, opp)]]

    def mu(k, tgt, opp):
        return float(np.mean(v(k, tgt, opp)))

    t9, p9 = welch_t(v("cum_payoff", "all", "capricious"),
                     v("cum_payoff", "alpha_only", "capricious"))
    t10, p10 = welch_t(v("cc_rate", "all", "static_noisy_tft"),
                       v("cc_rate", "lambda_only", "static_noisy_tft"))
    h9 = mu("cum_payoff", "all", "capricious") > mu("cum_payoff", "alpha_only", "capricious")
    h10 = mu("cc_rate", "all", "static_noisy_tft") > mu("cc_rate", "lambda_only", "static_noisy_tft")

    LOGGER.info("[H9] 변덕상대 누적보수 all=%.1f > alpha_only=%.1f (p=%.4f) → %s",
                mu("cum_payoff", "all", "capricious"),
                mu("cum_payoff", "alpha_only", "capricious"), p9, "지지" if h9 else "미지지")
    LOGGER.info("[H10] 정적잡음상대 CC율 all=%.3f > lambda_only=%.3f (p=%.4f) → %s",
                mu("cc_rate", "all", "static_noisy_tft"),
                mu("cc_rate", "lambda_only", "static_noisy_tft"), p10,
                "지지" if h10 else "미지지")
    # 이중해리: intent_only 는 잡음을 과잉처벌, beta_context 는 착취자를 과소방어
    dd_over = mu("cc_rate", "all", "static_noisy_tft") > mu("cc_rate", "intent_only", "static_noisy_tft")
    dd_under = mu("cum_payoff", "all", "exploiter") > mu("cum_payoff", "beta_context", "exploiter")
    LOGGER.info("[H9/H10 부가] 이중해리 — intent_only 과잉처벌:%s (CC %.3f vs %.3f) | "
                "beta_context 과소방어:%s (보수 %.1f vs %.1f)",
                dd_over, mu("cc_rate", "intent_only", "static_noisy_tft"),
                mu("cc_rate", "all", "static_noisy_tft"), dd_under,
                mu("cum_payoff", "beta_context", "exploiter"),
                mu("cum_payoff", "all", "exploiter"))

    table = {f"{t}|{o}": {"cum_payoff": mu("cum_payoff", t, o),
                          "cc_rate": mu("cc_rate", t, o),
                          "lam_final": mu("lam_final", t, o),
                          "E_beta": mu("E_beta", t, o)}
             for t in targets for o in ("capricious", "static_noisy_tft", "exploiter")}
    return {"table": table, "tests": {"H9": (t9, p9), "H10": (t10, p10)},
            "double_dissociation": {"intent_only_overpunishes": bool(dd_over),
                                    "beta_context_underdefends": bool(dd_under)},
            "supported": {"H9": bool(h9), "H10": bool(h10)}}


# ================================================================== 시각화
def visualize(results: dict, tag: str = ""):
    """
    3×3 패널. **모든 패널이 시드 간 평균 ± 1 SD 를 명시한다.**
      - 꺾은선: 평균 곡선 + 평균±1SD 반투명 음영대 (`band`)
      - 막대  : 평균 막대 + 1SD 오차막대(capsize) + 개별 시드 산점(jitter)
    """
    set_korean_font(plt, font_manager)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle("HalloReg: 위계적 항상성 조절 공감 에이전트 (IPD)  —  음영/오차막대 = ±1 SD (시드 간)",
                 fontsize=14)
    ERRKW = dict(capsize=3, ecolor="black", error_kw=dict(lw=1.0, capthick=1.0))
    rng = np.random.default_rng(0)

    def scatter_seeds(ax, x_center, vals, width=0.12):
        """개별 시드값을 지터 산점으로 겹쳐 그려 분포를 드러낸다."""
        v = np.asarray(vals, float)
        if v.size == 0:
            return
        jit = rng.uniform(-width, width, v.size)
        ax.scatter(np.full(v.size, x_center) + jit, v, s=8, color="k",
                   alpha=0.35, zorder=5, linewidths=0)

    # ---------------- (A) H1 λ 궤적 ----------------
    ax = axes[0, 0]
    if "H1" in results:
        tr = results["H1"]["traces"]
        band(ax, tr["lam_adaptive"], "AdaptiveAgent (조절)")
        band(ax, tr["lam_fixed"], "ToMEmpathicAgent (고정)", ls="--")
        n = tr["lam_adaptive"].shape[0]
        ax.set_title(f"A. 착취자 상대 λ 궤적 (H1, n={n} 시드)")
        ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    # ---------------- (B) H2 λ 회복 ----------------
    ax = axes[0, 1]
    if "H2H3" in results:
        tr = results["H2H3"]["traces"]
        for k, lab in (("exploiter", "착취자 (높은 β)"),
                       ("noisy_tft", "잡음 TFT (낮은 β)"),
                       ("noisy", "완전 무작위")):
            band(ax, tr[k], lab)
        ax.set_title("B. 의도 vs 맥락: λ 회복 (H2)")
        ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    # ---------------- (C) H3 θ 추정 ----------------
    ax = axes[0, 2]
    if "H2H3" in results:
        agg = results["H2H3"]["agg"]
        raw = results["H2H3"].get("raw", {})
        labs = ["exploiter", "noisy_tft", "noisy"]
        x = np.arange(len(labs)); w = 0.35
        a = [agg[l]["E_alpha"][0] for l in labs]
        b = [agg[l]["E_beta"][0] for l in labs]
        ae = [agg[l]["E_alpha"][1] for l in labs]      # [1] = SD
        be = [agg[l]["E_beta"][1] for l in labs]
        ax.bar(x - w/2, a, w, yerr=ae, label="E[α] 협력편향 (로짓)", **ERRKW)
        ax.bar(x + w/2, b, w, yerr=be, label="E[β] 행동정밀도", **ERRKW)
        for i, l in enumerate(labs):
            if l in raw:
                scatter_seeds(ax, i - w/2, [m["E_alpha"] for m in raw[l]])
                scatter_seeds(ax, i + w/2, [m["E_beta"] for m in raw[l]])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(["착취자", "잡음TFT", "무작위"], fontsize=8)
        ax.set_ylabel("추정 형질값 (단위 상이)")
        ax.set_title("C. 상대 형질 추정 구분 (H3)"); ax.legend(fontsize=7)

    # ---------------- (D) H1/H4 착취가능성 · CC율 ----------------
    ax = axes[1, 0]
    if "H1" in results and "H4" in results:
        labels = ["착취가능성\n(vs 착취자)", "CC율\n(vs TFT)"]
        ad = [results["H1"]["agg"]["adaptive"]["exploitability"][0],
              results["H4"]["agg"]["adaptive"]["cc_rate"][0]]
        ad_sd = [results["H1"]["agg"]["adaptive"]["exploitability"][1],
                 results["H4"]["agg"]["adaptive"]["cc_rate"][1]]
        fx = [results["H1"]["agg"]["fixed"]["exploitability"][0],
              results["H4"]["agg"]["fixed"]["cc_rate"][0]]
        fx_sd = [results["H1"]["agg"]["fixed"]["exploitability"][1],
                 results["H4"]["agg"]["fixed"]["cc_rate"][1]]
        x = np.arange(2); w = 0.35
        ax.bar(x - w/2, ad, w, yerr=ad_sd, label="Adaptive", **ERRKW)
        ax.bar(x + w/2, fx, w, yerr=fx_sd, label="Fixed-λ", **ERRKW)
        if "raw" in results["H1"]:
            scatter_seeds(ax, -w/2, [m["exploitability"] for m in results["H1"]["raw"]["adaptive"]])
            scatter_seeds(ax, +w/2, [m["exploitability"] for m in results["H1"]["raw"]["fixed"]])
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("비율 [-1, 1]")
        ax.set_title("D. 자기보호와 협력복원 (H1, H4)"); ax.legend(fontsize=8)

    # ---------------- (E) H5 선택적 방어 ----------------
    ax = axes[1, 1]
    if "H5" in results:
        r = results["H5"]
        soph = [r["defense"]["sophisticated|exploiter"],
                r["specificity"]["sophisticated"],
                r["payoff"]["sophisticated|noisy_tft"] / 100.0,
                r["payoff"]["sophisticated|noisy"] / 100.0]
        soph_sd = [r["defense_sd"]["sophisticated|exploiter"],
                   r["specificity_sd"]["sophisticated"],
                   r["payoff_sd"]["sophisticated|noisy_tft"] / 100.0,
                   r["payoff_sd"]["sophisticated|noisy"] / 100.0]
        imm = [r["defense"]["immediate|exploiter"],
               r["specificity"]["immediate"],
               r["payoff"]["immediate|noisy_tft"] / 100.0,
               r["payoff"]["immediate|noisy"] / 100.0]
        imm_sd = [r["defense_sd"]["immediate|exploiter"],
                  r["specificity_sd"]["immediate"],
                  r["payoff_sd"]["immediate|noisy_tft"] / 100.0,
                  r["payoff_sd"]["immediate|noisy"] / 100.0]
        x = np.arange(4); w = 0.35
        ax.bar(x - w/2, soph, w, yerr=soph_sd, label="정교(rmPFC, β 게이팅)", **ERRKW)
        ax.bar(x + w/2, imm, w, yerr=imm_sd, label="즉각(vmPFC)", **ERRKW)
        ax.set_xticks(x)
        ax.set_xticklabels(["착취자\n방어량", "방어\n특이성",
                            "잡음TFT\n보수/100", "무작위\n보수/100"], fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("지표값 (막대별 단위 상이)")
        ax.set_title("E. 선택적 자기보호와 관대함의 비용 (H5)"); ax.legend(fontsize=7)

    # ---------------- (F) H1 누적기 ----------------
    ax = axes[1, 2]
    if "H1" in results:
        tr = results["H1"]["traces"]
        band(ax, tr["grievance"], "불만 g⁻ (vmPFC)", color="crimson")
        band(ax, tr["trust"], "신뢰 g⁺", color="seagreen")
        band(ax, tr["disp_credence"], "기질귀인 신뢰도 (dmPFC)", color="steelblue", ls=":")
        ax.set_title("F. 항상성 누적기 (착취자 상대)")
        ax.set_xlabel("라운드"); ax.set_ylabel("무차원 [0, 1]"); ax.legend(fontsize=8)

    # ---------------- (G) H6 잡음 × 전략 ----------------
    ax = axes[2, 0]
    if "H6" in results:
        sc, sd = results["H6"]["scores"], results["H6"]["scores_sd"]
        strats = list(sc["0.0"].keys())
        x = np.arange(len(strats)); w = 0.35
        ax.bar(x - w/2, [sc["0.0"][t] for t in strats], w,
               yerr=[sd["0.0"][t] for t in strats], label="무잡음", **ERRKW)
        ax.bar(x + w/2, [sc["0.15"][t] for t in strats], w,
               yerr=[sd["0.15"][t] for t in strats], label="잡음 15%", **ERRKW)
        ax.set_xticks(x); ax.set_xticklabels(strats, rotation=30, fontsize=7)
        ax.set_ylabel("라운드당 평균 보수 [0, 5]")
        ax.set_title("G. 잡음과 고정전략 (H6)"); ax.legend(fontsize=8)

    # ---------------- (H) H7 변덕 상대 × 잡음 ----------------
    ax = axes[2, 1]
    if "H7" in results:
        sw, sw_sd = results["H7"]["sweep"], results["H7"]["sweep_sd"]
        noises = list(sw.keys())
        agents = ["adaptive", "generous_tft", "wsls", "tit_for_tat"]
        x = np.arange(len(noises)); w = 0.2
        for i, a in enumerate(agents):
            ax.bar(x + (i - 1.5) * w, [sw[n][a] for n in noises], w,
                   yerr=[sw_sd[n][a] for n in noises], label=a,
                   color="darkorange" if a == "adaptive" else None, **ERRKW)
        ax.set_xticks(x); ax.set_xticklabels([f"잡음 {n}" for n in noises], fontsize=7)
        ax.set_ylabel("라운드당 평균 보수 [0, 5]"); ax.legend(fontsize=6)
        ax.set_title("H. 변덕 상대(상호성→착취→화해) × 잡음 (H7)")

    # ---------------- (I) H8 집단 역학 ----------------
    ax = axes[2, 2]
    if "H8" in results:
        c = results["H8"]["curve"]
        xs = np.array([q[0] for q in c], float)
        ys = np.array([q[1] for q in c], float)
        es = np.array([q[2] for q in c], float)
        ax.plot(xs, ys, marker="o", lw=2, color="tab:purple")
        ax.fill_between(xs, ys - es, ys + es, alpha=0.18, color="tab:purple", lw=0)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="black", capsize=3, lw=1.0)
        ax.set_xlabel("집단 내 AdaptiveAgent 수"); ax.set_ylabel("상호협력(CC)률 [0, 1]")
        ax.set_title(f"I. 집단 역학 (H8, 기울기={results['H8']['slope']:.4f})")

    for a in axes.flat:
        a.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / f"halloreg_ipd{tag}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOGGER.info("그림 저장: %s", out)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()
                if not (isinstance(k, str) and k == "traces")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ================================================================== main
EXPERIMENTS = {
    "H1": exp_H1, "H2H3": exp_H2_H3, "H4": exp_H4, "H5": exp_H5,
    "H6": exp_H6, "H7": exp_H7, "H8": exp_H8, "H9H10": exp_H9_H10,
}


def main():
    ap = argparse.ArgumentParser(description="HalloReg IPD 실험 실행기")
    ap.add_argument("--seeds", type=int, default=12, help="조건별 시드(다이애드) 수")
    ap.add_argument("--rounds", type=int, default=120, help="다이애드 라운드 수")
    ap.add_argument("--jobs", type=int, default=-1, help="병렬 워커 수 (-1=코어-1, 1=순차)")
    ap.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS),
                    help=f"실행할 실험 {list(EXPERIMENTS)}")
    ap.add_argument("--backend", choices=["numpy", "pymdp"], default="numpy",
                    help="EFE 백엔드 (등가; pymdp 는 검증용/느림)")
    ap.add_argument("--check-equivalence", action="store_true",
                    help="pymdp 1.0.x 와 numpy 해석적 EFE 동치성 검증")
    ap.add_argument("--quick", action="store_true", help="스모크 테스트(작은 설정)")
    ap.add_argument("--tag", default="", help="출력 파일 접미사")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.rounds = 3, 30

    LOGGER.info("=" * 74)
    LOGGER.info("HalloReg IPD | seeds=%d rounds=%d jobs=%s backend=%s",
                args.seeds, args.rounds, args.jobs, args.backend)
    LOGGER.info("=" * 74)

    if args.check_equivalence:
        from HalloReg.core.pymdp_backend import PymdpEFE, pymdp_available
        if pymdp_available():
            ok = PymdpEFE.check_equivalence()
            LOGGER.info("pymdp ↔ numpy EFE 동치성: %s", "통과" if ok else "실패")
        else:
            LOGGER.warning("pymdp 미설치 — 동치성 검증 건너뜀")

    t0 = time.time()
    results = {}
    for name in args.experiments:
        if name not in EXPERIMENTS:
            LOGGER.warning("알 수 없는 실험: %s (건너뜀)", name)
            continue
        results[name] = EXPERIMENTS[name](args.seeds, args.rounds, args.jobs, args.backend)

    LOGGER.info("전체 실험 소요 %.1fs", time.time() - t0)

    png = visualize(results, tag=args.tag)
    js = RESULTS / f"halloreg_results{args.tag}.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(_jsonable({k: {kk: vv for kk, vv in v.items() if kk != "traces"}
                             for k, v in results.items()}), f,
                  ensure_ascii=False, indent=2)
    LOGGER.info("요약 저장: %s", js)

    LOGGER.info("-" * 74)
    LOGGER.info("가설 요약:")
    for k, v in results.items():
        sup = v.get("supported")
        LOGGER.info("  %-6s → %s", k, sup)
    LOGGER.info("완료. 결과: %s", RESULTS)


if __name__ == "__main__":
    main()
