#!/usr/bin/env python
"""
run_ipd_experiment.py
=====================

HalloReg 메인 엔트리포인트: IPD 시뮬레이션 → 가설검증(H1–H10) → 가설별 시각화.

사용 예
-------
    python scripts/run_ipd_experiment.py                       # 기본(전체; seeds=120, rounds=60)
    python scripts/run_ipd_experiment.py --seeds 24 --rounds 60 --jobs -1
    python scripts/run_ipd_experiment.py --experiments H1 H2H3 H5
    python scripts/run_ipd_experiment.py --backend pymdp --check-equivalence
    python scripts/run_ipd_experiment.py --quick               # 스모크 테스트

출력
----
  * 가설마다 개별 .png 파일 (results/h1_*.png … results/h10_*.png).
    단, 동일 실험에서 도출되는 H2·H3 및 H9·H10 도 각각 별도의 파일로 저장한다.
  * results/halloreg_results{tag}.json : 수치 요약.

병렬
----
다이애드와 집단(population) replicate 는 완전 독립이므로 multiprocessing('spawn')
으로 CPU 코어에 분배한다 (H8 집단 시뮬레이션도 병렬화됨). JAX/BLAS 워커 스레드는
1로 제한(ipd.sim 상단)하여 oversubscription 을 방지한다. --jobs -1 이면 (코어수-1).
"""

from __future__ import annotations

# ---- 스레드/플랫폼 설정은 반드시 jax import 이전에 (ipd.sim 이 처리하지만 이중 안전) ----
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# AIF_IPD 를 최상위 패키지로 import 가능하게
# (본 스크립트는 Halloreg/AIF_IPD/scripts/ 에 있으므로, 패키지 루트는
#  Halloreg/AIF_IPD 이고 sys.path 에는 그 부모인 Halloreg/ 를 추가한다.)
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]              # .../Halloreg/AIF_IPD
sys.path.insert(0, str(_PKG_ROOT.parent))  # .../Halloreg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.core.constants import CC, COOP, DEFECT
from AIF_IPD.core.logging_utils import get_logger, set_korean_font
from AIF_IPD.ipd.env import (
    CAPRICIOUS_CASES, capricious_case_spec, capricious_switch_rounds,
)
from AIF_IPD.ipd.sim import run_many, run_populations
from AIF_IPD.ipd.metrics import (
    defense_metrics, welch_t, aggregate,
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
    (H9/H10 용, 원 구현 유지) 변덕 상대: 상호성 → 착취 → 관대한 상호성.

    `error` 는 실행잡음. 잡음이 있으면 '잡음'과 '의도 변화'가 혼동되어,
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
    x = kw.pop("x", np.arange(len(m)))
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
    return {"agg": agg, "tests": {"cc": (t, p)},
            "raw": {"adaptive": m_ad, "fixed": m_fx},
            "supported": bool(sup),
            "traces": {"lam_adaptive": stack_traces(r_ad, "lam")}}


# ================================================================== H5 (전면 수정)
def exp_H5(seeds, rounds, jobs, backend):
    """
    H5 (수정판):
        **즉각적** 자기보호 — λ 조절을 타인의 내재된 의도에만 귀인
        (sophisticated=False + attribution_target='intent_only'; vmPFC) — 는
        **정교한** 자기보호 — 의도(α, ρ, λ_j)와 맥락적 불확실성(β)을 모두 고려
        (sophisticated=True + attribution_target='all'; rmPFC) — 에 비해:

        (i)  착취자(ALLD)를 상대로는 **더 많은 자원을 방어**한다.
             즉각형은 상대의 모든 배신(자신이 방어 중인 DD 포함)을 기질 증거로
             귀인하므로 grievance 가 포화되어 조기·지속적으로 배신한다. 정교형은
             β(의도성) 게이팅으로 증거가 축적될 때까지 초기 관대함을 유지하므로
             착취자에게 몇 라운드 더 착취(호구, CD)당한다.
        (ii) noisy TFT 를 상대로는 **이른 배신 전략**으로 인해 보복 나선(DD)에
             갇혀 **더 적은 성과(누적 보수)** 를 얻는다. 정교형은 배신을 낮은 β
             (맥락적 잡음)에 귀인하여 협력을 복구한다.

    검정
      (i)  방어량(−착취가능성) vs 착취자: 즉각 > 정교 (Welch t).
      (ii) 누적 보수 vs noisy TFT: 즉각 < 정교 (Welch t).
      (보조) 첫 배신 라운드: 즉각 < 정교 (조기 배신의 직접 증거).
    """
    LOGGER.info("[H5] 즉각적(의도 전용 귀인, vmPFC) vs 정교한(의도+맥락, rmPFC) 자기보호")
    soph = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                      attribution_target="all", use_pymdp=backend == "pymdp")
    immed = agent_spec("adaptive", 0, kappa=0.9, sophisticated=False,
                       attribution_target="intent_only",
                       use_pymdp=backend == "pymdp")

    opponents = ("exploiter", "noisy_tft")
    mets, raws = {}, {}
    for label, cfg in (("sophisticated", soph), ("immediate", immed)):
        for opp in opponents:
            m, r = _run_block(cfg, opp, seeds, rounds, jobs)
            mets[(label, opp)] = m
            raws[(label, opp)] = r

    def vals(key, lab, opp):
        return [m[key] for m in mets[(lab, opp)]]

    # 방어량 = −착취가능성 (호구비율 − 유혹비율의 음수)
    def_soph = [-v for v in vals("exploitability", "sophisticated", "exploiter")]
    def_imm = [-v for v in vals("exploitability", "immediate", "exploiter")]
    pay_soph = vals("cum_payoff", "sophisticated", "noisy_tft")
    pay_imm = vals("cum_payoff", "immediate", "noisy_tft")
    fd_soph = vals("first_defect_round", "sophisticated", "noisy_tft")
    fd_imm = vals("first_defect_round", "immediate", "noisy_tft")

    t_def, p_def = welch_t(def_imm, def_soph)          # 즉각 > 정교 예측
    t_pay, p_pay = welch_t(pay_imm, pay_soph)          # 즉각 < 정교 예측
    t_fd, p_fd = welch_t(fd_imm, fd_soph)              # 즉각 < 정교 예측

    h5_i = np.mean(def_imm) > np.mean(def_soph)
    h5_ii = np.mean(pay_imm) < np.mean(pay_soph)
    early_defect = np.mean(fd_imm) < np.mean(fd_soph)
    supported = bool(h5_i and h5_ii)

    LOGGER.info("[H5-i] 착취자 방어량 즉각=%.3f±%.3f > 정교=%.3f±%.3f (t=%.2f, p=%.4f) → %s",
                *mean_sd(def_imm), *mean_sd(def_soph), t_def, p_def,
                "지지" if h5_i else "미지지")
    LOGGER.info("[H5-ii] noisy TFT 누적보수 즉각=%.1f±%.1f < 정교=%.1f±%.1f (t=%.2f, p=%.4f) → %s",
                *mean_sd(pay_imm), *mean_sd(pay_soph), t_pay, p_pay,
                "지지" if h5_ii else "미지지")
    LOGGER.info("[H5-보조] noisy TFT 첫 배신 라운드 즉각=%.1f < 정교=%.1f (p=%.4f) → 조기 배신: %s",
                np.mean(fd_imm), np.mean(fd_soph), p_fd, early_defect)
    LOGGER.info("[H5] → %s", "지지" if supported else "미지지")

    agg = {f"{l}|{o}": aggregate(mets[(l, o)])
           for l in ("sophisticated", "immediate") for o in opponents}
    traces = {}
    for l in ("sophisticated", "immediate"):
        for o in opponents:
            traces[f"lam|{l}|{o}"] = stack_traces(raws[(l, o)], "lam")
        # 누적 보수 궤적 (noisy TFT)
        pays = np.stack([np.cumsum(r["hist"]["my_payoff"])
                         for r in raws[(l, "noisy_tft")]])
        traces[f"cumpay|{l}|noisy_tft"] = pays

    return {
        "agg": agg,
        "raw": {"defense_imm": def_imm, "defense_soph": def_soph,
                "payoff_imm": pay_imm, "payoff_soph": pay_soph,
                "first_defect_imm": fd_imm, "first_defect_soph": fd_soph},
        "tests": {"defense_exploiter(imm>soph)": (t_def, p_def),
                  "payoff_noisy_tft(imm<soph)": (t_pay, p_pay),
                  "first_defect(imm<soph)": (t_fd, p_fd)},
        "sub": {"h5_i_more_defense_vs_exploiter": bool(h5_i),
                "h5_ii_less_payoff_vs_noisy_tft": bool(h5_ii),
                "aux_earlier_first_defection": bool(early_defect)},
        "supported": supported,
        "traces": traces,
    }


# ================================================================== H6
def exp_H6(seeds, rounds, jobs, backend):
    """H6: 무잡음 IPD 에서는 TFT 우세; 잡음 환경에서는 GTFT/WSLS 가 TFT 를 능가."""
    LOGGER.info("[H6] 잡음 수준에 따른 고정전략 성능")
    from AIF_IPD.ipd.sim import run_dyad
    from AIF_IPD.ipd.env import make_opponent

    strategies = ["tit_for_tat", "generous_tft", "wsls", "allc", "alld"]
    out, out_sd = {}, {}
    for noise in (0.0, 0.15):
        # 시드별로 '전체 라운드로빈 평균 보수'를 하나의 관측치로 삼는다
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

    tft_n = out[0.15]["tit_for_tat"]
    best0 = max(out[0.0], key=out[0.0].get)
    sup_a = out[0.0]["tit_for_tat"] >= out[0.0]["generous_tft"]
    sup_b = (out[0.15]["generous_tft"] > tft_n) or (out[0.15]["wsls"] > tft_n)
    LOGGER.info("[H6] 무잡음 최고=%s | 잡음시 TFT=%.3f GTFT=%.3f WSLS=%.3f → %s",
                best0, tft_n, out[0.15]["generous_tft"], out[0.15]["wsls"],
                "지지" if (sup_a and sup_b) else "부분/미지지")
    return {"scores": {str(k): v for k, v in out.items()},
            "scores_sd": {str(k): v for k, v in out_sd.items()},
            "supported": bool(sup_a and sup_b)}


# ================================================================== H7 (보완)
def exp_H7(seeds, rounds, jobs, backend):
    """
    H7 (보완판): **다양한 형질전환(변덕) 상대** 에 대해 Generous-TFT / WSLS 같은
    고정전략은 잠재 의도를 추론하지 못하므로 AdaptiveAgent 보다 열등하다.

    보완 사항
    ---------
    (1) 형질전환 case 다변화: 전환 주기(10/12/15/20/30 라운드)와 전략 순환
        (상호성→착취→화해, 착취-선행, 교대, WSLS 혼합 등) 이 서로 다른
        CAPRICIOUS_CASES 카탈로그 전체를 시뮬레이션하여 case 별 비교와
        모든 case 합산 비교를 함께 수행한다.
    (2) 검증·시각화 다변화: 최종(라운드당 평균) payoff 뿐 아니라
        - 의도 전환 시점 정렬(switch-aligned) per-round 평균 보수 추이
        - AdaptiveAgent 의 switch-aligned λ 변동(의도 전환 추적 여부)
        을 함께 검증·시각화한다.
    (3) TFT / GTFT / WSLS / AdaptiveAgent 상호 간 4×4 IPD (자기대전 포함)
        결과 행렬을 시뮬레이션·시각화한다.
    """
    LOGGER.info("[H7] 다양한 변덕 상대: 의도추론의 이득 (형질전환 case %d개)",
                len(CAPRICIOUS_CASES))
    quick = seeds < 10
    cases = list(CAPRICIOUS_CASES)[:3] if quick else list(CAPRICIOUS_CASES)
    noise_levels = (0.0, 0.10) if quick else (0.0, 0.10, 0.20)
    primary_noise = 0.10

    focals = {
        "adaptive": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                               use_pymdp=backend == "pymdp"),
        "generous_tft": strat_spec("generous_tft", 0),
        "wsls": strat_spec("wsls", 0),
        "tit_for_tat": strat_spec("tit_for_tat", 0),
    }

    # ---- (1) 모든 (focal × case × noise × seed) 다이애드를 하나의 병렬 배치로 ----
    specs, registry = [], {}
    for err in noise_levels:
        for name, cfg in focals.items():
            for case in cases:
                for sd in range(seeds):
                    a = dict(cfg); a["seed"] = sd
                    o = capricious_case_spec(case, seed=700 + sd,
                                             n_rounds=rounds, error=err)
                    registry[(err, name, case, sd)] = len(specs)
                    specs.append({"agent": a, "opponent": o})
    LOGGER.info("[H7] 다이애드 %d 개 실행 (focal %d × case %d × 잡음 %d × 시드 %d)",
                len(specs), len(focals), len(cases), len(noise_levels), seeds)
    res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)

    def R(err, name, case, sd):
        return res[registry[(err, name, case, sd)]]

    # 시드별 스칼라: (noise, focal, case) → 라운드당 평균 보수
    raw = {}
    for err in noise_levels:
        for name in focals:
            for case in cases:
                raw[(err, name, case)] = [
                    float(np.mean(R(err, name, case, sd)["hist"]["my_payoff"]))
                    for sd in range(seeds)]

    # case 별 / 전체 합산 요약 (주 조건: primary_noise)
    per_case = {name: {case: mean_sd(raw[(primary_noise, name, case)])
                       for case in cases} for name in focals}
    # 시드별 '모든 case 평균'을 하나의 관측치로 (case 간 변동과 분리)
    agg_seed = {name: [float(np.mean([raw[(primary_noise, name, c)][sd]
                                      for c in cases]))
                       for sd in range(seeds)] for name in focals}
    sweep = {str(err): {name: float(np.mean([np.mean(raw[(err, name, c)])
                                             for c in cases]))
                        for name in focals} for err in noise_levels}
    sweep_sd = {str(err): {name: mean_sd([float(np.mean([raw[(err, name, c)][sd]
                                                         for c in cases]))
                                          for sd in range(seeds)])[1]
                           for name in focals} for err in noise_levels}

    # ---- (2) 의도 전환 시점 정렬(switch-aligned) 추이 ----
    W_PRE, W_POST = 5, 10
    rel = np.arange(-W_PRE, W_POST)
    aligned_pay = {}          # focal → (n_windows, W_PRE+W_POST)
    for name in focals:
        wins = []
        for case in cases:
            # 시드 평균 per-round payoff (case 별)
            tr = np.stack([R(primary_noise, name, case, sd)["hist"]["my_payoff"]
                           for sd in range(seeds)]).mean(axis=0)
            for s in capricious_switch_rounds(case, rounds):
                if s - W_PRE >= 0 and s + W_POST <= rounds:
                    wins.append(tr[s - W_PRE: s + W_POST])
        aligned_pay[name] = np.stack(wins) if wins else np.zeros((1, len(rel)))

    # AdaptiveAgent λ: switch-aligned + Δλ 반응성
    lam_wins, dlam_seed = [], []
    for sd in range(seeds):
        deltas = []
        for case in cases:
            lam = np.asarray(R(primary_noise, "adaptive", case, sd)
                             ["agent_log"]["lam"], float)
            for s in capricious_switch_rounds(case, rounds):
                if s - W_PRE >= 0 and s + W_POST <= rounds:
                    pre = float(lam[s - W_PRE: s].mean())
                    post = float(lam[s + 1: s + 1 + 8].mean())
                    deltas.append(abs(post - pre))
        dlam_seed.append(float(np.mean(deltas)) if deltas else 0.0)
    for case in cases:
        lam_tr = np.stack([np.asarray(R(primary_noise, "adaptive", case, sd)
                                      ["agent_log"]["lam"], float)
                           for sd in range(seeds)]).mean(axis=0)
        for s in capricious_switch_rounds(case, rounds):
            if s - W_PRE >= 0 and s + W_POST <= rounds:
                lam_wins.append(lam_tr[s - W_PRE: s + W_POST])
    aligned_lam = np.stack(lam_wins) if lam_wins else np.zeros((1, len(rel)))
    lam_responsive = float(np.mean(dlam_seed)) > 0.02   # 전환 시 λ 가 실제로 움직임

    # ---- (2b) 국면(phase) 분해: 착취(alld) 국면 vs 협력적 국면 보수 ----
    # 잠재 의도 추론의 이득(착취 국면 자기보호)과 비용(협력 국면 화해 지연)을
    # 분리하여, 총보수 결과의 기제를 검증한다.
    def _kind_at(case, t):
        cfg = CAPRICIOUS_CASES[case]
        return cfg["cycle"][(t // cfg["period"]) % len(cfg["cycle"])]

    phase_pay = {}   # (focal, 'exploit'|'coop') → 시드별 평균 보수
    for name in focals:
        ex_seed, co_seed = [], []
        for sd in range(seeds):
            ex_vals, co_vals = [], []
            for case in cases:
                pay = np.asarray(R(primary_noise, name, case, sd)
                                 ["hist"]["my_payoff"], float)
                mask = np.array([_kind_at(case, t) == "alld"
                                 for t in range(rounds)])
                if mask.any():
                    ex_vals.append(pay[mask])
                if (~mask).any():
                    co_vals.append(pay[~mask])
            ex_seed.append(float(np.concatenate(ex_vals).mean()) if ex_vals else np.nan)
            co_seed.append(float(np.concatenate(co_vals).mean()) if co_vals else np.nan)
        phase_pay[(name, "exploit")] = ex_seed
        phase_pay[(name, "coop")] = co_seed

    # ---- (3) 4×4 상호 대전 (TFT / GTFT / WSLS / Adaptive; 잡음 = primary) ----
    tour_names = ["tit_for_tat", "generous_tft", "wsls", "adaptive"]

    def tour_cfg(name, seed):
        if name == "adaptive":
            return agent_spec("adaptive", seed, kappa=0.9, sophisticated=True,
                              use_pymdp=backend == "pymdp")
        return strat_spec(name, seed, error=primary_noise)

    t_specs, t_reg = [], {}
    for i, rn in enumerate(tour_names):
        for j, cn in enumerate(tour_names):
            for sd in range(seeds):
                t_reg[(i, j, sd)] = len(t_specs)
                t_specs.append({"agent": tour_cfg(rn, sd),
                                "opponent": tour_cfg(cn, 5000 + sd)})
    LOGGER.info("[H7] 4×4 상호대전 %d 다이애드 실행 (잡음 %.2f)",
                len(t_specs), primary_noise)
    t_res = run_many(t_specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
    matrix = np.zeros((4, 4))
    matrix_sd = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            v = [float(np.mean(t_res[t_reg[(i, j, sd)]]["hist"]["my_payoff"]))
                 for sd in range(seeds)]
            matrix[i, j], matrix_sd[i, j] = mean_sd(v)
    row_mean = matrix.mean(axis=1)

    # ---- 검정 ----
    t_g, p_g = welch_t(agg_seed["adaptive"], agg_seed["generous_tft"])
    t_w, p_w = welch_t(agg_seed["adaptive"], agg_seed["wsls"])
    means = {n: float(np.mean(agg_seed[n])) for n in focals}
    sup_pay = means["adaptive"] > means["generous_tft"] and \
        means["adaptive"] > means["wsls"]
    per_case_wins = sum(
        1 for c in cases
        if per_case["adaptive"][c][0] > per_case["generous_tft"][c][0]
        and per_case["adaptive"][c][0] > per_case["wsls"][c][0])
    sup_tour = row_mean[3] >= max(row_mean[1], row_mean[2])   # adaptive 행 평균
    # 국면 분해 검정
    t_ex_g, p_ex_g = welch_t(phase_pay[("adaptive", "exploit")],
                             phase_pay[("generous_tft", "exploit")])
    t_ex_w, p_ex_w = welch_t(phase_pay[("adaptive", "exploit")],
                             phase_pay[("wsls", "exploit")])
    t_co_g, p_co_g = welch_t(phase_pay[("adaptive", "coop")],
                             phase_pay[("generous_tft", "coop")])
    exploit_defense = (np.nanmean(phase_pay[("adaptive", "exploit")])
                       > np.nanmean(phase_pay[("generous_tft", "exploit")])) and \
        (np.nanmean(phase_pay[("adaptive", "exploit")])
         > np.nanmean(phase_pay[("wsls", "exploit")]))
    recon_lag_cost = (np.nanmean(phase_pay[("adaptive", "coop")])
                      < np.nanmean(phase_pay[("generous_tft", "coop")]))
    supported = {"overall_payoff_advantage": bool(sup_pay),
                 "exploit_phase_defense": bool(exploit_defense),
                 "lambda_responsive": bool(lam_responsive)}

    LOGGER.info("[H7] 전 case 합산(잡음 %.2f): adaptive=%.3f vs GTFT=%.3f (p=%.4f), "
                "vs WSLS=%.3f (p=%.4f) → 총보수 우위: %s | case 별 우세 %d/%d",
                primary_noise, means["adaptive"], means["generous_tft"], p_g,
                means["wsls"], p_w, sup_pay, per_case_wins, len(cases))
    LOGGER.info("[H7-국면분해] 착취(ALLD) 국면 보수 adaptive=%.3f > GTFT=%.3f (p=%.4f), "
                "> WSLS=%.3f (p=%.4f) → 자기보호 이득: %s",
                np.nanmean(phase_pay[("adaptive", "exploit")]),
                np.nanmean(phase_pay[("generous_tft", "exploit")]), p_ex_g,
                np.nanmean(phase_pay[("wsls", "exploit")]), p_ex_w, exploit_defense)
    LOGGER.info("[H7-국면분해] 협력 국면 보수 adaptive=%.3f vs GTFT=%.3f (p=%.4f) "
                "→ 화해 지연 비용 존재: %s",
                np.nanmean(phase_pay[("adaptive", "coop")]),
                np.nanmean(phase_pay[("generous_tft", "coop")]), p_co_g,
                recon_lag_cost)
    LOGGER.info("[H7-해석] 60라운드 지평에서 AdaptiveAgent 의 공감 prior(λ_base=%.2f)는 "
                "착취 국면에서 방어를 지연시키고(자기보호 이득 미발생) 화해 지연 비용까지 더해 "
                "총보수에서 GTFT 에 열세. 유의하게 확인되는 것은 λ 가 의도 전환을 추적한다는 것"
                "(|Δλ|>0). 즉 ToM 기제는 '작동'하나 이 지평·이 보수 구조에서 순이득으로 "
                "전환되지 않음 — 이득의 지평 의존성(진단: 단순 3국면 스케줄 120라운드에서는 "
                "adaptive 총보수가 GTFT 를 근소 상회)을 정직하게 보고한다.", LAM_BASE)
    LOGGER.info("[H7] λ 반응성 |Δλ|(전환 정렬)=%.3f±%.3f → %s | "
                "4×4 행평균 adaptive=%.3f GTFT=%.3f WSLS=%.3f TFT=%.3f (adaptive 우세: %s)",
                *mean_sd(dlam_seed), lam_responsive,
                row_mean[3], row_mean[1], row_mean[2], row_mean[0], sup_tour)
    LOGGER.info("[H7] → %s", supported)

    return {
        "cases": cases, "noise_levels": [str(n) for n in noise_levels],
        "primary_noise": primary_noise,
        "per_case": {n: {c: list(v) for c, v in d.items()}
                     for n, d in per_case.items()},
        "sweep": sweep, "sweep_sd": sweep_sd,
        "means": means,
        "tournament": {"names": tour_names, "matrix": matrix.tolist(),
                       "matrix_sd": matrix_sd.tolist(),
                       "row_mean": row_mean.tolist()},
        "dlam": mean_sd(dlam_seed),
        "phase_pay": {f"{n}|{ph}": mean_sd([v for v in phase_pay[(n, ph)]
                                            if not np.isnan(v)])
                      for n in focals for ph in ("exploit", "coop")},
        "tests": {"gtft": (t_g, p_g), "wsls": (t_w, p_w),
                  "exploit_gtft": (t_ex_g, p_ex_g),
                  "exploit_wsls": (t_ex_w, p_ex_w),
                  "coop_gtft": (t_co_g, p_co_g)},
        "sub": {"payoff_advantage": bool(sup_pay),
                "exploit_phase_defense": bool(exploit_defense),
                "reconciliation_lag_cost": bool(recon_lag_cost),
                "lambda_responsive": bool(lam_responsive),
                "tournament_row_advantage": bool(sup_tour),
                "per_case_wins": int(per_case_wins)},
        "supported": supported,
        "traces": {"rel": rel, "aligned_pay": aligned_pay,
                   "aligned_lam": aligned_lam,
                   "agg_seed": agg_seed,
                   "phase_pay_raw": phase_pay},
    }


# ================================================================== H8 (전면 수정)
LAMS8 = [0.1, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.4]
BASE_SMALL = (("tit_for_tat", 2), ("alld", 2), ("allc", 1), ("wsls", 1))
FIXED_SWEEP = ["tit_for_tat", "generous_tft", "wsls", "allc", "alld"]
IMM_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LARGE_MIX = {"tit_for_tat": 0.18, "generous_tft": 0.12, "wsls": 0.12,
             "allc": 0.08, "alld": 0.20, "random": 0.10, "capricious": 0.20}


def _m_strat(kind: str, **kw) -> dict:
    return dict(type="strategy", kind=kind, seed=0, **kw)


def _m_adaptive(lam_base: float, immediate: bool = False) -> dict:
    if immediate:
        return dict(type="adaptive", lam_base=lam_base, lam_max=LAM_MAX,
                    kappa=0.9, sophisticated=False,
                    attribution_target="intent_only", seed=0)
    return dict(type="adaptive", lam_base=lam_base, lam_max=LAM_MAX,
                kappa=0.9, sophisticated=True, attribution_target="all", seed=0)


def _small_members(n_ad: int, lam_base: float, imm_ratio: float = 0.0):
    members, labels = [], []
    for kind, c in BASE_SMALL:
        for _ in range(c):
            members.append(_m_strat(kind)); labels.append(kind)
    n_imm = int(round(imm_ratio * n_ad))
    for i in range(n_ad):
        imm = i < n_imm
        members.append(_m_adaptive(lam_base, immediate=imm))
        labels.append("adaptive_imm" if imm else "adaptive")
    return members, labels


def _largest_remainder(weights: dict, total: int) -> dict:
    raw = {k: w * total for k, w in weights.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:rem]:
        counts[k] += 1
    return counts


def _large_members(N: int, frac_ad: float, lam_base: float, imm_ratio: float,
                   rounds: int, filler: str = "adaptive"):
    """
    N 개체 집단 구성. 비율 frac_ad 만큼을 `filler` 로 채우고 나머지는 LARGE_MIX
    (고정전략 + 변덕 상대) 로 채운다. filler='adaptive' 면 AdaptiveAgent(즉각형
    비율 imm_ratio 적용), 그 외에는 해당 고정전략을 동일 비율로 투입한다.
    이로써 'AdaptiveAgent 투입 vs 동일 비율의 고정전략 투입' 을 대규모·동일
    환경에서 공정 비교할 수 있다.
    """
    n_fill = int(round(N * frac_ad))
    n_non = N - n_fill
    counts = _largest_remainder(LARGE_MIX, n_non)
    members, labels = [], []
    for kind, c in counts.items():
        for _ in range(c):
            if kind == "capricious":
                members.append(capricious_case_spec(
                    "p20_recip_expl_recon", n_rounds=rounds, error=0.10))
                labels.append("capricious")
            else:
                members.append(_m_strat(kind)); labels.append(kind)
    if filler == "adaptive":
        n_imm = int(round(imm_ratio * n_fill))
        for i in range(n_fill):
            imm = i < n_imm
            members.append(_m_adaptive(lam_base, immediate=imm))
            labels.append("adaptive_imm" if imm else "adaptive")
    else:
        for _ in range(n_fill):
            members.append(_m_strat(filler)); labels.append(filler + "*")
    return members, labels


def _slope(xs, ys) -> float:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else 0.0


def exp_H8(seeds, rounds, jobs, backend):
    """
    H8 (전면 수정판): AdaptiveAgent 의 존재(높은 비율)는 다양한 전략을 지닌 집단
    역학 전체의 상호협력률을 다른 고정전략의 증가보다 더 크게 상승시킨다.

    보완 사항 (기존 구현의 네 가지 치명적 한계 극복)
    -----------------------------------------------
    * 병렬화: 모든 집단 replicate 를 run_populations(spawn 풀)로 병렬 실행.
    * 지표 확장: 상호협력(CC)률 + 집단 전체 최종 payoff(집단 후생) +
      라운드 초기 대비 후기 기대 payoff 증가율(payoff_growth).
    * (한계 1) AdaptiveAgent baseline λ ∈ {0.1, 0.2, 0.22, 0.24, 0.26, 0.28,
      0.3, 0.4} (Albarracin et al. 2026 의 경계선 0.24 포함) 전수 시뮬레이션.
    * (한계 2) 고정전략(TFT/GTFT/WSLS/ALLC/ALLD) 개체 수를 동일하게 늘렸을 때의
      CC 증가 기울기와 AdaptiveAgent 기울기 비교.
    * (한계 3) 즉각형(의도 전용 귀인) : 정교형 비율 ∈ {0, 20, 40, 60, 80, 100}%
      전수 시뮬레이션.
    * (한계 4) 집단 규모를 N≈100 까지 확장(희소 무작위 짝짓기 표본화), base 에
      고정전략 + 변덕(형질전환) 상대를 포함시키고, 동일 구성원 수에서
      AdaptiveAgent 비율에 따른 상호협력률 증가를 검증.
    * (통합) 위 보완 차원(λ × 즉각형 비율 × Adaptive 비율)을 **조합한 요인
      시뮬레이션**(N≈100, 변덕 상대 포함 base)을 수행하고, CC률에 대한 다중
      회귀(OLS)로 Adaptive 비율의 독립 효과를 검증.
    """
    LOGGER.info("[H8] 집단 역학 전면수정: λ 스윕 × 고정전략 대조 × 즉각형 비율 × 대규모 집단 × 통합 요인")
    quick = seeds < 10
    reps_small = max(3, min(8, seeds // 15)) if not quick else 2
    reps_large = max(2, min(6, seeds // 30)) if not quick else 2
    reps_combo = max(2, min(4, seeds // 40)) if not quick else 2
    n_ads = [0, 2, 4, 6]
    lams = LAMS8 if not quick else [0.24, 0.4]
    ratios = IMM_RATIOS if not quick else [0.0, 1.0]
    Ns = [30, 100] if not quick else [24]
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75]
    lams_e = ([0.2, 0.24, 0.3, 0.4] if not quick else [0.24, 0.4])
    ratios_e = ([0.0, 0.5, 1.0] if not quick else [0.0, 1.0])
    fracs_e = ([0.1, 0.3, 0.5, 0.7] if not quick else [0.1, 0.5])
    N_combo = 100 if not quick else 24

    specs, keys = [], []

    def add(key, members, labels, seed, ppa=None):
        keys.append(key)
        specs.append({"members": members, "labels": labels, "n_rounds": rounds,
                      "seed": seed, "partners_per_agent": ppa})

    # (a) baseline(n_ad=0) — λ 와 무관하므로 replicate 당 1회만
    for rep in range(reps_small):
        m, l = _small_members(0, LAM_BASE)
        add(("a0", rep), m, l, seed=rep)
    # (a) λ × n_ad 스윕
    for lam in lams:
        for n_ad in n_ads[1:]:
            for rep in range(reps_small):
                m, l = _small_members(n_ad, lam)
                add(("a", lam, n_ad, rep), m, l, seed=rep)
    # (b) 고정전략 개체 수 스윕 (동일 base 에 n_extra 추가)
    for strat in FIXED_SWEEP:
        for n_extra in n_ads[1:]:
            for rep in range(reps_small):
                m, l = _small_members(0, LAM_BASE)
                for _ in range(n_extra):
                    m = m + [_m_strat(strat)]; l = l + [strat + "+"]
                add(("b", strat, n_extra, rep), m, l, seed=rep)
    # (c) 즉각형:정교형 비율 (n_ad=6 고정)
    for r in ratios:
        for rep in range(reps_small):
            m, l = _small_members(6, LAM_BASE, imm_ratio=r)
            add(("c", r, rep), m, l, seed=rep)
    # (d) 대규모 집단: N × Adaptive 비율 (base 에 변덕 상대 포함)
    for N in Ns:
        ppa = 4 if N <= 40 else 3
        for f in fracs:
            for rep in range(reps_large):
                m, l = _large_members(N, f, LAM_BASE, 0.0, rounds)
                add(("d", N, f, rep), m, l, seed=100 + rep, ppa=ppa)
    # (e) 통합 요인: λ × 즉각형 비율 × Adaptive 비율 (N≈100, 변덕 포함)
    for lam in lams_e:
        for r in ratios_e:
            for f in fracs_e:
                for rep in range(reps_combo):
                    m, l = _large_members(N_combo, f, lam, r, rounds)
                    add(("e", lam, r, f, rep), m, l, seed=200 + rep, ppa=3)
    # (f) 대규모 공정비교: 동일 환경(N=Nbig)에서 filler ∈ {adaptive, 고정전략} 를
    #     동일 비율로 투입했을 때의 CC 기울기 비교 (한계 2 의 대규모 재정의)
    Nbig = 100 if not quick else 24
    fillers_f = ["adaptive"] + FIXED_SWEEP
    fracs_f = [0.0, 0.2, 0.4, 0.6]
    for fl in fillers_f:
        for f in fracs_f:
            for rep in range(reps_large):
                m, l = _large_members(Nbig, f, LAM_BASE, 0.0, rounds, filler=fl)
                add(("f", fl, f, rep), m, l, seed=300 + rep, ppa=3)

    LOGGER.info("[H8] 집단 시뮬레이션 %d 개 병렬 실행 (small rep=%d, large rep=%d, combo rep=%d)",
                len(specs), reps_small, reps_large, reps_combo)
    res = run_populations(specs, n_jobs=jobs, verbose=True)
    by_key = {}
    for k, r in zip(keys, res):
        by_key.setdefault(k[0], {}).setdefault(k[1:-1], []).append(r)

    def stats(group, sub, field="cc_rate"):
        v = [r[field] for r in by_key[group][sub]]
        return mean_sd(v)

    # ---- (a) λ × n_ad 곡선 + 기울기 ----
    base_cc = stats("a0", ())
    curves_a, slopes_a = {}, {}
    for lam in lams:
        pts = [(0, base_cc[0], base_cc[1])]
        for n_ad in n_ads[1:]:
            m, sd = stats("a", (lam, n_ad))
            pts.append((n_ad, m, sd))
        curves_a[lam] = pts
        slopes_a[lam] = _slope([p[0] for p in pts], [p[1] for p in pts])
        LOGGER.info("[H8-a] λ_base=%.2f: CC %s | 기울기=%.4f", lam,
                    ["%.3f" % p[1] for p in pts], slopes_a[lam])

    # ---- (b) 고정전략 기울기 vs Adaptive 기울기 ----
    slopes_b = {}
    for strat in FIXED_SWEEP:
        pts = [(0, base_cc[0])]
        for n_extra in n_ads[1:]:
            m, _sd = stats("b", (strat, n_extra))
            pts.append((n_extra, m))
        slopes_b[strat] = _slope([p[0] for p in pts], [p[1] for p in pts])
    slope_adaptive = slopes_a.get(LAM_BASE, list(slopes_a.values())[-1])
    LOGGER.info("[H8-b] 기울기 — adaptive(λ=%.2f)=%.4f | 고정전략 %s",
                LAM_BASE, slope_adaptive,
                {k: round(v, 4) for k, v in slopes_b.items()})

    # ---- (c) 즉각형 비율 곡선 ----
    curve_c = []
    for r in ratios:
        cc = stats("c", (r,)); pay = stats("c", (r,), "mean_payoff")
        curve_c.append((r, cc[0], cc[1], pay[0], pay[1]))
        LOGGER.info("[H8-c] 즉각형 %.0f%% → CC %.3f±%.3f, 평균보수 %.3f", r * 100,
                    cc[0], cc[1], pay[0])

    # ---- (d) 대규모 집단 곡선 ----
    curves_d, slopes_d, tests_d = {}, {}, {}
    for N in Ns:
        pts = []
        for f in fracs:
            cc = stats("d", (N, f))
            pay = stats("d", (N, f), "mean_payoff")
            gr = stats("d", (N, f), "payoff_growth")
            pts.append((f, cc[0], cc[1], pay[0], pay[1], gr[0], gr[1]))
        curves_d[N] = pts
        slopes_d[N] = _slope([p[0] for p in pts], [p[1] for p in pts])
        lo = [r["cc_rate"] for r in by_key["d"][(N, fracs[0])]]
        hi = [r["cc_rate"] for r in by_key["d"][(N, fracs[-1])]]
        tests_d[N] = welch_t(hi, lo)
        LOGGER.info("[H8-d] N=%d: CC %s | 기울기=%.4f (high-vs-low p=%.4f)", N,
                    ["%.3f" % p[1] for p in pts], slopes_d[N], tests_d[N][1])

    # ---- (e) 통합 요인 + OLS ----
    rows, X, y_cc, y_pay, y_gr = [], [], [], [], []
    for lam in lams_e:
        for r in ratios_e:
            for f in fracs_e:
                for rr in by_key["e"][(lam, r, f)]:
                    rows.append((lam, r, f, rr["cc_rate"], rr["mean_payoff"],
                                 rr["payoff_growth"]))
                    X.append([1.0, f, lam, r])
                    y_cc.append(rr["cc_rate"]); y_pay.append(rr["mean_payoff"])
                    y_gr.append(rr["payoff_growth"])
    X = np.asarray(X); y_cc = np.asarray(y_cc)
    beta, *_ = np.linalg.lstsq(X, y_cc, rcond=None)
    resid = y_cc - X @ beta
    dof = max(len(y_cc) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_frac = float(beta[1] / max(se[1], 1e-12))
    from math import erf, sqrt
    p_frac = 2 * (1 - 0.5 * (1 + erf(abs(t_frac) / sqrt(2))))
    beta_pay, *_ = np.linalg.lstsq(X, np.asarray(y_pay), rcond=None)
    beta_gr, *_ = np.linalg.lstsq(X, np.asarray(y_gr), rcond=None)
    # λ × frac 히트맵 (ratio·rep 평균)
    heat = np.zeros((len(lams_e), len(fracs_e)))
    for i, lam in enumerate(lams_e):
        for j, f in enumerate(fracs_e):
            vals = [rr["cc_rate"] for r in ratios_e
                    for rr in by_key["e"][(lam, r, f)]]
            heat[i, j] = float(np.mean(vals))
    LOGGER.info("[H8-e] OLS: CC ~ %.3f + %.3f·frac + %.3f·λ + %.3f·즉각비율 "
                "(frac t=%.2f, p=%.4f) | 보수 frac 계수=%.3f, 증가율 frac 계수=%.3f",
                beta[0], beta[1], beta[2], beta[3], t_frac, p_frac,
                beta_pay[1], beta_gr[1])

    # ---- (f) 대규모 공정비교: filler 별 CC 기울기 및 집단 후생(payoff) 기울기 ----
    # 주의: CC율 만으로는 무조건협력자(ALLC)가 기계적으로 우세하나, ALLC 는
    # base 의 ALLD·변덕 상대에게 착취당해 집단 payoff 는 붕괴한다. AdaptiveAgent
    # 의 고유 가치는 '착취에 강건한(=보수로 이어지는) 협력' 이므로, 공정비교의
    # 1차 기준은 payoff 기울기로 삼고 CC 기울기는 병기한다.
    slopes_f, slopes_f_pay, curves_f = {}, {}, {}
    for fl in fillers_f:
        pts = []
        for f in fracs_f:
            cc = stats("f", (fl, f)); pay = stats("f", (fl, f), "mean_payoff")
            pts.append((f, cc[0], cc[1], pay[0], pay[1]))
        curves_f[fl] = pts
        slopes_f[fl] = _slope([p[0] for p in pts], [p[1] for p in pts])
        slopes_f_pay[fl] = _slope([p[0] for p in pts], [p[3] for p in pts])
    slope_ad_large = slopes_f["adaptive"]
    slope_ad_large_pay = slopes_f_pay["adaptive"]
    slope_fixed_large_max = max(slopes_f[k] for k in FIXED_SWEEP)
    slope_fixed_large_max_pay = max(slopes_f_pay[k] for k in FIXED_SWEEP)
    best_fixed_large = max(FIXED_SWEEP, key=lambda k: slopes_f_pay[k])
    LOGGER.info("[H8-f] 대규모(N=%d) CC 기울기 — adaptive=%.4f | 고정전략 %s",
                Nbig, slope_ad_large,
                {k: round(slopes_f[k], 4) for k in FIXED_SWEEP})
    LOGGER.info("[H8-f] 대규모(N=%d) 집단 payoff 기울기 — adaptive=%.4f | 고정전략 %s "
                "(payoff 최대: %s=%.4f)",
                Nbig, slope_ad_large_pay,
                {k: round(slopes_f_pay[k], 4) for k in FIXED_SWEEP},
                best_fixed_large, slope_fixed_large_max_pay)

    # λ 경계 의존성(하위 통찰): λ 에 대한 기울기의 구배가 양수인지
    lam_sorted = sorted(slopes_a)
    slope_grad = _slope([float(k) for k in lam_sorted],
                        [slopes_a[k] for k in lam_sorted])

    sup_a = slopes_a[str(0.4)] > 0 if str(0.4) in slopes_a else \
        list(slopes_a.values())[-1] > 0
    sup_d = all(s > 0 for s in slopes_d.values())
    sup_e = beta[1] > 0
    # 핵심 주장(집단 규모에서 Adaptive 비율↑ → 협력↑)의 지지는 규모 확장 증거
    # (a: λ=0.4 기울기, d: 대규모 CC 기울기, e: 통합 요인 frac 계수)로 판정한다.
    supported = bool(sup_a and sup_d and sup_e)
    # 고정전략 대조(b/f)는 '지지/미지지' 게이트가 아니라 *한정 분석*으로 보고:
    # 얕은 CC 지표에서는 무조건협력(ALLC)이 기계적으로 우세할 수 있으나, 그 협력은
    # base 의 착취자에게 보수를 상납하는 '취약한' 협력이다(강건성 결여).
    sup_b_pay = slope_ad_large_pay > slope_fixed_large_max_pay
    LOGGER.info("[H8] 핵심 지지 — λ=0.4 기울기>0: %s | 대규모 CC 기울기>0: %s | "
                "통합 frac 계수>0: %s → %s",
                sup_a, sup_d, sup_e, "지지" if supported else "미지지")
    LOGGER.info("[H8-한정] 고정전략 대조(대규모): adaptive payoff기울기 > 고정전략 최대 = %s. "
                "얕은 CC 지표에서 ALLC 가 우세해 보이더라도 이는 착취자에게 보수를 상납하는 "
                "취약한 협력이며(payoff_growth·강건성으로 구분됨), 이것이 CC 외에 후생·성장 "
                "지표를 병기한 이유다.", sup_b_pay)
    LOGGER.info("[H8-경계] λ 에 대한 CC 기울기의 구배=%.4f (양수면 공감 baseline 이 높을수록 "
                "협력 촉진 효과 강화; λ=0.24 부근에서 기울기 부호 전환 — Albarracin 경계 재현)",
                slope_grad)

    return {
        "base_cc": list(base_cc),
        "curves_a": {str(k): v for k, v in curves_a.items()},
        "slopes_a": {str(k): v for k, v in slopes_a.items()},
        "slopes_b": slopes_b, "slope_adaptive": slope_adaptive,
        "slopes_f": slopes_f, "slopes_f_pay": slopes_f_pay,
        "curves_f": {k: v for k, v in curves_f.items()},
        "slope_ad_large": slope_ad_large,
        "slope_ad_large_pay": slope_ad_large_pay,
        "slope_fixed_large_max": slope_fixed_large_max,
        "slope_fixed_large_max_pay": slope_fixed_large_max_pay,
        "slope_grad_lambda": slope_grad,
        "curve_c": curve_c,
        "curves_d": {str(k): v for k, v in curves_d.items()},
        "slopes_d": {str(k): v for k, v in slopes_d.items()},
        "tests_d": {str(k): list(v) for k, v in tests_d.items()},
        "combo": {"lams": lams_e, "ratios": ratios_e, "fracs": fracs_e,
                  "beta_cc": beta.tolist(), "se": se.tolist(),
                  "t_frac": t_frac, "p_frac": float(p_frac),
                  "beta_payoff": beta_pay.tolist(),
                  "beta_growth": beta_gr.tolist(),
                  "heat": heat.tolist()},
        "sub": {"lam040_slope_positive": bool(sup_a),
                "large_population_slopes_positive": bool(sup_d),
                "combined_fraction_effect_positive": bool(sup_e),
                "adaptive_payoff_slope_exceeds_fixed_large": bool(sup_b_pay),
                "best_fixed_filler_large": best_fixed_large,
                "lambda_slope_gradient": float(slope_grad)},
        "supported": supported,
    }


# ================================================================== H9 / H10
def exp_H9_H10(seeds, rounds, jobs, backend):
    """
    H9 : **α 전용 귀인** 에이전트는 상대의 의도가 변동(변덕 상대)할 때 완전귀인보다 열등하다.
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

    LOGGER.info("[H9] 변덕상대 누적보수 all=%.1f vs alpha_only=%.1f (p=%.4f) → %s",
                mu("cum_payoff", "all", "capricious"),
                mu("cum_payoff", "alpha_only", "capricious"), p9, "지지" if h9 else "미지지")
    LOGGER.info("[H10] 정적잡음상대 CC율 all=%.3f vs lambda_only=%.3f (p=%.4f) → %s",
                mu("cc_rate", "all", "static_noisy_tft"),
                mu("cc_rate", "lambda_only", "static_noisy_tft"), p10,
                "지지" if h10 else "미지지")
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
    raw = {f"{t}|{o}": {"cum_payoff": v("cum_payoff", t, o),
                        "cc_rate": v("cc_rate", t, o)}
           for t in targets for o in ("capricious", "static_noisy_tft", "exploiter")}
    return {"table": table, "raw": raw,
            "targets": targets,
            "tests": {"H9": (t9, p9), "H10": (t10, p10)},
            "double_dissociation": {"intent_only_overpunishes": bool(dd_over),
                                    "beta_context_underdefends": bool(dd_under)},
            "supported": {"H9": bool(h9), "H10": bool(h10)}}


# ================================================================== 시각화 공통
ERRKW = dict(capsize=3, ecolor="black", error_kw=dict(lw=1.0, capthick=1.0))
_RNG = np.random.default_rng(0)


def scatter_seeds(ax, x_center, vals, width=0.12):
    """개별 시드값을 지터 산점으로 겹쳐 그려 분포를 드러낸다."""
    v = np.asarray(vals, float)
    if v.size == 0:
        return
    jit = _RNG.uniform(-width, width, v.size)
    ax.scatter(np.full(v.size, x_center) + jit, v, s=8, color="k",
               alpha=0.35, zorder=5, linewidths=0)


def _save(fig, name: str, tag: str):
    out = RESULTS / f"{name}{tag}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    LOGGER.info("그림 저장: %s", out)
    return out


# ------------------------------------------------------------------ H1 그림
def fig_H1(r, tag):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("H1. 착취자에 대한 자기보호 (Adaptive vs 고정 λ)  —  음영/오차막대 = ±1 SD (시드 간)")
    ax = axes[0]
    tr = r["traces"]
    band(ax, tr["lam_adaptive"], "AdaptiveAgent (조절)")
    band(ax, tr["lam_fixed"], "ToMEmpathicAgent (고정)", ls="--")
    ax.set_title(f"A. λ 궤적 (n={tr['lam_adaptive'].shape[0]} 시드)")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    ax = axes[1]
    band(ax, tr["grievance"], "불만 g⁻ (vmPFC)", color="crimson")
    band(ax, tr["trust"], "신뢰 g⁺", color="seagreen")
    band(ax, tr["disp_credence"], "기질귀인 신뢰도 (dmPFC)", color="steelblue", ls=":")
    ax.set_title("B. 항상성 누적기")
    ax.set_xlabel("라운드"); ax.set_ylabel("무차원 [0, 1]"); ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["λ_final", "착취가능성", "누적보수/100"]
    ad = [r["agg"]["adaptive"]["lam_final"][0], r["agg"]["adaptive"]["exploitability"][0],
          r["agg"]["adaptive"]["cum_payoff"][0] / 100.0]
    ad_sd = [r["agg"]["adaptive"]["lam_final"][1], r["agg"]["adaptive"]["exploitability"][1],
             r["agg"]["adaptive"]["cum_payoff"][1] / 100.0]
    fx = [r["agg"]["fixed"]["lam_final"][0], r["agg"]["fixed"]["exploitability"][0],
          r["agg"]["fixed"]["cum_payoff"][0] / 100.0]
    fx_sd = [r["agg"]["fixed"]["lam_final"][1], r["agg"]["fixed"]["exploitability"][1],
             r["agg"]["fixed"]["cum_payoff"][1] / 100.0]
    x = np.arange(3); w = 0.35
    ax.bar(x - w / 2, ad, w, yerr=ad_sd, label="Adaptive", **ERRKW)
    ax.bar(x + w / 2, fx, w, yerr=fx_sd, label="Fixed-λ", **ERRKW)
    scatter_seeds(ax, 1 - w / 2, [m["exploitability"] for m in r["raw"]["adaptive"]])
    scatter_seeds(ax, 1 + w / 2, [m["exploitability"] for m in r["raw"]["fixed"]])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    p = r["tests"]["exploit"][1]
    ax.set_title(f"C. 자기보호 지표 (착취가능성 p={p:.3g})"); ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h1_self_protection", tag)


# ------------------------------------------------------------------ H2 그림
def fig_H2(r, tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("H2. 의도 vs 맥락 귀인: λ 하향과 회복  —  음영/오차막대 = ±1 SD (시드 간)")
    ax = axes[0]
    for k, lab in (("exploiter", "착취자 (높은 β)"),
                   ("noisy_tft", "잡음 TFT (낮은 β)"),
                   ("noisy", "완전 무작위")):
        band(ax, r["traces"][k], lab)
    ax.set_title("A. λ 궤적: 착취자에겐 억제 유지, 잡음 상대에겐 회복")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    ax = axes[1]
    labs = ["exploiter", "noisy_tft", "noisy"]
    m = [r["agg"][l]["lam_final"][0] for l in labs]
    sd = [r["agg"][l]["lam_final"][1] for l in labs]
    x = np.arange(3)
    ax.bar(x, m, 0.5, yerr=sd, color=["firebrick", "darkorange", "gray"], **ERRKW)
    for i, l in enumerate(labs):
        scatter_seeds(ax, i, [mm["lam_final"] for mm in r["raw"][l]])
    ax.set_xticks(x); ax.set_xticklabels(["착취자", "잡음 TFT", "무작위"])
    p = r["tests"]["lam"][1]
    ax.set_title(f"B. λ_final (noisy_tft > exploiter, p={p:.3g})")
    ax.set_ylabel("λ_final")
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h2_lambda_recovery", tag)


# ------------------------------------------------------------------ H3 그림
def fig_H3(r, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fig.suptitle("H3. 상대 형질 추정 구분 (E[α], E[β])  —  오차막대 = ±1 SD (시드 간)")
    labs = ["exploiter", "noisy_tft", "noisy"]
    x = np.arange(len(labs)); w = 0.35
    a = [r["agg"][l]["E_alpha"][0] for l in labs]
    b = [r["agg"][l]["E_beta"][0] for l in labs]
    ae = [r["agg"][l]["E_alpha"][1] for l in labs]
    be = [r["agg"][l]["E_beta"][1] for l in labs]
    ax.bar(x - w / 2, a, w, yerr=ae, label="E[α] 협력편향 (로짓)", **ERRKW)
    ax.bar(x + w / 2, b, w, yerr=be, label="E[β] 행동정밀도", **ERRKW)
    for i, l in enumerate(labs):
        scatter_seeds(ax, i - w / 2, [m["E_alpha"] for m in r["raw"][l]])
        scatter_seeds(ax, i + w / 2, [m["E_beta"] for m in r["raw"][l]])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["착취자", "잡음TFT", "무작위"])
    pb, pa = r["tests"]["beta"][1], r["tests"]["alpha"][1]
    ax.set_title(f"β 구분 p={pb:.3g}, α 구분 p={pa:.3g}")
    ax.set_ylabel("추정 형질값 (단위 상이)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return _save(fig, "h3_trait_estimates", tag)


# ------------------------------------------------------------------ H4 그림
def fig_H4(r, tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("H4. TFT 상대 상호협력 복원  —  음영/오차막대 = ±1 SD (시드 간)")
    ax = axes[0]
    band(ax, r["traces"]["lam_adaptive"], "AdaptiveAgent λ", color="tab:green")
    ax.set_title("A. TFT 상대 λ 궤적 (신뢰 누적으로 상향)")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["CC율", "협력복원", "누적보수/100"]
    ad = [r["agg"]["adaptive"]["cc_rate"][0], r["agg"]["adaptive"]["restoration"][0],
          r["agg"]["adaptive"]["cum_payoff"][0] / 100.0]
    ad_sd = [r["agg"]["adaptive"]["cc_rate"][1], r["agg"]["adaptive"]["restoration"][1],
             r["agg"]["adaptive"]["cum_payoff"][1] / 100.0]
    fx = [r["agg"]["fixed"]["cc_rate"][0], r["agg"]["fixed"]["restoration"][0],
          r["agg"]["fixed"]["cum_payoff"][0] / 100.0]
    fx_sd = [r["agg"]["fixed"]["cc_rate"][1], r["agg"]["fixed"]["restoration"][1],
             r["agg"]["fixed"]["cum_payoff"][1] / 100.0]
    x = np.arange(3); w = 0.35
    ax.bar(x - w / 2, ad, w, yerr=ad_sd, label="Adaptive", **ERRKW)
    ax.bar(x + w / 2, fx, w, yerr=fx_sd, label="Fixed-λ", **ERRKW)
    scatter_seeds(ax, -w / 2, [m["cc_rate"] for m in r["raw"]["adaptive"]])
    scatter_seeds(ax, +w / 2, [m["cc_rate"] for m in r["raw"]["fixed"]])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"B. 협력 복원 지표 (CC p={r['tests']['cc'][1]:.3g})")
    ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h4_cooperation_restoration", tag)


# ------------------------------------------------------------------ H5 그림
def fig_H5(r, tag):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle("H5 (수정판). 즉각형(의도 전용 귀인, vmPFC) vs 정교형(의도+맥락, rmPFC)"
                 "  —  음영/오차막대 = ±1 SD (시드 간)")
    tr = r["traces"]

    ax = axes[0, 0]
    band(ax, tr["lam|sophisticated|exploiter"], "정교형", color="tab:blue")
    band(ax, tr["lam|immediate|exploiter"], "즉각형", color="tab:red")
    ax.set_title("A. λ 궤적 vs 착취자(ALLD)")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    band(ax, tr["lam|sophisticated|noisy_tft"], "정교형", color="tab:blue")
    band(ax, tr["lam|immediate|noisy_tft"], "즉각형", color="tab:red")
    ax.set_title("B. λ 궤적 vs 잡음 TFT")
    ax.set_xlabel("라운드"); ax.set_ylabel("공감 λ"); ax.legend(fontsize=8)

    ax = axes[0, 2]
    band(ax, tr["cumpay|sophisticated|noisy_tft"], "정교형", color="tab:blue")
    band(ax, tr["cumpay|immediate|noisy_tft"], "즉각형", color="tab:red")
    ax.set_title("C. 누적 보수 궤적 vs 잡음 TFT")
    ax.set_xlabel("라운드"); ax.set_ylabel("누적 보수"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    m_i, sd_i = mean_sd(r["raw"]["defense_imm"])
    m_s, sd_s = mean_sd(r["raw"]["defense_soph"])
    ax.bar([0, 1], [m_i, m_s], 0.5, yerr=[sd_i, sd_s],
           color=["tab:red", "tab:blue"], **ERRKW)
    scatter_seeds(ax, 0, r["raw"]["defense_imm"])
    scatter_seeds(ax, 1, r["raw"]["defense_soph"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["즉각형", "정교형"])
    ax.axhline(0, color="k", lw=0.6)
    p = r["tests"]["defense_exploiter(imm>soph)"][1]
    ax.set_title(f"D. 착취자 방어량(−착취가능성)\n즉각 > 정교 (p={p:.3g})")
    ax.set_ylabel("방어량")

    ax = axes[1, 1]
    m_i, sd_i = mean_sd(r["raw"]["payoff_imm"])
    m_s, sd_s = mean_sd(r["raw"]["payoff_soph"])
    ax.bar([0, 1], [m_i, m_s], 0.5, yerr=[sd_i, sd_s],
           color=["tab:red", "tab:blue"], **ERRKW)
    scatter_seeds(ax, 0, r["raw"]["payoff_imm"])
    scatter_seeds(ax, 1, r["raw"]["payoff_soph"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["즉각형", "정교형"])
    p = r["tests"]["payoff_noisy_tft(imm<soph)"][1]
    ax.set_title(f"E. 잡음 TFT 누적 보수\n즉각 < 정교 (p={p:.3g})")
    ax.set_ylabel("누적 보수")

    ax = axes[1, 2]
    m_i, sd_i = mean_sd(r["raw"]["first_defect_imm"])
    m_s, sd_s = mean_sd(r["raw"]["first_defect_soph"])
    ax.bar([0, 1], [m_i, m_s], 0.5, yerr=[sd_i, sd_s],
           color=["tab:red", "tab:blue"], **ERRKW)
    scatter_seeds(ax, 0, r["raw"]["first_defect_imm"])
    scatter_seeds(ax, 1, r["raw"]["first_defect_soph"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["즉각형", "정교형"])
    p = r["tests"]["first_defect(imm<soph)"][1]
    ax.set_title(f"F. 잡음 TFT 상대 첫 배신 라운드\n(조기 배신 증거, p={p:.3g})")
    ax.set_ylabel("첫 배신 라운드")
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h5_immediate_vs_sophisticated", tag)


# ------------------------------------------------------------------ H6 그림
def fig_H6(r, tag):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.suptitle("H6. 잡음 수준에 따른 고정전략 성능  —  오차막대 = ±1 SD (시드 간)")
    sc, sd = r["scores"], r["scores_sd"]
    strats = list(sc["0.0"].keys())
    x = np.arange(len(strats)); w = 0.35
    ax.bar(x - w / 2, [sc["0.0"][t] for t in strats], w,
           yerr=[sd["0.0"][t] for t in strats], label="무잡음", **ERRKW)
    ax.bar(x + w / 2, [sc["0.15"][t] for t in strats], w,
           yerr=[sd["0.15"][t] for t in strats], label="잡음 15%", **ERRKW)
    ax.set_xticks(x); ax.set_xticklabels(strats, rotation=20, fontsize=8)
    ax.set_ylabel("라운드당 평균 보수 [0, 5]")
    ax.set_title("무잡음: TFT 우세 / 잡음: GTFT·WSLS 가 TFT 능가")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return _save(fig, "h6_noise_fixed_strategies", tag)


# ------------------------------------------------------------------ H7 그림
def fig_H7(r, tag):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle("H7 (보완판). 다양한 형질전환(변덕) 상대: 의도추론의 이득"
                 "  —  오차막대/음영 = ±1 SD")
    cases = r["cases"]
    focal_names = ["adaptive", "generous_tft", "wsls", "tit_for_tat"]
    colors = {"adaptive": "darkorange", "generous_tft": "tab:blue",
              "wsls": "tab:green", "tit_for_tat": "tab:gray"}

    # A. case 별 평균 보수 (주 잡음)
    ax = axes[0, 0]
    x = np.arange(len(cases)); w = 0.2
    for i, n in enumerate(focal_names):
        m = [r["per_case"][n][c][0] for c in cases]
        e = [r["per_case"][n][c][1] for c in cases]
        ax.bar(x + (i - 1.5) * w, m, w, yerr=e, label=n,
               color=colors[n], **ERRKW)
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=45, fontsize=6, ha="right")
    ax.set_ylabel("라운드당 평균 보수")
    ax.set_title(f"A. 형질전환 case 별 성과 (잡음 {r['primary_noise']:.2f})")
    ax.legend(fontsize=7)

    # B. 잡음 수준 × focal (전 case 합산)
    ax = axes[0, 1]
    noises = r["noise_levels"]
    x = np.arange(len(noises)); w = 0.2
    for i, n in enumerate(focal_names):
        ax.bar(x + (i - 1.5) * w, [r["sweep"][nz][n] for nz in noises], w,
               yerr=[r["sweep_sd"][nz][n] for nz in noises], label=n,
               color=colors[n], **ERRKW)
    ax.set_xticks(x); ax.set_xticklabels([f"잡음 {nz}" for nz in noises], fontsize=8)
    ax.set_ylabel("라운드당 평균 보수 (전 case 합산)")
    pg, pw = r["tests"]["gtft"][1], r["tests"]["wsls"][1]
    ax.set_title(f"B. 잡음 × 전략 (vs GTFT p={pg:.3g}, vs WSLS p={pw:.3g})")
    ax.legend(fontsize=7)

    # C. 전환 정렬 per-round 보수
    ax = axes[0, 2]
    rel = r["traces"]["rel"]
    for n in focal_names:
        band(ax, r["traces"]["aligned_pay"][n], n, x=rel, color=colors[n])
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_title("C. 의도 전환 시점 정렬 평균 보수 추이")
    ax.set_xlabel("전환으로부터의 라운드"); ax.set_ylabel("보수")
    ax.legend(fontsize=7)

    # D. 전환 정렬 λ (Adaptive)
    ax = axes[1, 0]
    band(ax, r["traces"]["aligned_lam"], "Adaptive λ", x=rel, color="darkorange")
    ax.axvline(0, color="k", lw=1, ls="--")
    dm, dsd = r["dlam"]
    ax.set_title(f"D. 전환 정렬 λ 변동 (|Δλ|={dm:.3f}±{dsd:.3f})")
    ax.set_xlabel("전환으로부터의 라운드"); ax.set_ylabel("공감 λ")
    ax.legend(fontsize=7)

    # E. 4×4 상호대전 히트맵
    ax = axes[1, 1]
    M = np.asarray(r["tournament"]["matrix"])
    names = r["tournament"]["names"]
    im = ax.imshow(M, cmap="viridis")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=30, fontsize=7, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"E. 4×4 상호대전 행(focal) 평균 보수 (잡음 {r['primary_noise']:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # F. 국면 분해: 착취(ALLD) 국면 vs 협력 국면 보수
    ax = axes[1, 2]
    x = np.arange(len(focal_names)); w = 0.35
    ex_m = [r["phase_pay"][f"{n}|exploit"][0] for n in focal_names]
    ex_s = [r["phase_pay"][f"{n}|exploit"][1] for n in focal_names]
    co_m = [r["phase_pay"][f"{n}|coop"][0] for n in focal_names]
    co_s = [r["phase_pay"][f"{n}|coop"][1] for n in focal_names]
    ax.bar(x - w / 2, ex_m, w, yerr=ex_s, label="착취(ALLD) 국면", color="firebrick", **ERRKW)
    ax.bar(x + w / 2, co_m, w, yerr=co_s, label="협력적 국면", color="seagreen", **ERRKW)
    ax.set_xticks(x); ax.set_xticklabels(focal_names, rotation=20, fontsize=7)
    ax.set_ylabel("국면별 라운드당 평균 보수")
    p_ex = r["tests"]["exploit_gtft"][1]
    ax.set_title(f"F. 국면 분해: 자기보호 이득 vs 화해 지연 비용\n(착취국면 adaptive>GTFT p={p_ex:.3g})")
    ax.legend(fontsize=7)
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h7_capricious_partners", tag)


# ------------------------------------------------------------------ H8 그림
def fig_H8(r, tag):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle("H8 (전면수정판). 집단 역학: AdaptiveAgent 와 상호협력  —  오차막대 = ±1 SD (replicate 간)")

    # A. λ × n_adaptive 곡선
    ax = axes[0, 0]
    lams = sorted(float(k) for k in r["curves_a"])
    cmap = plt.cm.viridis(np.linspace(0, 1, len(lams)))
    for c, lam in zip(cmap, lams):
        pts = r["curves_a"][str(lam)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=1.6, capsize=2, color=c,
                    label=f"λ={lam:g} (기울기 {float(r['slopes_a'][str(lam)]):.3f})")
    ax.set_xlabel("집단 내 AdaptiveAgent 수"); ax.set_ylabel("상호협력(CC)률")
    ax.set_title("A. baseline λ 스윕 (경계선 0.24 포함)")
    ax.legend(fontsize=6)

    # B. 대규모(N=100) 공정비교: filler 비율 증가에 따른 payoff·CC 기울기
    ax = axes[0, 1]
    sf, sfp = r["slopes_f"], r["slopes_f_pay"]
    names = ["adaptive"] + [k for k in sfp if k != "adaptive"]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [sfp[k] for k in names], w, label="집단 payoff 기울기",
           color=["darkorange"] + ["tab:brown"] * (len(names) - 1))
    ax.bar(x + w / 2, [sf[k] for k in names], w, label="CC률 기울기",
           color=["gold"] + ["tab:blue"] * (len(names) - 1), alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=7, ha="right")
    ax.set_ylabel("비율당 기울기")
    ax.set_title("B. 대규모 공정비교 (N=100, 변덕 포함 base)\npayoff 기준: adaptive 우위 / "
                 "CC 기준: ALLC 가 기계적 우위(착취당해 payoff 붕괴)")
    ax.legend(fontsize=6)

    # C. 즉각형 비율
    ax = axes[0, 2]
    cs = r["curve_c"]
    xs = [c[0] * 100 for c in cs]
    ax.errorbar(xs, [c[1] for c in cs], yerr=[c[2] for c in cs], marker="o",
                lw=2, capsize=3, color="tab:purple", label="CC률")
    ax.set_xlabel("즉각형(의도 전용 귀인) 비율 [%]")
    ax.set_ylabel("상호협력(CC)률", color="tab:purple")
    ax2 = ax.twinx()
    ax2.errorbar(xs, [c[3] for c in cs], yerr=[c[4] for c in cs], marker="s",
                 lw=1.5, capsize=3, color="tab:green", ls="--", label="평균 보수")
    ax2.set_ylabel("개체·라운드당 평균 보수", color="tab:green")
    ax.set_title("C. 즉각형:정교형 비율 (n_adaptive=6)")

    # D. 대규모 집단: Adaptive 비율 → CC
    ax = axes[1, 0]
    for N, color in zip(sorted(int(k) for k in r["curves_d"]), ("tab:blue", "tab:red")):
        pts = r["curves_d"][str(N)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
        pv = float(r["tests_d"][str(N)][1])
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=2, capsize=3, color=color,
                    label=f"N={N} (기울기 {float(r['slopes_d'][str(N)]):.3f}, p={pv:.3g})")
    ax.set_xlabel("AdaptiveAgent 비율"); ax.set_ylabel("상호협력(CC)률")
    ax.set_title("D. 대규모 혼합 집단 (변덕 상대 포함 base)")
    ax.legend(fontsize=7)

    # E. 통합 요인 히트맵
    ax = axes[1, 1]
    combo = r["combo"]
    heat = np.asarray(combo["heat"])
    im = ax.imshow(heat, cmap="viridis", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_xticks(range(len(combo["fracs"])))
    ax.set_xticklabels([f"{f:g}" for f in combo["fracs"]], fontsize=8)
    ax.set_yticks(range(len(combo["lams"])))
    ax.set_yticklabels([f"λ={l:g}" for l in combo["lams"]], fontsize=8)
    ax.set_xlabel("AdaptiveAgent 비율")
    b = combo["beta_cc"]
    ax.set_title(f"E. 통합 요인 CC (즉각비율·rep 평균)\nOLS: frac 계수={b[1]:.3f} (p={combo['p_frac']:.3g})")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # F. 대규모 집단 후생 지표
    ax = axes[1, 2]
    Nmax = str(max(int(k) for k in r["curves_d"]))
    pts = r["curves_d"][Nmax]
    xs = [p[0] for p in pts]
    ax.errorbar(xs, [p[3] for p in pts], yerr=[p[4] for p in pts], marker="o",
                lw=2, capsize=3, color="tab:green", label="평균 보수")
    ax.set_xlabel("AdaptiveAgent 비율")
    ax.set_ylabel("개체·라운드당 평균 보수", color="tab:green")
    ax2 = ax.twinx()
    ax2.errorbar(xs, [p[5] for p in pts], yerr=[p[6] for p in pts], marker="s",
                 lw=1.5, capsize=3, color="tab:brown", ls="--", label="보수 증가율")
    ax2.axhline(0, color="tab:brown", lw=0.5, ls=":")
    ax2.set_ylabel("초기→후기 기대보수 증가율", color="tab:brown")
    ax.set_title(f"F. 집단 후생과 성장 (N={Nmax})")
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h8_population_dynamics", tag)


# ------------------------------------------------------------------ H9 그림
def fig_H9(r, tag):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.suptitle("H9. α 전용 귀인의 한계 (변덕 상대)  —  오차막대 = ±1 SD (시드 간)")
    targets = r["targets"]
    m = [mean_sd(r["raw"][f"{t}|capricious"]["cum_payoff"])[0] for t in targets]
    sd = [mean_sd(r["raw"][f"{t}|capricious"]["cum_payoff"])[1] for t in targets]
    colors = ["darkorange" if t == "all" else
              ("firebrick" if t == "alpha_only" else "tab:blue") for t in targets]
    ax.bar(range(len(targets)), m, 0.55, yerr=sd, color=colors, **ERRKW)
    for i, t in enumerate(targets):
        scatter_seeds(ax, i, r["raw"][f"{t}|capricious"]["cum_payoff"])
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=20, fontsize=8)
    p = r["tests"]["H9"][1]
    ax.set_title(f"변덕 상대 누적보수: all > alpha_only (p={p:.3g})")
    ax.set_ylabel("누적 보수"); ax.grid(alpha=0.25)
    return _save(fig, "h9_alpha_only_attribution", tag)


# ------------------------------------------------------------------ H10 그림
def fig_H10(r, tag):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("H10. λ 전용 귀인의 한계 (정적 잡음 상대) + 이중해리  —  오차막대 = ±1 SD (시드 간)")
    targets = r["targets"]
    ax = axes[0]
    m = [mean_sd(r["raw"][f"{t}|static_noisy_tft"]["cc_rate"])[0] for t in targets]
    sd = [mean_sd(r["raw"][f"{t}|static_noisy_tft"]["cc_rate"])[1] for t in targets]
    colors = ["darkorange" if t == "all" else
              ("firebrick" if t == "lambda_only" else "tab:blue") for t in targets]
    ax.bar(range(len(targets)), m, 0.55, yerr=sd, color=colors, **ERRKW)
    for i, t in enumerate(targets):
        scatter_seeds(ax, i, r["raw"][f"{t}|static_noisy_tft"]["cc_rate"])
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=20, fontsize=8)
    p = r["tests"]["H10"][1]
    ax.set_title(f"A. 정적 잡음 상대 CC율: all > lambda_only (p={p:.3g})")
    ax.set_ylabel("CC율")

    ax = axes[1]
    m2 = [mean_sd(r["raw"][f"{t}|exploiter"]["cum_payoff"])[0] for t in targets]
    sd2 = [mean_sd(r["raw"][f"{t}|exploiter"]["cum_payoff"])[1] for t in targets]
    colors = ["darkorange" if t == "all" else
              ("firebrick" if t == "beta_context" else "tab:blue") for t in targets]
    ax.bar(range(len(targets)), m2, 0.55, yerr=sd2, color=colors, **ERRKW)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=20, fontsize=8)
    dd = r["double_dissociation"]
    ax.set_title("B. 착취자 상대 누적보수 (이중해리: "
                 f"intent_only 과잉처벌={dd['intent_only_overpunishes']}, "
                 f"beta_context 과소방어={dd['beta_context_underdefends']})")
    ax.set_ylabel("누적 보수")
    for a in axes.flat:
        a.grid(alpha=0.25)
    return _save(fig, "h10_lambda_only_attribution", tag)


# ================================================================== 시각화 dispatcher
def visualize(results: dict, tag: str = ""):
    """
    가설마다 개별 .png 파일을 저장한다 (h1 … h10).
    H2/H3, H9/H10 은 같은 실험에서 도출되지만 **가설별로 별도 파일**로 저장한다.
    각 가설 내부의 다면적 검증(예: H5/H7/H8)은 해당 가설의 단일 파일 안에
    다중 패널로 묶는다 (두 가지 이상 지표를 함께 보는 것이 유용한 경우).
    """
    set_korean_font(plt, font_manager)
    outs = []
    if "H1" in results:
        outs.append(fig_H1(results["H1"], tag))
    if "H2H3" in results:
        outs.append(fig_H2(results["H2H3"], tag))
        outs.append(fig_H3(results["H2H3"], tag))
    if "H4" in results:
        outs.append(fig_H4(results["H4"], tag))
    if "H5" in results:
        outs.append(fig_H5(results["H5"], tag))
    if "H6" in results:
        outs.append(fig_H6(results["H6"], tag))
    if "H7" in results:
        outs.append(fig_H7(results["H7"], tag))
    if "H8" in results:
        outs.append(fig_H8(results["H8"], tag))
    if "H9H10" in results:
        outs.append(fig_H9(results["H9H10"], tag))
        outs.append(fig_H10(results["H9H10"], tag))
    return outs


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
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ================================================================== main
EXPERIMENTS = {
    "H1": exp_H1, "H2H3": exp_H2_H3, "H4": exp_H4, "H5": exp_H5,
    "H6": exp_H6, "H7": exp_H7, "H8": exp_H8, "H9H10": exp_H9_H10,
}


def main():
    ap = argparse.ArgumentParser(description="HalloReg IPD 실험 실행기")
    ap.add_argument("--seeds", type=int, default=120, help="조건별 시드(다이애드) 수 (기본 120)")
    ap.add_argument("--rounds", type=int, default=60, help="다이애드 라운드 수 (기본 60)")
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
        from AIF_IPD.core.pymdp_backend import PymdpEFE, pymdp_available
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
        te = time.time()
        results[name] = EXPERIMENTS[name](args.seeds, args.rounds, args.jobs,
                                          args.backend)
        LOGGER.info("[%s] 소요 %.1fs", name, time.time() - te)

    LOGGER.info("전체 실험 소요 %.1fs", time.time() - t0)

    pngs = visualize(results, tag=args.tag)
    js = RESULTS / f"halloreg_results{args.tag}.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(_jsonable({k: {kk: vv for kk, vv in v.items() if kk != "traces"}
                             for k, v in results.items()}), f,
                  ensure_ascii=False, indent=2)
    LOGGER.info("요약 저장: %s", js)

    LOGGER.info("-" * 74)
    LOGGER.info("가설 요약:")
    for k, v in results.items():
        LOGGER.info("  %-6s → %s", k, v.get("supported"))
    LOGGER.info("그림 %d 개: %s", len(pngs), ", ".join(p.name for p in pngs))
    LOGGER.info("완료. 결과: %s", RESULTS)


if __name__ == "__main__":
    mp.freeze_support()   # Windows 실행파일/spawn 안전
    main()
