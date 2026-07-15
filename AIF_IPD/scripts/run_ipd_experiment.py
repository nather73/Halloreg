#!/usr/bin/env python
"""
run_ipd_experiment.py
=====================

HalloReg 메인 엔트리포인트: IPD 시뮬레이션 → 가설검증(H1–H12 + H7H/H8E/GS)
→ 가설별 시각화.

v0.5 (Keystone 프런티어 판) — H12 추가:
  §H12 상주 조건부 Keystone 프런티어 — adaptive 의 '협력 유역 확장' 기여가
        상주 구성에 따라 부호가 뒤집히는 H8E(+0.67)·H11(adaptive<TFT) 상충을,
        고정 최대 유형집합 S*(11종) 위 단일 심플렉스에서 중립 filler(random)
        대치 기반 유역 확장 sub_widening 으로 재정의(원칙 A·B·D)해 인공물
        (차원·배경·척도 교란)을 제거하고, 위협축(R-P ρ_D·R-K ρ_C·R-N err)과
        중복축(R-D 협력자 다양성 m)을 따라 반응 곡면으로 정량화. [확증×3]
        C1 위협 단조성·C2 비대체성 단조성·C3 keystone 프런티어 교차 m*.
        [탐색] Shapley 순서무관 기여·ALLD 침입장벽 심화(random 앵커)·구조적
        사전 일치(원칙 E)·H8E/H11 좌표 정위. 내부 err/T 스윕(H8E 와 동형).
        설계: docs/H12_resident_conditioned_keystone_DESIGN.md.

v0.4 (Keystone 판) — 3축 확장:
  §17.1 GTFT 이중화: 확률론적(generous_tft) vs 횟수 기반(generous_tft_count)
        용서를 분리 구현, GTFT 등장 지점 전체(H6/H7/H7H/H8/H8E/H11/GS)에 병기.
  §17.2 H7 상대이점: GTFT 두 유형 + WSLS 대비 adaptive 의 payoff 우위를
        전환 주기 P × 지평 T 로 분해해 명시 (adaptive_advantage).
  §17.3 H11 Keystone: 치환 설계(ALLD 30% 고정) + fixed-λ 대조로 "adaptive 는
        승자가 아니라 조력자" 검증 — ΔCC·Δgap dose 기울기[확증 2] + ALLD
        침입장벽·클러스터 침입성·유역 확장 진화 프레임.

v0.2 (연구 보완판) — 6축 보완 이행:
  §1 통계  : 순열/부트스트랩 검정, Hedges' g/dz + 95% CI, Holm(확증)·BH-FDR(탐색),
            사전등록(experiments/hypotheses.yaml), 퇴화·절단 검정 교체, CRN 짝지음.
  §2 설계  : H5 요인 분해(2×2×2), H2 β-클램프/intent 절제, H7 잡음 대칭화(환경 계층),
            λ 반응성 순열 영가설, ToM 복원 연구(scripts/validate_tom_recovery.py).
  §3 단순성: H8E 복제자/Moran 역학, 매칭 분산 분해, 구성 민감도, 게임구조(GS) 스윕,
            비-ToM 학습 베이스라인(Q-러너/베이지안 BR/fictitious).
  §4 해석  : λ=0.24 '정성적 일치' 재보정 + 기울기 CI, ALLC 착취 이전(transfer) 회계,
            [확증]/[탐색] 태그 제도화.
  §5 시각화: CI 오차막대(검정 패널) vs 백분위 밴드(궤적), n 표기, 이중축 제거,
            조건부 기울기 플롯, PNG+PDF+캡션 JSON.
  §6 H7H  : 지평 의존성의 체계적 검증(전환수 고정 vs 주기 고정 족, T*, 히스테리시스).

사용 예
-------
    python scripts/run_ipd_experiment.py                       # 전체 (seeds=120, rounds=60)
    python scripts/run_ipd_experiment.py --experiments H5 H7H H8E
    python scripts/run_ipd_experiment.py --quick               # 스모크
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]              # .../Halloreg/AIF_IPD
sys.path.insert(0, str(_PKG_ROOT.parent))  # .../Halloreg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from AIF_IPD.core.constants import (CC, CD, DC, DD, COOP, DEFECT,
                                    my_action_from_state)
from AIF_IPD.core.logging_utils import get_logger, set_korean_font
from AIF_IPD.ipd.env import (
    CAPRICIOUS_CASES, CAPRICIOUS_PERIODS, DEFAULT_CAPRICIOUS_CASE,
    capricious_case_spec, capricious_switch_rounds,
    fixed_capricious_cases, capricious_cases_by_period,
)
from AIF_IPD.ipd.sim import run_many, run_populations
from AIF_IPD.ipd.metrics import (
    defense_metrics, aggregate,
    effect_size, effect_size_paired, perm_test, one_sample_perm,
    holm, bh_fdr, wilson_ci, prop_perm_test, hazard_perm_test,
    boot_ci, boot_mean_ci, ols_boot, slope_boot, fmt_es,
)
from AIF_IPD.ipd import evolution as evo

LOGGER = get_logger("HalloReg.run")
RESULTS = _PKG_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LAM_BASE = 0.4
LAM_MAX = 0.8

# ------------------------------------------------------------- 사전등록 로딩
_PREREG_PATH = _PKG_ROOT / "experiments" / "hypotheses.yaml"


def load_prereg() -> dict:
    """experiments/hypotheses.yaml 사전등록을 읽는다 (PyYAML 없으면 내장 사본)."""
    try:
        import yaml
        with open(_PREREG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        LOGGER.warning("PyYAML 미가용 — 내장 사전등록 사본 사용 (yaml 파일이 원본)")
        return {"alpha_family": 0.05, "fdr_q": 0.05,
                "hypotheses": {"H4": {"margin": 0.95, "floor": 0.30}}}


PREREG = load_prereg()


def prereg_of(h: str) -> dict:
    return (PREREG.get("hypotheses") or {}).get(h, {}) or {}


# ------------------------------------------------- 확증/탐색 결과 레지스트리
PRIMARY: list = []      # [{"hyp", "label", "p", "direction_met", "es", "tag"}]
EXPLORATORY: list = []  # [{"hyp", "label", "p", "tag"}]

# 현재 실행 중인 지평 태그 (main 루프가 설정; "T60"/"T240"/""(전지평 실험))
CUR_TAG: str = ""


def _tagged(label: str) -> str:
    return f"{label} [{CUR_TAG}]" if CUR_TAG else label


def register_primary(hyp: str, label: str, p: float, direction_met: bool,
                     es_str: str = ""):
    PRIMARY.append({"hyp": hyp, "label": _tagged(label), "p": float(p),
                    "direction_met": bool(direction_met), "es": es_str,
                    "tag": CUR_TAG})
    LOGGER.info("[확증][%s] %s — %s, 방향성립=%s (p_raw=%.4g; Holm 은 전 가설 종료 후)",
                hyp, _tagged(label), es_str, direction_met, p)


def register_exploratory(hyp: str, label: str, p: float, note: str = ""):
    EXPLORATORY.append({"hyp": hyp, "label": _tagged(label), "p": float(p),
                        "tag": CUR_TAG})
    if note:
        LOGGER.info("[탐색][%s] %s — %s (p_raw=%.4g; FDR 은 종료 후)",
                    hyp, _tagged(label), note, p)


def stable_seed(*key) -> int:
    """격자 셀 간 시드 독립화 (§1): 셀 키의 CRC32 기반 결정적 시드."""
    return zlib.crc32(repr(key).encode()) % (2 ** 31 - 1)


# ------------------------------------------------------------------ 스펙 헬퍼
def agent_spec(kind: str, seed: int, **kw) -> dict:
    base = dict(type=kind, seed=seed, lam_base=LAM_BASE)
    if kind == "adaptive":
        base.update(lam_max=LAM_MAX)
    base.update(kw)
    return base


def strat_spec(kind: str, seed: int, **kw) -> dict:
    return dict(type="strategy", kind=kind, seed=seed, **kw)


def capricious_spec(seed: int, rounds: int, error: float = 0.10) -> dict:
    """(H9/H10 용) 상호성 → 착취 → 관대한 상호성, 국면 = rounds/3."""
    phases = ("tit_for_tat", "alld", "generous_tft")
    cuts = [int(round(i * rounds / 3)) for i in range(3)]
    return dict(type="strategy", kind="tit_for_tat", seed=seed, error=error,
                schedule=[(c, ph) for c, ph in zip(cuts, phases)])


def stack_traces(results, key: str) -> np.ndarray:
    arrs = [np.asarray(r["agent_log"][key], float) for r in results]
    L = min(len(a) for a in arrs)
    return np.stack([a[:L] for a in arrs])


def mean_sd(vals) -> tuple:
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def _run_block(agent_cfg: dict, opp_kind: str, seeds: int, rounds: int,
               jobs: int, opp_cfg=None, extra_spec: dict | None = None):
    """
    동일 조건 seeds 개 시드 병렬 실행 → (metric dicts, raw results).
    상대 시드(1000+s)·noise_seed(3000+s) 를 조건 간 공유하는 CRN 짝지은 설계 —
    조건 간 비교는 짝지은 순열검정(부호뒤집기)이 기본이다 (§1).
    """
    specs = []
    for s in range(seeds):
        a = dict(agent_cfg); a["seed"] = s
        o = dict(opp_cfg) if opp_cfg else strat_spec(opp_kind, seed=1000 + s)
        o["seed"] = 1000 + s
        sp = {"agent": a, "opponent": o, "noise_seed": 3000 + s}
        if extra_spec:
            sp.update(extra_spec)
        specs.append(sp)
    res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
    mets = [defense_metrics(r["agent_log"], r["hist"], LAM_BASE) for r in res]
    return mets, res


def paired_stats(a, b, direction: str = "two-sided", seed: int = 0) -> dict:
    """짝지은(시드 정렬) 비교의 표준 패키지: 부호뒤집기 순열 p + dz 효과크기."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pt = perm_test(a, b, paired=True, seed=seed, alternative=direction)
    es = effect_size_paired(a - b, seed=seed + 1)
    return {"p": pt["p"], "stat": pt["stat"], "es": es,
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


# ================================================================== H1
def exp_H1(seeds, rounds, jobs, backend):
    """
    H1: AdaptiveAgent 는 착취자에게 자기보호하나 고정-λ 는 못 한다.
    [확증] 착취가능성 (짝지은 순열, adaptive < fixed)
    [탐색] λ_final < λ_base 단일표본 순열 (퇴화 두표본 검정 폐기, §1),
           누적보수 차이.
    """
    LOGGER.info("[H1] 착취자에 대한 자기보호")
    adaptive = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                          attribution_target="all", use_pymdp=backend == "pymdp")
    fixed = agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp")
    m_ad, r_ad = _run_block(adaptive, "exploiter", seeds, rounds, jobs)
    m_fx, r_fx = _run_block(fixed, "exploiter", seeds, rounds, jobs)

    expl_ad = [m["exploitability"] for m in m_ad]
    expl_fx = [m["exploitability"] for m in m_fx]
    pri = paired_stats(expl_ad, expl_fx, direction="less", seed=11)
    register_primary("H1", "착취가능성 adaptive<fixed", pri["p"],
                     pri["mean_a"] < pri["mean_b"], fmt_es(pri["es"], "dz"))

    lam_ad = [m["lam_final"] for m in m_ad]
    t_lam = one_sample_perm(lam_ad, mu0=LAM_BASE, alternative="less", seed=12)
    register_exploratory("H1", "λ_final < λ_base (단일표본)", t_lam["p"],
                         f"λ_final={np.mean(lam_ad):.3f} vs λ_base={LAM_BASE}")
    pay = paired_stats([m["cum_payoff"] for m in m_ad],
                       [m["cum_payoff"] for m in m_fx], "greater", seed=13)
    register_exploratory("H1", "누적보수 adaptive>fixed", pay["p"],
                         fmt_es(pay["es"], "dz"))

    supported = bool(pri["mean_a"] < pri["mean_b"])
    traces = {"lam_adaptive": stack_traces(r_ad, "lam"),
              "lam_fixed": stack_traces(r_fx, "lam"),
              "grievance": stack_traces(r_ad, "grievance"),
              "trust": stack_traces(r_ad, "trust"),
              "disp_credence": stack_traces(r_ad, "disp_credence")}
    return {"agg": {"adaptive": aggregate(m_ad), "fixed": aggregate(m_fx)},
            "primary": pri, "lam_one_sample": t_lam, "payoff": pay,
            "raw": {"expl_ad": expl_ad, "expl_fx": expl_fx, "lam_ad": lam_ad,
                    "pay_ad": [m["cum_payoff"] for m in m_ad],
                    "pay_fx": [m["cum_payoff"] for m in m_fx]},
            "supported": supported, "traces": traces}


# ================================================================== H2 / H3
def exp_H2_H3(seeds, rounds, jobs, backend):
    """
    H2 [확증] λ_final(noisy_tft) > λ_final(exploiter) — 짝지은 순열.
    H2 [탐색] 기제 절제 (§2): intent_only(맥락 귀인 불가)·beta_clamp(β 축 동결)
              에서 noisy_tft 의 λ 회복이 감쇠하는가 → 회복 차이를 '강화'가 아닌
              'β-귀인' 에 귀속.
    H3 [확증] E[β](exploiter) > E[β](noisy_tft). 복원 연구는 별도 스크립트.
    """
    LOGGER.info("[H2/H3] 의도 vs 맥락 귀인 + β-경로 절제")
    arms = {
        "all": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                          use_pymdp=backend == "pymdp"),
        "intent_only": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                                  attribution_target="intent_only",
                                  use_pymdp=backend == "pymdp"),
        "beta_clamp": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                                 beta_clamp=True, use_pymdp=backend == "pymdp"),
    }
    opps = ("exploiter", "noisy_tft", "noisy")
    M, R, traces = {}, {}, {}
    for arm, cfg in arms.items():
        for opp in opps if arm == "all" else ("noisy_tft", "exploiter"):
            m, r = _run_block(cfg, opp, seeds, rounds, jobs)
            M[(arm, opp)] = m; R[(arm, opp)] = r
    for opp in opps:
        traces[opp] = stack_traces(R[("all", opp)], "lam")
    traces["ablation_intent"] = stack_traces(R[("intent_only", "noisy_tft")], "lam")
    traces["ablation_clamp"] = stack_traces(R[("beta_clamp", "noisy_tft")], "lam")

    def v(arm, opp, key="lam_final"):
        return [m[key] for m in M[(arm, opp)]]

    h2 = paired_stats(v("all", "noisy_tft"), v("all", "exploiter"),
                      "greater", seed=21)
    register_primary("H2", "λ_final noisy_tft>exploiter", h2["p"],
                     h2["mean_a"] > h2["mean_b"], fmt_es(h2["es"], "dz"))
    # 절제: 회복량 = λ_final(noisy_tft) − λ_final(exploiter), arm 별
    rec = {arm: np.array(v(arm, "noisy_tft")) - np.array(v(arm, "exploiter"))
           for arm in arms}
    abl_i = paired_stats(rec["all"], rec["intent_only"], "greater", seed=22)
    abl_c = paired_stats(rec["all"], rec["beta_clamp"], "greater", seed=23)
    register_exploratory("H2", "회복량 all>intent_only (절제)", abl_i["p"],
                         fmt_es(abl_i["es"], "dz"))
    register_exploratory("H2", "회복량 all>beta_clamp (절제)", abl_c["p"],
                         fmt_es(abl_c["es"], "dz"))
    mech_attributed = (abl_i["mean_a"] > abl_i["mean_b"]) and \
        (abl_c["mean_a"] > abl_c["mean_b"])
    LOGGER.info("[H2-절제] 회복량 all=%.3f intent_only=%.3f beta_clamp=%.3f → "
                "β-귀인 경로 귀속: %s", rec["all"].mean(),
                rec["intent_only"].mean(), rec["beta_clamp"].mean(),
                mech_attributed)

    h3 = paired_stats(v("all", "exploiter", "E_beta"),
                      v("all", "noisy_tft", "E_beta"), "greater", seed=24)
    register_primary("H3", "E[β] exploiter>noisy_tft", h3["p"],
                     h3["mean_a"] > h3["mean_b"], fmt_es(h3["es"], "dz"))
    a3 = paired_stats(v("all", "noisy_tft", "E_alpha"),
                      v("all", "exploiter", "E_alpha"), "greater", seed=25)
    register_exploratory("H3", "E[α] noisy_tft>exploiter", a3["p"],
                         fmt_es(a3["es"], "dz"))
    LOGGER.info("[H3] 복원(recovery: 편향·RMSE·커버리지)은 "
                "scripts/validate_tom_recovery.py 의 별도 연구로 정량화 — "
                "본 검정은 판별(discrimination)만 주장한다 (§2)")

    agg = {opp: aggregate(M[("all", opp)]) for opp in opps}
    return {"agg": agg,
            "raw": {f"{a}|{o}": {"lam_final": v(a, o),
                                 "E_beta": v(a, o, "E_beta"),
                                 "E_alpha": v(a, o, "E_alpha")}
                    for (a, o) in M},
            "primary_H2": h2, "primary_H3": h3,
            "ablation": {"intent_only": abl_i, "beta_clamp": abl_c,
                         "recovery_means": {k: float(x.mean()) for k, x in rec.items()},
                         "mechanism_attributed": bool(mech_attributed)},
            "alpha_test": a3,
            "supported": {"H2": bool(h2["mean_a"] > h2["mean_b"]),
                          "H3": bool(h3["mean_a"] > h3["mean_b"])},
            "traces": traces}


# ================================================================== H4
def exp_H4(seeds, rounds, jobs, backend):
    """H4 [확증] CC율: adaptive ≥ margin×fixed ∧ ≥ floor (사전등록 상수)."""
    LOGGER.info("[H4] TFT 상대 상호협력 복원")
    pre = prereg_of("H4")
    margin = float(pre.get("margin", 0.95)); floor = float(pre.get("floor", 0.30))
    adaptive = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                          use_pymdp=backend == "pymdp")
    fixed = agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp")
    m_ad, r_ad = _run_block(adaptive, "tit_for_tat", seeds, rounds, jobs)
    m_fx, _ = _run_block(fixed, "tit_for_tat", seeds, rounds, jobs)
    cc_ad = [m["cc_rate"] for m in m_ad]; cc_fx = [m["cc_rate"] for m in m_fx]
    # 비열등성: (cc_ad − margin·cc_fx) 의 단측 짝지은 순열 (≥ 0)
    d = np.array(cc_ad) - margin * np.array(cc_fx)
    ni = one_sample_perm(d, mu0=0.0, alternative="greater", seed=41)
    ok = (np.mean(cc_ad) >= margin * np.mean(cc_fx)) and (np.mean(cc_ad) > floor)
    es = effect_size_paired(d, seed=42)
    register_primary("H4", f"CC 비열등성(margin={margin}, floor={floor})",
                     ni["p"], ok, fmt_es(es, "dz"))
    return {"agg": {"adaptive": aggregate(m_ad), "fixed": aggregate(m_fx)},
            "primary": {"p": ni["p"], "es": es, "margin": margin, "floor": floor},
            "raw": {"cc_ad": cc_ad, "cc_fx": cc_fx},
            "supported": bool(ok),
            "traces": {"lam_adaptive": stack_traces(r_ad, "lam")}}


# ================================================================== H5
def exp_H5(seeds, rounds, jobs, backend):
    """
    H5 [확증] 결합가설 (짝지은 순열 ×2):
        (i) 착취자 방어량: 즉각 > 정교, (ii) noisy TFT 누적보수: 즉각 < 정교.
    [탐색] 2×2×2 요인 분해 (sophisticated × attribution × dd_charge, §2) —
        군집(시드) 부트스트랩 회귀로 주효과·상호작용 CI.
        첫 배신: 절단 대응 — T 내 배신 비율(Wilson)+비율 순열, 위험곡선 순열.
    주의: 즉각형 정의(DD-충전)는 같은 리비전에서 도입된 조작 확인(manipulation
    check)이며, 창발적 예측(조기 배신·보복 나선)과 구분해 보고한다.
    """
    LOGGER.info("[H5] 즉각형(vmPFC) vs 정교형(rmPFC) — 확증 대비 + 요인 분해")
    kw = dict(kappa=0.9, use_pymdp=backend == "pymdp")

    def variant(soph: bool, intent: bool, dd: bool):
        return agent_spec("adaptive", 0, sophisticated=soph,
                          attribution_target="intent_only" if intent else "all",
                          dd_charges_grievance=dd, **kw)

    # ---- 확증 대비: 즉각형(F,T,T) vs 정교형(T,F,F) ----
    soph_cfg = variant(True, False, False)
    imm_cfg = variant(False, True, True)
    mets, raws = {}, {}
    for lab, cfg in (("sophisticated", soph_cfg), ("immediate", imm_cfg)):
        for opp in ("exploiter", "noisy_tft"):
            m, r = _run_block(cfg, opp, seeds, rounds, jobs)
            mets[(lab, opp)] = m; raws[(lab, opp)] = r

    def vals(key, lab, opp):
        return [m[key] for m in mets[(lab, opp)]]

    def_i = [-v for v in vals("exploitability", "immediate", "exploiter")]
    def_s = [-v for v in vals("exploitability", "sophisticated", "exploiter")]
    pay_i = vals("cum_payoff", "immediate", "noisy_tft")
    pay_s = vals("cum_payoff", "sophisticated", "noisy_tft")
    t_def = paired_stats(def_i, def_s, "greater", seed=51)
    t_pay = paired_stats(pay_i, pay_s, "less", seed=52)
    h5_i = t_def["mean_a"] > t_def["mean_b"]
    h5_ii = t_pay["mean_a"] < t_pay["mean_b"]
    # 결합가설 p: 두 단측 p 의 최대 (교차 기각역; 보수적)
    p_joint = max(t_def["p"], t_pay["p"])
    register_primary("H5", "결합: 방어(즉각>정교) ∧ 보수(즉각<정교)", p_joint,
                     bool(h5_i and h5_ii),
                     f"방어 {fmt_es(t_def['es'], 'dz')} | 보수 {fmt_es(t_pay['es'], 'dz')}")

    # ---- 절단 대응 첫 배신 (§1): T 내 배신 비율 + 위험곡선 ----
    T = rounds

    def fd_data(lab):
        fds = vals("first_defect_round", lab, "noisy_tft")
        times = np.minimum(np.asarray(fds, int), T)
        cens = np.asarray(fds) >= T
        return times, cens

    ti, ci_ = fd_data("immediate"); ts, cs = fd_data("sophisticated")
    prop = prop_perm_test(int(np.sum(~ci_)), len(ti), int(np.sum(~cs)), len(ts),
                          seed=53, alternative="greater")
    hz = hazard_perm_test(ti, ci_, ts, cs, horizon=T, seed=54)
    register_exploratory("H5", "T내 배신비율 즉각>정교 (절단 대응)", prop["p"],
                         f"{prop['p1']['p']:.2f} vs {prop['p2']['p']:.2f}")
    register_exploratory("H5", "첫 배신 위험곡선 차이 (로그랭크형)", hz["p"])

    # ---- [탐색] 2×2×2 요인 분해 ----
    fac_specs, fac_key = [], []
    for soph in (True, False):
        for intent in (False, True):
            for dd in (False, True):
                for opp in ("exploiter", "noisy_tft"):
                    fac_key.append((soph, intent, dd, opp))
                    fac_specs.append((variant(soph, intent, dd), opp))
    fac_out = {}
    for (key, (cfg, opp)) in zip(fac_key, fac_specs):
        m, _ = _run_block(cfg, opp, seeds, rounds, jobs)
        fac_out[key] = m

    def factorial_reg(opp: str, outcome: str, sign: float = 1.0):
        rows_X, rows_y, cl = [], [], []
        for soph in (True, False):
            for intent in (False, True):
                for dd in (False, True):
                    m = fac_out[(soph, intent, dd, opp)]
                    for sd_i, mm in enumerate(m):
                        s_c = 1.0 if soph else -1.0
                        i_c = 1.0 if intent else -1.0
                        d_c = 1.0 if dd else -1.0
                        rows_X.append([1, s_c, i_c, d_c,
                                       s_c * i_c, s_c * d_c, i_c * d_c])
                        rows_y.append(sign * mm[outcome])
                        cl.append(sd_i)
        return ols_boot(np.array(rows_X), np.array(rows_y),
                        cluster=np.array(cl), n_boot=1500, seed=55,
                        names=["const", "soph", "intent", "dd",
                               "soph×intent", "soph×dd", "intent×dd"])

    fx_def = factorial_reg("exploiter", "exploitability", sign=-1.0)  # 방어량
    fx_pay = factorial_reg("noisy_tft", "cum_payoff")
    for nm, b, ci in zip(fx_def["names"][1:], fx_def["beta"][1:], fx_def["ci"][1:]):
        LOGGER.info("[탐색][H5-요인|방어량] %s: %.3f [%.3f, %.3f]", nm, b, *ci)
    for nm, b, ci in zip(fx_pay["names"][1:], fx_pay["beta"][1:], fx_pay["ci"][1:]):
        LOGGER.info("[탐색][H5-요인|noisyTFT보수] %s: %.2f [%.2f, %.2f]", nm, b, *ci)
    LOGGER.info("[H5-주의] DD-충전 규칙은 같은 리비전 도입 — 본 검정 중 해당 경로는 "
                "조작 확인(manipulation check)이며, 조기 배신·위험곡선이 창발적 예측이다")

    traces = {}
    for l in ("sophisticated", "immediate"):
        for o in ("exploiter", "noisy_tft"):
            traces[f"lam|{l}|{o}"] = stack_traces(raws[(l, o)], "lam")
        traces[f"cumpay|{l}|noisy_tft"] = np.stack(
            [np.cumsum(r["hist"]["my_payoff"]) for r in raws[(l, "noisy_tft")]])

    fx_def.pop("boots", None); fx_pay.pop("boots", None)
    return {"tests": {"defense": t_def, "payoff": t_pay,
                      "prop_defect": prop, "hazard": hz},
            "factorial": {"defense": fx_def, "payoff": fx_pay},
            "raw": {"defense_imm": def_i, "defense_soph": def_s,
                    "payoff_imm": pay_i, "payoff_soph": pay_s,
                    "fd_imm": [int(x) for x in ti], "fd_cens_imm": ci_.tolist(),
                    "fd_soph": [int(x) for x in ts], "fd_cens_soph": cs.tolist()},
            "sub": {"h5_i": bool(h5_i), "h5_ii": bool(h5_ii)},
            "supported": bool(h5_i and h5_ii), "traces": traces}


# ================================================================== H6
def exp_H6(seeds, rounds, jobs, backend):
    """H6 [확증] 잡음 하 GTFT∨WSLS > TFT (시드 짝지은 라운드로빈 평균)."""
    LOGGER.info("[H6] 잡음 수준에 따른 고정전략 성능 (GTFT 확률/횟수 두 유형 병기)")
    from AIF_IPD.ipd.sim import run_dyad
    from AIF_IPD.ipd.env import make_opponent
    # generous_tft = 확률론적 용서, generous_tft_count = 횟수 기반 용서 (v0.4)
    strategies = ["tit_for_tat", "generous_tft", "generous_tft_count",
                  "wsls", "allc", "alld"]
    n_seed = max(4, seeds // 3)
    per_seed = {(nz, s): [] for nz in (0.0, 0.15) for s in strategies}
    for nz in (0.0, 0.15):
        for s in strategies:
            for sd in range(n_seed):
                pays = []
                for opp in strategies:
                    a = make_opponent(s, seed=sd, error=nz)
                    b = make_opponent(opp, seed=500 + sd, error=nz)
                    h = run_dyad(a, b, rounds)
                    pays.append(float(np.mean(h["my_payoff"])))
                per_seed[(nz, s)].append(float(np.mean(pays)))
    g_t = paired_stats(per_seed[(0.15, "generous_tft")],
                       per_seed[(0.15, "tit_for_tat")], "greater", seed=61)
    gc_t = paired_stats(per_seed[(0.15, "generous_tft_count")],
                        per_seed[(0.15, "tit_for_tat")], "greater", seed=610)
    w_t = paired_stats(per_seed[(0.15, "wsls")],
                       per_seed[(0.15, "tit_for_tat")], "greater", seed=62)
    p_pri = min(g_t["p"], w_t["p"]) * 2   # OR 결합(Bonferroni ×2, 보수적)
    ok = (g_t["mean_a"] > g_t["mean_b"]) or (w_t["mean_a"] > w_t["mean_b"])
    register_primary("H6", "잡음 하 GTFT(확률)∨WSLS>TFT", min(p_pri, 1.0), ok,
                     f"GTFT_prob {fmt_es(g_t['es'], 'dz')} | WSLS {fmt_es(w_t['es'], 'dz')}")
    # 횟수 기반 GTFT 는 탐색 지표로 병기 (확률론적 GTFT 와 직접 비교)
    register_exploratory("H6", "잡음 하 GTFT(횟수)>TFT", gc_t["p"],
                         f"{fmt_es(gc_t['es'], 'dz')} "
                         f"(Δ={gc_t['mean_a'] - gc_t['mean_b']:+.3f})")
    gcvsg = paired_stats(per_seed[(0.15, "generous_tft_count")],
                         per_seed[(0.15, "generous_tft")], "two-sided", seed=611)
    register_exploratory("H6", "GTFT 횟수 vs 확률 용서 (잡음0.15)", gcvsg["p"],
                         f"{fmt_es(gcvsg['es'], 'dz')} "
                         f"(횟수−확률={gcvsg['mean_a'] - gcvsg['mean_b']:+.3f})")
    scores = {str(nz): {s: mean_sd(per_seed[(nz, s)])[0] for s in strategies}
              for nz in (0.0, 0.15)}
    cis = {str(nz): {s: boot_mean_ci(per_seed[(nz, s)], seed=63)["ci"]
                     for s in strategies} for nz in (0.0, 0.15)}
    return {"scores": scores, "cis": cis, "n_seed": n_seed,
            "tests": {"gtft": g_t, "wsls": w_t},
            "raw": {f"{nz}|{s}": per_seed[(nz, s)]
                    for nz in (0.0, 0.15) for s in strategies},
            "supported": bool(ok)}


# ================================================================== H7
def exp_H7(seeds, rounds, jobs, backend):
    """
    H7 (v0.3 전면 확장) — 형질전환 case 격자: 주기 {10,30,60,120} × 순환족 8
    (고정전략 순환 6 + **AIF 국면 포함 이질 순환 2** — SwitchingAgent).

    [확증] 전 case 합산 라운드당 보수: adaptive > GTFT (짝지은 순열; 지평별).
    [탐색] (i) **주기 의존성**: 주기 P 별 Δ(P)=adaptive−GTFT (지지 부호가
           주기에 따라 상이하면 명시·시각화 — 본 리비전의 핵심 추가),
           (ii) 순환족 분해: AIF 국면 포함 case vs 고정전략 순환 case,
           (iii) λ 반응성(무작위 정렬 순열), (iv) 국면 분해,
           (v) 4×4 대전 (환경 계층 대칭 잡음), (vi) 비-ToM 베이스라인.

    잡음 설계 (v0.3): primary 잡음 0.10 에서 **전 focal × 전 case** 격자를
    실행하고, 잡음 스윕 {0.0, 0.20} 은 주 대비쌍(adaptive, GTFT)에 한정한다
    (32 case × 240 seeds 격자 확장에 따른 설계상 명시적 절충).

    퇴화 case 주의: 지평 T 가 주기 P 이하이면(예: T=60 의 p60/p120) 지평 내
    전환이 0회 — '무전환 대조'로 유지되며 전환 정렬·주기 반응 분석에서
    자동 제외되고, Δ(P) 해석에 전환 횟수를 병기한다.
    """
    LOGGER.info("[H7] 형질전환 case 격자 %d개 (주기 %s × 순환족) + 베이스라인",
                len(CAPRICIOUS_CASES), list(CAPRICIOUS_PERIODS))
    quick = seeds < 10
    if quick:
        cases = ["p10_recip_expl_recon", "p10_recip_expl_reconC",
                 "p30_flip", "p10_adaptive_expl"]
        noise_levels = (0.0, 0.10)
    else:
        cases = list(CAPRICIOUS_CASES)
        noise_levels = (0.0, 0.10, 0.20)
    primary_noise = 0.10
    # GTFT 두 유형 + WSLS 를 모두 스윕 대상에 포함 (v0.4 요구)
    sweep_focals = ("adaptive", "generous_tft", "generous_tft_count", "wsls")
    focals = {
        "adaptive": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                               use_pymdp=backend == "pymdp"),
        "generous_tft": strat_spec("generous_tft", 0),        # 확률론적 용서
        "generous_tft_count": strat_spec("generous_tft_count", 0),  # 횟수 기반
        "wsls": strat_spec("wsls", 0),
        "tit_for_tat": strat_spec("tit_for_tat", 0),
        "qlearner": dict(type="qlearner", seed=0),
        "bayes_br": dict(type="bayes_br", seed=0),
    }
    # adaptive 의 상대이점을 겨냥하는 주 비교군 (GTFT 두 유형 + WSLS)
    rival_focals = ("generous_tft", "generous_tft_count", "wsls")
    specs, registry = [], {}
    for err in noise_levels:
        focal_set = focals if err == primary_noise else \
            {k: focals[k] for k in sweep_focals}
        for name, cfg in focal_set.items():
            for case in cases:
                for sd in range(seeds):
                    a = dict(cfg); a["seed"] = sd
                    o = capricious_case_spec(case, seed=700 + sd,
                                             n_rounds=rounds, error=err)
                    registry[(err, name, case, sd)] = len(specs)
                    specs.append({"agent": a, "opponent": o,
                                  "noise_seed": 7000 + sd})
    LOGGER.info("[H7] 다이애드 %d 개 실행 (primary 잡음 전 focal, 스윕 잡음 %s 한정)",
                len(specs), sweep_focals)
    res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)

    def R(err, name, case, sd):
        return res[registry[(err, name, case, sd)]]

    def has(err, name):
        return (err, name, cases[0], 0) in registry

    raw = {(err, n, c): [float(np.mean(R(err, n, c, sd)["hist"]["my_payoff"]))
                         for sd in range(seeds)]
           for err in noise_levels for n in focals for c in cases
           if has(err, n)}
    agg_seed = {n: [float(np.mean([raw[(primary_noise, n, c)][sd] for c in cases]))
                    for sd in range(seeds)] for n in focals}
    per_case = {n: {c: (mean_sd(raw[(primary_noise, n, c)])[0],
                        boot_mean_ci(raw[(primary_noise, n, c)], seed=71)["ci"])
                    for c in cases} for n in focals}
    sweep = {str(e): {n: float(np.mean([np.mean(raw[(e, n, c)]) for c in cases]))
                      for n in focals if has(e, n)} for e in noise_levels}
    sweep_ci = {str(e): {n: boot_mean_ci(
        [float(np.mean([raw[(e, n, c)][sd] for c in cases]))
         for sd in range(seeds)], seed=72)["ci"]
        for n in focals if has(e, n)} for e in noise_levels}

    pri = paired_stats(agg_seed["adaptive"], agg_seed["generous_tft"],
                       "greater", seed=73)
    register_primary("H7", "전 case 합산 보수 adaptive>GTFT(확률)", pri["p"],
                     pri["mean_a"] > pri["mean_b"], fmt_es(pri["es"], "dz"))
    for base in ("generous_tft_count", "wsls", "tit_for_tat",
                 "qlearner", "bayes_br"):
        t = paired_stats(agg_seed["adaptive"], agg_seed[base], "two-sided",
                         seed=74)
        register_exploratory("H7", f"adaptive vs {base}", t["p"],
                             f"{fmt_es(t['es'], 'dz')} "
                             f"(Δ={t['mean_a'] - t['mean_b']:+.3f})")

    # ---- (신규 v0.4) adaptive 의 상대이점: 주기 × 지평별 payoff 우위 명시 ----
    # 각 rival(GTFT 확률/횟수, WSLS)에 대해, 전환 주기 P 별로 adaptive 가 라운드당
    # 보수를 더 획득하는지(짝지은 순열)와 전 case 합산 우위를 함께 보고한다.
    # '어느 전환 간격·어느 지평에서 adaptive 가 더 많은 payoff 를 얻는가' 를 명시.
    adaptive_advantage = {}
    for rival in rival_focals:
        overall = paired_stats(agg_seed["adaptive"], agg_seed[rival],
                               "greater", seed=740)
        by_period = {}
        for P in sorted({CAPRICIOUS_CASES[c]["period"] for c in cases}):
            cs = [c for c in cases if CAPRICIOUS_CASES[c]["period"] == P]
            d_seed = [float(np.mean([raw[(primary_noise, "adaptive", c)][sd]
                                     - raw[(primary_noise, rival, c)][sd]
                                     for c in cs])) for sd in range(seeds)]
            tt = one_sample_perm(d_seed, mu0=0.0, alternative="two-sided",
                                 seed=741 + P)
            ci = boot_mean_ci(d_seed, seed=742 + P)["ci"]
            n_sw = len(capricious_switch_rounds(cs[0], rounds))
            by_period[P] = {
                "delta_mean": float(np.mean(d_seed)), "ci": ci, "p": tt["p"],
                "switches_in_horizon": n_sw,
                "adaptive_wins": bool(np.mean(d_seed) > 0),
                "significant": bool(ci[0] > 0 or ci[1] < 0),
            }
        win_periods = [P for P, v in by_period.items() if v["adaptive_wins"]]
        adaptive_advantage[rival] = {
            "overall": {"delta": float(overall["mean_a"] - overall["mean_b"]),
                        "p": overall["p"], "es": fmt_es(overall["es"], "dz"),
                        "adaptive_wins": bool(overall["mean_a"] > overall["mean_b"])},
            "by_period": by_period,
            "win_periods": win_periods,
        }
        register_exploratory(
            "H7", f"adaptive 상대이점 vs {rival} (전 case)", overall["p"],
            f"Δ={overall['mean_a'] - overall['mean_b']:+.3f} "
            f"{fmt_es(overall['es'], 'dz')}; adaptive 우위 주기={win_periods or '없음'} "
            f"(지평 T={rounds})")
        LOGGER.info("[탐색][H7] adaptive vs %s @ T=%d: 전 case Δ=%+.3f; "
                    "주기별 우위 %s", rival, rounds,
                    overall["mean_a"] - overall["mean_b"],
                    {P: f"{v['delta_mean']:+.3f}" for P, v in by_period.items()})

    # ---- (신규) 주기 의존성: 주기별 Δ(P) = adaptive − GTFT ----
    periods_here = sorted({CAPRICIOUS_CASES[c]["period"] for c in cases})
    period_stats = {}
    for P in periods_here:
        cs = [c for c in cases if CAPRICIOUS_CASES[c]["period"] == P]
        d_seed = [float(np.mean([raw[(primary_noise, "adaptive", c)][sd]
                                 - raw[(primary_noise, "generous_tft", c)][sd]
                                 for c in cs])) for sd in range(seeds)]
        t = one_sample_perm(d_seed, mu0=0.0, alternative="two-sided",
                            seed=750 + P)
        ci = boot_mean_ci(d_seed, seed=751 + P)["ci"]
        n_sw = len(capricious_switch_rounds(cs[0], rounds))
        period_stats[P] = {
            "delta_mean": float(np.mean(d_seed)), "ci": ci, "p": t["p"],
            "n_cases": len(cs), "switches_in_horizon": n_sw,
            "supported": bool(np.mean(d_seed) > 0),
        }
        register_exploratory(
            "H7", f"주기 P={P} Δ(adaptive−GTFT)", t["p"],
            f"Δ={np.mean(d_seed):+.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
            f"(지평 내 전환 {n_sw}회{'; 무전환 대조' if n_sw == 0 else ''})")
    signs = {P: period_stats[P]["supported"] for P in periods_here}
    period_dependent = len(set(signs.values())) > 1
    if period_dependent:
        LOGGER.info("[H7] ⚠ 지지 여부가 전환 주기에 의존: %s — 그림·해석에 명시",
                    signs)
    else:
        LOGGER.info("[H7] 주기별 Δ 부호 일관: %s", signs)

    # ---- (신규) 순환족 분해: AIF 국면 포함 case vs 고정전략 순환 case ----
    fixed_cs = [c for c in cases if "adaptive" not in CAPRICIOUS_CASES[c]["cycle"]]
    aif_cs = [c for c in cases if "adaptive" in CAPRICIOUS_CASES[c]["cycle"]]
    family_split = {}
    for lbl, cs in (("fixed_cycles", fixed_cs), ("aif_cycles", aif_cs)):
        if not cs:
            continue
        d_seed = [float(np.mean([raw[(primary_noise, "adaptive", c)][sd]
                                 - raw[(primary_noise, "generous_tft", c)][sd]
                                 for c in cs])) for sd in range(seeds)]
        t = one_sample_perm(d_seed, mu0=0.0, alternative="two-sided", seed=760)
        family_split[lbl] = {"delta_mean": float(np.mean(d_seed)),
                             "ci": boot_mean_ci(d_seed, seed=761)["ci"],
                             "p": t["p"], "n_cases": len(cs)}
        register_exploratory("H7", f"순환족[{lbl}] Δ(adaptive−GTFT)", t["p"],
                             f"Δ={np.mean(d_seed):+.3f} (case {len(cs)}개)")

    # ---- λ 반응성: 전환 정렬 vs 무작위 정렬 순열 영가설 (§2) ----
    W_PRE, W_POST, K_NULL = 5, 10, 200
    rng0 = np.random.default_rng(99)
    z_seed, act_seed, null_mu_seed = [], [], []
    for sd in range(seeds):
        act, nulls = [], []
        for case in cases:
            lam = np.asarray(R(primary_noise, "adaptive", case, sd)
                             ["agent_log"]["lam"], float)
            switches = [s for s in capricious_switch_rounds(case, rounds)
                        if W_PRE <= s <= rounds - W_POST]
            excl = set()
            for s in switches:
                excl.update(range(s - W_PRE, s + W_POST))
            cand = [t for t in range(W_PRE, rounds - W_POST) if t not in excl]
            for s in switches:
                act.append(abs(lam[s + 1:s + 9].mean() - lam[s - W_PRE:s].mean()))
            if cand:
                ts = rng0.choice(cand, size=min(K_NULL, len(cand) * 3),
                                 replace=True)
                for t in ts:
                    nulls.append(abs(lam[t + 1:t + 9].mean()
                                     - lam[t - W_PRE:t].mean()))
        if act and nulls:
            mu, sdv = float(np.mean(nulls)), float(np.std(nulls) + 1e-9)
            act_seed.append(float(np.mean(act)))
            null_mu_seed.append(mu)
            z_seed.append((float(np.mean(act)) - mu) / sdv)
    lam_perm = one_sample_perm(np.array(act_seed) - np.array(null_mu_seed),
                               mu0=0.0, alternative="greater", seed=75)
    lam_responsive = bool(np.mean(z_seed) > 0 and lam_perm["p"] < 0.05)
    register_exploratory("H7", "λ 반응성 (전환 vs 무작위 정렬 순열)",
                         lam_perm["p"], f"z̄={np.mean(z_seed):.2f}")

    # ---- 국면 분해 ----
    def _kind_at(case, t):
        cfg = CAPRICIOUS_CASES[case]
        return cfg["cycle"][(t // cfg["period"]) % len(cfg["cycle"])]

    phase_pay = {}
    for name in focals:
        ex_s, co_s = [], []
        for sd in range(seeds):
            ex, co = [], []
            for case in cases:
                pay = np.asarray(R(primary_noise, name, case, sd)
                                 ["hist"]["my_payoff"], float)
                mask = np.array([_kind_at(case, t) == "alld"
                                 for t in range(rounds)])
                if mask.any():
                    ex.append(pay[mask])
                if (~mask).any():
                    co.append(pay[~mask])
            ex_s.append(float(np.concatenate(ex).mean()) if ex else np.nan)
            co_s.append(float(np.concatenate(co).mean()) if co else np.nan)
        phase_pay[(name, "exploit")] = ex_s
        phase_pay[(name, "coop")] = co_s
    px = paired_stats(phase_pay[("adaptive", "exploit")],
                      phase_pay[("generous_tft", "exploit")], "greater", seed=76)
    pc_ = paired_stats(phase_pay[("adaptive", "coop")],
                       phase_pay[("generous_tft", "coop")], "less", seed=77)
    register_exploratory("H7", "착취국면 방어 이득", px["p"], fmt_es(px["es"], "dz"))
    register_exploratory("H7", "협력국면 화해지연 비용", pc_["p"], fmt_es(pc_["es"], "dz"))

    # ---- 전환 정렬 궤적 (시드 우선 집계 — §5 불확실성/이질성 분리) ----
    rel = np.arange(-W_PRE, W_POST)
    aligned_seed, aligned_case = {}, {}
    for name in ("adaptive", "generous_tft", "generous_tft_count",
                 "wsls", "tit_for_tat"):
        per_seed_win, per_case_mean = [], {}
        for sd in range(seeds):
            wins = []
            for case in cases:
                tr = np.asarray(R(primary_noise, name, case, sd)
                                ["hist"]["my_payoff"], float)
                for s in capricious_switch_rounds(case, rounds):
                    if s - W_PRE >= 0 and s + W_POST <= rounds:
                        wins.append(tr[s - W_PRE:s + W_POST])
            if wins:
                per_seed_win.append(np.mean(wins, axis=0))
        aligned_seed[name] = np.stack(per_seed_win)
        for case in cases:
            trm = np.stack([np.asarray(R(primary_noise, name, case, sd)
                                       ["hist"]["my_payoff"], float)
                            for sd in range(seeds)]).mean(axis=0)
            wins = [trm[s - W_PRE:s + W_POST]
                    for s in capricious_switch_rounds(case, rounds)
                    if s - W_PRE >= 0 and s + W_POST <= rounds]
            if wins:
                per_case_mean[case] = np.mean(wins, axis=0)
        aligned_case[name] = per_case_mean
    lam_aligned_seed = []
    for sd in range(seeds):
        wins = []
        for case in cases:
            lam = np.asarray(R(primary_noise, "adaptive", case, sd)
                             ["agent_log"]["lam"], float)
            for s in capricious_switch_rounds(case, rounds):
                if s - W_PRE >= 0 and s + W_POST <= rounds:
                    wins.append(lam[s - W_PRE:s + W_POST])
        if wins:
            lam_aligned_seed.append(np.mean(wins, axis=0))
    lam_aligned_seed = np.stack(lam_aligned_seed)

    # ---- 5×5 상호대전: 환경 계층 잡음으로 전원 대칭 (§2; GTFT 두 유형 병기) ----
    tour_names = ["tit_for_tat", "generous_tft", "generous_tft_count",
                  "wsls", "adaptive"]

    def tour_cfg(name, seed):
        if name == "adaptive":
            return agent_spec("adaptive", seed, kappa=0.9, sophisticated=True,
                              use_pymdp=backend == "pymdp")
        return strat_spec(name, seed)          # 내부 error 없음 — 환경이 부과

    nt = len(tour_names)
    t_specs, t_reg = [], {}
    for i, rn in enumerate(tour_names):
        for j, cn in enumerate(tour_names):
            for sd in range(seeds):
                t_reg[(i, j, sd)] = len(t_specs)
                t_specs.append({"agent": tour_cfg(rn, sd),
                                "opponent": tour_cfg(cn, 5000 + sd),
                                "env_err_agent": primary_noise,
                                "env_err_opponent": primary_noise,
                                "noise_seed": 8000 + sd * 41 + i * 7 + j})
    t_res = run_many(t_specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
    matrix = np.zeros((nt, nt)); matrix_ci = np.zeros((nt, nt, 2))
    for i in range(nt):
        for j in range(nt):
            v = [float(np.mean(t_res[t_reg[(i, j, sd)]]["hist"]["my_payoff"]))
                 for sd in range(seeds)]
            matrix[i, j] = np.mean(v)
            matrix_ci[i, j] = boot_mean_ci(v, seed=78)["ci"]
    supported = {"overall_payoff_advantage": bool(pri["mean_a"] > pri["mean_b"]),
                 "exploit_phase_defense": bool(px["mean_a"] > px["mean_b"]),
                 "lambda_responsive": lam_responsive,
                 "by_period": signs,
                 "period_dependent": bool(period_dependent),
                 # rival(GTFT 확률/횟수, WSLS)별 전 case adaptive 우위 여부
                 "advantage_vs_rivals": {
                     rv: bool(adaptive_advantage[rv]["overall"]["adaptive_wins"])
                     for rv in rival_focals}}
    LOGGER.info("[H7] → %s (총보수의 지평 의존성은 H7H·지평 비교에서 [확증] 검증)",
                supported)
    return {"cases": cases, "noise_levels": [str(n) for n in noise_levels],
            "primary_noise": primary_noise, "primary": pri,
            "sweep_focals": list(sweep_focals),
            "rival_focals": list(rival_focals),
            "adaptive_advantage": adaptive_advantage,
            "per_case": {n: {c: [v[0], v[1]] for c, v in d.items()}
                         for n, d in per_case.items()},
            "period_stats": {str(P): v for P, v in period_stats.items()},
            "family_split": family_split,
            "sweep": sweep, "sweep_ci": sweep_ci,
            "phase_pay": {f"{n}|{ph}": mean_sd([x for x in phase_pay[(n, ph)]
                                                if not np.isnan(x)])
                          for n in focals for ph in ("exploit", "coop")},
            "phase_tests": {"exploit": px, "coop": pc_},
            "lambda_perm": {"p": lam_perm["p"], "z_mean": float(np.mean(z_seed)),
                            "act": float(np.mean(act_seed)),
                            "null": float(np.mean(null_mu_seed))},
            "tournament": {"names": tour_names, "matrix": matrix.tolist(),
                           "matrix_ci": matrix_ci.tolist(),
                           "symmetric_env_noise": primary_noise},
            "baselines": {b: float(np.mean(agg_seed[b]))
                          for b in ("qlearner", "bayes_br")},
            "means": {n: float(np.mean(agg_seed[n])) for n in focals},
            "supported": supported,
            "traces": {"rel": rel, "aligned_seed": aligned_seed,
                       "aligned_case": aligned_case,
                       "lam_aligned_seed": lam_aligned_seed,
                       "agg_seed": agg_seed}}


# ================================================================== H7H
def exp_H7H(seeds, rounds, jobs, backend):
    """
    H7H (§6) — 지평 의존성의 체계적 검증. (전지평 실험 — main 에서 1회 실행;
    `rounds` 는 목록으로 전달되며 내부 T 스윕이 요청 지평 {60, 240} 을 포함.)

    Δ(T) = payoff_adaptive − payoff_GTFT (CRN 짝지음) 를 두 스케줄 족에서 추정:
      족 a (전환수 고정): 국면 길이 ∝ T (3국면). 예측 Δ(T) ≈ b·T − c·k₀ → T 에 개선.
      족 b (주기 고정): 전환수 ∝ T. 예측 Δ(T) 는 근사 T-불변, 부호는 주기가 결정.
    [확증] 족 a 에서 dΔ/dT > 0 (시드 원자료 회귀 + 부트스트랩 CI 가 0 배제).
    [탐색] 족 b 기울기 ≈ 0 (CI 로 보고), T*(교차 지평) + 부트스트랩 CI,
           히스테리시스 조작(forgiveness↑, decay↓) → T* ↓ 방향성 예측.
    """
    LOGGER.info("[H7H] 지평 의존성: 스케줄 족 × T 스윕 + 히스테리시스 조작")
    quick = seeds < 10
    req = sorted(set(rounds)) if isinstance(rounds, (list, tuple)) else [int(rounds)]
    Ts = sorted(set([30, 60] + req))[:3] if quick else \
        sorted(set([60, 90, 120, 180, 240]) | set(req))
    noises_a = [0.10] if quick else [0.05, 0.10, 0.20]
    periods_b = [10] if quick else [10, 20, 30]
    primary_noise = 0.10

    def sched_a(seed, T, err):     # 전환수 고정 (3국면, 길이 ∝ T)
        return capricious_spec(seed, T, error=err)

    def sched_b(seed, T, period, err):   # 주기 고정
        phases = ["tit_for_tat", "alld", "generous_tft"]
        cuts = list(range(0, T, period))
        return dict(type="strategy", kind="tit_for_tat", seed=seed, error=err,
                    schedule=[(c, phases[k % 3]) for k, c in enumerate(cuts)])

    ad = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                    use_pymdp=backend == "pymdp")
    ad_fast = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                         forgiveness=0.15, grievance_decay=0.85,
                         use_pymdp=backend == "pymdp")   # 히스테리시스 완화
    gt = strat_spec("generous_tft", 0)
    gtc = strat_spec("generous_tft_count", 0)   # 횟수 기반 GTFT (탐색 병기)

    specs, reg = [], {}

    def add(key, focal, opp, T, sd):
        f = dict(focal); f["seed"] = sd
        reg[key] = (len(specs), T)
        specs.append({"agent": f, "opponent": opp, "noise_seed": 9000 + sd,
                      "_T": T})
    for T in Ts:
        for err in noises_a:
            for sd in range(seeds):
                add(("a", T, err, "adaptive", sd), ad,
                    sched_a(700 + sd, T, err), T, sd)
                add(("a", T, err, "gtft", sd), gt,
                    sched_a(700 + sd, T, err), T, sd)
        for sd in range(seeds):     # 횟수 기반 GTFT — primary 잡음 한정 (탐색)
            add(("a", T, primary_noise, "gtft_count", sd), gtc,
                sched_a(700 + sd, T, primary_noise), T, sd)
        for period in periods_b:
            for sd in range(seeds):
                add(("b", T, period, "adaptive", sd), ad,
                    sched_b(700 + sd, T, period, primary_noise), T, sd)
                add(("b", T, period, "gtft", sd), gt,
                    sched_b(700 + sd, T, period, primary_noise), T, sd)
        for sd in range(seeds):    # 히스테리시스 조작 (족 b, 주기=periods_b[0])
            add(("h", T, "fast", sd), ad_fast,
                sched_b(700 + sd, T, periods_b[0], primary_noise), T, sd)
    # T 가 다이애드마다 달라 run_many 를 T 별로 배치
    LOGGER.info("[H7H] 다이애드 %d 개 (T 별 배치 실행)", len(specs))
    results = [None] * len(specs)
    for T in Ts:
        idxs = [i for i, sp in enumerate(specs) if sp["_T"] == T]
        batch = [{k: v for k, v in specs[i].items() if k != "_T"} for i in idxs]
        out = run_many(batch, n_rounds=T, n_jobs=jobs, verbose=False)
        for i, r in zip(idxs, out):
            results[i] = r

    def pay(key):
        i, T = reg[key]
        return float(np.mean(results[i]["hist"]["my_payoff"]))

    # ---- 족 a: Δ(T) 시드 원자료 회귀 ----
    rows_T, rows_d, seed_cl = [], [], []
    delta_a = {}
    for T in Ts:
        ds = [pay(("a", T, primary_noise, "adaptive", sd))
              - pay(("a", T, primary_noise, "gtft", sd)) for sd in range(seeds)]
        delta_a[T] = (float(np.mean(ds)), boot_mean_ci(ds, seed=171)["ci"])
        rows_T += [T] * seeds; rows_d += ds; seed_cl += list(range(seeds))
    sl_a = slope_boot(rows_T, rows_d, cluster=np.array(seed_cl), seed=172)
    register_primary("H7H", "족a dΔ/dT>0", sl_a["p"],
                     sl_a["ci"][0] > 0,
                     f"기울기={sl_a['slope']:.4f} [{sl_a['ci'][0]:.4f}, {sl_a['ci'][1]:.4f}]")
    # T*: 선형 근사 근 + 시드 부트스트랩
    rngb = np.random.default_rng(173)
    d_mat = np.array([[pay(("a", T, primary_noise, "adaptive", sd))
                       - pay(("a", T, primary_noise, "gtft", sd))
                       for T in Ts] for sd in range(seeds)])
    Tarr = np.array(Ts, float)

    def t_star(mat):
        m = mat.mean(axis=0)
        b1, b0 = np.polyfit(Tarr, m, 1)
        return -b0 / b1 if b1 > 0 else np.nan
    ts_obs = t_star(d_mat)
    ts_boot = [t_star(d_mat[rngb.integers(0, seeds, seeds)]) for _ in range(2000)]
    ts_boot = [t for t in ts_boot if np.isfinite(t)]
    ts_ci = (list(np.percentile(ts_boot, [2.5, 97.5])) if ts_boot
             else [np.nan, np.nan])
    register_exploratory("H7H", "교차 지평 T*", 1.0,
                         f"T*={ts_obs:.0f} [{ts_ci[0]:.0f}, {ts_ci[1]:.0f}]")

    # ---- 족 a (탐색): 횟수 기반 GTFT 대비 Δ_count(T) — GTFT 두 유형 병기 ----
    delta_a_count = {}
    rows_Tc, rows_dc, cl_c = [], [], []
    for T in Ts:
        ds = [pay(("a", T, primary_noise, "adaptive", sd))
              - pay(("a", T, primary_noise, "gtft_count", sd))
              for sd in range(seeds)]
        delta_a_count[T] = (float(np.mean(ds)),
                            boot_mean_ci(ds, seed=1710)["ci"])
        rows_Tc += [T] * seeds; rows_dc += ds; cl_c += list(range(seeds))
    sl_ac = slope_boot(rows_Tc, rows_dc, cluster=np.array(cl_c), seed=1720)
    register_exploratory("H7H", "족a dΔ/dT (vs GTFT 횟수)", sl_ac["p"],
                         f"기울기={sl_ac['slope']:.4f} "
                         f"CI[{sl_ac['ci'][0]:.4f},{sl_ac['ci'][1]:.4f}]")

    # ---- 족 b: 주기별 기울기 (근사 0 예측) ----
    fam_b = {}
    for period in periods_b:
        rT, rd = [], []
        curve = {}
        for T in Ts:
            ds = [pay(("b", T, period, "adaptive", sd))
                  - pay(("b", T, period, "gtft", sd)) for sd in range(seeds)]
            curve[T] = (float(np.mean(ds)), boot_mean_ci(ds, seed=174)["ci"])
            rT += [T] * seeds; rd += ds
        sl = slope_boot(rT, rd, cluster=np.array(list(range(seeds)) * len(Ts)),
                        seed=175)
        fam_b[period] = {"curve": curve, "slope": sl}
        register_exploratory("H7H", f"족b(주기{period}) dΔ/dT≈0", sl["p"],
                             f"기울기={sl['slope']:.4f} CI[{sl['ci'][0]:.4f},{sl['ci'][1]:.4f}]")
    # ---- 히스테리시스 조작: forgiveness↑ → Δ 개선 (→ T*↓ 방향) ----
    hyst = {}
    for T in Ts:
        d_def = [pay(("b", T, periods_b[0], "adaptive", sd))
                 - pay(("b", T, periods_b[0], "gtft", sd)) for sd in range(seeds)]
        d_fast = [pay(("h", T, "fast", sd))
                  - pay(("b", T, periods_b[0], "gtft", sd)) for sd in range(seeds)]
        t = paired_stats(d_fast, d_def, "greater", seed=176)
        hyst[T] = {"delta_default": float(np.mean(d_def)),
                   "delta_fast": float(np.mean(d_fast)), "p": t["p"],
                   "dz": t["es"]["dz"]}
    hyst_dir = bool(np.mean([hyst[T]["delta_fast"] - hyst[T]["delta_default"]
                             for T in Ts]) > 0)
    register_exploratory("H7H", "forgiveness↑ → Δ 개선 (기제 조작)",
                         float(np.median([hyst[T]["p"] for T in Ts])),
                         f"방향성립={hyst_dir}")
    # 족 a 잡음 스윕 곡선 (탐색)
    noise_curves = {str(err): {str(T): mean_sd(
        [pay(("a", T, err, "adaptive", sd)) - pay(("a", T, err, "gtft", sd))
         for sd in range(seeds)])[0] for T in Ts} for err in noises_a}
    return {"Ts": Ts, "primary_noise": primary_noise,
            "family_a": {"delta": {str(T): [delta_a[T][0], delta_a[T][1]]
                                   for T in Ts},
                         "delta_count": {str(T): [delta_a_count[T][0],
                                                  delta_a_count[T][1]]
                                         for T in Ts},
                         "slope": sl_a, "slope_count": sl_ac,
                         "t_star": {"est": float(ts_obs), "ci": ts_ci}},
            "family_b": {str(p): {"curve": {str(T): [v[0], v[1]]
                                            for T, v in d["curve"].items()},
                                  "slope": d["slope"]}
                         for p, d in fam_b.items()},
            "hysteresis": {str(T): hyst[T] for T in Ts},
            "hysteresis_direction_ok": hyst_dir,
            "noise_curves": noise_curves,
            "supported": bool(sl_a["ci"][0] > 0)}


# ================================================================== H8
# GTFT 는 확률론적(generous_tft)·횟수 기반(generous_tft_count) 두 유형으로
# 분리해 병기한다 (v0.4) — 원 0.12 지분을 절반씩 배분해 총 구성 불변.
LARGE_MIX = {"tit_for_tat": 0.18, "generous_tft": 0.06,
             "generous_tft_count": 0.06, "wsls": 0.12,
             "allc": 0.08, "alld": 0.20, "random": 0.10, "capricious": 0.20}


def _largest_remainder(weights: dict, n: int) -> dict:
    raw = {k: w * n for k, w in weights.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = n - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:rem]:
        counts[k] += 1
    return counts


def _m_strat(kind: str, n_rounds: int, error: float = 0.10,
             case: str = DEFAULT_CAPRICIOUS_CASE) -> dict:
    if kind == "capricious":
        return capricious_case_spec(case, seed=0, n_rounds=n_rounds, error=error)
    err = 0.0 if kind in ("random",) else error
    return dict(type="strategy", kind=kind, seed=0, error=err)


def _m_adaptive(backend: str, sophisticated: bool = True,
                lam_base: float = LAM_BASE) -> dict:
    return agent_spec("adaptive", 0, kappa=0.9, sophisticated=sophisticated,
                      lam_base=lam_base,
                      attribution_target="all" if sophisticated else "intent_only",
                      dd_charges_grievance=None if sophisticated else True,
                      use_pymdp=backend == "pymdp")


def _large_members(n: int, n_adaptive: int, n_rounds: int, backend: str,
                   weights: dict | None = None, adaptive_kw: dict | None = None,
                   rand_capricious: bool = False, rng=None):
    """LARGE_MIX(또는 임의 가중) 기반 혼합 집단 구성원 + 라벨. (§3 구성 민감도 지원)"""
    weights = weights or LARGE_MIX
    counts = _largest_remainder(weights, n - n_adaptive)
    members, labels = [], []
    cases = fixed_capricious_cases()      # 집단 구성원은 고정전략 순환 case 만
    for kind, cnt in counts.items():
        for i in range(cnt):
            case = (str((rng or np.random.default_rng(0)).choice(cases))
                    if (rand_capricious and kind == "capricious")
                    else DEFAULT_CAPRICIOUS_CASE)
            members.append(_m_strat(kind, n_rounds, case=case))
            labels.append(kind)
    ad_cfg = _m_adaptive(backend, **(adaptive_kw or {}))
    for i in range(n_adaptive):
        members.append(dict(ad_cfg))
        labels.append("adaptive")
    return members, labels


def exp_H8(seeds, rounds, jobs, backend):
    """
    H8 (전면 재설계, §1·§3·§4) — 규모 확장 하 협력 구조.

    [확증] 통합 상호작용 회귀 CC ~ frac·λ + frac·ratio + λ + ratio 의
           λ=0.4 조건부 frac 기울기 CI 가 0 을 상회 (셀 군집 부트스트랩).
    [탐색] (a) λ×n_adaptive 소집단 — 기울기별 replicate 회귀+CI ("λ=0.24 는
           정성적 일치로 재보정"); (c) 즉각형 비율; (d) N=30 완전 라운드로빈 +
           N=100 매칭 재표집(분산 분해); (f) filler 공정비교 — filler 더미×frac
           상호작용 CI + ALLC 착취 이전 회계; (g) 구성 민감도(ALLD 비중,
           Dirichlet 무작위 구성, capricious case 무작위 배정).
    셀 시드는 stable_seed(cell, rep) 로 격자 간 독립화.
    """
    LOGGER.info("[H8] 규모 확장: 상호작용 회귀 + 역학 없는 매칭의 민감도")
    quick = seeds < 10
    reps_small = max(3, min(30, seeds // 4))
    reps_large = max(2, min(15, seeds // 8))
    reps_combo = max(2, min(8, seeds // 15))
    n_small, n_large = 12, (30 if quick else 100)

    all_specs, registry = [], {}

    def add(cell_key, rep, spec):
        spec = dict(spec)
        spec["seed"] = stable_seed(cell_key, rep)
        registry[(cell_key, rep)] = len(all_specs)
        all_specs.append(spec)

    def small_spec(n_ad, lam_b=LAM_BASE, soph=True, n=n_small):
        members, labels = [], []
        for k, c in _largest_remainder(
                {"tit_for_tat": .25, "alld": .25, "wsls": .25, "random": .25},
                n - n_ad).items():
            for _ in range(c):
                members.append(_m_strat(k, rounds)); labels.append(k)
        for _ in range(n_ad):
            members.append(_m_adaptive(backend, sophisticated=soph,
                                       lam_base=lam_b))
            labels.append("adaptive" if soph else "adaptive_imm")
        return {"members": members, "labels": labels, "n_rounds": rounds,
                "partners_per_agent": 3}

    # (a) λ 스윕 × n_adaptive
    lam_grid = [0.1, 0.24, 0.4, 0.6] if quick else \
        [0.0, 0.1, 0.2, 0.24, 0.3, 0.4, 0.6, 0.8]
    nad_grid_small = [0, 3, 6, 9]
    for lam in lam_grid:
        for n_ad in nad_grid_small:
            for rep in range(reps_small):
                add(("a", lam, n_ad), rep, small_spec(n_ad, lam_b=lam))
    # (c) 즉각/정교 비율
    ratio_grid = [0.0, 0.5, 1.0] if quick else [0.0, 0.25, 0.5, 0.75, 1.0]
    for ratio in ratio_grid:
        for rep in range(reps_small):
            n_ad = 6
            n_imm = int(round(ratio * n_ad))
            members, labels = [], []
            for k, c in _largest_remainder(
                    {"tit_for_tat": .25, "alld": .25, "wsls": .25,
                     "random": .25}, n_small - n_ad).items():
                for _ in range(c):
                    members.append(_m_strat(k, rounds)); labels.append(k)
            for i in range(n_ad):
                soph = i >= n_imm
                members.append(_m_adaptive(backend, sophisticated=soph))
                labels.append("adaptive" if soph else "adaptive_imm")
            add(("c", ratio), rep, {"members": members, "labels": labels,
                                    "n_rounds": rounds,
                                    "partners_per_agent": 3})
    # (d) 대규모: N=30 완전 라운드로빈(표본화 오차 0 기준선) + N=100 매칭 재표집
    nad_frac_grid = [0.0, 0.25, 0.5] if quick else [0.0, 0.10, 0.25, 0.50, 0.75]
    for frac in nad_frac_grid:
        n_ad30 = int(round(frac * 30))
        members, labels = _large_members(30, n_ad30, rounds, backend)
        for rep in range(reps_large):
            add(("d30", frac), rep, {"members": members, "labels": labels,
                                     "n_rounds": rounds,
                                     "partners_per_agent": None})
        if not quick:
            n_ad100 = int(round(frac * 100))
            members, labels = _large_members(100, n_ad100, rounds, backend)
            for rep in range(reps_large):
                add(("d100", frac), rep,
                    {"members": members, "labels": labels, "n_rounds": rounds,
                     "partners_per_agent": 6, "match_resamples": 3})
    # (e) 통합 조합 격자 (상호작용 추정용): frac × λ × ratio
    frac_e = [0.0, 0.25, 0.5] if quick else [0.0, 0.25, 0.5, 0.75]
    lam_e = [0.24, 0.4] if quick else [0.1, 0.24, 0.4, 0.6]
    ratio_e = [0.0, 1.0] if quick else [0.0, 0.5, 1.0]
    for frac in frac_e:
        for lam in lam_e:
            for ratio in ratio_e:
                n_ad = int(round(frac * n_small))
                n_imm = int(round(ratio * n_ad))
                members, labels = [], []
                for k, c in _largest_remainder(
                        {"tit_for_tat": .25, "alld": .25, "wsls": .25,
                         "random": .25}, n_small - n_ad).items():
                    for _ in range(c):
                        members.append(_m_strat(k, rounds)); labels.append(k)
                for i in range(n_ad):
                    soph = i >= n_imm
                    members.append(_m_adaptive(backend, sophisticated=soph,
                                               lam_base=lam))
                    labels.append("adaptive" if soph else "adaptive_imm")
                for rep in range(reps_combo):
                    add(("e", frac, lam, ratio), rep,
                        {"members": members, "labels": labels,
                         "n_rounds": rounds, "partners_per_agent": 3})
    # (f) filler 공정비교 (N=30 라운드로빈; adaptive 대신 고정전략 투입)
    fillers = ["generous_tft", "generous_tft_count", "allc"] if quick else \
        ["generous_tft", "generous_tft_count", "tit_for_tat", "allc", "wsls"]
    for frac in nad_frac_grid:
        n_f = int(round(frac * 30))
        for filler in fillers:
            base_members, base_labels = _large_members(30, 0, rounds, backend)
            members = base_members[:30 - n_f] + \
                [_m_strat(filler, rounds) for _ in range(n_f)]
            labels = base_labels[:30 - n_f] + [f"filler_{filler}"] * n_f
            for rep in range(reps_large):
                add(("f", frac, filler), rep,
                    {"members": members, "labels": labels, "n_rounds": rounds,
                     "partners_per_agent": None})
    # (g) 구성 민감도
    sens_reps = max(2, reps_large // 2)
    alld_shares = [0.10, 0.30] if quick else [0.10, 0.20, 0.30]
    for share in alld_shares:
        w = dict(LARGE_MIX); w["alld"] = share
        tot = sum(v for k, v in w.items() if k != "alld")
        for k in w:
            if k != "alld":
                w[k] = w[k] / tot * (1 - share)
        for frac in ([0.0, 0.5] if quick else [0.0, 0.25, 0.5]):
            members, labels = _large_members(30, int(round(frac * 30)), rounds,
                                             backend, weights=w)
            for rep in range(sens_reps):
                add(("g_alld", share, frac), rep,
                    {"members": members, "labels": labels, "n_rounds": rounds,
                     "partners_per_agent": None})
    n_dir = 4 if quick else 20
    rng_dir = np.random.default_rng(881)
    base_w = np.array([LARGE_MIX[k] for k in LARGE_MIX])
    dir_keys = list(LARGE_MIX)
    for di in range(n_dir):
        w = dict(zip(dir_keys, rng_dir.dirichlet(10.0 * base_w)))
        for frac in [0.0, 0.5]:
            members, labels = _large_members(30, int(round(frac * 30)), rounds,
                                             backend, weights=w)
            for rep in range(max(2, sens_reps // 2)):
                add(("g_dir", di, frac), rep,
                    {"members": members, "labels": labels, "n_rounds": rounds,
                     "partners_per_agent": None})
    rng_cap = np.random.default_rng(882)
    for frac in ([0.0, 0.5] if quick else [0.0, 0.25, 0.5]):
        members, labels = _large_members(30, int(round(frac * 30)), rounds,
                                         backend, rand_capricious=True,
                                         rng=rng_cap)
        for rep in range(sens_reps):
            add(("g_cap", frac), rep,
                {"members": members, "labels": labels, "n_rounds": rounds,
                 "partners_per_agent": None})

    LOGGER.info("[H8] 집단 %d 개 병렬 실행 (reps: small=%d large=%d combo=%d)",
                len(all_specs), reps_small, reps_large, reps_combo)
    t0 = time.time()
    out = run_populations(all_specs, n_jobs=jobs, verbose=True)
    LOGGER.info("[H8] 집단 실행 %.1fs", time.time() - t0)

    def cells(prefix):
        ks = sorted({k for (k, _r) in registry if k[0] == prefix},
                    key=lambda x: tuple(str(v) for v in x))
        return ks

    def reps_of(cell, key="cc_rate"):
        return [out[registry[(cell, r)]][key]
                for r in range(_n_reps(cell))]

    def _n_reps(cell):
        return sum(1 for (k, _r) in registry if k == cell)

    # ---- (a) λ×n_ad — replicate 원자료 기울기 + CI (§1(iv), §4 재보정) ----
    lam_slopes = {}
    for lam in lam_grid:
        xs, ys, cl = [], [], []
        for n_ad in nad_grid_small:
            cell = ("a", lam, n_ad)
            for r in range(_n_reps(cell)):
                xs.append(n_ad / n_small)
                ys.append(out[registry[(cell, r)]]["cc_rate"])
                cl.append(str(cell) + str(r))
        sl = slope_boot(xs, ys, seed=stable_seed("slope", lam))
        lam_slopes[lam] = sl
        sig = "0 배제" if (sl["ci"][0] > 0 or sl["ci"][1] < 0) else "0 포함"
        LOGGER.info("[탐색][H8a] λ=%.2f: CC~frac 기울기=%.3f [%.3f, %.3f] (%s)",
                    lam, sl["slope"], sl["ci"][0], sl["ci"][1], sig)
    sign_flip = (lam_slopes[min(lam_grid)]["slope"] < 0 <
                 lam_slopes[max(lam_grid)]["slope"])
    LOGGER.info("[탐색][H8a] 저λ 음성→고λ 양성 부호 구조: %s — Albarracin λ=0.24 "
                "경계와의 관계는 '정성적 일치' 로만 주장 (§4 재보정; 원 설정 "
                "재구현 대응 스윕은 선택 과제)", sign_flip)

    # ---- (e) 통합 상호작용 회귀 [확증] ----
    X, y, cl = [], [], []
    for cell in cells("e"):
        _, frac, lam, ratio = cell
        for r in range(_n_reps(cell)):
            X.append([1.0, frac, lam, ratio, frac * lam, frac * ratio])
            y.append(out[registry[(cell, r)]]["cc_rate"])
            cl.append(str(cell))
    reg_e = ols_boot(np.array(X), np.array(y), cluster=np.array(cl),
                     n_boot=2000, seed=stable_seed("H8e"),
                     names=["const", "frac", "lam", "ratio",
                            "frac×lam", "frac×ratio"])
    boots = reg_e.pop("boots")
    cond04 = reg_e["beta"][1] + 0.4 * reg_e["beta"][4]
    cond04_b = boots[:, 1] + 0.4 * boots[:, 4]
    cond04_ci = [float(np.percentile(cond04_b, 2.5)),
                 float(np.percentile(cond04_b, 97.5))]
    p_cond = float(np.clip(2 * min((cond04_b <= 0).mean(),
                                   (cond04_b >= 0).mean()),
                           1.0 / len(cond04_b), 1.0))
    register_primary("H8", "λ=0.4 조건부 frac 기울기 > 0", p_cond,
                     cond04_ci[0] > 0,
                     f"β={cond04:.3f} [{cond04_ci[0]:.3f}, {cond04_ci[1]:.3f}]")
    cond_slopes = {}
    for lam in sorted({c[2] for c in cells("e")}):
        cb = boots[:, 1] + lam * boots[:, 4]
        cond_slopes[lam] = {"est": float(reg_e["beta"][1]
                                         + lam * reg_e["beta"][4]),
                            "ci": [float(np.percentile(cb, 2.5)),
                                   float(np.percentile(cb, 97.5))]}

    # ---- (c) 비율 ----
    ratio_out = {}
    for ratio in ratio_grid:
        cell = ("c", ratio)
        cc = reps_of(cell); pay = reps_of(cell, "mean_payoff")
        ratio_out[ratio] = {"cc": [mean_sd(cc)[0], boot_mean_ci(cc, seed=83)["ci"]],
                            "payoff": [mean_sd(pay)[0],
                                       boot_mean_ci(pay, seed=84)["ci"]]}
    rx, ry = [], []
    for ratio in ratio_grid:
        for v in reps_of(("c", ratio)):
            rx.append(ratio); ry.append(v)
    sl_ratio = slope_boot(rx, ry, seed=85)
    register_exploratory("H8", "즉각형 비율 → CC 기울기", sl_ratio["p"],
                         f"{sl_ratio['slope']:.3f} CI[{sl_ratio['ci'][0]:.3f},{sl_ratio['ci'][1]:.3f}]")

    # ---- (d) 대규모 + 매칭 분산 분해 (§3) ----
    large_curves = {}
    for pref, N in (("d30", 30), ("d100", 100)):
        if not cells(pref):
            continue
        cur = {}
        for cell in cells(pref):
            frac = cell[1]
            cc = reps_of(cell); pay = reps_of(cell, "mean_payoff")
            cur[frac] = {"cc": [mean_sd(cc)[0], boot_mean_ci(cc, seed=86)["ci"]],
                         "payoff": [mean_sd(pay)[0],
                                    boot_mean_ci(pay, seed=87)["ci"]]}
        xs, ys = [], []
        for cell in cells(pref):
            for v in reps_of(cell):
                xs.append(cell[1]); ys.append(v)
        large_curves[pref] = {"curve": cur, "N": N,
                              "slope": slope_boot(xs, ys, seed=88)}
    match_decomp = None
    if cells("d100"):
        within, between = [], []
        for cell in cells("d100"):
            reps_cc = []
            for r in range(_n_reps(cell)):
                m = out[registry[(cell, r)]].get("cc_by_matching", [])
                if len(m) > 1:
                    within.append(float(np.var(m, ddof=1)))
                reps_cc.append(out[registry[(cell, r)]]["cc_rate"])
            if len(reps_cc) > 1:
                between.append(float(np.var(reps_cc, ddof=1)))
        match_decomp = {"var_within_matching": float(np.mean(within)) if within else None,
                        "var_between_replicates": float(np.mean(between)) if between else None}
        LOGGER.info("[탐색][H8d] 분산 분해 — 매칭 내: %s / replicate 간: %s",
                    match_decomp["var_within_matching"],
                    match_decomp["var_between_replicates"])

    # ---- (f) filler 공정비교: 통합 회귀 filler 더미 × frac (§1(iv)) ----
    Xf, yf, clf = [], [], []
    filler_names = sorted({c[2] for c in cells("f")})
    for cell in cells("d30"):
        frac = cell[1]
        for v in reps_of(cell):
            Xf.append([1.0, frac] + [0.0] * (2 * len(filler_names)))
            yf.append(v); clf.append(str(cell))
    for cell in cells("f"):
        _, frac, filler = cell
        fi = filler_names.index(filler)
        row = [1.0, frac] + [0.0] * (2 * len(filler_names))
        row[2 + fi] = 1.0
        row[2 + len(filler_names) + fi] = frac
        for v in reps_of(cell):
            Xf.append(row); yf.append(v); clf.append(str(cell))
    names_f = (["const", "frac"] + [f"d_{f}" for f in filler_names]
               + [f"frac×{f}" for f in filler_names])
    reg_f = ols_boot(np.array(Xf), np.array(yf), cluster=np.array(clf),
                     n_boot=1500, seed=89, names=names_f)
    reg_f.pop("boots", None)
    filler_inter = {}
    for fi, f in enumerate(filler_names):
        j = 2 + len(filler_names) + fi
        filler_inter[f] = {"est": reg_f["beta"][j], "ci": reg_f["ci"][j],
                           "p": reg_f["p"][j]}
        register_exploratory("H8", f"frac×{f} 상호작용 (adaptive 대비)",
                             reg_f["p"][j],
                             f"{reg_f['beta'][j]:.3f} CI[{reg_f['ci'][j][0]:.3f},{reg_f['ci'][j][1]:.3f}]")
    # ALLC 착취 이전 회계 (§4): filler=allc 셀에서 ALLD 로의 이전 정량화
    transfer = None
    allc_cells = [c for c in cells("f") if c[2] == "allc" and c[1] >= 0.5]
    if allc_cells:
        cell = allc_cells[-1]
        gains, ne_pay, grp = [], [], []
        for r in range(_n_reps(cell)):
            o = out[registry[(cell, r)]]
            gains.append(o["alld_exploit_gain_total"])
            ne_pay.append(o["nonexploiter_mean_payoff"])
            grp.append(o["group_payoff"])
        transfer = {"cell": str(cell),
                    "alld_exploit_gain_total": mean_sd(gains)[0],
                    "share_of_group_payoff": float(np.mean(gains) / np.mean(grp)),
                    "nonexploiter_mean_payoff": mean_sd(ne_pay)[0]}
        LOGGER.info("[탐색][H8f] ALLC 투입 셀: 집단 후생 중 %.1f%% 가 ALLD 착취 "
                    "이전(transfer); 착취자 제외 평균 보수=%.3f (§4 측정화)",
                    100 * transfer["share_of_group_payoff"],
                    transfer["nonexploiter_mean_payoff"])

    # ---- (g) 민감도: frac 기울기 부호의 강건성 ----
    sens = {}
    for share in alld_shares:
        xs, ys = [], []
        for cell in cells("g_alld"):
            if cell[1] == share:
                for v in reps_of(cell):
                    xs.append(cell[2]); ys.append(v)
        sens[f"alld_{share}"] = slope_boot(xs, ys, seed=90)
    xs, ys = [], []
    for cell in cells("g_dir"):
        for v in reps_of(cell):
            xs.append(cell[2]); ys.append(v)
    sens["dirichlet_pooled"] = slope_boot(xs, ys, seed=91)
    xs, ys = [], []
    for cell in cells("g_cap"):
        for v in reps_of(cell):
            xs.append(cell[1]); ys.append(v)
    sens["capricious_random"] = slope_boot(xs, ys, seed=92)
    for k, sl in sens.items():
        register_exploratory("H8", f"민감도[{k}] frac 기울기", sl["p"],
                             f"{sl['slope']:.3f} CI[{sl['ci'][0]:.3f},{sl['ci'][1]:.3f}]")
    robust_pos = all(sl["slope"] > 0 for sl in sens.values())

    lam_curves = {}
    for lam in lam_grid:
        lam_curves[lam] = {n_ad: [mean_sd(reps_of(("a", lam, n_ad)))[0],
                                  boot_mean_ci(reps_of(("a", lam, n_ad)),
                                               seed=93)["ci"]]
                           for n_ad in nad_grid_small}
    supported = {"conditional_slope_lam04": bool(cond04_ci[0] > 0),
                 "sign_structure_qualitative": bool(sign_flip),
                 "sensitivity_slope_positive": bool(robust_pos)}
    LOGGER.info("[H8] → %s", supported)
    return {"reps": {"small": reps_small, "large": reps_large,
                     "combo": reps_combo},
            "lam_grid": lam_grid, "nad_grid": nad_grid_small,
            "n_small": n_small,
            "lam_curves": {str(k): {str(n): v for n, v in d.items()}
                           for k, d in lam_curves.items()},
            "lam_slopes": {str(k): v for k, v in lam_slopes.items()},
            "interaction_reg": reg_e, "conditional_slopes":
                {str(k): v for k, v in cond_slopes.items()},
            "primary": {"p": p_cond, "est": float(cond04), "ci": cond04_ci},
            "ratio": {str(k): v for k, v in ratio_out.items()},
            "ratio_slope": sl_ratio,
            "large": {k: {"N": v["N"], "slope": v["slope"],
                          "curve": {str(f): c for f, c in v["curve"].items()}}
                      for k, v in large_curves.items()},
            "match_decomposition": match_decomp,
            "filler_regression": reg_f, "filler_interactions": filler_inter,
            "allc_transfer": transfer,
            "sensitivity": sens,
            "supported": supported}


# ================================================================== H8E
# 유형별 고정 색 (모든 복제자/Moran 그림에서 전 유형 legend 명시 — v0.3 요구)
TYPE_COLORS = {
    "tit_for_tat": "#1f77b4", "generous_tft": "#2ca02c",
    "generous_tft_count": "#98df8a", "wsls": "#17becf",
    "allc": "#bcbd22", "alld": "#d62728", "random": "#7f7f7f",
    "capricious": "#9467bd", "adaptive": "#ff7f0e", "adaptive_imm": "#8c564b",
    "tom_fixed": "#e377c2",
}


def _resident_from_mix(names, mix):
    w = np.zeros(len(names))
    for kind, v in mix.items():
        if kind in names:
            w[names.index(kind)] = v
    return w / w.sum()


def _growth_boot(raw, resident, inv_idx, n_boot=2000, seed=0):
    """Π 시드 부트스트랩으로 침입 성장률 g 의 점추정·95% CI·양측 p."""
    pi_seeds = raw.shape[2]
    g_obs = evo.invasion_growth(raw.mean(axis=2), resident, inv_idx)
    rng = np.random.default_rng(seed)
    gb = np.array([evo.invasion_growth(
        raw[:, :, rng.integers(0, pi_seeds, pi_seeds)].mean(axis=2),
        resident, inv_idx) for _ in range(n_boot)])
    ci = [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]
    pv = float(np.clip(2 * min((gb <= 0).mean(), (gb >= 0).mean()),
                       1.0 / n_boot, 1.0))
    return {"g": float(g_obs), "ci": ci, "p": pv}


def exp_H8E(seeds, rounds, jobs, backend):
    """
    H8E (v0.3 전면 확장) — 진화적 동역학: 평균장 복제자 + 경험보수 Moran.

    Π 격자 : env_error ∈ {0.0, 0.05, 0.10, 0.15, 0.20} × T ∈ {60, 240},
             pi_seeds = 50 (요구 사양), 순서쌍 대칭 재사용으로 다이애드 절반화.

    [확증] 기준 혼합 상주집단(LARGE_MIX)에 대한 adaptive 침입 성장률 > 0 —
           **지평별 2개** (T=60, T=240; primary 잡음 0.10). v0.1.2 관찰
           (T=60 에서 g<0)과 H7H 의 T*≈170 에 비추어 지평 의존성이 예상되며,
           지지 여부는 지평별로 정직하게 분리 보고한다.
    [탐색] (a) g(err, T) 전 격자 (즉각형 병기),
           (b) **역방향 침입**: 정교형 adaptive 다수 상주집단에 각 고정전략이
               침입할 때의 성장률 — 순수/혼합(85%) 상주 두 구성, 전 격자,
               복제자 궤적으로 동역학 검증,
           (c) **상주 구성 민감도**: Dirichlet 무작위 구성 + ALLD 비중 스윕에서
               adaptive 침입 가능 구성 비율과 g~ALLD 비중 기울기,
           (d) 복제자 궤적(전 유형 legend), **끌개(attractor) 구조**,
               **협력 유역(cooperation basin)** — 3-유형 부분계의 상태공간
               위상 초상(벡터장 + 유역 지도)과 전 격자 유역 비율,
           (e) Moran 평균장 교차검증 (지평별).
    """
    LOGGER.info("[H8E] 복제자 동역학 + Moran — (err × T) 격자")
    quick = seeds < 10
    pi_seeds = 4 if quick else 50
    env_errors = [0.10] if quick else [0.0, 0.05, 0.10, 0.15, 0.20]
    req = sorted(set(rounds)) if isinstance(rounds, (list, tuple)) else [int(rounds)]
    Ts = req[:1] if quick else sorted(set([60, 240]) | set(req))
    primary_err = 0.10 if 0.10 in env_errors else env_errors[0]

    def type_specs_for(T):
        return {
            "tit_for_tat": _m_strat("tit_for_tat", T),
            "generous_tft": _m_strat("generous_tft", T),
            "generous_tft_count": _m_strat("generous_tft_count", T),
            "wsls": _m_strat("wsls", T),
            "allc": _m_strat("allc", T),
            "alld": _m_strat("alld", T),
            "random": _m_strat("random", T),
            "capricious": _m_strat("capricious", T),
            "adaptive": _m_adaptive(backend, sophisticated=True),
            "adaptive_imm": _m_adaptive(backend, sophisticated=False),
        }

    est = {}
    for ti, T in enumerate(Ts):
        for ei, err in enumerate(env_errors):
            t0 = time.time()
            est[(err, T)] = evo.estimate_payoff_matrix(
                type_specs_for(T), n_rounds=T, seeds=pi_seeds, n_jobs=jobs,
                env_error=err, symmetric=True,
                seed_offset=1000 * (ti * len(env_errors) + ei))
            LOGGER.info("[H8E] Π(err=%.2f, T=%d) 추정 %.1fs", err, T,
                        time.time() - t0)
    names = est[(primary_err, Ts[0])]["names"]
    k = len(names)
    i_ad, i_imm = names.index("adaptive"), names.index("adaptive_imm")
    fixed_names = [n for n in names if n not in ("adaptive", "adaptive_imm")]
    resident_w = _resident_from_mix(names, LARGE_MIX)

    # ---- (확증 + a) 기준 혼합 상주집단 침입 격자 ----
    inv_grid = {}
    for (err, T), e in est.items():
        inv_grid[(err, T)] = {
            "adaptive": _growth_boot(e["raw"], resident_w, i_ad,
                                     seed=stable_seed("g_ad", err, T)),
            "adaptive_imm": _growth_boot(e["raw"], resident_w, i_imm,
                                         seed=stable_seed("g_im", err, T)),
        }
    supported_by_T = {}
    for T in Ts:
        g = inv_grid[(primary_err, T)]["adaptive"]
        register_primary("H8E", f"adaptive 침입 성장률 > 0 (T={T})", g["p"],
                         g["ci"][0] > 0,
                         f"g={g['g']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]")
        supported_by_T[f"T{T}"] = bool(g["ci"][0] > 0)
        g_i = inv_grid[(primary_err, T)]["adaptive_imm"]
        register_exploratory("H8E", f"즉각형 침입 성장률 (T={T})", g_i["p"],
                             f"g={g_i['g']:.3f}")
    horizon_dependent = len(set(supported_by_T.values())) > 1
    if horizon_dependent:
        LOGGER.info("[H8E] ⚠ 침입 지지 여부가 지평에 의존: %s — H7H 의 "
                    "T* 기제(학습·화해 비용의 상환)와 정합 여부를 해석에 명시",
                    supported_by_T)

    # ---- (b) 역방향 침입: adaptive 다수 상주집단에 고정전략 침입 ----
    invaders = fixed_names + ["adaptive_imm"]
    res_pure = np.eye(k)[i_ad]
    res_mixed = 0.85 * res_pure + 0.15 * _resident_from_mix(
        names, {n: LARGE_MIX.get(n, 0.0) for n in fixed_names})
    res_mixed /= res_mixed.sum()
    reverse = {}
    for (err, T), e in est.items():
        rv = {}
        for inv in invaders:
            ii = names.index(inv)
            rv[inv] = {
                "pure": _growth_boot(e["raw"], res_pure, ii,
                                     seed=stable_seed("rvP", err, T, inv)),
                "mixed": _growth_boot(e["raw"], res_mixed, ii,
                                      seed=stable_seed("rvM", err, T, inv)),
            }
        reverse[(err, T)] = rv
    for T in Ts:
        can = [inv for inv in invaders
               if reverse[(primary_err, T)][inv]["pure"]["ci"][0] > 0]
        cannot = [inv for inv in invaders
                  if reverse[(primary_err, T)][inv]["pure"]["ci"][1] < 0]
        register_exploratory(
            "H8E", f"역침입 (T={T}, 순수 adaptive 상주)", 1.0,
            f"침입 가능(CI>0): {can or '없음'} / 격퇴(CI<0): {cannot or '없음'}")
    # 역침입 동역학 검증: adaptive 다수 + 침입자 5% 초기점의 복제자 궤적
    reverse_traj = {}
    for T in Ts:
        Pi = est[(primary_err, T)]["Pi"]
        tr = {}
        for inv in invaders:
            x0 = np.full(k, 0.02 / (k - 2))
            x0[i_ad] = 0.93
            x0[names.index(inv)] = 0.05
            x0 /= x0.sum()
            tr[inv] = evo.replicator_trajectory(Pi, x0, steps=400)[
                :, names.index(inv)]
        reverse_traj[T] = tr

    # ---- (c) 상주 구성 민감도 (Π 대수 — 무비용) ----
    n_comp = 8 if quick else 100
    rng_c = np.random.default_rng(808)
    comp_out = {}
    alld_sweep_shares = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for (err, T), e in est.items():
        Pi = e["Pi"]
        gs, alld_share = [], []
        for _ in range(n_comp):
            w7 = rng_c.dirichlet(np.ones(len(fixed_names)))
            w = np.zeros(k)
            for nm, v in zip(fixed_names, w7):
                w[names.index(nm)] = v
            gs.append(evo.invasion_growth(Pi, w, i_ad))
            alld_share.append(float(w7[fixed_names.index("alld")]))
        sl = slope_boot(alld_share, gs, seed=stable_seed("comp", err, T))
        sweep_g = []
        for sh in alld_sweep_shares:
            mix = {nm: LARGE_MIX.get(nm, 0.0) for nm in fixed_names}
            tot = sum(v for nm, v in mix.items() if nm != "alld")
            for nm in mix:
                if nm != "alld":
                    mix[nm] = mix[nm] / tot * (1 - sh)
            mix["alld"] = sh
            sweep_g.append(evo.invasion_growth(
                Pi, _resident_from_mix(names, mix), i_ad))
        comp_out[(err, T)] = {
            "frac_invadable": float(np.mean(np.array(gs) > 0)),
            "g_vs_alld_slope": sl,
            "g_samples": gs, "alld_share_samples": alld_share,
            "alld_sweep": {"shares": alld_sweep_shares, "g": sweep_g},
        }
    for T in Ts:
        co = comp_out[(primary_err, T)]
        register_exploratory(
            "H8E", f"구성 민감도 (T={T})", co["g_vs_alld_slope"]["p"],
            f"침입 가능 구성 비율={co['frac_invadable']:.2f}, "
            f"g~ALLD 비중 기울기={co['g_vs_alld_slope']['slope']:.3f}")

    # ---- (d) 복제자 궤적 / 끌개 / 협력 유역 ----
    coop_idx = [names.index(t) for t in
                ("tit_for_tat", "generous_tft", "generous_tft_count",
                 "wsls", "allc", "adaptive", "adaptive_imm")]
    x0s = {}
    x_mix = resident_w * 0.9; x_mix[i_ad] = 0.1
    x0s["mix+10%ad"] = x_mix / x_mix.sum()
    x0s["uniform"] = np.ones(k) / k
    x_alld = np.ones(k) * 0.05
    x_alld[names.index("alld")] = 1.0 - 0.05 * (k - 1)
    x0s["alld_heavy"] = x_alld
    trajs, attractors, basins = {}, {}, {}
    n_basin = 40 if quick else 400
    n_attr = 150 if quick else 1500       # [v0.6.3] 끌개 구성: 층화 표집, n=1500
    for T in Ts:
        Pi = est[(primary_err, T)]["Pi"]
        trajs[T] = {lbl: evo.replicator_trajectory(Pi, x0, steps=500)
                    for lbl, x0 in x0s.items()}
        attractors[T] = evo.attractor_analysis(
            Pi, names, n_samples=n_attr, steps=800,
            seed=stable_seed("attr", T))
        top = attractors[T]["attractors"][:3]
        register_exploratory(
            "H8E", f"끌개 구조 (T={T}; 층화 n={n_attr})", 1.0,
            "; ".join(f"균등유역 {a['basin_frac']:.2f}/코너유입 {a['corner_frac_conv']:.2f}"
                      " → " + ", ".join(
                f"{names[i]}={c:.2f}" for i, c in enumerate(a["composition"])
                if c > 0.05) for a in top))
        # 코너 강건성: 유형별 지배 초기 집단이 1위 끌개로 흡수되는 비율
        cc_conv = attractors[T]["corner_convergence"]
        register_exploratory(
            "H8E", f"코너 강건성 (T={T})", 1.0,
            "; ".join(f"{nm}→A0:{d.get(0, 0.0):.2f}"
                      for nm, d in sorted(cc_conv.items())))
    for (err, T), e in est.items():
        Pi = e["Pi"]; CCm = e["CC"]
        keep = [i for i in range(k) if i not in (i_ad, i_imm)]
        rng_b = np.random.default_rng(802)
        X0 = rng_b.dirichlet(np.ones(k), size=n_basin)
        X0k = rng_b.dirichlet(np.ones(len(keep)), size=n_basin)
        # [지표 개정 A] 이분 유역 → 종착 조성의 행동적 CC율 xᵀ·CCm·x.
        cc_with = evo.re_terminal_cc(Pi, CCm, X0)
        cc_without = evo.re_terminal_cc(Pi[np.ix_(keep, keep)],
                                        CCm[np.ix_(keep, keep)], X0k)
        basins[(err, T)] = {"with": cc_with, "without": cc_without,
                            "widening": float(cc_with - cc_without)}
    for T in Ts:
        b = basins[(primary_err, T)]
        register_exploratory("H8E", f"협력 CC-widening (T={T}, adaptive 유−무)",
                             1.0, f"ΔCC={b['widening']:+.3f} "
                             f"({b['with']:.2f} vs {b['without']:.2f})")

    # 3-유형 부분계 상태공간: (adaptive, alld, tit_for_tat) / (adaptive, alld, allc)
    subsystems = {"ad_alld_tft": ("adaptive", "alld", "tit_for_tat"),
                  "ad_alld_allc": ("adaptive", "alld", "allc")}
    tern = {}
    n_grid = 9 if quick else 23
    for T in Ts:
        Pi = est[(primary_err, T)]["Pi"]
        for key, subset in subsystems.items():
            Pi3, sub_names = evo.restrict_matrix(Pi, names, subset)
            coop3 = [i for i, nm in enumerate(sub_names) if nm != "alld"]
            bm = evo.basin_map_3(Pi3, coop3, n=n_grid, steps=800)
            starts = evo.ternary_grid(4)
            tr3 = [evo.replicator_trajectory(Pi3, x0, steps=600)
                   for x0 in starts]
            tern[(key, T)] = {"names": sub_names, "map": bm,
                              "trajs": tr3, "Pi3": Pi3}

    # ---- (e) Moran (평균장 교차검증; 지평별) ----
    moran = {}
    n_moran = 2 if quick else 10
    gens = 10 if quick else 30
    for T in Ts:
        e = est[(primary_err, T)]
        init = np.round(np.r_[resident_w[:k - 2] * 0.9, [0.10, 0.0]]
                        * 100).astype(int)
        init[0] += 100 - init.sum()
        mf = np.stack([evo.moran_process(e["Pi"], e["Pi_sd"], init,
                                         generations=gens, mutation=0.01,
                                         seed=810 + r)
                       for r in range(n_moran)])
        ad_final = mf[:, -1, i_ad]
        moran[T] = {"freq": mf, "final_adaptive_mean": float(ad_final.mean()),
                    "grew_frac": float(np.mean(ad_final > init[i_ad] / 100)),
                    "reps": n_moran, "generations": gens}
        register_exploratory("H8E", f"Moran adaptive 성장 (T={T})", 1.0,
                             f"최종 빈도={ad_final.mean():.2f} (초기 0.10)")

    def gk(d):        # (err, T) 튜플 키 → 문자열 키 (JSON 직렬화)
        return {f"err{err}|T{T}": v for (err, T), v in d.items()}

    supported = {
        "invasion_by_T": supported_by_T,
        "horizon_dependent": bool(horizon_dependent),
        "reverse_alld_repelled": {
            f"T{T}": bool(reverse[(primary_err, T)]["alld"]["pure"]["ci"][1] < 0)
            for T in Ts},
        "cc_widening_positive": {
            f"T{T}": bool(basins[(primary_err, T)]["widening"] > 0) for T in Ts},
    }
    LOGGER.info("[H8E] → %s", supported)
    return {"names": names, "Ts": Ts, "env_errors": env_errors,
            "primary_err": primary_err, "pi_seeds": pi_seeds,
            "Pi": {f"err{err}|T{T}": e["Pi"].tolist()
                   for (err, T), e in est.items()},
            "invasion_grid": gk({key: v for key, v in inv_grid.items()}),
            "reverse_invasion": gk({key: {inv: {kk: vv for kk, vv in d.items()}
                                          for inv, d in rv.items()}
                                    for key, rv in reverse.items()}),
            "composition": gk({key: {kk: vv for kk, vv in v.items()
                                     if kk not in ("g_samples",
                                                   "alld_share_samples")}
                               for key, v in comp_out.items()}),
            "attractors": {f"T{T}": {"attractors": a["attractors"][:8]}
                           for T, a in attractors.items()},
            "basins": gk(basins),
            "moran": {f"T{T}": {kk: vv for kk, vv in m.items() if kk != "freq"}
                      for T, m in moran.items()},
            "supported": supported,
            "traces": {"replicator": trajs, "moran": moran,
                       "reverse_traj": reverse_traj,
                       "comp_raw": comp_out, "ternary": tern,
                       "attractor_full": attractors,
                       "est_keys": list(est.keys()),
                       "Pi_mats": {key: est[key]["Pi"] for key in est}}}


# ================================================================== H11
# Keystone(핵심종) 검정 — "AdaptiveAgent 는 승자가 아니라 조력자" 프레임.
#
# H7(경쟁 열세)·H8E(g<0 침입 실패)는 'adaptive 가 스스로 이기는가' 를 기각했을
# 뿐이다. H11 은 'adaptive 가 남(협력자)을 이롭게 하는가' 를 정면으로 겨냥한다:
#   주장 A — 협력자에게 ALLD 대비 상대적 이점 (gap = coop payoff − ALLD payoff)
#   주장 B — 집단 협력률(CC) 증진
#
# 구성(composition) 인공물 차단: 비-swap 구성(특히 ALLD 30%)을 고정한 **치환
# 설계**. 핵심 대조 control-1 = fixed-λ ToMEmpathic (λ=0.4) — ToM·공감 기계를
# 동일하게 두고 λ 자기조절만 제거하므로 treatment−ctrl1 이 'λ 위계적 조절이
# 남을 돕는다' 는 가설의 순수 효과다.
H11_ARMS = {
    "treatment":         "adaptive",       # 정교형 AdaptiveAgent (all 귀인)
    "ctrl1_fixed_lambda": "tom_fixed",     # fixed-λ ToMEmpathic (λ=0.4) — 핵심 대조
    "ctrl2_immediate":   "adaptive_imm",   # 즉각형 (sophisticated=False)
    "ctrl3_extra_tft":   "swap_tft",       # 추가 TFT (조건부 협력자 증량 대조)
    "ctrl4_allc":        "swap_allc",      # ALLC (순진한 협력자 대조)
}
# 기저 조건부 협력자 라벨 (gap 계산의 분자; GTFT 확률/횟수 두 유형 모두 포함)
H11_COOP_LABELS = ("tit_for_tat", "generous_tft", "generous_tft_count", "wsls")


def _h11_swap_spec(arm: str, rounds: int, backend: str) -> dict:
    if arm == "treatment":
        return _m_adaptive(backend, sophisticated=True)
    if arm == "ctrl1_fixed_lambda":
        return agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp")
    if arm == "ctrl2_immediate":
        return _m_adaptive(backend, sophisticated=False)
    if arm == "ctrl3_extra_tft":
        return _m_strat("tit_for_tat", rounds)
    if arm == "ctrl4_allc":
        return _m_strat("allc", rounds)
    raise ValueError(arm)


def _h11_members(arm: str, n_swap: int, rounds: int, backend: str,
                 base_counts: dict, n_total: int):
    """치환 설계 집단: 비-swap 구성 고정 + swap 슬롯 + random filler 패딩."""
    members, labels = [], []
    for kind, cnt in base_counts.items():
        for _ in range(cnt):
            members.append(_m_strat(kind, rounds)); labels.append(kind)
    swap = _h11_swap_spec(arm, rounds, backend)
    for _ in range(n_swap):
        members.append(dict(swap)); labels.append(H11_ARMS[arm])
    n_fill = n_total - len(members)
    for _ in range(n_fill):
        members.append(_m_strat("random", rounds)); labels.append("random")
    assert len(members) == n_total
    return members, labels


def _h11_gap(o: dict, base_counts: dict) -> float:
    """주장 A 의 조작적 정의: 기저 조건부 협력자 가중평균 보수 − ALLD 보수."""
    blp = o["by_label_payoff"]
    w_tot = sum(base_counts[k] for k in H11_COOP_LABELS)
    coop = sum(blp.get(k, 0.0) * base_counts[k]
               for k in H11_COOP_LABELS) / max(w_tot, 1)
    return float(coop - blp.get("alld", 0.0))


def exp_H11(seeds, rounds, jobs, backend):
    """
    H11 [확증×2] Keystone 검정 — 치환 설계 + fixed-λ 대조 + 진화 프레임.

    집단 템플릿 (N=30, 완전 라운드로빈 — 매칭 잡음 제거):
      ALLD 9 (30%, 착취 압력 상수) + 조건부 협력자 12
      (TFT 4, GTFT확률 2, GTFT횟수 2, WSLS 4) + swap n∈{0,3,6,9} + random filler.
      dose 0 셀은 전 arm 공유(Δ≡0 앵커; CRN 짝지음은 stable_seed("H11pop",
      dose, rep) 를 arm 간 공유하여 달성).

    [확증·주장 B] ΔCC = CC(treat) − CC(ctrl1) 의 dose 기울기 > 0 (rep 군집 부트).
    [확증·주장 A] Δgap = gap(treat) − gap(ctrl1) 의 dose 기울기 > 0.
    [탐색] ALLD 억제·협력자 보호·이전(transfer) 회계 가드·즉각형/추가TFT/ALLC
           arm 대조·다이애드 기제(adaptive→ALLD DD 방어율, adaptive→협력자 CC
           유지율)·진화 프레임(ALLD 침입장벽 심화, 협력자 클러스터 침입성,
           후보 유형별 협력 유역 확장 — adaptive vs 고정전략).
    """
    LOGGER.info("[H11] Keystone 검정: 치환 설계 %d arms × dose + 진화 프레임",
                len(H11_ARMS))
    quick = seeds < 10
    if quick:
        n_total = 15
        base_counts = {"alld": 5, "tit_for_tat": 2, "generous_tft": 1,
                       "generous_tft_count": 1, "wsls": 2}
        doses = [0, 2, 4]
        reps = 2
        pi_seeds, n_basin, basin_steps, n_boot_basin = 4, 40, 300, 8
    else:
        n_total = 30
        base_counts = {"alld": 9, "tit_for_tat": 4, "generous_tft": 2,
                       "generous_tft_count": 2, "wsls": 4}
        doses = [0, 3, 6, 9]
        reps = max(4, min(12, seeds // 20))
        pi_seeds, n_basin, basin_steps, n_boot_basin = 30, 200, 500, 60

    # ---------------- 1) 치환 설계 집단 실행 ----------------
    all_specs, registry = [], {}

    def add(cell, rep, spec):
        registry[(cell, rep)] = len(all_specs)
        all_specs.append(spec)

    # dose 0: 전 arm 공유 (swap 없음 — arm 무관 동일 집단)
    m0, l0 = _h11_members("treatment", 0, rounds, backend, base_counts, n_total)
    for rep in range(reps):
        add(("dose0",), rep, {"members": m0, "labels": l0, "n_rounds": rounds,
                              "partners_per_agent": None,
                              "seed": stable_seed("H11pop", 0, rep)})
    for arm in H11_ARMS:
        for dose in doses[1:]:
            members, labels = _h11_members(arm, dose, rounds, backend,
                                           base_counts, n_total)
            for rep in range(reps):
                add((arm, dose), rep,
                    {"members": members, "labels": labels, "n_rounds": rounds,
                     "partners_per_agent": None,
                     "seed": stable_seed("H11pop", dose, rep)})  # arm 간 CRN 공유
    LOGGER.info("[H11] 집단 %d 개 병렬 실행 (reps=%d, doses=%s, N=%d 라운드로빈)",
                len(all_specs), reps, doses, n_total)
    t0 = time.time()
    out = run_populations(all_specs, n_jobs=jobs, verbose=True)
    LOGGER.info("[H11] 집단 실행 %.1fs", time.time() - t0)

    def O(cell, rep):
        return out[registry[(cell, rep)]]

    def cell_of(arm, dose):
        return ("dose0",) if dose == 0 else (arm, dose)

    # ---------------- 2) DV 추출 ----------------
    cc = {}; gap = {}; alld_pay = {}; coop_pay = {}
    transfer_share = {}
    for arm in H11_ARMS:
        for dose in doses:
            cell = cell_of(arm, dose)
            cc[(arm, dose)] = [O(cell, r)["cc_rate"] for r in range(reps)]
            gap[(arm, dose)] = [_h11_gap(O(cell, r), base_counts)
                                for r in range(reps)]
            alld_pay[(arm, dose)] = [O(cell, r)["by_label_payoff"]
                                     .get("alld", np.nan) for r in range(reps)]
            coop_pay[(arm, dose)] = {
                k: [O(cell, r)["by_label_payoff"].get(k, np.nan)
                    for r in range(reps)] for k in H11_COOP_LABELS}
            transfer_share[(arm, dose)] = [
                O(cell, r)["alld_exploit_gain_total"]
                / max(O(cell, r)["group_payoff"], 1e-9) for r in range(reps)]

    # ---------------- 3) 확증: Δ(treat−ctrl1) dose 기울기 ----------------
    def delta_slope(dv, arm_a, arm_b, seed_key):
        xs, ys, cl = [], [], []
        for dose in doses:
            for r in range(reps):
                xs.append(dose / n_total)
                ys.append(dv[(arm_a, dose)][r] - dv[(arm_b, dose)][r])
                cl.append(f"rep{r}")
        return slope_boot(xs, ys, cluster=np.array(cl),
                          seed=stable_seed("H11", seed_key))

    sl_cc = delta_slope(cc, "treatment", "ctrl1_fixed_lambda", "ccB")
    sl_gap = delta_slope(gap, "treatment", "ctrl1_fixed_lambda", "gapA")
    register_primary("H11", "주장B ΔCC(treat−fixedλ) dose 기울기 > 0",
                     sl_cc["p"], sl_cc["ci"][0] > 0,
                     f"β={sl_cc['slope']:.3f} "
                     f"[{sl_cc['ci'][0]:.3f}, {sl_cc['ci'][1]:.3f}]")
    register_primary("H11", "주장A Δgap(treat−fixedλ) dose 기울기 > 0",
                     sl_gap["p"], sl_gap["ci"][0] > 0,
                     f"β={sl_gap['slope']:.3f} "
                     f"[{sl_gap['ci'][0]:.3f}, {sl_gap['ci'][1]:.3f}]")

    # 다른 대조 arm 대비 Δ 기울기 (탐색)
    delta_slopes_ctrl = {}
    for arm in ("ctrl2_immediate", "ctrl3_extra_tft", "ctrl4_allc"):
        s_cc = delta_slope(cc, "treatment", arm, f"cc_{arm}")
        s_gp = delta_slope(gap, "treatment", arm, f"gap_{arm}")
        delta_slopes_ctrl[arm] = {"cc": s_cc, "gap": s_gp}
        register_exploratory("H11", f"ΔCC(treat−{arm}) dose 기울기", s_cc["p"],
                             f"β={s_cc['slope']:.3f} "
                             f"CI[{s_cc['ci'][0]:.3f},{s_cc['ci'][1]:.3f}]")

    # arm 별 절대 dose 기울기 (탐색; 곡선 시각화용)
    arm_slopes = {}
    for arm in H11_ARMS:
        xs, ys = [], []
        for dose in doses:
            for r in range(reps):
                xs.append(dose / n_total); ys.append(cc[(arm, dose)][r])
        arm_slopes[arm] = slope_boot(xs, ys, seed=stable_seed("H11abs", arm))

    # ---------------- 4) 탐색: 기제 (최대 dose 셀) ----------------
    dmax = doses[-1]
    # (i) ALLD 억제: payoff_ALLD(treat) < payoff_ALLD(ctrl1)
    supp = paired_stats(alld_pay[("treatment", dmax)],
                        alld_pay[("ctrl1_fixed_lambda", dmax)], "less",
                        seed=stable_seed("H11supp"))
    register_exploratory("H11", "ALLD 억제 (treat<fixedλ, 최대 dose)",
                         supp["p"], f"{fmt_es(supp['es'], 'dz')} "
                         f"(Δ={supp['mean_a'] - supp['mean_b']:+.3f})")
    # (ii) 협력자 보호: 유형별 payoff(treat) > payoff(ctrl1)
    protect = {}
    for k in H11_COOP_LABELS:
        t = paired_stats(coop_pay[("treatment", dmax)][k],
                         coop_pay[("ctrl1_fixed_lambda", dmax)][k], "greater",
                         seed=stable_seed("H11prot", k))
        protect[k] = {"delta": float(t["mean_a"] - t["mean_b"]), "p": t["p"]}
        register_exploratory("H11", f"협력자 보호[{k}] (treat>fixedλ)", t["p"],
                             f"Δ={t['mean_a'] - t['mean_b']:+.3f}")
    # (iii) 이전(transfer) 회계 가드: CC 상승이 착취 이전의 착시인지
    tr_t = float(np.mean(transfer_share[("treatment", dmax)]))
    tr_c = float(np.mean(transfer_share[("ctrl1_fixed_lambda", dmax)]))
    register_exploratory("H11", "이전 회계 가드 (착취 이전 비중)", 1.0,
                         f"treat={tr_t:.3f} vs fixedλ={tr_c:.3f} "
                         f"(감소 시 CC 상승이 실질)")
    # (iv) 다이애드 기제: adaptive→ALLD DD 방어율 / adaptive→협력자 CC 유지율
    def _mech(arm, actor):
        dd_alld, cc_coop = [], []
        for r in range(reps):
            db = O(cell_of(arm, dmax), r).get("dyad_behavior", {})
            key = f"{actor}→alld"
            if key in db:
                dd_alld.append(db[key]["dd"])
            ccs = [db[f"{actor}→{k}"]["cc"] for k in H11_COOP_LABELS
                   if f"{actor}→{k}" in db]
            if ccs:
                cc_coop.append(float(np.mean(ccs)))
        return dd_alld, cc_coop
    mech = {}
    for arm, actor in (("treatment", "adaptive"),
                       ("ctrl1_fixed_lambda", "tom_fixed"),
                       ("ctrl2_immediate", "adaptive_imm")):
        dd_alld, cc_coop = _mech(arm, actor)
        mech[actor] = {"dd_vs_alld": mean_sd(dd_alld),
                       "cc_vs_coop": mean_sd(cc_coop),
                       "dd_raw": dd_alld, "cc_raw": cc_coop}
    if mech["adaptive"]["dd_raw"] and mech["tom_fixed"]["dd_raw"]:
        m_dd = paired_stats(mech["adaptive"]["dd_raw"],
                            mech["tom_fixed"]["dd_raw"], "greater",
                            seed=stable_seed("H11dd"))
        register_exploratory("H11", "선택적 방어: adaptive→ALLD DD율 > fixedλ",
                             m_dd["p"],
                             f"{mech['adaptive']['dd_vs_alld'][0]:.2f} vs "
                             f"{mech['tom_fixed']['dd_vs_alld'][0]:.2f}")
        m_cc = paired_stats(mech["adaptive"]["cc_raw"],
                            mech["tom_fixed"]["cc_raw"], "two-sided",
                            seed=stable_seed("H11cc"))
        register_exploratory("H11", "협력 지탱: adaptive→협력자 CC율 (vs fixedλ)",
                             m_cc["p"],
                             f"{mech['adaptive']['cc_vs_coop'][0]:.2f} vs "
                             f"{mech['tom_fixed']['cc_vs_coop'][0]:.2f}")

    # ---------------- 5) 진화 프레임 (Π 추정 후 대수 연산) ----------------
    type_specs = {
        "tit_for_tat": _m_strat("tit_for_tat", rounds),
        "generous_tft": _m_strat("generous_tft", rounds),
        "generous_tft_count": _m_strat("generous_tft_count", rounds),
        "wsls": _m_strat("wsls", rounds),
        "allc": _m_strat("allc", rounds),
        "alld": _m_strat("alld", rounds),
        "random": _m_strat("random", rounds),
        "adaptive": _m_adaptive(backend, sophisticated=True),
        "tom_fixed": agent_spec("tom_empathic", 0,
                                use_pymdp=backend == "pymdp"),
    }
    t0 = time.time()
    est = evo.estimate_payoff_matrix(type_specs, n_rounds=rounds,
                                     seeds=pi_seeds, n_jobs=jobs,
                                     env_error=0.10, symmetric=True,
                                     seed_offset=41_000)
    LOGGER.info("[H11] Π(9유형, T=%d) 추정 %.1fs", rounds, time.time() - t0)
    names = est["names"]; raw = est["raw"]; cc_raw = est["CC_raw"]; k = len(names)
    i_alld = names.index("alld")

    coop_mix = {"tit_for_tat": .30, "generous_tft": .15,
                "generous_tft_count": .15, "wsls": .30, "allc": .10}
    res_coop = _resident_from_mix(names, coop_mix)

    def mixed_resident(extra: str, share: float = 0.20):
        w = (1 - share) * res_coop.copy()
        w[names.index(extra)] += share
        return w / w.sum()

    # (a) ALLD 침입장벽: 협력자-only vs 협력자+adaptive vs 협력자+fixedλ
    residents = {"coop_only": res_coop,
                 "coop+adaptive": mixed_resident("adaptive"),
                 "coop+tom_fixed": mixed_resident("tom_fixed")}
    barrier = {lbl: _growth_boot(raw, w, i_alld,
                                 seed=stable_seed("H11bar", lbl))
               for lbl, w in residents.items()}
    # 짝지은 부트: Δg = g_ALLD(coop_only) − g_ALLD(coop+adaptive) > 0
    #             (adaptive 를 섞으면 ALLD 성장률이 더 음수 = 장벽 심화)
    def _barrier_deepening_boot(extra, n_boot=2000):
        rng = np.random.default_rng(stable_seed("H11deep", extra))
        w_mix = residents[f"coop+{extra}"]
        d_obs = (evo.invasion_growth(raw.mean(axis=2), res_coop, i_alld)
                 - evo.invasion_growth(raw.mean(axis=2), w_mix, i_alld))
        db = []
        for _ in range(n_boot):
            idx = rng.integers(0, pi_seeds, pi_seeds)
            Pi_b = raw[:, :, idx].mean(axis=2)
            db.append(evo.invasion_growth(Pi_b, res_coop, i_alld)
                      - evo.invasion_growth(Pi_b, w_mix, i_alld))
        db = np.asarray(db)
        ci = [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]
        pv = float(np.clip(2 * min((db <= 0).mean(), (db >= 0).mean()),
                           1.0 / n_boot, 1.0))
        return {"delta": float(d_obs), "ci": ci, "p": pv}
    deepening = {extra: _barrier_deepening_boot(extra)
                 for extra in ("adaptive", "tom_fixed")}
    register_exploratory(
        "H11", "ALLD 침입장벽 심화 (adaptive 혼합)", deepening["adaptive"]["p"],
        f"Δg={deepening['adaptive']['delta']:+.3f} "
        f"CI[{deepening['adaptive']['ci'][0]:.3f},"
        f"{deepening['adaptive']['ci'][1]:.3f}] "
        f"(fixedλ 대조 Δg={deepening['tom_fixed']['delta']:+.3f})")

    # (b) 협력자 클러스터 침입성: ALLD-heavy 상주에 클러스터 침입
    res_alld = _resident_from_mix(names, {"alld": 0.8, "random": 0.2})
    w_cl_only = res_coop
    w_cl_ad = mixed_resident("adaptive")
    def _cluster_boot(w_cluster, key, n_boot=2000):
        rng = np.random.default_rng(stable_seed("H11cl", key))
        g_obs = evo.cluster_invasion_growth(raw.mean(axis=2), res_alld,
                                            w_cluster)
        gb = np.asarray([evo.cluster_invasion_growth(
            raw[:, :, rng.integers(0, pi_seeds, pi_seeds)].mean(axis=2),
            res_alld, w_cluster) for _ in range(n_boot)])
        return {"g": float(g_obs),
                "ci": [float(np.percentile(gb, 2.5)),
                       float(np.percentile(gb, 97.5))],
                "p": float(np.clip(2 * min((gb <= 0).mean(),
                                           (gb >= 0).mean()),
                                   1.0 / n_boot, 1.0))}
    cluster_inv = {"coop_only": _cluster_boot(w_cl_only, "only"),
                   "coop+adaptive": _cluster_boot(w_cl_ad, "ad")}
    register_exploratory(
        "H11", "협력자 클러스터 침입성 (ALLD-heavy 상주)",
        cluster_inv["coop+adaptive"]["p"],
        f"g(only)={cluster_inv['coop_only']['g']:+.3f} vs "
        f"g(+adaptive)={cluster_inv['coop+adaptive']['g']:+.3f}")

    # (c) 후보 유형별 협력 CC-widening: adaptive vs 고정전략들 (벡터화 부트)
    # [지표 개정 A] 이분 유역 → 종착 조성의 행동적 CC율 xᵀ·CCm·x.
    candidates = ["adaptive", "tom_fixed", "tit_for_tat", "generous_tft",
                  "generous_tft_count", "wsls"]
    def _widening_all(Pi_full, CC_full, rng):
        """공통 초기점으로 후보별 CC-widening(X) = CC(S) − CC(S∖X) (행동적 CC율)."""
        X0_full = rng.dirichlet(np.ones(k), size=n_basin)
        b_full = evo.re_terminal_cc(Pi_full, CC_full, X0_full, steps=basin_steps)
        X0_red = rng.dirichlet(np.ones(k - 1), size=n_basin)
        wid = {}
        for cand in candidates:
            keep = [i for i in range(k) if names[i] != cand]
            b_wo = evo.re_terminal_cc(Pi_full[np.ix_(keep, keep)],
                                      CC_full[np.ix_(keep, keep)], X0_red,
                                      steps=basin_steps)
            wid[cand] = b_full - b_wo
        return wid
    rng_w = np.random.default_rng(stable_seed("H11wid"))
    wid_pt = _widening_all(raw.mean(axis=2), cc_raw.mean(axis=2), rng_w)
    wid_boot = {c: [] for c in candidates}
    for _ in range(n_boot_basin):
        idx = rng_w.integers(0, pi_seeds, pi_seeds)
        wb = _widening_all(raw[:, :, idx].mean(axis=2),
                           cc_raw[:, :, idx].mean(axis=2), rng_w)
        for c in candidates:
            wid_boot[c].append(wb[c])
    widening = {}
    for c in candidates:
        b = np.asarray(wid_boot[c])
        widening[c] = {"est": float(wid_pt[c]),
                       "ci": [float(np.percentile(b, 2.5)),
                              float(np.percentile(b, 97.5))]}
    # adaptive 의 유역 확장 > 각 고정전략 (짝지은 부트 — 동일 boot 인덱스 순회)
    wid_vs_fixed = {}
    for c in [x for x in candidates if x != "adaptive"]:
        db = np.asarray(wid_boot["adaptive"]) - np.asarray(wid_boot[c])
        pv = float(np.clip(2 * min((db <= 0).mean(), (db >= 0).mean()),
                           1.0 / max(n_boot_basin, 1), 1.0))
        wid_vs_fixed[c] = {
            "delta": float(wid_pt["adaptive"] - wid_pt[c]),
            "ci": [float(np.percentile(db, 2.5)),
                   float(np.percentile(db, 97.5))],
            "p": pv}
        register_exploratory(
            "H11", f"협력 CC-widening: adaptive vs {c}", pv,
            f"Δ={wid_vs_fixed[c]['delta']:+.3f} "
            f"CI[{wid_vs_fixed[c]['ci'][0]:.3f},{wid_vs_fixed[c]['ci'][1]:.3f}]")

    # ---------------- 6) 판정·반환 ----------------
    supported = {
        "claim_B_cc_slope": bool(sl_cc["ci"][0] > 0),
        "claim_A_gap_slope": bool(sl_gap["ci"][0] > 0),
        "keystone_joint": bool(sl_cc["ci"][0] > 0 and sl_gap["ci"][0] > 0),
        "alld_suppression": bool(supp["mean_a"] < supp["mean_b"]),
        "barrier_deepening_adaptive": bool(deepening["adaptive"]["ci"][0] > 0),
        "widening_gt_all_fixed": bool(all(
            v["delta"] > 0 for v in wid_vs_fixed.values())),
    }
    LOGGER.info("[H11] → %s", supported)
    curves_cc = {arm: {str(d): [mean_sd(cc[(arm, d)])[0],
                                boot_mean_ci(cc[(arm, d)],
                                             seed=stable_seed("H11ci", arm, d))
                                ["ci"]] for d in doses} for arm in H11_ARMS}
    curves_gap = {arm: {str(d): [mean_sd(gap[(arm, d)])[0],
                                 boot_mean_ci(gap[(arm, d)],
                                              seed=stable_seed("H11cig", arm,
                                                               d))["ci"]]
                        for d in doses} for arm in H11_ARMS}
    return {"arms": {a: H11_ARMS[a] for a in H11_ARMS},
            "doses": doses, "n_total": n_total, "reps": reps,
            "base_counts": base_counts,
            "curves_cc": curves_cc, "curves_gap": curves_gap,
            "primary_B_cc_slope": sl_cc, "primary_A_gap_slope": sl_gap,
            "delta_slopes_ctrl": delta_slopes_ctrl,
            "arm_abs_slopes": arm_slopes,
            "alld_suppression": {"test": {kk: supp[kk] for kk in
                                          ("p", "mean_a", "mean_b")},
                                 "by_dose": {
                                     arm: {str(d): mean_sd(alld_pay[(arm, d)])
                                           for d in doses}
                                     for arm in ("treatment",
                                                 "ctrl1_fixed_lambda")}},
            "cooperator_protection": protect,
            "coop_pay_max_dose": {
                arm: {kk: mean_sd(coop_pay[(arm, dmax)][kk])
                      for kk in H11_COOP_LABELS}
                for arm in ("treatment", "ctrl1_fixed_lambda")},
            "transfer_guard": {"treatment": tr_t, "ctrl1_fixed_lambda": tr_c},
            "mechanism": {a: {kk: v for kk, v in m.items()
                              if not kk.endswith("_raw")}
                          for a, m in mech.items()},
            "evolution": {"pi_seeds": pi_seeds, "names": names,
                          "alld_barrier": barrier,
                          "barrier_deepening": deepening,
                          "cluster_invasion": cluster_inv,
                          "basin_widening": widening,
                          "widening_vs_fixed": wid_vs_fixed},
            "supported": supported,
            "traces": {"mech": mech, "Pi": est["Pi"]}}


# ================================================================== H12
# H12 — 상주 조건부 Keystone 프런티어 (설계: docs/H12_resident_conditioned_keystone_DESIGN.md)
#
# 배경: adaptive 의 '협력 유역 확장' 기여가 상주 구성에 따라 부호가 뒤집힌다
#       (H8E: Δ=+0.67 @T=240 / H11: adaptive < TFT). H12 는 이 상충을
#       (1) 인공물(차원 재정규화·중복·지지 불일치)을 지표 재정의로 제거하고
#       (2) 실질 성분을 '상주 구성을 독립변인으로 하는 반응 곡면' 으로 정량화해
#       adaptive 가 keystone 이 되는 조건 경계를 확정한다.
#
# 설계 규범 (§3):
#   원칙 A — 고정 최대 유형집합 S*(11종) 위 단일 심플렉스. 상주 = 그 위 분포.
#   원칙 B — 중립 filler(random) 대치로 유형 기여 측정 (제거 아님).
#   원칙 C — 위협축(ρ_D·ρ_C·err)·중복축(협력자 다양성 m)의 명시적 매개변수화.
#   원칙 D — estimand 를 고정 Π*·고정 초기점 사전 위 하나의 범함수로 못박음.
#   원칙 E — 균등(interior) 사전과 ALLD-heavy 구조적(structural) 사전 병기.
#
# 확증 지표 (§6.1): [C1] 위협 단조성(R-P 기울기>0) · [C2] 중복 단조성
#   (R-D 비대체성 기울기<0) · [C3] 교차 경계 존재(R-D keystone 프런티어 m*).
H12_CANON = ["tit_for_tat", "generous_tft", "generous_tft_count", "wsls",
             "allc", "alld", "random", "capricious",
             "adaptive", "adaptive_imm", "tom_fixed"]          # S* (11종)
H12_NONCOOP = ("alld", "random", "capricious")                  # 위협·중립
H12_FIXED_COOP = ["tit_for_tat", "generous_tft", "generous_tft_count",
                  "wsls", "allc"]                               # best_fixed 후보
H12_M_PROFILES = {                                              # 중복 축 m
    1: ["tit_for_tat"],
    2: ["tit_for_tat", "generous_tft"],
    3: ["tit_for_tat", "generous_tft", "wsls"],
    5: ["tit_for_tat", "generous_tft", "wsls",
        "generous_tft_count", "allc"],
}


def _h12_type_specs(T, backend):
    """정준 지지집합 S*(11종) 의 Π* 추정용 유형 스펙."""
    return {
        "tit_for_tat": _m_strat("tit_for_tat", T),
        "generous_tft": _m_strat("generous_tft", T),
        "generous_tft_count": _m_strat("generous_tft_count", T),
        "wsls": _m_strat("wsls", T),
        "allc": _m_strat("allc", T),
        "alld": _m_strat("alld", T),
        "random": _m_strat("random", T),
        "capricious": _m_strat("capricious", T),
        "adaptive": _m_adaptive(backend, sophisticated=True),
        "adaptive_imm": _m_adaptive(backend, sophisticated=False),
        "tom_fixed": agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp"),
    }


def _h12_resident_mean(names, rho_D, rho_C, m_profile, slot_type, slot_share):
    """
    §4.1 배경 조성 + 대치 슬롯 → S* 위 상주 조성(초기점 사전 평균).
      배경 = ρ_D·ALLD + ρ_C·capricious + (1−ρ_D−ρ_C−s)·[다양성 프로파일 m]
      대치 슬롯 = s·e_{slot_type}   (슬롯 지분 s 는 **협력자 몫에서만** 차감 —
                                     ALLD·capricious 지분 불변, §4.1)
    """
    w = np.zeros(len(names))

    def ix(nm):
        return names.index(nm)

    w[ix("alld")] += rho_D
    w[ix("capricious")] += rho_C
    coop_budget = max(1.0 - rho_D - rho_C, 0.0)
    m_budget = max(coop_budget - slot_share, 0.0)
    if m_profile:
        for nm in m_profile:
            w[ix(nm)] += m_budget / len(m_profile)
    else:
        w[ix("random")] += m_budget
    w[ix(slot_type)] += slot_share
    s = w.sum()
    return w / s if s > 0 else np.ones(len(names)) / len(names)


def _h12_prior_points(U0, resident_mean, blend, alld_idx=None, corner=0.0):
    """
    공통 U0(Dirichlet(1) 배치)를 상주 평균으로 결정적 대응(원칙 D·5.4):
        X0 = (1−blend)·U0 + blend·mean   (mean 은 슬롯이 인코딩된 상주 조성)
    corner>0 이면 구조적 사전(원칙 E): mean 을 ALLD 코너로 당긴다
        mean ← corner·e_ALLD + (1−corner)·mean.
    U0 를 전 상주·슬롯·부트에 공유하므로 '슬롯만 바뀌는' 짝지음이 성립한다.
    """
    mean = resident_mean.copy()
    if corner > 0 and alld_idx is not None:
        e = np.zeros_like(mean)
        e[alld_idx] = 1.0
        mean = corner * e + (1.0 - corner) * mean
    X0 = (1.0 - blend) * U0 + blend * mean[None, :]
    X0 = np.clip(X0, 1e-12, None)
    return X0 / X0.sum(axis=1, keepdims=True)


def exp_H12(seeds, rounds, jobs, backend):
    """
    H12 [확증×3] 상주 조건부 Keystone 프런티어.

    Π* 격자 : env_error ∈ {0,.05,.10,.15,.20} × T ∈ {60,240}, 정준 지지집합
              S*(11종), 대칭 재사용, pi_seeds≥30 (원칙 A — 1회 캐시·전 상주 공유).

    지표 재정의 (§5.1, v0.6.1): CC-widening(X | 배경 B, 슬롯 s)
        = re_terminal_cc(Π*, CCm, P_B, 슬롯=X) − re_terminal_cc(Π*, CCm, P_B, 슬롯=random)
      종착 조성의 행동적 CC율 CC(x)=xᵀ·CCm·x (이분 유역 폐기; 라벨-행동 괴리 제거).
      지지집합은 항상 S*(11종) 불변 — 슬롯 유형만 X↔random. 슬롯은 초기점 사전
      P_B 의 평균 성분으로 인코딩되어 차원·척도가 완전히 고정된다(2.1–2.3 제거).

    [확증] C1 위협 단조성(R-P ρ_D 기울기 > 0) · C2 중복 단조성(R-D 다양성
           기울기 < 0) · C3 교차 경계 m* 존재(R-D keystone 프런티어).
    [탐색] R-K(변덕)·R-N(잡음) 축 · Shapley φ(adaptive) vs φ(TFT)(순서 무관
           기여, C={ad,TFT,GTFT,WSLS}) · ALLD 침입장벽 심화 Δg(random 앵커) ·
           구조적 사전(원칙 E) 결론 일치 · H8E/H11 좌표 정위(triangulate) ·
           R-K 변덕 대응의 정교(adaptive) vs 즉각(adaptive_imm) 분해.
    """
    import math
    from itertools import combinations

    LOGGER.info("[H12] 상주 조건부 Keystone 프런티어 — 고정 심플렉스 S*(11종) 반응 곡면")
    quick = seeds < 10
    pi_seeds = 4 if quick else 30
    env_errors = [0.10] if quick else [0.0, 0.05, 0.10, 0.15, 0.20]
    req = sorted(set(rounds)) if isinstance(rounds, (list, tuple)) else [int(rounds)]
    Ts = req[:1] if quick else sorted(set([60, 240]) | set(req))
    primary_err = 0.10 if 0.10 in env_errors else env_errors[0]
    T_main = Ts[-1]                          # keystone 은 긴 지평(H8E +0.67 @T=240)
    slot_share = 0.20                        # 대치 슬롯 지분 (§4.1 예시)
    blend = 0.30                             # interior 사전(원칙 E: 대역 질문)
    corner_struct, blend_struct = 0.60, 0.45  # 구조적 사전(원칙 E: 침입 질문)
    n_basin = 40 if quick else 300
    basin_steps = 250 if quick else 500
    n_boot = 40 if quick else 300            # basin 곡면 부트 (Π-시드 CRN)
    n_boot_g = 200 if quick else 2000        # 침입장벽 부트 (저비용)
    n_boot_s = 20 if quick else 120          # 구조적 사전·Shapley 부트

    # ---------------- 1) Π*(err × T) 추정 (원칙 A) ----------------
    est = {}
    for ti, T in enumerate(Ts):
        for ei, err in enumerate(env_errors):
            t0 = time.time()
            est[(err, T)] = evo.estimate_payoff_matrix(
                _h12_type_specs(T, backend), n_rounds=T, seeds=pi_seeds,
                n_jobs=jobs, env_error=err, symmetric=True,
                seed_offset=61_000 + 1000 * (ti * len(env_errors) + ei))
            LOGGER.info("[H12] Π*(err=%.2f, T=%d, 11종) 추정 %.1fs", err, T,
                        time.time() - t0)
    names = est[(primary_err, T_main)]["names"]
    k = len(names)
    coop_idx = [i for i, nm in enumerate(names) if nm not in H12_NONCOOP]
    i_alld = names.index("alld")
    raw_main = est[(primary_err, T_main)]["raw"]
    cc_raw_main = est[(primary_err, T_main)]["CC_raw"]
    Pi_full_main = raw_main.mean(axis=2)
    CC_full_main = cc_raw_main.mean(axis=2)

    # 공통 초기점 U0 (원칙 D·5.4 — 전 상주·슬롯·부트 공유)
    U0 = np.random.default_rng(stable_seed("H12U0")).dirichlet(np.ones(k),
                                                               size=n_basin)

    def basin_slot(Pi, CCm, rho_D, rho_C, m_profile, slot, corner=0.0, bl=None):
        # [지표 개정 A] 이분 유역 → 종착 조성의 행동적 CC율 xᵀ·CCm·x.
        rm = _h12_resident_mean(names, rho_D, rho_C, m_profile, slot, slot_share)
        X0 = _h12_prior_points(U0, rm, blend if bl is None else bl,
                               alld_idx=i_alld, corner=corner)
        return evo.re_terminal_cc(Pi, CCm, X0, steps=basin_steps)

    # ---------------- 2) 축 곡면 부트 엔진 (Π-시드 CRN) ----------------
    def axis_surface(raw, cc_raw, coord_specs, slots, seed, corner=0.0, bl=None,
                     n_b=n_boot):
        coords = [c for c, _ in coord_specs]
        S = len(slots)
        Pi_f = raw.mean(axis=2); CC_f = cc_raw.mean(axis=2)
        pt = np.array([[basin_slot(Pi_f, CC_f, cs["rho_D"], cs["rho_C"],
                                   cs["m_profile"], sl, corner, bl)
                        for sl in slots] for _, cs in coord_specs])   # (C,S)
        rng = np.random.default_rng(seed)
        ps = raw.shape[2]
        boot = np.empty((n_b, len(coords), S))
        for b in range(n_b):
            idx = rng.integers(0, ps, ps)               # Π·CCm 공통 시드(CRN)
            Pi_b = raw[:, :, idx].mean(axis=2)
            CC_b = cc_raw[:, :, idx].mean(axis=2)
            for cidx, (_, cs) in enumerate(coord_specs):
                for si, sl in enumerate(slots):
                    boot[b, cidx, si] = basin_slot(Pi_b, CC_b, cs["rho_D"], cs["rho_C"],
                                                   cs["m_profile"], sl, corner, bl)
        return {"coords": coords, "slots": list(slots), "point": pt, "boot": boot}

    def si(surf, sl):
        return surf["slots"].index(sl)

    def marginal(surf, X):                    # sub_widening(X) = basin(X) − basin(random)
        xi, ri = si(surf, X), si(surf, "random")
        return surf["point"][:, xi] - surf["point"][:, ri], \
            surf["boot"][:, :, xi] - surf["boot"][:, :, ri]

    def irreplace(surf, X, Y):                # sub_widening(X) − sub_widening(Y) (random 상쇄)
        xi, yi = si(surf, X), si(surf, Y)
        return surf["point"][:, xi] - surf["point"][:, yi], \
            surf["boot"][:, :, xi] - surf["boot"][:, :, yi]

    def keystone_delta(surf):                 # basin(adaptive) − max_fixed basin(fixed)
        ai = si(surf, "adaptive")
        fis = [si(surf, f) for f in H12_FIXED_COOP]
        pt = surf["point"][:, ai] - surf["point"][:, fis].max(axis=1)
        bt = surf["boot"][:, :, ai] - surf["boot"][:, :, fis].max(axis=2)
        return pt, bt

    def slope_ci(coords, bt, pt):
        c = np.asarray(coords, float)
        sl_obs = float(np.polyfit(c, pt, 1)[0])
        sls = np.array([np.polyfit(c, bt[b], 1)[0] for b in range(bt.shape[0])])
        ci = [float(np.percentile(sls, 2.5)), float(np.percentile(sls, 97.5))]
        p = float(np.clip(2 * min((sls <= 0).mean(), (sls >= 0).mean()),
                          1.0 / len(sls), 1.0))
        return {"slope": sl_obs, "ci": ci, "p": p, "coords": list(coords),
                "curve": [float(x) for x in pt],
                "curve_ci": [[float(np.percentile(bt[:, c2], 2.5)),
                              float(np.percentile(bt[:, c2], 97.5))]
                             for c2 in range(bt.shape[1])]}

    def crossing_ci(coords, bt, pt):
        c = np.asarray(coords, float)

        def root(v):
            for i in range(len(c) - 1):
                a, b2 = v[i], v[i + 1]
                if a == 0:
                    return float(c[i])
                if (a < 0) != (b2 < 0):
                    return float(c[i] + (a / (a - b2)) * (c[i + 1] - c[i]))
            return None
        r_obs = root(pt)
        roots = [root(bt[b]) for b in range(bt.shape[0])]
        found = [x for x in roots if x is not None]
        frac = len(found) / max(len(roots), 1)
        ci = ([float(np.percentile(found, 2.5)),
               float(np.percentile(found, 97.5))] if found
              else [float("nan"), float("nan")])
        within = (r_obs is not None and c.min() <= r_obs <= c.max())
        return {"crossing": r_obs, "ci": ci, "frac_found": float(frac),
                "within_range": bool(within),
                "range": [float(c.min()), float(c.max())]}

    def pack_axis(surf):
        ai, ri = si(surf, "adaptive"), si(surf, "random")
        fis = [si(surf, f) for f in H12_FIXED_COOP]

        def ci_of(colfn):
            arr = np.array([colfn(surf["boot"][b])
                            for b in range(surf["boot"].shape[0])])
            return [[float(np.percentile(arr[:, c2], 2.5)),
                     float(np.percentile(arr[:, c2], 97.5))]
                    for c2 in range(arr.shape[1])]
        m_pt, m_bt = marginal(surf, "adaptive")
        marg = slope_ci(surf["coords"], m_bt, m_pt)
        return {
            "coords": list(surf["coords"]),
            "basin_adaptive": [float(x) for x in surf["point"][:, ai]],
            "ci_adaptive": ci_of(lambda B: B[:, ai]),
            "basin_random": [float(x) for x in surf["point"][:, ri]],
            "ci_random": ci_of(lambda B: B[:, ri]),
            "basin_best_fixed": [float(x) for x in surf["point"][:, fis].max(axis=1)],
            "ci_best_fixed": ci_of(lambda B: B[:, fis].max(axis=1)),
            "marginal_adaptive": {"curve": marg["curve"], "slope": marg["slope"],
                                  "ci": marg["ci"], "curve_ci": marg["curve_ci"]},
        }

    # ---------------- 3) 축별 상주족 (§4.2) ----------------
    m_def = H12_M_PROFILES[3]                 # TFT+GTFT_prob+WSLS (기본 중복 프로파일)
    slots_core = ["adaptive", "random", "tit_for_tat", "generous_tft",
                  "generous_tft_count", "wsls", "allc"]
    slots_rk = slots_core + ["adaptive_imm"]

    RP = [(rd, dict(rho_D=rd, rho_C=0.0, m_profile=m_def))
          for rd in ([0.0, 0.2, 0.4] if quick else [0.0, 0.1, 0.2, 0.3, 0.4])]
    RK = [(rc, dict(rho_D=0.2, rho_C=rc, m_profile=m_def))
          for rc in ([0.0, 0.2] if quick else [0.0, 0.1, 0.2, 0.3])]
    div_counts = [1, 3] if quick else [1, 2, 3, 5]
    RD = [(dc, dict(rho_D=0.2, rho_C=0.1, m_profile=H12_M_PROFILES[dc]))
          for dc in div_counts]

    t0 = time.time()
    surf_RP = axis_surface(raw_main, cc_raw_main, RP, slots_core, stable_seed("H12RP"))
    surf_RK = axis_surface(raw_main, cc_raw_main, RK, slots_rk, stable_seed("H12RK"))
    surf_RD = axis_surface(raw_main, cc_raw_main, RD, slots_core, stable_seed("H12RD"))
    LOGGER.info("[H12] R-P/R-K/R-D 곡면 부트 %.1fs", time.time() - t0)

    # R-N (env 잡음 축 — 좌표마다 다른 Π*(err) 재사용; §4.2 R-N)
    def rn_surface(seed):
        cs = dict(rho_D=0.2, rho_C=0.1, m_profile=m_def)
        rng = np.random.default_rng(seed)
        pt, boot = [], []
        for err in env_errors:
            raw_e = est[(err, T_main)]["raw"]
            cc_raw_e = est[(err, T_main)]["CC_raw"]
            ps = raw_e.shape[2]
            Pi_f = raw_e.mean(axis=2); CC_f = cc_raw_e.mean(axis=2)
            pt.append([basin_slot(Pi_f, CC_f, cs["rho_D"], cs["rho_C"],
                                  cs["m_profile"], sl) for sl in slots_core])
            bb = np.empty((n_boot, len(slots_core)))
            for b in range(n_boot):
                idx = rng.integers(0, ps, ps)
                Pi_b = raw_e[:, :, idx].mean(axis=2)
                CC_b = cc_raw_e[:, :, idx].mean(axis=2)
                bb[b] = [basin_slot(Pi_b, CC_b, cs["rho_D"], cs["rho_C"],
                                    cs["m_profile"], sl) for sl in slots_core]
            boot.append(bb)
        return {"coords": list(env_errors), "slots": list(slots_core),
                "point": np.array(pt), "boot": np.stack(boot, axis=1)}
    surf_RN = rn_surface(stable_seed("H12RN")) if len(env_errors) > 1 else None

    # ---------------- 4) 확증 지표 C1·C2·C3 ----------------
    pt, bt = marginal(surf_RP, "adaptive")
    c1 = slope_ci(surf_RP["coords"], bt, pt)
    register_primary("H12", "[C1] 위협축(R-P ρ_D) 한계 keystone 기울기 > 0",
                     c1["p"], c1["ci"][0] > 0,
                     f"β={c1['slope']:+.3f} [{c1['ci'][0]:+.3f},{c1['ci'][1]:+.3f}]")

    pt2, bt2 = irreplace(surf_RD, "adaptive", "tit_for_tat")
    c2 = slope_ci(surf_RD["coords"], bt2, pt2)
    register_primary("H12", "[C2] 중복축(R-D 다양성) 비대체성 기울기 < 0",
                     c2["p"], c2["ci"][1] < 0,
                     f"β={c2['slope']:+.3f} [{c2['ci'][0]:+.3f},{c2['ci'][1]:+.3f}]")

    ptk, btk = keystone_delta(surf_RD)
    c3 = crossing_ci(surf_RD["coords"], btk, ptk)
    c3_p = float(np.clip(1.0 - c3["frac_found"], 1.0 / max(n_boot, 1), 1.0))
    register_primary(
        "H12", "[C3] R-D keystone 프런티어(교차 m*) 존재",
        c3_p, bool(c3["within_range"] and c3["frac_found"] > 0.5),
        (f"m*={c3['crossing']:.2f} CI[{c3['ci'][0]:.2f},{c3['ci'][1]:.2f}] "
         f"(부트 교차검출 {c3['frac_found']:.0%})" if c3["within_range"]
         else f"관측 범위 {c3['range']} 내 교차 부재 (외삽 금지, §6.3)"))

    # keystone_delta 곡선(그림용) — R-D·R-P
    def pack_keystone(surf):
        pt_, bt_ = keystone_delta(surf)
        return {"coords": list(surf["coords"]),
                "curve": [float(x) for x in pt_],
                "curve_ci": [[float(np.percentile(bt_[:, c2], 2.5)),
                              float(np.percentile(bt_[:, c2], 97.5))]
                             for c2 in range(bt_.shape[1])]}
    ks_RD = pack_keystone(surf_RD)
    ks_RP = pack_keystone(surf_RP)
    c3_RP = crossing_ci(surf_RP["coords"], *keystone_delta(surf_RP)[::-1])

    # ---------------- 5) 탐색: R-K·R-N 축 기울기 ----------------
    ptk_rk, btk_rk = marginal(surf_RK, "adaptive")
    c1_rk = slope_ci(surf_RK["coords"], btk_rk, ptk_rk)
    register_exploratory("H12", "[C1보조] 위협축(R-K ρ_C) 한계 keystone 기울기",
                         c1_rk["p"], f"β={c1_rk['slope']:+.3f} "
                         f"CI[{c1_rk['ci'][0]:+.3f},{c1_rk['ci'][1]:+.3f}]")
    # R-K 변덕 대응 분해: 정교(adaptive) vs 즉각(adaptive_imm) — H5 연결 (§6.2)
    pti, bti = marginal(surf_RK, "adaptive_imm")
    c1_rk_imm = slope_ci(surf_RK["coords"], bti, pti)
    register_exploratory("H12", "R-K 변덕대응 정교(soph) vs 즉각(imm) 기울기 분해",
                         1.0, f"β_soph={c1_rk['slope']:+.3f} vs "
                         f"β_imm={c1_rk_imm['slope']:+.3f} "
                         "(정교>즉각이면 rmPFC 정교함 기여)")
    rn_pack = None
    if surf_RN is not None:
        pt_rn, bt_rn = marginal(surf_RN, "adaptive")
        c1_rn = slope_ci(surf_RN["coords"], bt_rn, pt_rn)
        register_exploratory("H12", "위협축(R-N env 잡음) 한계 keystone 기울기",
                             c1_rn["p"], f"β={c1_rn['slope']:+.3f} "
                             f"CI[{c1_rn['ci'][0]:+.3f},{c1_rn['ci'][1]:+.3f}]")
        rn_pack = pack_axis(surf_RN)
        rn_pack["c1_slope"] = c1_rn

    # ---------------- 6) 탐색: Shapley 한계 기여 (§5.2) ----------------
    C_set = ["adaptive", "tit_for_tat", "generous_tft", "wsls"]
    i_cap = names.index("capricious")

    def basin_coalition(Pi, CCm, coalition, rho_D=0.2, rho_C=0.1):
        w = np.zeros(k)
        w[i_alld] += rho_D
        w[i_cap] += rho_C
        budget = max(1.0 - rho_D - rho_C, 0.0)
        if coalition:
            for nm in coalition:
                w[names.index(nm)] += budget / len(coalition)
        else:
            w[names.index("random")] += budget
        w = w / w.sum()
        X0 = _h12_prior_points(U0, w, blend, alld_idx=i_alld, corner=0.0)
        return evo.re_terminal_cc(Pi, CCm, X0, steps=basin_steps)

    def shapley(Pi, CCm):
        phi = {x: 0.0 for x in C_set}
        n = len(C_set)
        for x in C_set:
            oc = [c for c in C_set if c != x]
            for r in range(len(oc) + 1):
                for Tsub in combinations(oc, r):
                    wgt = (math.factorial(len(Tsub))
                           * math.factorial(n - len(Tsub) - 1)
                           / math.factorial(n))
                    phi[x] += wgt * (basin_coalition(Pi, CCm, list(Tsub) + [x])
                                     - basin_coalition(Pi, CCm, list(Tsub)))
        return phi
    t0 = time.time()
    phi_pt = shapley(Pi_full_main, CC_full_main)
    rng_sh = np.random.default_rng(stable_seed("H12shap"))
    phi_boot = {x: [] for x in C_set}
    for _ in range(n_boot_s):
        idx = rng_sh.integers(0, pi_seeds, pi_seeds)
        Pi_b = raw_main[:, :, idx].mean(axis=2)
        CC_b = cc_raw_main[:, :, idx].mean(axis=2)
        pb = shapley(Pi_b, CC_b)
        for x in C_set:
            phi_boot[x].append(pb[x])
    LOGGER.info("[H12] Shapley(2^4) 부트 %.1fs", time.time() - t0)
    dphi = np.array(phi_boot["adaptive"]) - np.array(phi_boot["tit_for_tat"])
    shap = {
        "C_set": C_set,
        "phi": {x: float(phi_pt[x]) for x in C_set},
        "phi_ci": {x: [float(np.percentile(phi_boot[x], 2.5)),
                       float(np.percentile(phi_boot[x], 97.5))] for x in C_set},
        "adaptive_minus_tft": {
            "delta": float(phi_pt["adaptive"] - phi_pt["tit_for_tat"]),
            "ci": [float(np.percentile(dphi, 2.5)),
                   float(np.percentile(dphi, 97.5))],
            "p": float(np.clip(2 * min((dphi <= 0).mean(), (dphi >= 0).mean()),
                               1.0 / len(dphi), 1.0))}}
    register_exploratory(
        "H12", "Shapley φ(adaptive) − φ(TFT) (C={ad,TFT,GTFT,WSLS}, 순서무관)",
        shap["adaptive_minus_tft"]["p"],
        f"Δφ={shap['adaptive_minus_tft']['delta']:+.3f} "
        f"CI[{shap['adaptive_minus_tft']['ci'][0]:+.3f},"
        f"{shap['adaptive_minus_tft']['ci'][1]:+.3f}]")

    # ---------------- 7) 탐색: ALLD 침입장벽 치환 회계 (§5.3) ----------------
    def barrier_family(raw_e, rho_D, rho_C, m_profile, key):
        out = {}
        for X in ("random", "adaptive", "tom_fixed"):
            w = _h12_resident_mean(names, rho_D, rho_C, m_profile, X, slot_share)
            out[X] = _growth_boot(raw_e, w, i_alld, n_boot=n_boot_g,
                                  seed=stable_seed("H12bar", key, X))
        # Δg = g(random 슬롯) − g(X 슬롯): 양수면 X 가 장벽 심화 (전략 효과 순수 귀속)
        out["deepen_adaptive"] = float(out["random"]["g"] - out["adaptive"]["g"])
        out["deepen_tom_fixed"] = float(out["random"]["g"] - out["tom_fixed"]["g"])
        return out
    barrier = {
        "R-P_high(ρ_D=0.4)": barrier_family(raw_main, 0.4, 0.0, m_def, "RPhi"),
        "R-K_high(ρ_C=0.3)": barrier_family(raw_main, 0.2, 0.3, m_def, "RKhi"),
        "R-D_low(m=1)": barrier_family(raw_main, 0.2, 0.1, H12_M_PROFILES[1], "RDlo"),
        "R-D_high(m=5)": barrier_family(raw_main, 0.2, 0.1, H12_M_PROFILES[5], "RDhi"),
    }
    register_exploratory(
        "H12", "ALLD 침입장벽 심화 Δg(adaptive, random 앵커) — R-K 고위협",
        1.0, f"Δg={barrier['R-K_high(ρ_C=0.3)']['deepen_adaptive']:+.3f} "
        f"(fixedλ 대조 {barrier['R-K_high(ρ_C=0.3)']['deepen_tom_fixed']:+.3f})")

    # ---------------- 8) 탐색: 구조적 사전 결론 일치 (원칙 E) ----------------
    surf_RP_s = axis_surface(raw_main, cc_raw_main, RP, slots_core, stable_seed("H12RPs"),
                             corner=corner_struct, bl=blend_struct, n_b=n_boot_s)
    surf_RD_s = axis_surface(raw_main, cc_raw_main, RD, slots_core, stable_seed("H12RDs"),
                             corner=corner_struct, bl=blend_struct, n_b=n_boot_s)
    c1_s = slope_ci(surf_RP_s["coords"], *marginal(surf_RP_s, "adaptive")[::-1])
    c2_s = slope_ci(surf_RD_s["coords"], *irreplace(surf_RD_s, "adaptive",
                                                    "tit_for_tat")[::-1])
    agree = {
        "C1": {"interior": c1["slope"], "structural": c1_s["slope"],
               "same_sign": bool((c1["slope"] > 0) == (c1_s["slope"] > 0))},
        "C2": {"interior": c2["slope"], "structural": c2_s["slope"],
               "same_sign": bool((c2["slope"] < 0) == (c2_s["slope"] < 0))}}
    register_exploratory(
        "H12", "사전 강건성(원칙 E): interior vs 구조적(ALLD코너) 결론 일치", 1.0,
        f"C1 일치={agree['C1']['same_sign']} "
        f"(구조 β={c1_s['slope']:+.3f}), C2 일치={agree['C2']['same_sign']} "
        f"(구조 β={c2_s['slope']:+.3f})")

    # ---------------- 9) 탐색: H8E·H11 좌표 정위 (§6.2) ----------------
    def delta_at(rho_D, rho_C, m_profile):
        b_ad = basin_slot(Pi_full_main, CC_full_main, rho_D, rho_C, m_profile, "adaptive")
        b_best = max(basin_slot(Pi_full_main, CC_full_main, rho_D, rho_C, m_profile, f)
                     for f in H12_FIXED_COOP)
        b_rnd = basin_slot(Pi_full_main, CC_full_main, rho_D, rho_C, m_profile, "random")
        return {"marginal_adaptive": float(b_ad - b_rnd),
                "keystone_delta": float(b_ad - b_best)}
    triang = {
        "H8E_corner(ρ_C=0.3,저중복 m=1)": delta_at(0.2, 0.3, H12_M_PROFILES[1]),
        "H11_corner(ρ_C=0,고중복 m=5)": delta_at(0.3, 0.0, H12_M_PROFILES[5]),
    }
    register_exploratory(
        "H12", "H8E/H11 좌표 정위 (곡면 위 두 점으로 재현)", 1.0,
        f"H8E 코너 keystoneΔ={triang['H8E_corner(ρ_C=0.3,저중복 m=1)']['keystone_delta']:+.3f} "
        f"vs H11 코너={triang['H11_corner(ρ_C=0,고중복 m=5)']['keystone_delta']:+.3f}")

    # ---------------- 10) 판정·반환 ----------------
    supported = {
        "C1_threat_monotone": bool(c1["ci"][0] > 0),
        "C2_redundancy_monotone": bool(c2["ci"][1] < 0),
        "C3_frontier_exists": bool(c3["within_range"] and c3["frac_found"] > 0.5),
        "prior_robust_C1": bool(agree["C1"]["same_sign"]),
        "prior_robust_C2": bool(agree["C2"]["same_sign"]),
    }
    LOGGER.info("[H12] → %s", supported)
    return {
        "names": names, "config": {"pi_seeds": pi_seeds, "T_main": T_main,
                                   "primary_err": primary_err,
                                   "env_errors": env_errors, "Ts": Ts,
                                   "slot_share": slot_share, "blend": blend,
                                   "n_basin": n_basin, "n_boot": n_boot},
        "axes": {"R-P": pack_axis(surf_RP), "R-K": pack_axis(surf_RK),
                 "R-D": pack_axis(surf_RD),
                 **({"R-N": rn_pack} if rn_pack else {})},
        "confirmatory": {"C1": c1, "C2": c2, "C3": c3, "C3_p": c3_p,
                         "C1_R-K": c1_rk, "C1_R-K_imm": c1_rk_imm},
        "keystone_frontier": {"R-D": ks_RD, "R-D_crossing": c3,
                              "R-P": ks_RP, "R-P_crossing": c3_RP},
        "shapley": shap,
        "barrier": barrier,
        "prior_agreement": agree,
        "triangulation": triang,
        "supported": supported,
        "traces": {"Pi_main": Pi_full_main.tolist()}}


# ================================================================== GS
def exp_GS(seeds, rounds, jobs, backend):
    """
    GS (§3 게임구조 일반화, 전면 [탐색]) — 협력지수 CI=(R−P)/(T−S) 스윕.
    CI ∈ {0.4(현행), 0.5, 0.6}. (제안서의 '0.6 현행' 은 착오 — 현행 R=3,P=1,
    T=5,S=0 은 CI=0.4. 2R>T+S 를 만족하는 상향 스윕으로 조정.)
    핵심 결론 3개의 부호 강건성 확인: H1 착취가능성, H7 총보수 열세,
    H8 소집단 frac→CC 방향.
    """
    LOGGER.info("[GS] 게임구조(협력지수) 스윕")
    quick = seeds < 10
    cis = [0.4, 0.5, 0.6]
    n_seed = max(3, seeds // 3)
    # 고정전략 순환 case 3개 (짧은 주기 위주 — T=60 에서도 전환 존재)
    cases = ["p10_recip_expl_recon", "p30_flip", "p10_expl_first"]
    out = {}
    for ci in cis:
        extra = {"game_ci": ci}
        ad = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                        use_pymdp=backend == "pymdp")
        fx = agent_spec("tom_empathic", 0, use_pymdp=backend == "pymdp")
        m_ad, _ = _run_block(ad, "exploiter", n_seed, rounds, jobs,
                             extra_spec=extra)
        m_fx, _ = _run_block(fx, "exploiter", n_seed, rounds, jobs,
                             extra_spec=extra)
        d_expl = (np.array([m["exploitability"] for m in m_ad])
                  - np.array([m["exploitability"] for m in m_fx]))
        # H7 미니: adaptive vs GTFT(확률/횟수 두 유형), 3 case
        specs = []
        gs_focals = (("adaptive", ad),
                     ("gtft", strat_spec("generous_tft", 0)),
                     ("gtft_count", strat_spec("generous_tft_count", 0)))
        for nm, cfg in gs_focals:
            for case in cases:
                for sd in range(n_seed):
                    a = dict(cfg); a["seed"] = sd
                    specs.append({"agent": a,
                                  "opponent": capricious_case_spec(
                                      case, seed=700 + sd, n_rounds=rounds,
                                      error=0.10),
                                  "noise_seed": 7000 + sd, "game_ci": ci})
        res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
        third = len(res) // 3
        pay_ad = np.mean([np.mean(r["hist"]["my_payoff"]) for r in res[:third]])
        pay_gt = np.mean([np.mean(r["hist"]["my_payoff"])
                          for r in res[third:2 * third]])
        pay_gtc = np.mean([np.mean(r["hist"]["my_payoff"])
                           for r in res[2 * third:]])
        # H8 미니: 소집단 frac {0, 0.5} → CC 방향
        pops = []
        for n_ad in (0, 6):
            for rep in range(max(2, n_seed // 2)):
                members, labels = [], []
                for kk, c in _largest_remainder(
                        {"tit_for_tat": .25, "alld": .25, "wsls": .25,
                         "random": .25}, 12 - n_ad).items():
                    for _ in range(c):
                        members.append(_m_strat(kk, rounds)); labels.append(kk)
                for _ in range(n_ad):
                    members.append(_m_adaptive(backend)); labels.append("adaptive")
                pops.append({"members": members, "labels": labels,
                             "n_rounds": rounds, "partners_per_agent": 3,
                             "seed": stable_seed("gs", ci, n_ad, rep),
                             "game_ci": ci})
        pout = run_populations(pops, n_jobs=jobs, verbose=False)
        halfp = len(pout) // 2
        cc0 = np.mean([o["cc_rate"] for o in pout[:halfp]])
        cc6 = np.mean([o["cc_rate"] for o in pout[halfp:]])
        out[str(ci)] = {"expl_diff_mean": float(np.mean(d_expl)),
                        "expl_diff_ci": boot_mean_ci(d_expl, seed=901)["ci"],
                        "h7_pay_adaptive": float(pay_ad),
                        "h7_pay_gtft": float(pay_gt),
                        "h7_pay_gtft_count": float(pay_gtc),
                        "h8_cc_frac0": float(cc0), "h8_cc_frac05": float(cc6),
                        "h8_direction_positive": bool(cc6 > cc0)}
        register_exploratory("GS", f"CI={ci} 강건성", 1.0,
                             f"착취Δ={np.mean(d_expl):+.3f}, "
                             f"H7 {pay_ad:.2f} vs GTFT확률 {pay_gt:.2f} / "
                             f"GTFT횟수 {pay_gtc:.2f}, "
                             f"H8 방향 {'양' if cc6 > cc0 else '음'}")
    signs = [out[str(ci)]["expl_diff_mean"] < 0 for ci in cis]
    return {"cis": [str(c) for c in cis], "results": out,
            "robust_h1_sign": bool(all(signs)),
            "supported": None}   # 확증 지표 없음 (사전등록)


# ================================================================== H9/H10
def exp_H9_H10(seeds, rounds, jobs, backend):
    """
    귀인 범위 절제: all vs α-only vs λ-only.
    H9 [확증] 변덕 상대 보수: all > α-only. H10 [확증] 정적 상대 CC: all > λ-only.
    [탐색] 비-ToM 베이스라인(qlearner/bayes_br)의 변덕 상대 보수 병기 (§3).
    """
    LOGGER.info("[H9/H10] 귀인 범위 절제 + 베이스라인")
    kw = dict(kappa=0.9, sophisticated=True, use_pymdp=backend == "pymdp")
    variants = {
        "all": agent_spec("adaptive", 0, attribution_target="all", **kw),
        "alpha_only": agent_spec("adaptive", 0, attribution_target="alpha_only", **kw),
        "lambda_only": agent_spec("adaptive", 0, attribution_target="lambda_only", **kw),
    }
    res_cap, res_static, mets_static = {}, {}, {}
    for name, cfg in variants.items():
        specs = [{"agent": {**cfg, "seed": s},
                  "opponent": capricious_spec(1000 + s, rounds),
                  "noise_seed": 3000 + s} for s in range(seeds)]
        res_cap[name] = run_many(specs, n_rounds=rounds, n_jobs=jobs,
                                 verbose=False)
        m, r = _run_block(cfg, "noisy_tft", seeds, rounds, jobs)
        mets_static[name] = m; res_static[name] = r
    base_cap = {}
    for bname in ("qlearner", "bayes_br"):
        specs = [{"agent": {"type": bname, "seed": s},
                  "opponent": capricious_spec(1000 + s, rounds),
                  "noise_seed": 3000 + s} for s in range(seeds)]
        rr = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
        base_cap[bname] = [float(np.mean(r["hist"]["my_payoff"])) for r in rr]

    pay = {n: [float(np.mean(r["hist"]["my_payoff"])) for r in res_cap[n]]
           for n in variants}
    lam_react = {n: [float(np.std(np.asarray(r["agent_log"]["lam"], float)))
                     for r in res_cap[n]] for n in variants}
    cc_static = {n: [m["cc_rate"] for m in mets_static[n]] for n in variants}

    h9 = paired_stats(pay["all"], pay["alpha_only"], "greater", seed=191)
    register_primary("H9", "변덕 상대 보수 all>α-only", h9["p"],
                     h9["mean_a"] > h9["mean_b"], fmt_es(h9["es"], "dz"))
    h10 = paired_stats(cc_static["all"], cc_static["lambda_only"], "greater",
                       seed=192)
    register_primary("H10", "정적 상대 CC all>λ-only", h10["p"],
                     h10["mean_a"] > h10["mean_b"], fmt_es(h10["es"], "dz"))
    t_react = paired_stats(lam_react["all"], lam_react["alpha_only"],
                           "greater", seed=193)
    register_exploratory("H9", "λ 반응성(std) all>α-only", t_react["p"],
                         fmt_es(t_react["es"], "dz"))
    for bname, v in base_cap.items():
        t = paired_stats(pay["all"], v, "two-sided", seed=194)
        register_exploratory("H9", f"all vs {bname} (비-ToM 베이스라인)",
                             t["p"], fmt_es(t["es"], "dz"))
    lam_cap_traces = {n: np.stack([np.asarray(r["agent_log"]["lam"], float)
                                   for r in res_cap[n]]) for n in variants}
    return {"payoff_capricious": {n: [mean_sd(v)[0],
                                      boot_mean_ci(v, seed=195)["ci"]]
                                  for n, v in pay.items()},
            "baseline_capricious": {n: [mean_sd(v)[0],
                                        boot_mean_ci(v, seed=196)["ci"]]
                                    for n, v in base_cap.items()},
            "lam_react": {n: mean_sd(v)[0] for n, v in lam_react.items()},
            "cc_static": {n: [mean_sd(v)[0], boot_mean_ci(v, seed=197)["ci"]]
                          for n, v in cc_static.items()},
            "tests": {"H9": h9, "H10": h10, "react": t_react},
            "raw": {"pay": pay, "cc_static": cc_static},
            "supported": {"H9": bool(h9["mean_a"] > h9["mean_b"]),
                          "H10": bool(h10["mean_a"] > h10["mean_b"])},
            "traces": {"lam_capricious": lam_cap_traces}}


# ================================================================== VP / ABA / ORE
# v0.6 확장 (3축) — 고정 payoff·이분법·단순 복제자·신경구현 부재의 한계 보완.
#
#   §VP  가변 페이오프(연속 협력–경쟁 트레이드오프; Pisauro et al. 2022) —
#         CI 를 극단(음수·1 이상)까지 변동시키는 환경에서, 현재 보수로 자·타 효용을
#         맥락-의존 계산하는 adaptive 가 payoff-무감 고정전략 대비 얻는 성능·협력
#         유역 확장을 검증. H7/H8E/H12 의 '고정 payoff' 교란을 제거.
#   §ABA A→B→A 의도 전환 추적 복구 — 상대가 협력적 상호성(A)→착취(B)→다시 A 로
#         복귀할 때 adaptive 가 의도 추론을 **복구**하는지, β-맥락 절제·통제권·비-ToM
#         대비 우월한지 검증(Spiering 2025 자기/타인 귀인 포함).
#   §ORE Optimal Replicator Equation(Bravetti & Padilla 2018) — 개체 수준 RE 를
#         집단 수준 경쟁까지 확장한 ORE 가 협력 유역을 넓히는지, adaptive 의 한계적
#         기여가 ORE 하에서 더 뚜렷한지 검증. Bravetti Fig.1 재현 포함.


# ---- 가변 페이오프 공통 유형/상대 카탈로그 ----
def _vp_focal_specs(backend):
    """VP focal(주인공) 후보: 맥락-의존 adaptive + payoff-무감 고정전략들."""
    ad = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                    controllability=True, use_pymdp=backend == "pymdp")
    return {
        "adaptive": ad,
        "tft": strat_spec("tit_for_tat", 0),
        "gtft": strat_spec("generous_tft", 0),
        "wsls": strat_spec("wsls", 0),
        "allc": strat_spec("allc", 0),
        "alld": strat_spec("alld", 0),
    }


_VP_OPP_PANEL = ["tit_for_tat", "generous_tft", "wsls", "allc", "alld", "random"]
_VP_REGIMES = ["mild_pd", "harsh_pd", "coop_harmony", "deadlock",
               "oscillate", "blocks_c_i_c", "aba_coop_deadlock"]
# '시변(time-varying)' 레짐 — 어떤 단일 고정전략도 전 맥락에서 최적일 수 없어
# 맥락-의존 adaptive 의 우위가 드러나는 확증 대상. (고정-극단 레짐은 대조군)
_VP_VARIABLE = ["oscillate", "blocks_c_i_c", "aba_coop_deadlock"]
_VP_EXTREME_FIXED = ["harsh_pd", "coop_harmony", "deadlock"]   # 극단이나 고정 — 대조


def exp_VP(seeds, rounds, jobs, backend):
    """
    §VP 가변 페이오프(연속 협력–경쟁). CI 를 극단(음수·1 이상)까지 변동.
    [확증] C-VP1 변동/극단 레짐에서 adaptive 의 payoff 우위(best 고정전략 대비) > 0.
           C-VP2 변동/극단 레짐에서 adaptive 의 협력 유역 확장 sub_widening > 0
                 (중립 filler=random 대치 기준; RE·ORE 병기).
    [탐색] 레짐별 우위 분해, ORE vs RE 유역, 맥락(CI) 추정 오차.
    내부에서 CI 레짐 × 지평 격자를 자체 추정(GLOBAL).
    """
    from AIF_IPD.ipd.variable_payoff import (
        run_variable_many, estimate_variable_payoff_matrix, CI_REGIMES)
    Ts = rounds if isinstance(rounds, (list, tuple)) else [rounds]
    vp_seeds = max(3, min(int(seeds), 24))       # 순차-in-dyad 이므로 시드 상한
    T = int(min(max(Ts), 120))                   # 대표 지평(비용 관리; 지평 목록 명시)
    focal = _vp_focal_specs(backend)
    LOGGER.info("[VP] 가변 CI 레짐=%d × focal=%d × 상대=%d, seeds=%d, T=%d",
                len(_VP_REGIMES), len(focal), len(_VP_OPP_PANEL), vp_seeds, T)

    # ---- (1) focal × 상대 × 레짐 payoff 격자 ----
    pay = {reg: {f: [] for f in focal} for reg in _VP_REGIMES}
    for reg in _VP_REGIMES:
        specs, reg_idx = [], {}
        for fname, fcfg in focal.items():
            for opp in _VP_OPP_PANEL:
                for s in range(vp_seeds):
                    a = dict(fcfg); a["seed"] = stable_seed("vp", reg, fname, s)
                    o = strat_spec(opp, seed=stable_seed("vp_o", reg, opp, s))
                    reg_idx[(fname, opp, s)] = len(specs)
                    specs.append({"agent": a, "opponent": o,
                                  "env_err_agent": 0.05, "env_err_opponent": 0.05,
                                  "noise_seed": stable_seed("vp_n", reg, fname, opp, s)})
        res = run_variable_many(specs, reg, T, n_jobs=jobs)
        for fname in focal:
            # focal 의 상대 패널 평균 보수(시드별) — 상대 전반의 성능
            for s in range(vp_seeds):
                vals = [float(np.mean(res[reg_idx[(fname, opp, s)]]["hist"]["my_payoff"]))
                        for opp in _VP_OPP_PANEL]
                pay[reg][fname].append(float(np.mean(vals)))

    # 레짐별 adaptive 우위 = adaptive − best 단일 고정전략 (평균 기준으로 선택;
    # 시드별 max 는 승자의 저주 편향 → 평균 최고 전략 하나를 골라 짝지어 비교)
    advantage = {}
    for reg in _VP_REGIMES:
        fixed_names = [f for f in focal if f != "adaptive"]
        fixed_means = {f: float(np.mean(pay[reg][f])) for f in fixed_names}
        best_f = max(fixed_means, key=fixed_means.get)       # 평균 최고 고정전략 1개
        best_series = np.array(pay[reg][best_f])
        adv = np.array(pay[reg]["adaptive"]) - best_series
        st = paired_stats(pay[reg]["adaptive"], list(best_series),
                          "greater", seed=stable_seed("vp_adv", reg))
        advantage[reg] = {"mean": float(adv.mean()),
                          "ci": boot_mean_ci(adv, seed=stable_seed("vpc", reg))["ci"],
                          "p": st["p"], "es": st["es"], "raw": adv.tolist(),
                          "best_fixed": best_f}

    # 확증 C-VP1: 변동/극단 레짐 통합에서 adaptive 우위 > 0 (레짐 평균 후 단일표본)
    var_adv = np.concatenate([np.array(advantage[r]["raw"]) for r in _VP_VARIABLE])
    t_cvp1 = one_sample_perm(var_adv, mu0=0.0, alternative="greater",
                             seed=stable_seed("cvp1"))
    register_primary("VP", "C-VP1 변동레짐 adaptive payoff 우위>0", t_cvp1["p"],
                     bool(var_adv.mean() > 0),
                     f"Δpay={var_adv.mean():.3f} [{boot_mean_ci(var_adv, seed=7)['ci'][0]:.3f},"
                     f"{boot_mean_ci(var_adv, seed=7)['ci'][1]:.3f}]")
    # 대조: 고정 mild_pd 에서의 우위(한계 재현 — 여기선 유의하지 않을 수 있음)
    mild_adv = np.array(advantage["mild_pd"]["raw"])
    register_exploratory("VP", "대조: mild_pd(고정) adaptive 우위",
                         one_sample_perm(mild_adv, 0.0, alternative="greater", seed=8)["p"],
                         f"Δpay={mild_adv.mean():.3f}")

    # ---- (2) 협력 확장 CC-widening (종착 행동적 CC율; 중립 filler=random 대치) ----
    # [지표 개정 v0.6.1] 이분 유역 폐기 → 종착 조성의 행동적 CC율 xᵀ·CCm·x.
    # [검정 구성 개정 v0.6.2] 확증 replicate 단위 = **시드**(레짐 아님) — 레짐 수준
    # 집계(n=3)는 검정력이 구조적으로 부족(순열 p 하한 ~0.125).
    basin = {}
    seed_sw_re, seed_sw_ore = [], []
    for reg in _VP_REGIMES:
        base_specs = {
            "tft": strat_spec("tit_for_tat", 0), "gtft": strat_spec("generous_tft", 0),
            "wsls": strat_spec("wsls", 0), "allc": strat_spec("allc", 0),
            "alld": strat_spec("alld", 0), "random": strat_spec("random", 0),
        }
        # 슬롯 = {adaptive, random(중립 filler)} 두 조건에서 Π·CCm 추정
        specs_ad = dict(base_specs); specs_ad["slot"] = dict(focal["adaptive"])
        specs_rd = dict(base_specs); specs_rd["slot"] = strat_spec("random", 0)
        Pi_ad = estimate_variable_payoff_matrix(specs_ad, reg, T, vp_seeds,
                                                env_error=0.05, n_jobs=jobs)
        Pi_rd = estimate_variable_payoff_matrix(specs_rd, reg, T, vp_seeds,
                                                env_error=0.05, n_jobs=jobs)
        X0 = np.random.default_rng(stable_seed("vpX", reg)).dirichlet(
            np.ones(len(Pi_ad["names"])), size=150)
        re_ad = evo.re_terminal_cc(Pi_ad["Pi"], Pi_ad["CC"], X0)
        re_rd = evo.re_terminal_cc(Pi_rd["Pi"], Pi_rd["CC"], X0)
        ore_ad = evo.ore_terminal_cc(Pi_ad["Pi"], Pi_ad["CC"], X0[:80])
        ore_rd = evo.ore_terminal_cc(Pi_rd["Pi"], Pi_rd["CC"], X0[:80])
        # 시드 수준 replicate (확증 기반; 동일 X0·동일 시드 인덱스로 짝지음)
        s_ad = evo.per_seed_terminal_cc(Pi_ad["raw"], Pi_ad["CC_raw"], X0[:80])
        s_rd = evo.per_seed_terminal_cc(Pi_rd["raw"], Pi_rd["CC_raw"], X0[:80])
        if reg in _VP_VARIABLE:
            seed_sw_re.append(s_ad["re"] - s_rd["re"])
            seed_sw_ore.append(s_ad["ore"] - s_rd["ore"])
        basin[reg] = {"re_cc_widening": float(re_ad - re_rd),
                      "ore_cc_widening": float(ore_ad - ore_rd),
                      "re_ad": re_ad, "re_rd": re_rd,
                      "ore_ad": ore_ad, "ore_rd": ore_rd,
                      "seed_re_widening_mean": float((s_ad["re"] - s_rd["re"]).mean()),
                      "seed_ore_widening_mean": float((s_ad["ore"] - s_rd["ore"]).mean())}
    # 확증 C-VP2: 변동 레짐에서 RE CC-widening > 0 (시드×레짐 단일표본)
    sw = np.concatenate(seed_sw_re)
    t_cvp2 = one_sample_perm(sw, 0.0, alternative="greater", seed=stable_seed("cvp2"))
    ci_v2 = boot_mean_ci(sw, seed=stable_seed("cvp2ci"))["ci"]
    register_primary("VP", "C-VP2 변동레짐 협력 CC-widening>0",
                     t_cvp2["p"], bool(sw.mean() > 0),
                     f"CC_widening={sw.mean():.3f} [{ci_v2[0]:.3f},{ci_v2[1]:.3f}] "
                     f"(n={len(sw)} 시드×레짐)")
    sw_ore = np.concatenate(seed_sw_ore)
    register_exploratory("VP", "ORE CC-widening(변동레짐)",
                         one_sample_perm(sw_ore, 0.0, alternative="greater", seed=9)["p"],
                         f"ORE CC_widening={sw_ore.mean():.3f}")

    return {"regimes": _VP_REGIMES, "variable": _VP_VARIABLE, "T": T,
            "seeds": vp_seeds, "metric": "behavioral_CC_rate",
            "replicate_unit": "seed_x_regime",
            "cvp1_replicates": var_adv.tolist(), "cvp1_p": t_cvp1["p"],
            "cvp2_replicates": sw.tolist(), "cvp2_p": t_cvp2["p"],
            "cvp2_ci": ci_v2,
            "pay_mean": {reg: {f: mean_sd(pay[reg][f])[0] for f in focal}
                         for reg in _VP_REGIMES},
            "advantage": advantage, "basin": basin,
            "confirmatory": {"C_VP1_payoff_advantage": {
                                "delta": float(var_adv.mean()),
                                "ci": boot_mean_ci(var_adv, seed=7)["ci"],
                                "p": t_cvp1["p"], "supported": bool(var_adv.mean() > 0)},
                             "C_VP2_cc_widening": {
                                "cc_widening": float(sw.mean()), "ci": ci_v2,
                                "p": t_cvp2["p"], "n": int(len(sw)),
                                "supported": bool(sw.mean() > 0 and t_cvp2["p"] < 0.05)}},
            "supported": {"C_VP1": bool(var_adv.mean() > 0 and t_cvp1["p"] < 0.05),
                          "C_VP2": bool(sw.mean() > 0 and t_cvp2["p"] < 0.05)}}


# ------------------------------------------------------------------ ABA
def exp_ABA(seeds, rounds, jobs, backend):
    """
    §ABA A→B→A 의도 전환 추적 복구. 상대: 협력적 상호성(A=TFT)→착취(B=ALLD)→다시 A.

    [핵심 기제 — 정직한 보고] 전(前)등록 예상과 달리, 완전한 allostatic 에이전트는
    배신(B) 후 자기보호 상태에 고착되어 A 복귀 시에도 **자동으로 협력을 재개하지
    않는다**(히스테리시스). 이는 자기 방어적 배신이 상대의 개심(改心)을 관측할 기회를
    스스로 차단하는 '배신 후 선택적 관측' 함정(Selective observation following
    betrayal)의 재현이다. 복구는 **용서(forgiveness)=재탐색 성향**에 의해 게이팅되며,
    본 실험은 이 용량-반응을 확증한다.

    [확증] C-ABA1 충분한 용서 하에서 의도추론이 복구된다: 高용서 adaptive 의 복구오차
                 |pc(A3)−pc(A1)|<0.15 이고 CC 복원비 CC(A3)/CC(A1)>0.70.
           C-ABA2 복구는 용서에 단조 증가(용량-반응): 복구오차∼용서 기울기<0.
    [탐색] β-맥락 절제·통제권 on/off 의 복구 영향(방향 불문 정직 보고), 비-ToM
           베이스라인의 행동 복구, 자기충족 함정(A3 배신율↔복구실패 상관).
    내부에서 지평 목록을 함께 다룸(GLOBAL).
    """
    Ts = rounds if isinstance(rounds, (list, tuple)) else [rounds]
    from AIF_IPD.ipd.variable_payoff import aba_schedule
    FORG = [0.02, 0.05, 0.15, 0.30]          # 용서(재탐색) 수준 스윕
    out_by_T = {}
    for T in Ts:
        sched = aba_schedule("tit_for_tat", "alld", T)
        third = max(T // 3, 1)
        A1 = slice(0, third); A3 = slice(2 * third, T)

        def opp_spec(s):
            return dict(type="strategy", kind="tit_for_tat",
                        seed=1000 + s, error=0.05, schedule=sched)

        def run_variant(cfg):
            specs = [{"agent": {**cfg, "seed": s}, "opponent": opp_spec(s),
                      "noise_seed": 3000 + s} for s in range(seeds)]
            return run_many(specs, n_rounds=T, n_jobs=jobs, verbose=False)

        def traits(rr, key):
            return np.stack([np.asarray(rr[i]["agent_log"][key], float)
                             for i in range(len(rr))])

        def cc_phase(rr, ph):
            return np.array([float(np.mean(r["hist"]["state"][ph] == CC)) for r in rr])

        def defect_phase(rr, ph):  # focal 배신율(A3 자기충족 함정 지표)
            return np.array([float(np.mean((r["hist"]["state"][ph] // 2) == DEFECT))
                             for r in rr])

        def metrics(rr):
            pc = traits(rr, "pred_coop")
            a1 = pc[:, A1].mean(axis=1); a3 = pc[:, A3].mean(axis=1)
            cc1 = cc_phase(rr, A1); cc3 = cc_phase(rr, A3)
            return {"rec_err": np.abs(a3 - a1), "pc_A1": a1, "pc_A3": a3,
                    "cc_A1": cc1, "cc_A3": cc3,
                    "cc_ratio": cc3 / np.clip(cc1, 1e-6, None),
                    "def_A3": defect_phase(rr, A3),
                    "pred_coop": pc, "disp_cred": traits(rr, "disp_credence"),
                    "lam": traits(rr, "lam"), "control": traits(rr, "control")}

        # 용서 스윕 (adaptive, 통제권 on)
        forg = {}
        for fg in FORG:
            cfg = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                             controllability=True, forgiveness=fg,
                             use_pymdp=backend == "pymdp")
            forg[fg] = metrics(run_variant(cfg))

        # 절제 변형 (기본 용서에서): β-절제 / 통제권 off
        base_cfg = dict(kappa=0.9, sophisticated=True, use_pymdp=backend == "pymdp")
        abl = {
            "beta_clamp": metrics(run_variant(agent_spec(
                "adaptive", 0, controllability=True, beta_clamp=True, **base_cfg))),
            "noctrl": metrics(run_variant(agent_spec(
                "adaptive", 0, controllability=False, **base_cfg))),
        }
        # 비-ToM 베이스라인 (행동 복구만)
        base_cc = {}
        for bname in ("qlearner", "bayes_br"):
            rr = run_variant({"type": bname})
            base_cc[bname] = {"cc_A1": cc_phase(rr, A1), "cc_A3": cc_phase(rr, A3)}
        out_by_T[T] = {"forg": forg, "abl": abl, "base_cc": base_cc, "third": third}

    # 확증은 최장 지평 기준(짧은 지평은 국면이 짧아 복구 관측 불충분)
    Tref = max(Ts)
    ref = out_by_T[Tref]
    hi = max(FORG)
    rec_hi = ref["forg"][hi]["rec_err"]
    cc_ratio_hi = ref["forg"][hi]["cc_ratio"]

    # C-ABA1: 高용서에서 복구 — 복구오차<0.15 & CC 복원비>0.70
    t_aba1 = one_sample_perm(rec_hi, mu0=0.15, alternative="less",
                             seed=stable_seed("aba1"))
    cc_ratio_ok = bool(np.mean(cc_ratio_hi) > 0.70)
    aba1_supported = bool(np.mean(rec_hi) < 0.15 and cc_ratio_ok)
    register_primary("ABA", "C-ABA1 高용서 하 의도추론 복구(오차<0.15 & CC복원비>0.70)",
                     t_aba1["p"], aba1_supported,
                     f"복구오차={np.mean(rec_hi):.3f}, CC복원비={np.mean(cc_ratio_hi):.2f}")

    # C-ABA2: 용량-반응 — 복구오차가 용서에 단조 감소(기울기<0)
    xs, ys = [], []
    for fg in FORG:
        r = out_by_T[Tref]["forg"][fg]["rec_err"]
        xs.extend([fg] * len(r)); ys.extend(list(r))
    sl = slope_boot(np.array(xs), np.array(ys), seed=stable_seed("aba2"))
    aba2_supported = bool(sl["slope"] < 0 and sl["ci"][1] < 0)
    register_primary("ABA", "C-ABA2 복구오차∼용서 기울기<0(용량-반응)",
                     sl["p"], aba2_supported,
                     f"slope={sl['slope']:.3f} [{sl['ci'][0]:.3f},{sl['ci'][1]:.3f}]")

    # 탐색: 절제 변형(정직 보고), 자기충족 함정, 비-ToM 행동 복구
    dflt = out_by_T[Tref]["forg"][0.05]
    for name, m in ref["abl"].items():
        t = paired_stats(list(m["rec_err"]), list(dflt["rec_err"]),
                         "two-sided", seed=stable_seed("abaabl", name))
        register_exploratory("ABA", f"{name} vs 기본(용서0.05) 복구오차",
                             t["p"], f"Δ={t['mean_a']-t['mean_b']:+.3f}")
    # 자기충족 함정: A3 배신율↔복구오차 상관(저용서에서 강해야)
    lo = out_by_T[Tref]["forg"][0.02]
    trap = slope_boot(lo["def_A3"], lo["rec_err"], seed=stable_seed("abatrap"))
    register_exploratory("ABA", "저용서: A3 배신율→복구오차 기울기(자기충족 함정)",
                         trap["p"], f"slope={trap['slope']:.3f}")
    for b, d in ref["base_cc"].items():
        t = paired_stats(list(d["cc_A3"]), list(d["cc_A1"]), "two-sided",
                         seed=stable_seed("abab", b))
        register_exploratory("ABA", f"{b} 행동 CC 복원 A3 vs A1", t["p"],
                             f"ΔCC={t['mean_a']-t['mean_b']:+.3f}")

    return {"Ts": Ts, "Tref": Tref, "forg_levels": FORG, "by_T": out_by_T,
            "confirmatory": {
                "C_ABA1_recovery": {"rec_err_hi": float(np.mean(rec_hi)),
                                    "ci": boot_mean_ci(rec_hi, seed=1)["ci"],
                                    "cc_ratio_hi": float(np.mean(cc_ratio_hi)),
                                    "p": t_aba1["p"], "supported": aba1_supported},
                "C_ABA2_dose_response": {"slope": sl["slope"], "ci": sl["ci"],
                                         "p": sl["p"], "supported": aba2_supported}},
            "supported": {"C_ABA1": aba1_supported, "C_ABA2": aba2_supported}}


# ------------------------------------------------------------------ ORE
def _ore_type_specs(backend):
    ad = agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                    controllability=True, use_pymdp=backend == "pymdp")
    return {"adaptive": ad, "tft": strat_spec("tit_for_tat", 0),
            "gtft": strat_spec("generous_tft", 0), "wsls": strat_spec("wsls", 0),
            "allc": strat_spec("allc", 0), "alld": strat_spec("alld", 0)}


def exp_ORE(seeds, rounds, jobs, backend):
    """
    §ORE 집단 수준 경쟁(Optimal Replicator Equation; Bravetti & Padilla 2018).

    [지표 개정] 이분 협력 유역(협력-라벨 점유>0.5)은 (i) 임의 임계 이분화, (ii) 협력
    '라벨'과 실제 '행동'의 괴리(deadlock 에서 협력-라벨 유형이 실제로는 배신)라는 두
    결함이 있어 **폐기**한다. 대신 종착 조성의 **행동적 CC율** CC(x)=xᵀ·CCm·x (연속,
    관측된 상호협력에서 직접 유도)를 확증·시각화 지표로 병기한다.

    [확증] C-ORE1 ORE 종착 CC율 > RE 종착 CC율 (동일 Π·CCm·초기점, 레짐 전반 짝지음).
           C-ORE2 ORE 하 adaptive 한계 기여(CC-widening: adaptive vs random 슬롯 종착
                  CC율 차)>0.
    [탐색] Bravetti Fig.1 재현(2-유형 RE vs ORE), 레짐별 ΔCC, RE 하 CC-widening,
           Π·CCm 시드 부트스트랩.
    내부에서 CI 레짐 격자 자체 추정(GLOBAL).
    """
    from AIF_IPD.ipd.variable_payoff import estimate_variable_payoff_matrix
    Ts = rounds if isinstance(rounds, (list, tuple)) else [rounds]
    ore_seeds = max(3, min(int(seeds), 24))
    T = int(min(max(Ts), 120))
    regimes = ["mild_pd", "harsh_pd", "deadlock", "oscillate"]
    LOGGER.info("[ORE] RE vs ORE 종착 CC율, 레짐=%d, seeds=%d, T=%d",
                len(regimes), ore_seeds, T)

    # ---- (A) Bravetti Fig.1/2 재현 (2-유형) ----
    repro = {}
    for name, (R, Tt, P, S, x0) in {
            "model1": (4, 5, 1, 0, 0.3), "model2": (3, 4, 2, 0, 0.2),
            "model3": (3.5, 4, 0.5, 0.5, 0.1)}.items():
        repro[name] = evo.ore_two_type(R=R, T=Tt, P=P, S=S, xC0=x0,
                                       tau=1.5, steps=500, sweeps=90)

    # ---- (B) K-유형 HalloReg: 레짐별 RE vs ORE 종착 CC율 + adaptive CC-widening ----
    # [검정 구성 개정 v0.6.2] 확증 replicate 단위 = **시드**(레짐이 아님). 레짐 수준
    # 짝지음(n=3~4)은 순열 p 하한이 ~1/2^n(예: n=4 → 0.0625)이라 효과가 전 레짐에서
    # 일관되게 양수여도 유의에 도달할 수 없다(구조적 검정력 결손). 시드 s 마다
    # Π_s·CCm_s 로 CC율을 계산하면 (레짐×시드) 관측이 생겨 올바른 검정력을 얻는다.
    per_reg = {}
    re_list, ore_list, sw_re_list, sw_ore_list = [], [], [], []
    seed_re, seed_ore, seed_sw_re, seed_sw_ore = [], [], [], []   # 시드 수준 pooled
    for reg in regimes:
        specs = _ore_type_specs(backend)
        Pest = estimate_variable_payoff_matrix(specs, reg, T, ore_seeds,
                                               env_error=0.05, n_jobs=jobs)
        Pi, CCm, names = Pest["Pi"], Pest["CC"], Pest["names"]
        pi_raw, cc_raw = Pest["raw"], Pest["CC_raw"]
        rng = np.random.default_rng(stable_seed("oreX", reg))
        X0 = rng.dirichlet(np.ones(len(names)), size=160)
        re_cc = evo.re_terminal_cc(Pi, CCm, X0)
        ore_cc = evo.ore_terminal_cc(Pi, CCm, X0[:100])

        # adaptive CC-widening: adaptive 슬롯 vs random 슬롯 (Π·CCm 재추정)
        base = {k: dict(v) for k, v in specs.items() if k != "adaptive"}
        base_rand = dict(base); base_rand["random"] = strat_spec("random", 0)
        Pr = estimate_variable_payoff_matrix(base_rand, reg, T, ore_seeds,
                                             env_error=0.05, n_jobs=jobs)
        Xr = rng.dirichlet(np.ones(len(Pr["names"])), size=160)
        re_ad, ore_ad = re_cc, ore_cc
        re_rd = evo.re_terminal_cc(Pr["Pi"], Pr["CC"], Xr)
        ore_rd = evo.ore_terminal_cc(Pr["Pi"], Pr["CC"], Xr[:100])

        # --- 시드 수준 replicate (확증의 기반) ---
        s_ad = evo.per_seed_terminal_cc(pi_raw, cc_raw, X0[:100])
        s_rd = evo.per_seed_terminal_cc(Pr["raw"], Pr["CC_raw"], Xr[:100])
        seed_re.append(s_ad["re"]); seed_ore.append(s_ad["ore"])
        seed_sw_re.append(s_ad["re"] - s_rd["re"])       # 시드 짝지음(동일 시드 인덱스)
        seed_sw_ore.append(s_ad["ore"] - s_rd["ore"])

        per_reg[reg] = {
            "names": names, "cc_self": np.diag(CCm).tolist(),
            "re_cc": re_cc, "ore_cc": ore_cc,
            "re_cc_ci": boot_mean_ci(s_ad["re"], seed=stable_seed("oreCIre", reg))["ci"],
            "ore_cc_ci": boot_mean_ci(s_ad["ore"], seed=stable_seed("oreCIore", reg))["ci"],
            "re_cc_widening": float(re_ad - re_rd),
            "ore_cc_widening": float(ore_ad - ore_rd),
            "seed_re_mean": float(s_ad["re"].mean()),
            "seed_ore_mean": float(s_ad["ore"].mean()),
            "seed_re": s_ad["re"].tolist(), "seed_ore": s_ad["ore"].tolist(),
            "seed_sw_ore": (s_ad["ore"] - s_rd["ore"]).tolist(),
            "seed_sw_re": (s_ad["re"] - s_rd["re"]).tolist(),
            "ore_minus_re_cc": float(ore_cc - re_cc)}
        re_list.append(re_cc); ore_list.append(ore_cc)
        sw_re_list.append(re_ad - re_rd); sw_ore_list.append(ore_ad - ore_rd)

    # 시드×레짐 pooled replicate (레짐 내 시드 짝지음 유지)
    pooled_re = np.concatenate(seed_re); pooled_ore = np.concatenate(seed_ore)
    pooled_sw_ore = np.concatenate(seed_sw_ore)
    pooled_sw_re = np.concatenate(seed_sw_re)

    # 확증 C-ORE1: ORE 종착 CC율 > RE 종착 CC율 (시드×레짐 짝지음)
    t_ore1 = paired_stats(list(pooled_ore), list(pooled_re), "greater",
                          seed=stable_seed("core1"))
    d_ore1 = float(pooled_ore.mean() - pooled_re.mean())
    ci1 = boot_mean_ci(pooled_ore - pooled_re, seed=stable_seed("core1ci"))["ci"]
    register_primary("ORE", "C-ORE1 ORE 종착 CC율>RE 종착 CC율", t_ore1["p"],
                     bool(d_ore1 > 0),
                     f"ΔCC={d_ore1:+.3f} [{ci1[0]:.3f},{ci1[1]:.3f}] "
                     f"(n={len(pooled_re)} 시드×레짐)")
    # 확증 C-ORE2: ORE 하 adaptive CC-widening > 0 (시드×레짐 단일표본)
    t_ore2 = one_sample_perm(pooled_sw_ore, 0.0, alternative="greater",
                             seed=stable_seed("core2"))
    ci2 = boot_mean_ci(pooled_sw_ore, seed=stable_seed("core2ci"))["ci"]
    register_primary("ORE", "C-ORE2 ORE 하 adaptive CC-widening>0", t_ore2["p"],
                     bool(pooled_sw_ore.mean() > 0),
                     f"CC_widening_ORE={pooled_sw_ore.mean():.3f} "
                     f"[{ci2[0]:.3f},{ci2[1]:.3f}] (n={len(pooled_sw_ore)})")
    register_exploratory("ORE", "RE 하 adaptive CC-widening",
                         one_sample_perm(pooled_sw_re, 0.0, alternative="greater",
                                         seed=stable_seed("oreSWre"))["p"],
                         f"CC_widening_RE={pooled_sw_re.mean():.3f}")
    # 탐색: 레짐 수준 일관성(전 레짐 부호 일치 여부 — 효과의 강건성)
    register_exploratory("ORE", "레짐 수준 ΔCC 부호 일관성",
                         1.0, f"{int(np.sum(np.array(ore_list) > np.array(re_list)))}"
                              f"/{len(regimes)} 레짐에서 ORE>RE")

    ore_arr, re_arr = np.array(ore_list), np.array(re_list)
    sw_ore = np.array(sw_ore_list)

    return {"regimes": regimes, "T": T, "seeds": ore_seeds, "metric": "behavioral_CC_rate",
            "replicate_unit": "seed_x_regime",
            "reproduction": {k: {"re_end": v["re_end"], "ore_end": v["ore_end"],
                                 "re_xC": v["re_xC"].tolist(), "ore_xC": v["ore_xC"].tolist(),
                                 "re_fit": v["re_fit"].tolist(), "ore_fit": v["ore_fit"].tolist()}
                             for k, v in repro.items()},
            "per_regime": per_reg,
            "confirmatory": {
                "C_ORE1_cc_rate": {"delta": d_ore1, "ci": ci1, "p": t_ore1["p"],
                                   "n": int(len(pooled_re)),
                                   "supported": bool(d_ore1 > 0 and t_ore1["p"] < 0.05)},
                "C_ORE2_adaptive_cc_widening": {"cc_widening": float(pooled_sw_ore.mean()),
                                                "ci": ci2, "p": t_ore2["p"],
                                                "n": int(len(pooled_sw_ore)),
                                                "supported": bool(pooled_sw_ore.mean() > 0
                                                                  and t_ore2["p"] < 0.05)}},
            "supported": {"C_ORE1": bool(d_ore1 > 0 and t_ore1["p"] < 0.05),
                          "C_ORE2": bool(pooled_sw_ore.mean() > 0 and t_ore2["p"] < 0.05)}}


# =================================================================== 시각화
def _kfont():
    try:
        set_korean_font(plt, font_manager)
    except Exception:
        pass


def _save(fig, name: str, caption: str, n_note: str = ""):
    """PNG + 벡터 PDF + 캡션 메타데이터 JSON 동시 출력 (§5)."""
    png = RESULTS / f"{name}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / f"{name}.pdf", bbox_inches="tight")
    with open(RESULTS / f"{name}.caption.json", "w", encoding="utf-8") as f:
        json.dump({"figure": name, "caption": caption, "n": n_note,
                   "uncertainty": "밴드=시드 16–84 백분위(유계 존중), "
                                  "오차막대=평균의 부트스트랩 95% CI"},
                  f, ensure_ascii=False, indent=2)
    plt.close(fig)
    LOGGER.info("그림 저장: %s (+pdf, caption)", png)


def band(ax, mat: np.ndarray, label: str, color=None, ls="-"):
    """궤적 밴드: 시드 분포의 16–84 백분위 (유계 지표에서 경계 초과 없음, §5)."""
    m = mat.mean(axis=0)
    lo, hi = np.percentile(mat, [16, 84], axis=0)
    x = np.arange(mat.shape[1])
    ln, = ax.plot(x, m, label=label, color=color, ls=ls, lw=1.8)
    ax.fill_between(x, lo, hi, alpha=0.18, color=ln.get_color())
    return ln


def bar_ci(ax, xs, means, cis, labels=None, colors=None, width=0.6):
    """평균 + 부트스트랩 95% CI 오차막대 (검정 주석 패널 표준, §5)."""
    means = np.asarray(means, float)
    err = np.array([[m - c[0], c[1] - m] for m, c in zip(means, cis)]).T
    ax.bar(xs, means, width=width, yerr=err, capsize=4,
           color=colors, alpha=0.85)
    if labels:
        ax.set_xticks(xs); ax.set_xticklabels(labels)


def _n_note(ax, n, loc="lower right"):
    ax.annotate(f"n={n}", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="0.4")


# --------------------------------------------------------------- 그림들
def fig_H1(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    t = d["traces"]
    band(ax[0], t["lam_adaptive"], "adaptive λ", color="C0")
    band(ax[0], t["lam_fixed"], "fixed λ", color="C1", ls="--")
    ax[0].axhline(LAM_BASE, color="0.6", ls=":", lw=1)
    ax[0].set_title("λ 궤적 (착취자 상대)"); ax[0].set_xlabel("라운드")
    ax[0].set_ylabel("λ"); ax[0].legend(fontsize=8)
    _n_note(ax[0], t["lam_adaptive"].shape[0])
    band(ax[1], t["grievance"], "grievance g⁻", color="C3")
    band(ax[1], t["trust"], "trust g⁺", color="C2")
    ax[1].set_title("항상성 적분기"); ax[1].set_xlabel("라운드"); ax[1].legend(fontsize=8)
    r = d["raw"]
    means = [np.mean(r["expl_ad"]), np.mean(r["expl_fx"])]
    cis = [boot_mean_ci(r["expl_ad"])["ci"], boot_mean_ci(r["expl_fx"])["ci"]]
    bar_ci(ax[2], [0, 1], means, cis, ["adaptive", "fixed"],
           colors=["C0", "C1"])
    ax[2].set_title(f"착취가능성 [확증] ({d['primary']['es'] if isinstance(d['primary'].get('es'),str) else fmt_es(d['primary']['es'],'dz')})")
    ax[2].set_ylabel("exploitability"); _n_note(ax[2], len(r["expl_ad"]))
    _save(fig, "h1_self_protection" + tag,
          "H1: adaptive 는 착취자에게 λ 를 낮춰 자기보호하나 고정-λ 는 못 한다. "
          "밴드=시드 16–84 백분위, 막대=평균±부트스트랩 95% CI.",
          f"seeds={len(r['expl_ad'])}")


def fig_H2_H3(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for opp, c in zip(("exploiter", "noisy_tft", "noisy"), ("C3", "C2", "C1")):
        if opp in d["traces"]:
            band(ax[0], d["traces"][opp], opp, color=c)
    ax[0].axhline(LAM_BASE, color="0.6", ls=":")
    ax[0].set_title("λ 궤적: 의도 vs 맥락"); ax[0].set_xlabel("라운드")
    ax[0].set_ylabel("λ"); ax[0].legend(fontsize=8)
    # β/α 판별
    raw = d["raw"]
    b_ex = raw["all|exploiter"]["E_beta"]; b_nt = raw["all|noisy_tft"]["E_beta"]
    bar_ci(ax[1], [0, 1], [np.mean(b_ex), np.mean(b_nt)],
           [boot_mean_ci(b_ex)["ci"], boot_mean_ci(b_nt)["ci"]],
           ["exploiter", "noisy_tft"], colors=["C3", "C2"])
    ax[1].set_title("E[β] 판별 [확증]"); ax[1].set_ylabel("E[β]")
    _n_note(ax[1], len(b_ex))
    # 절제 패널 (§2): 회복량 all vs intent_only vs beta_clamp
    rec = d["ablation"]["recovery_means"]
    order = ["all", "intent_only", "beta_clamp"]
    ax[2].bar(range(3), [rec[k] for k in order],
              color=["C0", "C4", "C5"], alpha=0.85)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(order, fontsize=8)
    ax[2].set_title("β-경로 절제: λ 회복량\n(all>절제 → β-귀인 귀속)")
    ax[2].set_ylabel("λ 회복 (noisy−exploiter)")
    ax[2].annotate("mechanism_attributed=" + str(d["ablation"]["mechanism_attributed"]),
                   xy=(0.5, 0.95), xycoords="axes fraction", ha="center",
                   fontsize=8, color="0.3")
    _save(fig, "h2_h3_intent_context" + tag,
          "H2/H3: noisy TFT 는 λ 회복(맥락 귀인), 착취자는 높은 E[β](의도 귀인). "
          "절제 패널: intent_only/beta_clamp 에서 회복 감쇠 → 회복을 β-귀인에 귀속.",
          f"seeds={len(b_ex)}")


def fig_H4(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    band(ax[0], d["traces"]["lam_adaptive"], "adaptive λ", color="C0")
    ax[0].axhline(LAM_BASE, color="0.6", ls=":")
    ax[0].set_title("TFT 상대 λ"); ax[0].set_xlabel("라운드"); ax[0].set_ylabel("λ")
    r = d["raw"]
    bar_ci(ax[1], [0, 1], [np.mean(r["cc_ad"]), np.mean(r["cc_fx"])],
           [boot_mean_ci(r["cc_ad"])["ci"], boot_mean_ci(r["cc_fx"])["ci"]],
           ["adaptive", "fixed"], colors=["C0", "C1"])
    ax[1].axhline(d["primary"]["floor"], color="r", ls="--", lw=1,
                  label=f"floor={d['primary']['floor']}")
    ax[1].set_title("CC율 비열등성 [확증]"); ax[1].set_ylabel("CC rate")
    ax[1].legend(fontsize=8); _n_note(ax[1], len(r["cc_ad"]))
    _save(fig, "h4_cooperation_recovery" + tag,
          "H4: adaptive 의 CC율은 fixed 대비 비열등(margin 0.95)하며 floor 초과.",
          f"seeds={len(r['cc_ad'])}")


def fig_H5(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    r = d["raw"]
    bar_ci(ax[0], [0, 1], [np.mean(r["defense_imm"]), np.mean(r["defense_soph"])],
           [boot_mean_ci(r["defense_imm"])["ci"], boot_mean_ci(r["defense_soph"])["ci"]],
           ["즉각", "정교"], colors=["C3", "C0"])
    ax[0].set_title("착취자 방어량 (즉각>정교)"); ax[0].set_ylabel("−exploitability")
    _n_note(ax[0], len(r["defense_imm"]))
    bar_ci(ax[1], [0, 1], [np.mean(r["payoff_imm"]), np.mean(r["payoff_soph"])],
           [boot_mean_ci(r["payoff_imm"])["ci"], boot_mean_ci(r["payoff_soph"])["ci"]],
           ["즉각", "정교"], colors=["C3", "C0"])
    ax[1].set_title("noisy TFT 누적보수 (즉각<정교)"); ax[1].set_ylabel("cum payoff")
    _n_note(ax[1], len(r["payoff_imm"]))
    # 요인 forest (§5): 방어량 회귀 계수 CI
    fx = d["factorial"]["defense"]
    names = fx["names"][1:]; betas = fx["beta"][1:]; cis = fx["ci"][1:]
    yy = np.arange(len(names))
    for i, (b, ci) in enumerate(zip(betas, cis)):
        ax[2].plot([ci[0], ci[1]], [i, i], color="0.4")
        ax[2].plot(b, i, "o", color="C0")
    ax[2].axvline(0, color="r", ls="--", lw=1)
    ax[2].set_yticks(yy); ax[2].set_yticklabels(names, fontsize=8)
    ax[2].set_title("2×2×2 요인 (방어량) [탐색]")
    ax[2].set_xlabel("계수 ±코딩 (부트 95% CI)")
    _save(fig, "h5_immediate_vs_sophisticated" + tag,
          "H5: 즉각형(vmPFC)은 방어량↑·화해보수↓. Forest=요인 분해 계수 CI. "
          "DD-충전 경로는 조작 확인이며 조기 배신·위험곡선이 창발적 예측.",
          f"seeds={len(r['defense_imm'])}")


def fig_H6(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    strategies = list(next(iter(d["scores"].values())).keys())
    for k, nz in enumerate(("0.0", "0.15")):
        means = [d["scores"][nz][s] for s in strategies]
        cis = [d["cis"][nz][s] for s in strategies]
        bar_ci(ax[k], range(len(strategies)), means, cis, strategies,
               colors=[f"C{i}" for i in range(len(strategies))])
        ax[k].set_title(f"잡음 {nz}"); ax[k].tick_params(axis="x", labelrotation=30)
        _n_note(ax[k], d["n_seed"])
    ax[0].set_ylabel("라운드당 평균 보수")
    _save(fig, "h6_noise_robustness" + tag,
          "H6: 잡음 하에서 GTFT/WSLS 가 TFT 를 앞선다 (라운드로빈 평균±부트 CI). "
          "generous_tft=확률론적 용서, generous_tft_count=횟수 기반(인내 임계) "
          "용서 — 두 유형 모두 병기.",
          f"seeds={d['n_seed']}")


def fig_H7(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    t = d["traces"]; rel = t["rel"]
    # (a) 전환 정렬 보수 — 시드 CI 밴드 + case 얇은선 (§5). GTFT 두 유형 + WSLS 병기.
    aln_series = [("adaptive", "C0"), ("generous_tft", "C1"),
                  ("generous_tft_count", "C4"), ("wsls", "C2")]
    for name, c in aln_series:
        if name not in t["aligned_seed"]:
            continue
        mat = t["aligned_seed"][name]
        m = mat.mean(axis=0); lo, hi = np.percentile(mat, [2.5, 97.5], axis=0)
        lbl = {"generous_tft": "GTFT(확률)",
               "generous_tft_count": "GTFT(횟수)"}.get(name, name)
        ax[0, 0].plot(rel, m, color=c, lw=2, label=lbl)
        ax[0, 0].fill_between(rel, lo, hi, alpha=0.14, color=c)
    for case, arr in t["aligned_case"]["adaptive"].items():
        ax[0, 0].plot(rel, arr, color="C0", lw=0.5, alpha=0.2)
    ax[0, 0].axvline(0, color="0.5", ls="--"); ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title("전환 정렬 보수 (밴드=시드 95%CI, 얇은선=case)")
    ax[0, 0].set_xlabel("전환 상대 라운드")
    _n_note(ax[0, 0], t["aligned_seed"]["adaptive"].shape[0])
    # (b) λ 전환 정렬
    lam = t["lam_aligned_seed"]; m = lam.mean(axis=0)
    lo, hi = np.percentile(lam, [16, 84], axis=0)
    ax[0, 1].plot(rel, m, color="C0", lw=2)
    ax[0, 1].fill_between(rel, lo, hi, alpha=0.2, color="C0")
    ax[0, 1].axvline(0, color="0.5", ls="--")
    ax[0, 1].set_title(f"λ 전환 반응 (z̄={d['lambda_perm']['z_mean']:.2f}, "
                       f"p={d['lambda_perm']['p']:.3f})")
    ax[0, 1].set_xlabel("전환 상대 라운드"); ax[0, 1].set_ylabel("λ")
    # (c) 총보수 막대 (focal + 베이스라인)
    means = d["means"]
    order = ["adaptive", "generous_tft", "generous_tft_count", "wsls",
             "tit_for_tat", "qlearner", "bayes_br"]
    order = [o for o in order if o in means]
    ags = t["agg_seed"]
    bar_ci(ax[0, 2], range(len(order)), [means[o] for o in order],
           [boot_mean_ci(ags[o])["ci"] for o in order], order,
           colors=["C0"] + ["0.6"] * (len(order) - 1))
    ax[0, 2].tick_params(axis="x", labelrotation=30)
    ax[0, 2].set_title(f"전 case 총보수 [확증] adaptive vs GTFT "
                       f"({fmt_es(d['primary']['es'], 'dz')})")
    ax[0, 2].set_ylabel("라운드당 평균 보수")
    # (d, 신규 v0.4) 주기 × rival 별 adaptive 상대이점: '어느 전환 간격에서
    # adaptive 가 GTFT(확률/횟수)·WSLS 대비 payoff 를 더 얻는가' 를 명시.
    adv = d.get("adaptive_advantage", {})
    rivals = d.get("rival_focals", ["generous_tft", "generous_tft_count", "wsls"])
    rival_lbl = {"generous_tft": "GTFT(확률)",
                 "generous_tft_count": "GTFT(횟수)", "wsls": "WSLS"}
    rcolors = {"generous_tft": "C1", "generous_tft_count": "C4", "wsls": "C2"}
    if adv:
        Ps = sorted({int(P) for rv in rivals
                     for P in adv[rv]["by_period"]})
        nr = len(rivals); w = 0.8 / max(nr, 1)
        for ri, rv in enumerate(rivals):
            bp = adv[rv]["by_period"]
            dm = [bp[P]["delta_mean"] for P in Ps]
            err = np.array([[bp[P]["delta_mean"] - bp[P]["ci"][0],
                             bp[P]["ci"][1] - bp[P]["delta_mean"]]
                            for P in Ps]).T
            xs = np.arange(len(Ps)) + (ri - (nr - 1) / 2) * w
            bars = ax[1, 0].bar(xs, dm, width=w, yerr=err, capsize=2,
                                color=rcolors.get(rv, f"C{ri}"),
                                alpha=0.88, label=rival_lbl.get(rv, rv))
            # 유의(CI 0 배제)한 우위는 별표로 강조
            for x, P in zip(xs, Ps):
                if bp[P]["significant"] and bp[P]["adaptive_wins"]:
                    ax[1, 0].text(x, bp[P]["ci"][1], "*", ha="center",
                                  va="bottom", fontsize=9, color="k")
        ax[1, 0].axhline(0, color="0.4", lw=1)
        ax[1, 0].set_xticks(range(len(Ps)))
        sw0 = adv[rivals[0]]["by_period"]
        ax[1, 0].set_xticklabels(
            [f"P={P}\n({sw0[P]['switches_in_horizon']}회"
             + ("; 무전환" if sw0[P]['switches_in_horizon'] == 0 else "") + ")"
             for P in Ps], fontsize=8)
        ax[1, 0].legend(fontsize=7)
        ax[1, 0].set_title("주기별 adaptive 상대이점 Δ (양수=adaptive 우위)\n"
                           "vs GTFT(확률/횟수)·WSLS [탐색]; *=CI 0 배제")
        ax[1, 0].set_ylabel("Δ payoff (라운드당)")
    else:
        ax[1, 0].axis("off")
    # (e, 신규) case 격자 히트맵: 주기 × 순환족의 Δ
    fam_names, per_names = [], []
    for c in d["cases"]:
        fam = c.split("_", 1)[1]
        if fam not in fam_names:
            fam_names.append(fam)
    per_names = sorted({CAPRICIOUS_CASES[c]["period"] for c in d["cases"]})
    grid = np.full((len(per_names), len(fam_names)), np.nan)
    for c in d["cases"]:
        P = CAPRICIOUS_CASES[c]["period"]; fam = c.split("_", 1)[1]
        da = d["per_case"]["adaptive"][c][0] - d["per_case"]["generous_tft"][c][0]
        grid[per_names.index(P), fam_names.index(fam)] = da
    vmax = np.nanmax(np.abs(grid)) or 1.0
    im = ax[1, 1].imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    ax[1, 1].set_xticks(range(len(fam_names)))
    ax[1, 1].set_xticklabels(fam_names, rotation=40, fontsize=7, ha="right")
    ax[1, 1].set_yticks(range(len(per_names)))
    ax[1, 1].set_yticklabels([f"P={P}" for P in per_names], fontsize=8)
    for i in range(len(per_names)):
        for j in range(len(fam_names)):
            if np.isfinite(grid[i, j]):
                ax[1, 1].text(j, i, f"{grid[i, j]:+.2f}", ha="center",
                              va="center", fontsize=7)
    ax[1, 1].set_title("case 격자 Δ (adaptive−GTFT)\n"
                       "열: 순환족 (adaptive_* = AIF 국면 포함)")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    # (f) 대전 행렬 (대칭 환경 잡음; GTFT 두 유형 포함)
    mat = np.array(d["tournament"]["matrix"]); names = d["tournament"]["names"]
    nt = len(names)
    im = ax[1, 2].imshow(mat, cmap="viridis")
    ax[1, 2].set_xticks(range(nt))
    ax[1, 2].set_xticklabels(names, rotation=35, fontsize=7, ha="right")
    ax[1, 2].set_yticks(range(nt)); ax[1, 2].set_yticklabels(names, fontsize=7)
    for i in range(nt):
        for j in range(nt):
            ax[1, 2].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                          color="w", fontsize=7)
    ax[1, 2].set_title(f"{nt}×{nt} 대전 (대칭 잡음 "
                       f"{d['tournament']['symmetric_env_noise']})")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
    _save(fig, "h7_capricious_partners" + tag,
          "H7(v0.4): 주기 {10,30,60,120} × 순환족(AIF 국면·횟수GTFT 포함) case "
          "격자. λ 는 전환에 반응(무작위 정렬 순열 대비). (d) 패널은 주기 × "
          "rival(GTFT 확률/횟수, WSLS)별 adaptive payoff 상대이점을 명시 — "
          "어느 전환 간격·지평에서 adaptive 가 더 많은 보수를 얻는지 표시. "
          "대전은 환경 계층 대칭 잡음.",
          f"cases={len(d['cases'])}, noise={d['primary_noise']}")


def fig_H7H(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    Ts = d["Ts"]
    # 족 a: Δ(T) + 회귀선 + T*
    fa = d["family_a"]["delta"]
    m = [fa[str(T)][0] for T in Ts]
    err = np.array([[fa[str(T)][0] - fa[str(T)][1][0],
                     fa[str(T)][1][1] - fa[str(T)][0]] for T in Ts]).T
    ax[0, 0].errorbar(Ts, m, yerr=err, fmt="o-", capsize=4, color="C0",
                      label="Δ vs GTFT(확률) [확증]")
    fac = d["family_a"].get("delta_count")
    if fac:
        mc = [fac[str(T)][0] for T in Ts]
        errc = np.array([[fac[str(T)][0] - fac[str(T)][1][0],
                          fac[str(T)][1][1] - fac[str(T)][0]] for T in Ts]).T
        ax[0, 0].errorbar(Ts, mc, yerr=errc, fmt="s--", capsize=3, ms=4,
                          color="C4", label="Δ vs GTFT(횟수) [탐색]")
    ax[0, 0].axhline(0, color="r", ls="--", lw=1)
    ts = d["family_a"]["t_star"]
    if np.isfinite(ts["est"]):
        ax[0, 0].axvline(ts["est"], color="C2", ls=":",
                         label=f"T*={ts['est']:.0f} [{ts['ci'][0]:.0f},{ts['ci'][1]:.0f}]")
    ax[0, 0].legend(fontsize=7)
    sl = d["family_a"]["slope"]
    ax[0, 0].set_title(f"족a 전환수 고정 [확증]\n기울기={sl['slope']:.4f} "
                       f"[{sl['ci'][0]:.4f},{sl['ci'][1]:.4f}]")
    ax[0, 0].set_xlabel("지평 T"); ax[0, 0].set_ylabel("Δ = adaptive−GTFT")
    # 족 b: 주기별 곡선
    for period, dd in d["family_b"].items():
        cur = dd["curve"]
        ax[0, 1].plot(Ts, [cur[str(T)][0] for T in Ts], "o-",
                      label=f"주기 {period} (기울기 {dd['slope']['slope']:.4f})")
    ax[0, 1].axhline(0, color="r", ls="--", lw=1); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("족b 주기 고정 (근사 T-불변 예측)")
    ax[0, 1].set_xlabel("지평 T"); ax[0, 1].set_ylabel("Δ")
    # 히스테리시스
    hy = d["hysteresis"]
    ax[1, 0].plot(Ts, [hy[str(T)]["delta_default"] for T in Ts], "o-",
                  color="C1", label="기본 (forgiveness=0.05)")
    ax[1, 0].plot(Ts, [hy[str(T)]["delta_fast"] for T in Ts], "s-",
                  color="C2", label="완화 (forgiveness=0.15)")
    ax[1, 0].axhline(0, color="r", ls="--", lw=1); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title(f"히스테리시스 조작 (방향성립={d['hysteresis_direction_ok']})")
    ax[1, 0].set_xlabel("지평 T"); ax[1, 0].set_ylabel("Δ")
    # 잡음 곡선
    for err_l, cur in d["noise_curves"].items():
        ax[1, 1].plot(Ts, [cur[str(T)] for T in Ts], "o-", label=f"잡음 {err_l}")
    ax[1, 1].axhline(0, color="r", ls="--", lw=1); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("족a 잡음 스윕"); ax[1, 1].set_xlabel("지평 T")
    ax[1, 1].set_ylabel("Δ")
    _save(fig, "h7h_horizon_dependence" + tag,
          "H7H: Δ(T) 를 두 스케줄 족에서 분리 추정. 족a(전환수 고정)는 T 에 선형 "
          "개선, 족b(주기 고정)는 근사 T-불변. forgiveness↑→Δ개선(기제 조작).",
          f"Ts={Ts}")


def fig_H8(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    # (a) λ 곡선
    for lam, cur in d["lam_curves"].items():
        nad = sorted(cur, key=lambda x: int(x))
        ax[0, 0].plot([int(n) for n in nad], [cur[n][0] for n in nad], "o-",
                      label=f"λ={lam}", lw=1.2)
    ax[0, 0].set_title("(a) λ×n_adaptive → CC"); ax[0, 0].set_xlabel("n_adaptive")
    ax[0, 0].set_ylabel("CC rate"); ax[0, 0].legend(fontsize=7, ncol=2)
    # (a') 기울기 CI (λ별)
    lams = sorted(d["lam_slopes"], key=lambda x: float(x))
    for i, lam in enumerate(lams):
        sl = d["lam_slopes"][lam]
        ax[0, 1].plot([sl["ci"][0], sl["ci"][1]], [i, i], color="0.4")
        ax[0, 1].plot(sl["slope"], i, "o", color="C0")
    ax[0, 1].axvline(0, color="r", ls="--", lw=1)
    ax[0, 1].set_yticks(range(len(lams))); ax[0, 1].set_yticklabels(lams, fontsize=8)
    ax[0, 1].set_title("(a') λ별 frac 기울기 CI\n('정성적 일치'로 재보정)")
    ax[0, 1].set_xlabel("CC~frac 기울기"); ax[0, 1].set_ylabel("λ")
    # (e) 조건부 기울기 플롯 (§5: 이중축 대체)
    cs = d["conditional_slopes"]
    ll = sorted(cs, key=lambda x: float(x))
    ax[0, 2].plot([float(x) for x in ll], [cs[x]["est"] for x in ll], "o-", color="C0")
    ax[0, 2].fill_between([float(x) for x in ll],
                          [cs[x]["ci"][0] for x in ll],
                          [cs[x]["ci"][1] for x in ll], alpha=0.2, color="C0")
    ax[0, 2].axhline(0, color="r", ls="--", lw=1)
    ax[0, 2].axvline(0.4, color="C2", ls=":", label="λ=0.4 [확증]")
    ax[0, 2].set_title("(e) 조건부 frac 기울기 (상호작용 회귀)")
    ax[0, 2].set_xlabel("λ"); ax[0, 2].set_ylabel("∂CC/∂frac"); ax[0, 2].legend(fontsize=8)
    # (c) 비율
    rr = sorted(d["ratio"], key=lambda x: float(x))
    ax[1, 0].errorbar([float(x) for x in rr], [d["ratio"][x]["cc"][0] for x in rr],
                      yerr=[[d["ratio"][x]["cc"][0] - d["ratio"][x]["cc"][1][0] for x in rr],
                            [d["ratio"][x]["cc"][1][1] - d["ratio"][x]["cc"][0] for x in rr]],
                      fmt="o-", capsize=3, color="C0")
    ax[1, 0].set_title("(c) 즉각형 비율 → CC"); ax[1, 0].set_xlabel("즉각형 비율")
    ax[1, 0].set_ylabel("CC rate")
    # (d) 대규모 곡선 (이중축 제거 — CC/payoff 상하 분리)
    for pref, mk in (("d30", "o-"), ("d100", "s--")):
        if pref in d["large"]:
            cur = d["large"][pref]["curve"]
            fr = sorted(cur, key=lambda x: float(x))
            ax[1, 1].plot([float(x) for x in fr], [cur[x]["cc"][0] for x in fr],
                          mk, label=f"N={d['large'][pref]['N']}")
    ax[1, 1].set_title("(d) 대규모 frac→CC"); ax[1, 1].set_xlabel("adaptive 비율")
    ax[1, 1].set_ylabel("CC rate"); ax[1, 1].legend(fontsize=8)
    for pref, mk in (("d30", "o-"), ("d100", "s--")):
        if pref in d["large"]:
            cur = d["large"][pref]["curve"]
            fr = sorted(cur, key=lambda x: float(x))
            ax[1, 2].plot([float(x) for x in fr], [cur[x]["payoff"][0] for x in fr],
                          mk, label=f"N={d['large'][pref]['N']}")
    ax[1, 2].set_title("(d) 대규모 frac→후생"); ax[1, 2].set_xlabel("adaptive 비율")
    ax[1, 2].set_ylabel("평균 보수"); ax[1, 2].legend(fontsize=8)
    # (f) filler 상호작용 CI
    fi = d["filler_interactions"]
    fk = list(fi)
    for i, f in enumerate(fk):
        ax[2, 0].plot(fi[f]["ci"], [i, i], color="0.4")
        ax[2, 0].plot(fi[f]["est"], i, "o", color="C3")
    ax[2, 0].axvline(0, color="r", ls="--", lw=1)
    ax[2, 0].set_yticks(range(len(fk))); ax[2, 0].set_yticklabels(fk, fontsize=8)
    ax[2, 0].set_title("(f) frac×filler 상호작용\n(adaptive 대비 초과 기여)")
    ax[2, 0].set_xlabel("상호작용 계수 CI")
    # (f') ALLC 이전 회계
    if d.get("allc_transfer"):
        tr = d["allc_transfer"]
        ax[2, 1].bar([0, 1], [tr["share_of_group_payoff"],
                              1 - tr["share_of_group_payoff"]],
                     color=["C3", "C2"])
        ax[2, 1].set_xticks([0, 1])
        ax[2, 1].set_xticklabels(["ALLD 착취\n이전", "잔여 후생"], fontsize=8)
        ax[2, 1].set_title(f"(f) ALLC 착시 측정\n비착취 평균={tr['nonexploiter_mean_payoff']:.2f}")
        ax[2, 1].set_ylabel("집단 후생 비중")
    # (g) 민감도 기울기
    sens = d["sensitivity"]; sk = list(sens)
    for i, k in enumerate(sk):
        ax[2, 2].plot(sens[k]["ci"], [i, i], color="0.4")
        ax[2, 2].plot(sens[k]["slope"], i, "o", color="C0")
    ax[2, 2].axvline(0, color="r", ls="--", lw=1)
    ax[2, 2].set_yticks(range(len(sk))); ax[2, 2].set_yticklabels(sk, fontsize=7)
    ax[2, 2].set_title("(g) 구성 민감도: frac 기울기"); ax[2, 2].set_xlabel("기울기 CI")
    _save(fig, "h8_scaling_cooperation" + tag,
          "H8: λ=0.4 조건부 frac 기울기[확증]. 이중축 제거·조건부 기울기 플롯. "
          "ALLC 착시를 착취 이전 비중으로 측정. 구성 민감도로 부호 강건성 지도화.",
          f"reps small={d['reps']['small']}/large={d['reps']['large']}")


def _tern_frame(ax, sub_names):
    """3-유형 심플렉스 테두리 + 꼭짓점 라벨."""
    V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(V[:, 0], V[:, 1], color="0.3", lw=1)
    ax.text(-0.03, -0.04, sub_names[0], ha="right", fontsize=8,
            color=TYPE_COLORS.get(sub_names[0], "k"))
    ax.text(1.03, -0.04, sub_names[1], ha="left", fontsize=8,
            color=TYPE_COLORS.get(sub_names[1], "k"))
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, sub_names[2], ha="center", fontsize=8,
            color=TYPE_COLORS.get(sub_names[2], "k"))
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.0)
    ax.set_aspect("equal"); ax.axis("off")


def _tern_panel(ax, td, title, mode="phase"):
    """
    3-유형 부분계 패널.
    mode="phase": 궤적(회색) + 벡터장 화살표 + 끌개 점 — Replicator 상태 공간.
    mode="basin": 격자 초기점을 종착 협력 점유율로 채색 — cooperation basin.
    """
    sub_names = td["names"]; bm = td["map"]
    _tern_frame(ax, sub_names)
    if mode == "basin":
        gx, gy = evo.ternary_xy(bm["grid"])
        sc = ax.scatter(gx, gy, c=bm["coop_share"], cmap="RdYlGn", vmin=0,
                        vmax=1, s=14, marker="h", lw=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, label="종착 협력 점유율")
    else:
        for tr in td["trajs"]:
            px, py = evo.ternary_xy(tr)
            ax.plot(px, py, color="0.55", lw=0.7, alpha=0.7)
            ax.plot(px[0], py[0], ".", color="0.55", ms=3)
        # 끌개: 유역 지도의 종착점 군집 (0.02 양자화)
        ends = bm["ends"]
        keys = np.round(ends / 0.02).astype(int)
        uniq, cnt = np.unique(keys, axis=0, return_counts=True)
        for u, c in zip(uniq, cnt):
            e = u * 0.02
            ex, ey = evo.ternary_xy(np.asarray(e, float))
            ax.plot(ex, ey, "*", ms=6 + 14 * (c / cnt.sum()), color="C3",
                    mec="k", mew=0.4, zorder=5)
    ax.set_title(title, fontsize=9)


def fig_H8E(d, tag=""):
    """H8E 시각화 3부작: 침입 구조 / 복제자 동역학 / 상태공간·유역."""
    _kfont()
    names = d["names"]; Ts = d["Ts"]; errs = d["env_errors"]
    perr = d["primary_err"]; t = d["traces"]
    Tmain = Ts[-1]

    # ============ 그림 1: 보수 구조와 침입 (h8e_invasion_structure) ============
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for col, T in enumerate(Ts[:2]):
        Pi = np.array(t["Pi_mats"][(perr, T)])
        im = ax[0, col].imshow(Pi, cmap="RdYlGn")
        ax[0, col].set_xticks(range(len(names)))
        ax[0, col].set_xticklabels(names, rotation=45, fontsize=7, ha="right")
        ax[0, col].set_yticks(range(len(names)))
        ax[0, col].set_yticklabels(names, fontsize=7)
        ax[0, col].set_title(f"Π (행=focal, err={perr}, T={T})")
        fig.colorbar(im, ax=ax[0, col], fraction=0.046)
    if len(Ts) < 2:
        ax[0, 1].axis("off")
    # 침입 성장률 격자 g(err, T)
    G = np.array([[d["invasion_grid"][f"err{e}|T{T}"]["adaptive"]["g"]
                   for T in Ts] for e in errs])
    vmax = np.max(np.abs(G)) or 1.0
    im = ax[0, 2].imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax[0, 2].set_xticks(range(len(Ts)))
    ax[0, 2].set_xticklabels([f"T={T}" for T in Ts])
    ax[0, 2].set_yticks(range(len(errs)))
    ax[0, 2].set_yticklabels([f"err={e}" for e in errs], fontsize=8)
    for i, e in enumerate(errs):
        for j, T in enumerate(Ts):
            g = d["invasion_grid"][f"err{e}|T{T}"]["adaptive"]
            star = "*" if (g["ci"][0] > 0 or g["ci"][1] < 0) else ""
            ax[0, 2].text(j, i, f"{g['g']:+.2f}{star}", ha="center",
                          va="center", fontsize=8)
    ax[0, 2].set_title("adaptive 침입 성장률 g(err, T) [확증: err=0.10]\n"
                       "(*: 부트 95% CI 가 0 배제)")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)
    # 역방향 침입 forest (순수 adaptive 상주, primary err, 지평별)
    inv_names = [n for n in names if n not in ("adaptive",)]
    inv_names = [n for n in inv_names if n in d["reverse_invasion"]
                 [f"err{perr}|T{Ts[0]}"]]
    yy = np.arange(len(inv_names))
    off = np.linspace(-0.18, 0.18, len(Ts))
    for ti, T in enumerate(Ts):
        rv = d["reverse_invasion"][f"err{perr}|T{T}"]
        for i, inv in enumerate(inv_names):
            g = rv[inv]["pure"]
            ax[1, 0].plot(g["ci"], [i + off[ti]] * 2, color=f"C{ti}", lw=1.4)
            ax[1, 0].plot(g["g"], i + off[ti], "o", ms=4, color=f"C{ti}",
                          label=f"T={T}" if i == 0 else None)
    ax[1, 0].axvline(0, color="r", ls="--", lw=1)
    ax[1, 0].set_yticks(yy); ax[1, 0].set_yticklabels(inv_names, fontsize=8)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("역방향 침입: 순수 adaptive 상주집단에 대한\n"
                       "각 전략의 침입 성장률 (err=0.10)")
    ax[1, 0].set_xlabel("g (부트 95% CI)")
    # 구성 민감도: g vs 상주 ALLD 비중 (Dirichlet 표본 + 구조적 스윕)
    for ti, T in enumerate(Ts):
        co = t["comp_raw"][(perr, T)]
        ax[1, 1].scatter(co["alld_share_samples"], co["g_samples"], s=8,
                         alpha=0.4, color=f"C{ti}",
                         label=f"T={T} (침입가능 {co['frac_invadable']:.0%})")
        sw = co["alld_sweep"]
        ax[1, 1].plot(sw["shares"], sw["g"], "-", color=f"C{ti}", lw=1.8)
    ax[1, 1].axhline(0, color="r", ls="--", lw=1); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("상주 구성 민감도 (Dirichlet 무작위 구성 + ALLD 스윕)")
    ax[1, 1].set_xlabel("상주 ALLD 비중"); ax[1, 1].set_ylabel("adaptive 침입 g")
    # 침입 가능 구성 비율 히트맵 (err × T)
    F = np.array([[t["comp_raw"][(e, T)]["frac_invadable"] for T in Ts]
                  for e in errs])
    im = ax[1, 2].imshow(F, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax[1, 2].set_xticks(range(len(Ts)))
    ax[1, 2].set_xticklabels([f"T={T}" for T in Ts])
    ax[1, 2].set_yticks(range(len(errs)))
    ax[1, 2].set_yticklabels([f"err={e}" for e in errs], fontsize=8)
    for i in range(len(errs)):
        for j in range(len(Ts)):
            ax[1, 2].text(j, i, f"{F[i, j]:.2f}", ha="center", va="center",
                          color="w", fontsize=8)
    ax[1, 2].set_title("adaptive 침입 가능 상주 구성의 비율")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
    _save(fig, "h8e_invasion_structure" + tag,
          "H8E-1: 쌍별 보수 Π(지평별), 침입 성장률의 (err×T) 격자[확증: "
          "err=0.10 지평별], 역방향 침입(adaptive 상주 → 고정전략 침입), "
          "상주 구성 민감도.",
          f"Π seeds={d['pi_seeds']}, errs={errs}, Ts={Ts}")

    # ============ 그림 2: 복제자·Moran 동역학 (h8e_replicator_dynamics) ============
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for col, T in enumerate(Ts[:2]):
        tr = t["replicator"][T]["mix+10%ad"]
        for i, nm in enumerate(names):
            ax[0, col].plot(tr[:, i], lw=(2.2 if nm == "adaptive" else 1.2),
                            color=TYPE_COLORS.get(nm), label=nm)
        ax[0, col].set_title(f"복제자 궤적 (기준혼합+10% adaptive, T={T})")
        ax[0, col].set_xlabel("세대"); ax[0, col].set_ylabel("빈도")
        ax[0, col].legend(fontsize=7, ncol=2)   # 전 유형 legend 명시 (v0.3)
    if len(Ts) < 2:
        ax[0, 1].axis("off")
    # 역침입 동역학 (T=Tmain): adaptive 다수 + 침입자 5% 초기점
    for inv, curve in t["reverse_traj"][Tmain].items():
        ax[0, 2].plot(curve, lw=1.4, color=TYPE_COLORS.get(inv), label=inv)
    ax[0, 2].axhline(0.05, color="0.6", ls=":", lw=1)
    ax[0, 2].set_title(f"역침입 복제자 궤적 (침입자 빈도, T={Tmain})\n"
                       "초기 5% — 0.05 선 위로 성장 시 침입 성공")
    ax[0, 2].set_xlabel("세대"); ax[0, 2].set_ylabel("침입자 빈도")
    ax[0, 2].legend(fontsize=7, ncol=2)
    # 끌개 구조: 상위 끌개의 조성 누적막대 + 유역 비율
    for col, T in enumerate(Ts[:2]):
        ats = d["attractors"][f"T{T}"]["attractors"][:5]
        bottoms = np.zeros(len(ats))
        for i, nm in enumerate(names):
            vals = [a["composition"][i] for a in ats]
            ax[1, col].bar(range(len(ats)), vals, bottom=bottoms,
                           color=TYPE_COLORS.get(nm), label=nm, width=0.7)
            bottoms += np.array(vals)
        ax[1, col].set_xticks(range(len(ats)))
        ax[1, col].set_xticklabels([f"유역\n{a['basin_frac']:.0%}"
                                    for a in ats], fontsize=8)
        ax[1, col].set_title(f"복제자 끌개 조성 (Dirichlet 초기점, T={T})")
        ax[1, col].set_ylabel("조성")
        if col == 0:
            ax[1, col].legend(fontsize=6, ncol=2)
    if len(Ts) < 2:
        ax[1, 1].axis("off")
    # Moran 교차검증 (지평별)
    for ti, T in enumerate(Ts):
        mf = np.array(t["moran"][T]["freq"])
        mad = mf[:, :, names.index("adaptive")]
        m = mad.mean(axis=0); lo, hi = np.percentile(mad, [16, 84], axis=0)
        ax[1, 2].plot(m, color=f"C{ti}", lw=2,
                      label=f"T={T} (최종 {t['moran'][T]['final_adaptive_mean']:.2f})")
        ax[1, 2].fill_between(range(len(m)), lo, hi, alpha=0.15,
                              color=f"C{ti}")
    ax[1, 2].axhline(0.10, color="0.6", ls=":", lw=1)
    ax[1, 2].set_title("Moran adaptive 빈도 (평균장 교차검증)")
    ax[1, 2].set_xlabel("세대"); ax[1, 2].set_ylabel("adaptive 빈도")
    ax[1, 2].legend(fontsize=8)
    _save(fig, "h8e_replicator_dynamics" + tag,
          "H8E-2: 복제자 궤적(전 유형 legend 명시)·역침입 동역학·끌개 조성"
          "(유역 비율)·Moran 교차검증. 지평 T={60,240} 병렬 비교.",
          f"Π seeds={d['pi_seeds']}")

    # ============ 그림 3: 상태공간·협력 유역 (h8e_state_space_basins) ============
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    key_a, key_b = "ad_alld_tft", "ad_alld_allc"
    for col, T in enumerate(Ts[:2]):
        _tern_panel(ax[0, col], t["ternary"][(key_a, T)],
                    f"상태공간 위상 초상 (adaptive–alld–TFT, T={T})\n"
                    "회색=궤적, ★=끌개(크기∝유역)", mode="phase")
    if len(Ts) < 2:
        ax[0, 1].axis("off")
    _tern_panel(ax[0, 2], t["ternary"][(key_a, Tmain)],
                f"끌개 유역 위상도 (adaptive–alld–TFT, T={Tmain})", mode="basin")
    _tern_panel(ax[1, 0], t["ternary"][(key_b, Tmain)],
                f"끌개 유역 위상도 (adaptive–alld–ALLC, T={Tmain})", mode="basin")
    # 협력 유역 비율 히트맵 (adaptive 포함, 전 격자)
    BW = np.array([[d["basins"][f"err{e}|T{T}"]["with"] for T in Ts]
                   for e in errs])
    im = ax[1, 1].imshow(BW, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[1, 1].set_xticks(range(len(Ts)))
    ax[1, 1].set_xticklabels([f"T={T}" for T in Ts])
    ax[1, 1].set_yticks(range(len(errs)))
    ax[1, 1].set_yticklabels([f"err={e}" for e in errs], fontsize=8)
    for i in range(len(errs)):
        for j in range(len(Ts)):
            ax[1, 1].text(j, i, f"{BW[i, j]:.2f}", ha="center", va="center",
                          fontsize=8)
    ax[1, 1].set_title("협력-지배 끌개 유역 비율 (9유형; 위상 구조)")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    # 유역 확장 (with − without adaptive)
    DW = np.array([[d["basins"][f"err{e}|T{T}"]["widening"] for T in Ts]
                   for e in errs])
    vmax = np.max(np.abs(DW)) or 1.0
    im = ax[1, 2].imshow(DW, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    ax[1, 2].set_xticks(range(len(Ts)))
    ax[1, 2].set_xticklabels([f"T={T}" for T in Ts])
    ax[1, 2].set_yticks(range(len(errs)))
    ax[1, 2].set_yticklabels([f"err={e}" for e in errs], fontsize=8)
    for i in range(len(errs)):
        for j in range(len(Ts)):
            ax[1, 2].text(j, i, f"{DW[i, j]:+.2f}", ha="center", va="center",
                          fontsize=8)
    ax[1, 2].set_title("협력 CC-widening Δ (adaptive 유 − 무)\n종착 행동적 CC율")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
    _save(fig, "h8e_state_space_basins" + tag,
          "H8E-3: Replicator 상태 공간(3-유형 위상 초상 + 끌개), cooperation "
          "basin 지도, 협력 유역 비율·확장의 (err×T) 격자.",
          f"basin n={len(t['ternary'][(key_a, Tmain)]['map']['grid'])} 격자점")


def fig_H11(d, tag=""):
    """H11 Keystone 검정 시각화: 치환 설계 dose 곡선 + 기제 + 진화 프레임."""
    _kfont()
    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    doses = d["doses"]; arms = list(d["arms"])
    arm_lbl = {"treatment": "adaptive (treat)",
               "ctrl1_fixed_lambda": "fixed-λ ToM (ctrl1)",
               "ctrl2_immediate": "즉각형 (ctrl2)",
               "ctrl3_extra_tft": "추가 TFT (ctrl3)",
               "ctrl4_allc": "ALLC (ctrl4)"}
    arm_col = {"treatment": "C0", "ctrl1_fixed_lambda": "C1",
               "ctrl2_immediate": "C4", "ctrl3_extra_tft": "C2",
               "ctrl4_allc": "C3"}

    def _dose_panel(a, curves, ylabel, title):
        for arm in arms:
            cur = curves[arm]
            m = [cur[str(dd)][0] for dd in doses]
            err = np.array([[cur[str(dd)][0] - cur[str(dd)][1][0],
                             cur[str(dd)][1][1] - cur[str(dd)][0]]
                            for dd in doses]).T
            a.errorbar(doses, m, yerr=err, fmt="o-", capsize=3, ms=4,
                       color=arm_col[arm], label=arm_lbl[arm], lw=1.4)
        a.set_xlabel("swap dose (개체 수)"); a.set_ylabel(ylabel)
        a.set_title(title); a.legend(fontsize=7)

    # (a) dose→CC 곡선 (주장 B 원자료)
    _dose_panel(ax[0, 0], d["curves_cc"], "집단 CC rate",
                "(a) dose→집단 CC [주장 B 원자료]\n(dose0 은 전 arm 공유)")
    # (b) Δ 기울기 forest: treat−각 대조 (CC)
    rows = [("vs fixed-λ [확증]", d["primary_B_cc_slope"])]
    rows += [(f"vs {arm_lbl[a2]}", d["delta_slopes_ctrl"][a2]["cc"])
             for a2 in d["delta_slopes_ctrl"]]
    for i, (lbl, sl) in enumerate(rows):
        ax[0, 1].plot(sl["ci"], [i, i], color="0.4")
        ax[0, 1].plot(sl["slope"], i, "o",
                      color="C0" if "확증" in lbl else "0.5")
    ax[0, 1].axvline(0, color="r", ls="--", lw=1)
    ax[0, 1].set_yticks(range(len(rows)))
    ax[0, 1].set_yticklabels([r[0] for r in rows], fontsize=8)
    ax[0, 1].set_title("(b) ΔCC dose 기울기 (treat−대조)\n"
                       "[확증: vs fixed-λ — λ 자기조절의 순수 효과]")
    ax[0, 1].set_xlabel("∂ΔCC/∂frac (부트 95% CI)")
    # (c) dose→gap 곡선 (주장 A 원자료)
    _dose_panel(ax[0, 2], d["curves_gap"], "gap = coop − ALLD 보수",
                "(c) dose→협력자 상대이점 gap [주장 A 원자료]")
    # (d) Δgap 기울기 forest
    rows = [("vs fixed-λ [확증]", d["primary_A_gap_slope"])]
    rows += [(f"vs {arm_lbl[a2]}", d["delta_slopes_ctrl"][a2]["gap"])
             for a2 in d["delta_slopes_ctrl"]]
    for i, (lbl, sl) in enumerate(rows):
        ax[1, 0].plot(sl["ci"], [i, i], color="0.4")
        ax[1, 0].plot(sl["slope"], i, "o",
                      color="C0" if "확증" in lbl else "0.5")
    ax[1, 0].axvline(0, color="r", ls="--", lw=1)
    ax[1, 0].set_yticks(range(len(rows)))
    ax[1, 0].set_yticklabels([r[0] for r in rows], fontsize=8)
    ax[1, 0].set_title("(d) Δgap dose 기울기 (treat−대조)")
    ax[1, 0].set_xlabel("∂Δgap/∂frac (부트 95% CI)")
    # (e) ALLD 억제 + 협력자 보호 (최대 dose, treat vs ctrl1)
    dmax = str(doses[-1])
    sup = d["alld_suppression"]["by_dose"]
    cats = ["alld"] + list(d["coop_pay_max_dose"]["treatment"])
    tvals = [sup["treatment"][dmax][0]] + \
        [d["coop_pay_max_dose"]["treatment"][k][0]
         for k in d["coop_pay_max_dose"]["treatment"]]
    cvals = [sup["ctrl1_fixed_lambda"][dmax][0]] + \
        [d["coop_pay_max_dose"]["ctrl1_fixed_lambda"][k][0]
         for k in d["coop_pay_max_dose"]["ctrl1_fixed_lambda"]]
    xx = np.arange(len(cats)); w = 0.38
    ax[1, 1].bar(xx - w / 2, tvals, w, color="C0", label="treat(adaptive)")
    ax[1, 1].bar(xx + w / 2, cvals, w, color="C1", label="ctrl1(fixed-λ)")
    ax[1, 1].set_xticks(xx)
    ax[1, 1].set_xticklabels(cats, rotation=30, fontsize=7, ha="right")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("(e) 최대 dose 라벨별 보수\n"
                       "ALLD 억제(좌) + 협력자 보호(우 4개) [탐색]")
    ax[1, 1].set_ylabel("라운드당 보수")
    # (f) 다이애드 기제
    mech = d["mechanism"]
    actors = [a for a in ("adaptive", "tom_fixed", "adaptive_imm") if a in mech]
    xx = np.arange(len(actors)); w = 0.38
    dd_m = [mech[a]["dd_vs_alld"][0] for a in actors]
    dd_s = [mech[a]["dd_vs_alld"][1] for a in actors]
    cc_m = [mech[a]["cc_vs_coop"][0] for a in actors]
    cc_s = [mech[a]["cc_vs_coop"][1] for a in actors]
    ax[1, 2].bar(xx - w / 2, dd_m, w, yerr=dd_s, capsize=3, color="C3",
                 label="→ALLD DD율 (선택적 방어)")
    ax[1, 2].bar(xx + w / 2, cc_m, w, yerr=cc_s, capsize=3, color="C2",
                 label="→협력자 CC율 (협력 지탱)")
    ax[1, 2].set_xticks(xx); ax[1, 2].set_xticklabels(actors, fontsize=8)
    ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title("(f) keystone 기제: 착취자에게만 문을 닫고\n"
                       "협력자와는 CC 를 지탱하는가 [탐색]")
    ax[1, 2].set_ylabel("비율")
    # (g) ALLD 침입장벽 (진화 프레임)
    ev = d["evolution"]; bl = list(ev["alld_barrier"])
    for i, lbl in enumerate(bl):
        g = ev["alld_barrier"][lbl]
        ax[2, 0].plot(g["ci"], [i, i], color="0.4")
        ax[2, 0].plot(g["g"], i, "o", color="C0")
    ax[2, 0].axvline(0, color="r", ls="--", lw=1)
    ax[2, 0].set_yticks(range(len(bl)))
    ax[2, 0].set_yticklabels(bl, fontsize=8)
    dp = ev["barrier_deepening"]["adaptive"]
    ax[2, 0].set_title("(g) ALLD 침입 성장률 (상주 구성별)\n"
                       f"장벽 심화 Δg={dp['delta']:+.3f} "
                       f"CI[{dp['ci'][0]:.3f},{dp['ci'][1]:.3f}]")
    ax[2, 0].set_xlabel("g_ALLD (음수=침입 격퇴)")
    # (h) 협력자 클러스터 침입성
    cl = ev["cluster_invasion"]; ck = list(cl)
    for i, lbl in enumerate(ck):
        g = cl[lbl]
        ax[2, 1].plot(g["ci"], [i, i], color="0.4")
        ax[2, 1].plot(g["g"], i, "o", color="C2")
    ax[2, 1].axvline(0, color="r", ls="--", lw=1)
    ax[2, 1].set_yticks(range(len(ck))); ax[2, 1].set_yticklabels(ck, fontsize=8)
    ax[2, 1].set_title("(h) ALLD-heavy 상주에 대한\n협력자 클러스터 침입 성장률")
    ax[2, 1].set_xlabel("g_cluster (양수=침입 가능)")
    # (i) 후보별 협력 유역 확장
    wid = ev["basin_widening"]; wk = list(wid)
    for i, c in enumerate(wk):
        v = wid[c]
        col = "C0" if c == "adaptive" else ("C1" if c == "tom_fixed" else "0.5")
        ax[2, 2].plot(v["ci"], [i, i], color="0.4")
        ax[2, 2].plot(v["est"], i, "o", color=col)
    ax[2, 2].axvline(0, color="r", ls="--", lw=1)
    ax[2, 2].set_yticks(range(len(wk))); ax[2, 2].set_yticklabels(wk, fontsize=8)
    ok = d["supported"]["widening_gt_all_fixed"]
    ax[2, 2].set_title("(i) 유형별 협력 CC-widening(X)\n"
                       f"adaptive > 전 고정전략: {'성립' if ok else '미성립'} [탐색]")
    ax[2, 2].set_xlabel("Δ 종착 CC율 (부트 95% CI)")
    _save(fig, "h11_keystone" + tag,
          "H11: Keystone 검정 — 치환 설계(비-swap 구성·ALLD 30% 고정, dose0 "
          "전 arm 공유)에서 ΔCC·Δgap(treat−fixedλ) dose 기울기[확증 2]. "
          "탐색: ALLD 억제·협력자 보호·다이애드 기제(선택적 방어/협력 지탱)·"
          "이전 회계 가드·진화 프레임(ALLD 침입장벽 심화, 협력자 클러스터 "
          "침입성, 후보 유형별 협력 유역 확장 — adaptive vs 고정전략).",
          f"reps={d['reps']}, N={d['n_total']}, Π seeds={d['evolution']['pi_seeds']}")


# ==================================================================== fig H12
def fig_H12(d, tag=""):
    """H12 상주 조건부 Keystone 프런티어 시각화 (3×3)."""
    _kfont()
    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    axes = d["axes"]

    def _basin_panel(a, key, xlabel, title, c1=None):
        if key not in axes:
            a.set_visible(False)
            return
        ax_d = axes[key]
        xs = ax_d["coords"]

        def _band(vals, cis, color, lbl):
            m = np.array(vals)
            lo = np.array([c[0] for c in cis])
            hi = np.array([c[1] for c in cis])
            a.plot(xs, m, "o-", color=color, lw=1.6, ms=4, label=lbl)
            a.fill_between(xs, lo, hi, color=color, alpha=0.15)
        _band(ax_d["basin_adaptive"], ax_d["ci_adaptive"], "C1", "slot=adaptive")
        _band(ax_d["basin_best_fixed"], ax_d["ci_best_fixed"], "C2",
              "slot=best fixed coop")
        _band(ax_d["basin_random"], ax_d["ci_random"], "0.5", "slot=random(중립)")
        a.set_xlabel(xlabel)
        a.set_ylabel("종착 행동적 CC율 xᵀ·CCm·x")
        sub = ""
        if c1 is not None:
            sub = f"\n한계 keystone 기울기 β={c1['slope']:+.3f} " \
                  f"[{c1['ci'][0]:+.3f},{c1['ci'][1]:+.3f}]"
        a.set_title(title + sub, fontsize=9)
        a.legend(fontsize=7)

    conf = d["confirmatory"]
    # (a) R-P 위협축 (ρ_D)
    _basin_panel(ax[0, 0], "R-P", "ALLD 지분 ρ_D",
                 "(a) R-P 위협축(착취 압력) [확증 C1]", conf["C1"])
    # (b) R-K 변덕축 (ρ_C)
    _basin_panel(ax[0, 1], "R-K", "capricious 지분 ρ_C",
                 "(b) R-K 위협축(변덕) [C1 보조]", conf["C1_R-K"])
    # (c) R-N 잡음축 (err)
    _basin_panel(ax[0, 2], "R-N", "환경 잡음 err",
                 "(c) R-N 위협축(환경 잡음)",
                 axes.get("R-N", {}).get("c1_slope"))

    # (d) R-D 중복축 (다양성 m) — basin 곡선
    _basin_panel(ax[1, 0], "R-D", "협력자 다양성 m (가짓수)",
                 "(d) R-D 중복축(협력자 다양성) [상충 해소 핵심축]")
    # (e) R-D 비대체성: sub_widening(adaptive) − sub_widening(TFT) [확증 C2]
    rd = axes["R-D"]
    xs = rd["coords"]
    c2 = conf["C2"]
    m = np.array(c2["curve"])
    lo = np.array([c[0] for c in c2["curve_ci"]])
    hi = np.array([c[1] for c in c2["curve_ci"]])
    ax[1, 1].plot(xs, m, "o-", color="C3", lw=1.8, ms=5)
    ax[1, 1].fill_between(xs, lo, hi, color="C3", alpha=0.18)
    ax[1, 1].axhline(0, color="0.4", ls="--", lw=1)
    ax[1, 1].set_xlabel("협력자 다양성 m (가짓수)")
    ax[1, 1].set_ylabel("CC-widening(adaptive) − CC-widening(TFT)")
    ax[1, 1].set_title(f"(e) 비대체성(irreplaceability) [확증 C2]\n"
                       f"기울기 β={c2['slope']:+.3f} "
                       f"[{c2['ci'][0]:+.3f},{c2['ci'][1]:+.3f}] "
                       f"(<0 = 다양성↑ → adaptive 대체)", fontsize=9)
    # (f) Keystone 프런티어: basin(adaptive) − best_fixed vs m (교차 m*) [확증 C3]
    kf = d["keystone_frontier"]["R-D"]
    cr = d["keystone_frontier"]["R-D_crossing"]
    m = np.array(kf["curve"])
    lo = np.array([c[0] for c in kf["curve_ci"]])
    hi = np.array([c[1] for c in kf["curve_ci"]])
    ax[1, 2].plot(kf["coords"], m, "o-", color="C0", lw=1.8, ms=5,
                  label="R-D: adaptive − best_fixed")
    ax[1, 2].fill_between(kf["coords"], lo, hi, color="C0", alpha=0.16)
    kp = d["keystone_frontier"]["R-P"]
    ax[1, 2].plot(kp["coords"], kp["curve"], "s--", color="C4", lw=1.3, ms=4,
                  alpha=0.8, label="R-P: (참고, ρ_D 축)")
    ax[1, 2].axhline(0, color="r", ls="--", lw=1)
    if cr["within_range"] and cr["crossing"] is not None:
        ax[1, 2].axvline(cr["crossing"], color="C2", lw=1.4)
        ax[1, 2].axvspan(cr["ci"][0], cr["ci"][1], color="C2", alpha=0.12)
        sub = f"\nm* = {cr['crossing']:.2f} " \
              f"CI[{cr['ci'][0]:.2f},{cr['ci'][1]:.2f}] (부트 {cr['frac_found']:.0%})"
    else:
        sub = f"\n관측 범위 {cr['range']} 내 교차 부재 (외삽 금지)"
    ax[1, 2].set_xlabel("협력자 다양성 m (가짓수)")
    ax[1, 2].set_ylabel("keystone Δ = CC율(adaptive) − best_fixed")
    ax[1, 2].set_title("(f) Keystone 프런티어 [확증 C3]" + sub, fontsize=9)
    ax[1, 2].legend(fontsize=7)

    # (g) Shapley φ (순서무관 기여)
    sh = d["shapley"]
    cs = sh["C_set"]
    xx = np.arange(len(cs))
    vals = [sh["phi"][x] for x in cs]
    err = np.array([[sh["phi"][x] - sh["phi_ci"][x][0],
                     sh["phi_ci"][x][1] - sh["phi"][x]] for x in cs]).T
    cols = ["C0" if x == "adaptive" else "0.6" for x in cs]
    ax[2, 0].bar(xx, vals, yerr=err, capsize=4, color=cols, alpha=0.85)
    ax[2, 0].set_xticks(xx)
    ax[2, 0].set_xticklabels(cs, rotation=20, fontsize=7, ha="right")
    dphi = sh["adaptive_minus_tft"]
    ax[2, 0].set_title(f"(g) Shapley φ (C={{ad,TFT,GTFT,WSLS}})\n"
                       f"Δφ(ad−TFT)={dphi['delta']:+.3f} "
                       f"[{dphi['ci'][0]:+.3f},{dphi['ci'][1]:+.3f}] [탐색]",
                       fontsize=9)
    ax[2, 0].set_ylabel("협력 CC율 한계 기여 φ (Shapley)")

    # (h) ALLD 침입장벽 심화 Δg (random 앵커) — 상주족별
    bar = d["barrier"]
    fams = list(bar)
    xx = np.arange(len(fams))
    w = 0.38
    dg_ad = [bar[f]["deepen_adaptive"] for f in fams]
    dg_tf = [bar[f]["deepen_tom_fixed"] for f in fams]
    ax[2, 1].bar(xx - w / 2, dg_ad, w, color="C0", label="adaptive")
    ax[2, 1].bar(xx + w / 2, dg_tf, w, color="C1", label="fixed-λ ToM")
    ax[2, 1].axhline(0, color="0.4", ls="--", lw=1)
    ax[2, 1].set_xticks(xx)
    ax[2, 1].set_xticklabels(fams, rotation=25, fontsize=6.5, ha="right")
    ax[2, 1].set_ylabel("Δg = g(random 슬롯) − g(X 슬롯)")
    ax[2, 1].set_title("(h) ALLD 침입장벽 심화 Δg (random 앵커 대비)\n"
                       "양수 = 슬롯 유형이 장벽 심화 [탐색]", fontsize=9)
    ax[2, 1].legend(fontsize=7)

    # (i) H8E/H11 좌표 정위 + 사전 강건성
    tr = d["triangulation"]
    keys = list(tr)
    xx = np.arange(len(keys))
    kd = [tr[kk]["keystone_delta"] for kk in keys]
    ma = [tr[kk]["marginal_adaptive"] for kk in keys]
    ax[2, 2].bar(xx - 0.2, kd, 0.38, color="C0", label="keystone Δ (ad−best_fixed)")
    ax[2, 2].bar(xx + 0.2, ma, 0.38, color="C2", label="marginal (ad−random)")
    ax[2, 2].axhline(0, color="r", ls="--", lw=1)
    ax[2, 2].set_xticks(xx)
    ax[2, 2].set_xticklabels(["H8E 코너\n(고변덕·저중복)", "H11 코너\n(무변덕·고중복)"],
                             fontsize=7)
    ag = d["prior_agreement"]
    ax[2, 2].set_title("(i) H8E/H11 좌표 정위 (곡면 위 두 점)\n"
                       f"사전 강건성: C1 일치={ag['C1']['same_sign']}, "
                       f"C2 일치={ag['C2']['same_sign']} [탐색]", fontsize=9)
    ax[2, 2].set_ylabel("CC율 차이")
    ax[2, 2].legend(fontsize=7)

    sup = d["supported"]
    _save(fig, "h12_keystone_frontier" + tag,
          "H12: 상주 조건부 Keystone 프런티어. 고정 심플렉스 S*(11종)·중립 filler "
          "대치로 차원·배경·척도 인공물을 제거한 뒤(원칙 A·B), 위협축(R-P ρ_D "
          "[확증 C1], R-K ρ_C, R-N err)과 중복축(R-D 다양성 m)을 따라 대치 기반 "
          "협력 CC-widening 을 곡면으로 추정(행동적 CC율). [확증] C1 위협 단조성·C2 "
          "비대체성 단조성·C3 keystone 프런티어 교차 m*. [탐색] Shapley 순서무관 "
          "기여·ALLD 침입장벽 심화(random 앵커)·구조적 사전 일치·H8E(고변덕·저중복) "
          "와 H11(무변덕·고중복)을 곡면 위 두 점으로 재현·정위. "
          f"판정: C1={sup['C1_threat_monotone']}, C2={sup['C2_redundancy_monotone']}, "
          f"C3={sup['C3_frontier_exists']}.",
          f"Π* seeds={d['config']['pi_seeds']}, S*=11종, T={d['config']['T_main']}, "
          f"slot_share={d['config']['slot_share']}, n_basin={d['config']['n_basin']}")


def fig_GS(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cis = d["cis"]; R = d["results"]
    ax[0].bar(range(len(cis)), [R[c]["expl_diff_mean"] for c in cis],
              color=["C0" if R[c]["expl_diff_mean"] < 0 else "C3" for c in cis])
    ax[0].axhline(0, color="0.5", lw=1); ax[0].set_xticks(range(len(cis)))
    ax[0].set_xticklabels([f"CI={c}" for c in cis])
    ax[0].set_title("H1 착취가능성 차 (adaptive−fixed)")
    ax[0].set_ylabel("Δ exploitability")
    w = 0.26
    ax[1].bar(np.arange(len(cis)) - w, [R[c]["h7_pay_adaptive"] for c in cis],
              w, label="adaptive", color="C0")
    ax[1].bar(np.arange(len(cis)), [R[c]["h7_pay_gtft"] for c in cis],
              w, label="GTFT(확률)", color="C1")
    ax[1].bar(np.arange(len(cis)) + w,
              [R[c].get("h7_pay_gtft_count", np.nan) for c in cis],
              w, label="GTFT(횟수)", color="C4")
    ax[1].set_xticks(range(len(cis))); ax[1].set_xticklabels([f"CI={c}" for c in cis])
    ax[1].set_title("H7 총보수 (3 case; GTFT 두 유형)"); ax[1].legend(fontsize=7)
    w = 0.35
    ax[2].bar(np.arange(len(cis)) - w / 2, [R[c]["h8_cc_frac0"] for c in cis],
              w, label="frac=0", color="0.6")
    ax[2].bar(np.arange(len(cis)) + w / 2, [R[c]["h8_cc_frac05"] for c in cis],
              w, label="frac=0.5", color="C0")
    ax[2].set_xticks(range(len(cis))); ax[2].set_xticklabels([f"CI={c}" for c in cis])
    ax[2].set_title("H8 소집단 CC (frac 방향)"); ax[2].legend(fontsize=8)
    _save(fig, "gs_game_structure" + tag,
          "GS[탐색]: 협력지수 CI 스윕에서 핵심 결론(H1 부호, H7 열세, H8 방향)의 "
          "강건성 지도화. 확증 지표 아님(사전등록).",
          f"CI∈{d['cis']}")


def fig_H9_H10(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    pc = d["payoff_capricious"]; order = ["all", "alpha_only", "lambda_only"]
    bar_ci(ax[0], range(3), [pc[k][0] for k in order],
           [pc[k][1] for k in order], order, colors=["C0", "C4", "C5"])
    # 베이스라인 병기
    bc = d["baseline_capricious"]
    for i, (bn, v) in enumerate(bc.items()):
        ax[0].axhline(v[0], color=f"C{6+i}", ls="--", lw=1, label=bn)
    ax[0].set_title("H9 변덕 상대 보수 [확증]\n(+비-ToM 베이스라인)")
    ax[0].set_ylabel("라운드당 보수"); ax[0].legend(fontsize=7)
    ax[0].tick_params(axis="x", labelrotation=20)
    react = d["lam_react"]
    ax[1].bar(range(3), [react[k] for k in order], color=["C0", "C4", "C5"])
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(order, fontsize=8, rotation=20)
    ax[1].set_title("λ 반응성 (std)"); ax[1].set_ylabel("std(λ)")
    cs = d["cc_static"]
    bar_ci(ax[2], range(3), [cs[k][0] for k in order],
           [cs[k][1] for k in order], order, colors=["C0", "C4", "C5"])
    ax[2].set_title("H10 정적 상대 CC [확증]"); ax[2].set_ylabel("CC rate")
    ax[2].tick_params(axis="x", labelrotation=20)
    _save(fig, "h9_h10_attribution_scope" + tag,
          "H9/H10: 완전 귀인(all)이 α-only(변덕 추적)·λ-only(정적 CC)를 모두 "
          "앞선다. 비-ToM 베이스라인 병기로 ToM 이득을 학습 이득과 분리.",
          f"seeds={pc['all'][1] and ''}")


# ==================================================================== main
# 지평별 실행 실험 (T=60/240 각각) vs 전지평 실험 (내부 T 스윕; 1회 실행)
# -------------------------------------------------------------- VP / ABA / ORE 그림
def _conf_panel(ax, groups, labels, threshold, direction, stats_txt, title,
                ylabel, colors=None, thr_label=None):
    """
    확증 replicate 패널: 그룹별 지터 산점 + 그룹/전체 평균±부트95%CI + 임계선 +
    통계 주석. direction: 'greater'(임계 초과 지지) / 'less'(임계 미만 지지).
    """
    rng = np.random.default_rng(0)
    pooled = np.concatenate([np.asarray(g, float) for g in groups])
    for gi, g in enumerate(groups):
        g = np.asarray(g, float)
        xj = gi + rng.uniform(-0.16, 0.16, len(g))
        ax.scatter(xj, g, s=14, alpha=0.45,
                   color=(colors[gi] if colors else "C0"), zorder=2)
        ci = boot_mean_ci(g, seed=101 + gi)["ci"]
        ax.errorbar([gi], [g.mean()], yerr=[[g.mean() - ci[0]], [ci[1] - g.mean()]],
                    fmt="o", color="k", capsize=4, ms=6, zorder=3)
    # pooled 평균 CI (우측 별도 위치)
    xp = len(groups) - 0.5 + 0.9
    cip = boot_mean_ci(pooled, seed=7)["ci"]
    ax.errorbar([xp], [pooled.mean()],
                yerr=[[pooled.mean() - cip[0]], [cip[1] - pooled.mean()]],
                fmt="D", color="C3", capsize=5, ms=8, zorder=4, label="전체 평균±95%CI")
    ax.axhline(threshold, color="k", ls="--", lw=1.2,
               label=thr_label or f"판정 임계 {threshold:g}")
    ax.set_xticks(list(range(len(groups))) + [xp])
    ax.set_xticklabels(list(labels) + ["pooled"], rotation=20, fontsize=8)
    ok = (pooled.mean() > threshold) if direction == "greater" else (pooled.mean() < threshold)
    ax.set_title(title + ("  [방향 성립 ✓]" if ok else "  [방향 불성립 ✗]"), fontsize=10)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="best")
    ax.annotate(stats_txt, xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


def fig_VP_confirmatory(d, tag=""):
    """[확증 전용] C-VP1 / C-VP2 replicate 분포·CI·p 를 명시적으로 시각화."""
    _kfont()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    vs = d["seeds"]
    # C-VP1: 시변 레짐별 per-seed payoff 우위 (레짐 순서로 분할)
    rep1 = np.asarray(d["cvp1_replicates"], float)
    g1 = [rep1[i * vs:(i + 1) * vs] for i in range(len(d["variable"]))]
    ci1 = boot_mean_ci(rep1, seed=7)["ci"]
    _conf_panel(ax[0], g1, d["variable"], 0.0, "greater",
                f"Δpay={rep1.mean():+.3f} [{ci1[0]:.3f},{ci1[1]:.3f}]\n"
                f"p_raw={d['cvp1_p']:.4g}, n={len(rep1)} (시드×레짐)\n"
                f"Holm 보정은 요약 로그 참조",
                "[확증 C-VP1] 시변 레짐 adaptive payoff 우위 > 0",
                "Δ 라운드당 보수 (adaptive − best 고정)",
                colors=["C0", "C1", "C2"], thr_label="0 (우위 없음)")
    # C-VP2: 시변 레짐별 per-seed CC-widening
    rep2 = np.asarray(d["cvp2_replicates"], float)
    g2 = [rep2[i * vs:(i + 1) * vs] for i in range(len(d["variable"]))]
    _conf_panel(ax[1], g2, d["variable"], 0.0, "greater",
                f"CC-widening={rep2.mean():+.3f} "
                f"[{d['cvp2_ci'][0]:.3f},{d['cvp2_ci'][1]:.3f}]\n"
                f"p_raw={d['cvp2_p']:.4g}, n={len(rep2)} (시드×레짐)",
                "[확증 C-VP2] 시변 레짐 협력 CC-widening > 0",
                "Δ 종착 행동적 CC율 (adaptive − random 슬롯)",
                colors=["C0", "C1", "C2"], thr_label="0 (확장 없음)")
    _save(fig, "vp_confirmatory" + tag,
          "§VP 확증 전용 시각화: C-VP1(좌) 시변 레짐 payoff 우위와 C-VP2(우) 협력 "
          "CC-widening 의 시드×레짐 replicate 분포·그룹/전체 평균±부트95%CI·판정 "
          "임계(0)·순열 p. 전체 평균 CI 가 0 을 배제하고 p<0.05(Holm) 면 지지.",
          f"seeds={vs}, T={d['T']}, replicate=시드×레짐")


def fig_VP(d, tag=""):
    _kfont()
    regs = d["regimes"]; adv = d["advantage"]; basin = d["basin"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    # (a) 레짐별 adaptive payoff 우위(best 고정 대비) + CI
    xs = np.arange(len(regs))
    means = [adv[r]["mean"] for r in regs]
    cis = [adv[r]["ci"] for r in regs]
    cols = ["0.6" if r == "mild_pd" else "C0" for r in regs]
    bar_ci(ax[0], xs, means, cis, regs, colors=cols)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("(a) adaptive payoff 우위 = adaptive − best 고정전략\n"
                    "[확증 C-VP1] 변동/극단 레짐(파랑)")
    ax[0].set_ylabel("Δ 라운드당 보수"); ax[0].tick_params(axis="x", rotation=30)
    _n_note(ax[0], d["seeds"])
    # (b) 레짐별 focal 유형 payoff 프로파일 (맥락-의존 vs 무감)
    focal = list(next(iter(d["pay_mean"].values())).keys())
    for f in focal:
        ys = [d["pay_mean"][r][f] for r in regs]
        ax[1].plot(xs, ys, marker="o", label=f,
                   lw=2.2 if f == "adaptive" else 1.2,
                   color="C0" if f == "adaptive" else None,
                   zorder=3 if f == "adaptive" else 1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(regs, rotation=30)
    ax[1].set_title("(b) CI 레짐별 focal 보수 프로파일")
    ax[1].set_ylabel("라운드당 보수"); ax[1].legend(fontsize=7, ncol=2)
    # (c) 협력 CC-widening (종착 행동적 CC율; RE vs ORE)
    w = 0.38
    ax[2].bar(xs - w / 2, [basin[r]["re_cc_widening"] for r in regs], w,
              label="RE CC-widening", color="C1", alpha=0.85)
    ax[2].bar(xs + w / 2, [basin[r]["ore_cc_widening"] for r in regs], w,
              label="ORE CC-widening", color="C2", alpha=0.85)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(regs, rotation=30)
    ax[2].set_title("(c) adaptive 협력 CC-widening\n[확증 C-VP2] filler=random 대치")
    ax[2].set_ylabel("Δ 종착 행동적 CC율"); ax[2].legend(fontsize=8)
    _save(fig, "vp_variable_payoff" + tag,
          "§VP 가변 페이오프(연속 협력–경쟁): CI 를 극단(음수·1 이상)까지 변동시키는 "
          "환경에서 맥락-의존 효용을 계산하는 adaptive 가 payoff-무감 고정전략 대비 얻는 "
          "성능 우위(a,b)와 협력 확장(c). 지표는 이분 유역을 폐기하고 종착 조성의 "
          "행동적 CC율 xᵀ·CCm·x(연속·라벨무관)로 개정. 고정 mild_pd(회색) 대비 변동/극단 "
          "레짐에서 우위가 나타남 — H7/H8E/H12 '고정 payoff' 교란 제거.",
          f"seeds={d['seeds']}, T={d['T']}, 지표=행동적 CC율")
    fig_VP_confirmatory(d, tag)


def fig_ABA(d, tag=""):
    _kfont()
    Tref = d["Tref"]; ref = d["by_T"][Tref]; third = ref["third"]
    FORG = d["forg_levels"]; forg = ref["forg"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    # (a) pred_coop 궤적 (A→B→A): 저용서 vs 고용서 (히스테리시스 vs 복구)
    lo, hi = min(FORG), max(FORG)
    band(ax[0], forg[lo]["pred_coop"], f"용서={lo} (저)", color="C3", ls="--")
    band(ax[0], forg[hi]["pred_coop"], f"용서={hi} (고)", color="C0")
    for c in (third, 2 * third):
        ax[0].axvline(c, color="0.5", ls=":", lw=1)
    ax[0].set_title("(a) 상대 협력확률 추론 pred_coop\nA(협력)→B(착취)→A(협력)")
    ax[0].set_xlabel("라운드"); ax[0].set_ylabel("pred_coop"); ax[0].legend(fontsize=8)
    ax[0].annotate("A1", (third * 0.4, 0.03), color="0.4")
    ax[0].annotate("B", (third * 1.4, 0.03), color="0.4")
    ax[0].annotate("A3", (third * 2.4, 0.03), color="0.4")
    _n_note(ax[0], forg[hi]["pred_coop"].shape[0])
    # (b) 용량-반응: 용서별 복구오차 + CC 복원비 [확증 C-ABA1/2]
    xs = np.arange(len(FORG))
    rec_means = [forg[fg]["rec_err"].mean() for fg in FORG]
    rec_cis = [boot_mean_ci(forg[fg]["rec_err"], seed=i)["ci"] for i, fg in enumerate(FORG)]
    bar_ci(ax[1], xs, rec_means, rec_cis, [str(f) for f in FORG], colors="C0")
    ax[1].axhline(0.15, color="k", ls="--", lw=1, label="복구 임계 0.15")
    ax[1].set_title("(b) 복구오차 vs 용서(재탐색)\n[확증 C-ABA2 기울기<0, C-ABA1 高용서<0.15]")
    ax[1].set_xlabel("용서(forgiveness)"); ax[1].set_ylabel("복구오차 |pc(A3)−pc(A1)|")
    ax[1].legend(fontsize=8); _n_note(ax[1], len(forg[FORG[0]]["rec_err"]))
    # (c) CC 복원: A1 vs A3, 용서 수준 + 절제/베이스라인
    groups = [f"용서{f}" for f in FORG] + list(ref["abl"].keys()) + list(ref["base_cc"].keys())
    cc1, cc3 = [], []
    for fg in FORG:
        cc1.append(forg[fg]["cc_A1"].mean()); cc3.append(forg[fg]["cc_A3"].mean())
    for name, m in ref["abl"].items():
        cc1.append(m["cc_A1"].mean()); cc3.append(m["cc_A3"].mean())
    for b, m in ref["base_cc"].items():
        cc1.append(m["cc_A1"].mean()); cc3.append(m["cc_A3"].mean())
    xg = np.arange(len(groups)); w = 0.4
    ax[2].bar(xg - w / 2, cc1, w, label="phase A1(전)", color="0.7")
    ax[2].bar(xg + w / 2, cc3, w, label="phase A3(복귀)", color="C0")
    ax[2].set_xticks(xg); ax[2].set_xticklabels(groups, rotation=35, ha="right", fontsize=7)
    ax[2].set_title("(c) CC 복원: A1 vs A3")
    ax[2].set_ylabel("CC 율"); ax[2].legend(fontsize=8)
    _save(fig, "aba_intent_recovery" + tag,
          "§ABA A→B→A 의도 전환 추적 복구: 배신(B) 후 자기보호 고착으로 A 복귀 시 "
          "자동 복구가 일어나지 않는 히스테리시스(배신 후 선택적 관측 함정)를, 용서"
          "(재탐색) 수준이 게이팅한다. 高용서에서 pred_coop·CC 가 복원되며(a,c), 복구"
          "오차는 용서에 단조 감소한다(b, 용량-반응). β-절제·통제권·비-ToM 는 대조.",
          f"seeds={len(forg[FORG[0]]['rec_err'])}, T={Tref}")
    fig_ABA_confirmatory(d, tag)


def fig_ABA_confirmatory(d, tag=""):
    """[확증 전용] C-ABA1 / C-ABA2 를 명시적으로 시각화."""
    _kfont()
    Tref = d["Tref"]; forg = d["by_T"][Tref]["forg"]; FORG = d["forg_levels"]
    hi = max(FORG)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    conf = d["confirmatory"]
    # (a) C-ABA1a: 高용서 복구오차 < 0.15
    rec = np.asarray(forg[hi]["rec_err"], float)
    _conf_panel(ax[0], [rec], [f"용서={hi}"], 0.15, "less",
                f"복구오차={rec.mean():.3f} "
                f"[{conf['C_ABA1_recovery']['ci'][0]:.3f},"
                f"{conf['C_ABA1_recovery']['ci'][1]:.3f}]" + "\n"
                f"p_raw={conf['C_ABA1_recovery']['p']:.4g}, n={len(rec)}, T={Tref}",
                "[확증 C-ABA1a] 高용서 의도추론 복구오차 < 0.15",
                "복구오차 |pc(A3)−pc(A1)|", thr_label="복구 임계 0.15")
    # (b) C-ABA1b: 高용서 CC 복원비 > 0.70
    ccr = np.asarray(forg[hi]["cc_ratio"], float)
    _conf_panel(ax[1], [ccr], [f"용서={hi}"], 0.70, "greater",
                f"CC복원비={ccr.mean():.2f}, n={len(ccr)}, T={Tref}" + "\n"
                "(C-ABA1 은 (a)AND(b) 결합 판정)",
                "[확증 C-ABA1b] 高용서 CC 복원비 CC(A3)/CC(A1) > 0.70",
                "CC 복원비", thr_label="복원 임계 0.70")
    # (c) C-ABA2: 용량-반응 산점 + 기울기 CI 밴드
    rng = np.random.default_rng(1)
    xs_all, ys_all = [], []
    for fg in FORG:
        r = np.asarray(forg[fg]["rec_err"], float)
        ax[2].scatter(fg + rng.uniform(-0.008, 0.008, len(r)), r,
                      s=14, alpha=0.45, color="C0")
        xs_all.extend([fg] * len(r)); ys_all.extend(r.tolist())
    xs_all = np.asarray(xs_all); ys_all = np.asarray(ys_all)
    sl = conf["C_ABA2_dose_response"]
    xg = np.linspace(min(FORG), max(FORG), 50)
    b0 = ys_all.mean() - sl["slope"] * xs_all.mean()
    ax[2].plot(xg, b0 + sl["slope"] * xg, color="C3", lw=2,
               label=f"기울기={sl['slope']:.3f}")
    ax[2].fill_between(xg, b0 + sl["ci"][0] * (xg - xs_all.mean()) + sl["slope"] * xs_all.mean(),
                       b0 + sl["ci"][1] * (xg - xs_all.mean()) + sl["slope"] * xs_all.mean(),
                       color="C3", alpha=0.15, label="기울기 95%CI 밴드")
    ax[2].set_title("[확증 C-ABA2] 복구오차∼용서 기울기 < 0"
                    + ("  [방향 성립 ✓]" if sl["slope"] < 0 else "  [✗]"), fontsize=10)
    ax[2].set_xlabel("용서(forgiveness)"); ax[2].set_ylabel("복구오차")
    ax[2].legend(fontsize=8)
    ax[2].annotate(f"slope={sl['slope']:.3f} [{sl['ci'][0]:.3f},{sl['ci'][1]:.3f}]\n"
                   f"p_raw={sl['p']:.4g}",
                   xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, va="bottom",
                   bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    _save(fig, "aba_confirmatory" + tag,
          "§ABA 확증 전용 시각화: C-ABA1(a·b 결합 — 高용서 복구오차<0.15 AND CC복원비"
          ">0.70)과 C-ABA2(c — 복구오차∼용서 용량-반응 기울기<0, 부트 CI 상한<0). "
          "replicate=시드, 판정은 최장 지평 기준.",
          f"seeds={len(rec)}, Tref={Tref}")


def fig_ORE(d, tag=""):
    _kfont()
    repro = d["reproduction"]; per = d["per_regime"]; regs = d["regimes"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    # (a) Bravetti Fig.1 재현: model1 의 RE vs ORE x_C(t)
    for m, color in [("model1", "C0"), ("model2", "C1"), ("model3", "C2")]:
        r = repro[m]
        t = np.linspace(0, 1, len(r["re_xC"]))
        ax[0].plot(t, r["re_xC"], color=color, ls="--", lw=1.4,
                   label=f"{m} RE")
        ax[0].plot(t, r["ore_xC"], color=color, ls="-", lw=2.0,
                   label=f"{m} ORE")
    ax[0].set_title("(a) Bravetti&Padilla 재현: 협력자 빈도 x_C(t)\nRE(점선)→소멸, ORE(실선)→창발")
    ax[0].set_xlabel("정규화 시간 t/τ"); ax[0].set_ylabel("x_C"); ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_ylim(-0.02, 1.02)
    # (b) 레짐별 RE vs ORE 종착 행동적 CC율 (CI 오차막대)
    xs = np.arange(len(regs)); w = 0.38
    re_m = [per[r]["re_cc"] for r in regs]; ore_m = [per[r]["ore_cc"] for r in regs]
    re_ci = [per[r]["re_cc_ci"] for r in regs]; ore_ci = [per[r]["ore_cc_ci"] for r in regs]
    err_re = np.array([[max(0.0, m - c[0]), max(0.0, c[1] - m)]
                       for m, c in zip(re_m, re_ci)]).T
    err_ore = np.array([[max(0.0, m - c[0]), max(0.0, c[1] - m)]
                        for m, c in zip(ore_m, ore_ci)]).T
    ax[1].bar(xs - w / 2, re_m, w, yerr=err_re, capsize=3, label="RE", color="C1", alpha=0.85)
    ax[1].bar(xs + w / 2, ore_m, w, yerr=err_ore, capsize=3, label="ORE", color="C2", alpha=0.85)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(regs, rotation=20)
    ax[1].set_title("(b) 종착 행동적 CC율: RE vs ORE\n[확증 C-ORE1] ORE>RE")
    ax[1].set_ylabel("종착 CC율  xᵀ·CCm·x"); ax[1].legend(fontsize=8)
    # (c) adaptive CC-widening: RE vs ORE (레짐별)
    sw_re = [per[r]["re_cc_widening"] for r in regs]
    sw_ore = [per[r]["ore_cc_widening"] for r in regs]
    ax[2].bar(xs - w / 2, sw_re, w, label="RE", color="C1", alpha=0.85)
    ax[2].bar(xs + w / 2, sw_ore, w, label="ORE", color="C2", alpha=0.85)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(regs, rotation=20)
    ax[2].set_title("(c) adaptive 한계 기여 CC-widening\n[확증 C-ORE2] ORE 하 > 0")
    ax[2].set_ylabel("Δ 종착 CC율 (adaptive−random)"); ax[2].legend(fontsize=8)
    _save(fig, "ore_optimal_replicator" + tag,
          "§ORE 집단 수준 경쟁(Optimal Replicator Equation; Bravetti&Padilla 2018): "
          "2-유형 재현(a; RE→배신 지배, ORE→협력 창발). 지표는 이분 유역을 폐기하고 "
          "종착 조성의 **행동적 CC율** xᵀ·CCm·x(연속·라벨무관)로 개정: ORE 가 실제 "
          "상호협력을 RE 대비 높이고(b), adaptive 의 한계 기여가 뚜렷(c). deadlock 은 "
          "CC율이 낮게 유지되어(라벨이 아닌 행동 반영) 과거 이분 유역의 인공적 고점이 제거됨.",
          f"seeds={d['seeds']}, T={d['T']}, 지표=행동적 CC율")
    fig_ORE_confirmatory(d, tag)


def fig_ORE_confirmatory(d, tag=""):
    """[확증 전용] C-ORE1 / C-ORE2 를 명시적으로 시각화."""
    _kfont()
    regs = d["regimes"]; per = d["per_regime"]; conf = d["confirmatory"]
    cols = [f"C{i}" for i in range(len(regs))]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # (a) C-ORE1: 시드×레짐 짝지은 RE vs ORE 산점 (y=x 위 = ORE 우위)
    lo, hi_v = 1.0, 0.0
    for ci_, reg in enumerate(regs):
        re_s = np.asarray(per[reg]["seed_re"], float)
        ore_s = np.asarray(per[reg]["seed_ore"], float)
        ax[0].scatter(re_s, ore_s, s=18, alpha=0.6, color=cols[ci_], label=reg)
        lo = min(lo, re_s.min(), ore_s.min()); hi_v = max(hi_v, re_s.max(), ore_s.max())
    pad = 0.03 * (hi_v - lo + 1e-9)
    ax[0].plot([lo - pad, hi_v + pad], [lo - pad, hi_v + pad], "k--", lw=1.2,
               label="y=x (효과 없음)")
    c1 = conf["C_ORE1_cc_rate"]
    ax[0].set_title("[확증 C-ORE1] ORE 종착 CC율 > RE 종착 CC율"
                    + ("  [방향 성립 ✓]" if c1["delta"] > 0 else "  [✗]"), fontsize=10)
    ax[0].set_xlabel("RE 종착 CC율 (시드별)"); ax[0].set_ylabel("ORE 종착 CC율 (시드별)")
    ax[0].legend(fontsize=7)
    ax[0].annotate(f"ΔCC={c1['delta']:+.3f} [{c1['ci'][0]:.3f},{c1['ci'][1]:.3f}]" + "\n"
                   + f"p_raw={c1['p']:.4g}, n={c1['n']} (시드×레짐 짝지음)",
                   xy=(0.02, 0.98), xycoords="axes fraction", fontsize=8, va="top",
                   bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    # (b) C-ORE2: 레짐별 시드 CC-widening replicate + pooled CI
    g2 = [np.asarray(per[reg]["seed_sw_ore"], float) for reg in regs]
    c2 = conf["C_ORE2_adaptive_cc_widening"]
    _conf_panel(ax[1], g2, regs, 0.0, "greater",
                f"CC-widening={c2['cc_widening']:+.3f} "
                f"[{c2['ci'][0]:.3f},{c2['ci'][1]:.3f}]" + "\n"
                + f"p_raw={c2['p']:.4g}, n={c2['n']} (시드×레짐)",
                "[확증 C-ORE2] ORE 하 adaptive CC-widening > 0",
                "Δ 종착 CC율 (adaptive − random 슬롯)",
                colors=cols, thr_label="0 (기여 없음)")
    _save(fig, "ore_confirmatory" + tag,
          "§ORE 확증 전용 시각화: C-ORE1(좌) 시드×레짐 짝지은 RE vs ORE 종착 CC율 "
          "산점(y=x 위 = ORE 우위)과 C-ORE2(우) ORE 하 adaptive CC-widening 의 "
          "레짐별 replicate 분포·평균±부트95%CI·판정 임계(0)·순열 p.",
          f"seeds={d['seeds']}, T={d['T']}, replicate=시드×레짐")


EXPERIMENTS = {
    "H1": (exp_H1, fig_H1),
    "H2H3": (exp_H2_H3, fig_H2_H3),
    "H4": (exp_H4, fig_H4),
    "H5": (exp_H5, fig_H5),
    "H6": (exp_H6, fig_H6),
    "H7": (exp_H7, fig_H7),
    "H7H": (exp_H7H, fig_H7H),
    "H8": (exp_H8, fig_H8),
    "H8E": (exp_H8E, fig_H8E),
    "H9H10": (exp_H9_H10, fig_H9_H10),
    "H11": (exp_H11, fig_H11),
    "H12": (exp_H12, fig_H12),
    "GS": (exp_GS, fig_GS),
    "VP": (exp_VP, fig_VP),
    "ABA": (exp_ABA, fig_ABA),
    "ORE": (exp_ORE, fig_ORE),
}
GLOBAL_EXPERIMENTS = {"H7H", "H8E", "H12", "VP", "ABA", "ORE"}   # 내부 err/T/레짐 스윕 — rounds 목록을 통째로 전달


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k != "traces"}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def apply_corrections():
    """전 가설 종료 후: 1차 지표 Holm(가족오류율), 탐색 지표 BH-FDR (§1)."""
    if PRIMARY:
        praw = [x["p"] for x in PRIMARY]
        for x, ph in zip(PRIMARY, holm(praw)):
            x["p_holm"] = ph
    if EXPLORATORY:
        praw = [x["p"] for x in EXPLORATORY]
        for x, q in zip(EXPLORATORY, bh_fdr(praw)):
            x["q_fdr"] = q


# ------------------------------------------------------ 지평(T) 비교·해석
def _horizon_metrics(name: str, out: dict) -> dict:
    """지평 비교용 핵심 지표 추출 (실험별)."""
    try:
        if name == "H1":
            r = out["raw"]
            return {"착취가능성 Δ(adaptive−fixed)":
                    float(np.mean(r["expl_ad"]) - np.mean(r["expl_fx"]))}
        if name == "H2H3":
            rec = out["ablation"]["recovery_means"]
            return {"λ 회복량(all)": float(rec["all"]),
                    "E[β] 판별 Δ": float(out["primary_H3"]["mean_a"]
                                         - out["primary_H3"]["mean_b"])}
        if name == "H4":
            r = out["raw"]
            return {"CC(adaptive)": float(np.mean(r["cc_ad"])),
                    "CC(fixed)": float(np.mean(r["cc_fx"]))}
        if name == "H5":
            r = out["raw"]
            return {"방어량 Δ(즉각−정교)":
                    float(np.mean(r["defense_imm"]) - np.mean(r["defense_soph"])),
                    "noisyTFT 라운드당 보수 Δ(즉각−정교)":
                    float((np.mean(r["payoff_imm"]) - np.mean(r["payoff_soph"]))
                          / max(len(out["traces"].get("cumpay|immediate|noisy_tft",
                                                      [[0]])[0]), 1))}
        if name == "H6":
            sc = out["scores"]["0.15"]
            return {"잡음0.15 GTFT−TFT": float(sc["generous_tft"]
                                               - sc["tit_for_tat"])}
        if name == "H7":
            m = out["means"]
            d = {"전 case Δ(adaptive−GTFT확률)": float(m["adaptive"]
                                                      - m["generous_tft"])}
            if "generous_tft_count" in m:
                d["전 case Δ(adaptive−GTFT횟수)"] = float(
                    m["adaptive"] - m["generous_tft_count"])
            if "wsls" in m:
                d["전 case Δ(adaptive−WSLS)"] = float(m["adaptive"]
                                                      - m["wsls"])
            for P, v in out["period_stats"].items():
                d[f"Δ(P={P})"] = float(v["delta_mean"])
            return d
        if name == "H8":
            return {"λ=0.4 조건부 frac 기울기": float(out["primary"]["est"])}
        if name == "H11":
            return {"주장B ΔCC dose 기울기":
                    float(out["primary_B_cc_slope"]["slope"]),
                    "주장A Δgap dose 기울기":
                    float(out["primary_A_gap_slope"]["slope"]),
                    "ALLD 장벽 심화 Δg": float(
                        out["evolution"]["barrier_deepening"]["adaptive"]
                        ["delta"])}
        if name == "H9H10":
            pc = out["payoff_capricious"]; cs = out["cc_static"]
            return {"변덕 보수 Δ(all−α-only)":
                    float(pc["all"][0] - pc["alpha_only"][0]),
                    "정적 CC Δ(all−λ-only)":
                    float(cs["all"][0] - cs["lambda_only"][0])}
        if name == "GS":
            return {f"CI={c} H8 방향": out["results"][c]["h8_direction_positive"]
                    for c in out["cis"]}
    except Exception as e:                                    # pragma: no cover
        LOGGER.warning("지평 지표 추출 실패 (%s): %s", name, e)
    return {}


def _flat_supported(sup) -> dict:
    if isinstance(sup, dict):
        return {k: v for k, v in sup.items() if isinstance(v, (bool, np.bool_))}
    if sup is None:
        return {}
    return {"supported": bool(sup)}


def interpret_horizons(name: str, outs: dict) -> dict:
    """
    T=60 vs T=240 결과 비교 + 해석 (지지 여부가 유의미하게 달라지는 경우 명시).
    outs : {T: 실험 출력}
    """
    Ts = sorted(outs)
    sup = {f"T{T}": _flat_supported(outs[T].get("supported")) for T in Ts}
    mets = {f"T{T}": _horizon_metrics(name, outs[T]) for T in Ts}
    keys = set().union(*(set(v) for v in sup.values()))
    flips = {k: {f"T{T}": sup[f"T{T}"].get(k) for T in Ts}
             for k in keys
             if len({sup[f"T{T}"].get(k) for T in Ts}) > 1}
    if flips:
        text = (f"{name}: 지지 여부가 지평 의존적 — {flips}. "
                "H7H 의 기제(Δ(T) ≈ b·T − c·k: 학습·화해의 고정비용이 긴 "
                "지평에서 상환, T*≈170 부근 교차)와 정합하는지 지표 방향으로 "
                "확인할 것. 짧은 지평(T=60)의 결론을 긴 지평으로 외삽하지 말 것.")
    else:
        text = (f"{name}: 지지 여부가 두 지평(T={Ts})에서 불변. "
                "핵심 지표의 크기 변화는 metrics_by_T 참조.")
    LOGGER.info("[지평 비교][%s] %s", name, text)
    return {"supported_by_T": sup, "metrics_by_T": mets,
            "support_flips": _jsonable(flips), "interpretation": text}


def fig_horizon_overview(summary: dict, Ts: list):
    """확증(1차) 지표의 지평별 판정 개관 그림."""
    if len(Ts) < 2 or not PRIMARY:
        return
    _kfont()
    rows = {}
    for x in PRIMARY:
        base = x["label"].rsplit(" [", 1)[0]
        rows.setdefault((x["hyp"], base), {})[x.get("tag") or "전지평"] = x
    labels = [f"[{h}] {b}" for (h, b) in rows]
    cols = [f"T{T}" for T in Ts] + ["전지평"]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2))
    alpha_f = PREREG.get("alpha_family", 0.05)
    for i, key in enumerate(rows):
        for j, col in enumerate(cols):
            x = rows[key].get(col)
            if x is None:
                continue
            ok_dir = x.get("direction_met")
            ok_p = x.get("p_holm", 1.0) < alpha_f
            c = "C2" if (ok_dir and ok_p) else ("C1" if ok_dir else "C3")
            ax.scatter(j, i, s=160, color=c, marker="s")
            ax.text(j, i, "" if ok_dir and ok_p else ("d" if ok_dir else "×"),
                    ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("확증(1차) 지표 지평별 판정 — 초록=방향+Holm 유의, "
                 "주황(d)=방향만, 빨강(×)=방향 미성립")
    _save(fig, "horizon_overview",
          "지평(T=60/240)별 확증 지표 판정 개관. 지지가 지평 의존적인 가설은 "
          "행 내 색 변화로 드러난다 (H7H 의 Δ(T) 기제 참조).",
          f"Ts={Ts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=240)
    ap.add_argument("--rounds", type=int, nargs="+", default=[60, 240],
                    help="지평 목록 — 각 지평에서 전 실험을 반복 (기본 60 240)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--backend", choices=["numpy", "pymdp"], default="numpy")
    ap.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true",
                    help="스모크: seeds=3, rounds=[30]")
    ap.add_argument("--check-equivalence", action="store_true",
                    help="pymdp↔numpy EFE 등가성만 검증하고 종료")
    args = ap.parse_args()
    if args.check_equivalence:
        from AIF_IPD.core.pymdp_backend import PymdpEFE, pymdp_available
        if not pymdp_available():
            LOGGER.warning("pymdp(JAX) 미설치 — 등가성 검증 불가")
            return
        ok = PymdpEFE.check_equivalence()
        LOGGER.info("pymdp↔numpy EFE 등가성: %s", "통과" if ok else "실패")
        return
    if args.quick:
        args.seeds, args.rounds = 3, [30]
    Ts = sorted(set(args.rounds))
    LOGGER.info("=== HalloReg v0.3 — seeds=%d rounds=%s jobs=%d backend=%s ===",
                args.seeds, Ts, args.jobs, args.backend)
    LOGGER.info("사전등록: %s (α_family=%s, FDR q=%s)", _PREREG_PATH.name,
                PREREG.get("alpha_family"), PREREG.get("fdr_q"))

    global CUR_TAG
    summary = {"config": {**vars(args), "rounds": Ts}, "results": {}}
    for name in args.experiments:
        if name not in EXPERIMENTS:
            LOGGER.warning("알 수 없는 실험 '%s' 건너뜀", name)
            continue
        fn, figfn = EXPERIMENTS[name]
        t0 = time.time()
        LOGGER.info("──────── 실험 %s 시작 ────────", name)
        if name in GLOBAL_EXPERIMENTS:            # 내부 T 스윕 — 1회
            CUR_TAG = ""
            out = fn(args.seeds, Ts, args.jobs, args.backend)
            try:
                figfn(out)
            except Exception as e:
                LOGGER.exception("그림 %s 실패: %s", name, e)
            summary["results"][name] = _jsonable(
                {k: v for k, v in out.items() if k != "traces"})
        else:                                     # 지평별 반복
            outs = {}
            entry = {}
            for T in Ts:
                CUR_TAG = f"T{T}"
                LOGGER.info("──── %s @ T=%d ────", name, T)
                out = fn(args.seeds, T, args.jobs, args.backend)
                try:
                    figfn(out, tag=f"_T{T}")
                except Exception as e:
                    LOGGER.exception("그림 %s(T=%d) 실패: %s", name, T, e)
                outs[T] = out
                entry[f"T{T}"] = _jsonable(
                    {k: v for k, v in out.items() if k != "traces"})
            CUR_TAG = ""
            if len(Ts) > 1:
                entry["horizon_comparison"] = _jsonable(
                    interpret_horizons(name, outs))
            summary["results"][name] = entry
        LOGGER.info("──────── 실험 %s 완료 (%.1fs) ────────", name,
                    time.time() - t0)

    apply_corrections()
    summary["confirmatory"] = PRIMARY
    summary["exploratory"] = EXPLORATORY
    try:
        fig_horizon_overview(summary, Ts)
    except Exception as e:
        LOGGER.exception("지평 개관 그림 실패: %s", e)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("═════════ 확증(1차) 지표 — Holm 보정 ═════════")
    for x in PRIMARY:
        verdict = "지지" if x.get("direction_met") and \
            x.get("p_holm", 1) < PREREG.get("alpha_family", 0.05) else "미지지/주의"
        LOGGER.info("[확증][%s] %s | %s | p_holm=%.4g → %s",
                    x["hyp"], x["label"], x["es"], x.get("p_holm", np.nan), verdict)
    LOGGER.info("═════════ 탐색 지표 — BH-FDR (q=%s) ═════════",
                PREREG.get("fdr_q"))
    n_sig = sum(1 for x in EXPLORATORY if x.get("q_fdr", 1) < PREREG.get("fdr_q", 0.05))
    LOGGER.info("탐색 지표 %d개 중 FDR 통과 %d개", len(EXPLORATORY), n_sig)
    LOGGER.info("결과 저장: %s", RESULTS / "summary.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
