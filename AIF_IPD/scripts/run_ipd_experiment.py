#!/usr/bin/env python
"""
run_ipd_experiment.py
=====================

HalloReg 메인 엔트리포인트: IPD 시뮬레이션 → 가설검증(H1–H10 + H7H/H8E/GS)
→ 가설별 시각화.

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

from AIF_IPD.core.constants import CC, CD, DC, DD, COOP, DEFECT
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
    LOGGER.info("[H6] 잡음 수준에 따른 고정전략 성능")
    from AIF_IPD.ipd.sim import run_dyad
    from AIF_IPD.ipd.env import make_opponent
    strategies = ["tit_for_tat", "generous_tft", "wsls", "allc", "alld"]
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
    w_t = paired_stats(per_seed[(0.15, "wsls")],
                       per_seed[(0.15, "tit_for_tat")], "greater", seed=62)
    p_pri = min(g_t["p"], w_t["p"]) * 2   # OR 결합(Bonferroni ×2, 보수적)
    ok = (g_t["mean_a"] > g_t["mean_b"]) or (w_t["mean_a"] > w_t["mean_b"])
    register_primary("H6", "잡음 하 GTFT∨WSLS>TFT", min(p_pri, 1.0), ok,
                     f"GTFT {fmt_es(g_t['es'], 'dz')} | WSLS {fmt_es(w_t['es'], 'dz')}")
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
        cases = ["p10_recip_expl_recon", "p30_flip", "p10_adaptive_expl"]
        noise_levels = (0.0, 0.10)
    else:
        cases = list(CAPRICIOUS_CASES)
        noise_levels = (0.0, 0.10, 0.20)
    primary_noise = 0.10
    sweep_focals = ("adaptive", "generous_tft")   # 비-primary 잡음의 스윕 대상
    focals = {
        "adaptive": agent_spec("adaptive", 0, kappa=0.9, sophisticated=True,
                               use_pymdp=backend == "pymdp"),
        "generous_tft": strat_spec("generous_tft", 0),
        "wsls": strat_spec("wsls", 0),
        "tit_for_tat": strat_spec("tit_for_tat", 0),
        "qlearner": dict(type="qlearner", seed=0),
        "bayes_br": dict(type="bayes_br", seed=0),
    }
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
    register_primary("H7", "전 case 합산 보수 adaptive>GTFT", pri["p"],
                     pri["mean_a"] > pri["mean_b"], fmt_es(pri["es"], "dz"))
    for base in ("wsls", "tit_for_tat", "qlearner", "bayes_br"):
        t = paired_stats(agg_seed["adaptive"], agg_seed[base], "two-sided",
                         seed=74)
        register_exploratory("H7", f"adaptive vs {base}", t["p"],
                             fmt_es(t["es"], "dz"))

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
    for name in ("adaptive", "generous_tft", "wsls", "tit_for_tat"):
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

    # ---- 4×4 상호대전: 환경 계층 잡음으로 전원 대칭 (§2) ----
    tour_names = ["tit_for_tat", "generous_tft", "wsls", "adaptive"]

    def tour_cfg(name, seed):
        if name == "adaptive":
            return agent_spec("adaptive", seed, kappa=0.9, sophisticated=True,
                              use_pymdp=backend == "pymdp")
        return strat_spec(name, seed)          # 내부 error 없음 — 환경이 부과

    t_specs, t_reg = [], {}
    for i, rn in enumerate(tour_names):
        for j, cn in enumerate(tour_names):
            for sd in range(seeds):
                t_reg[(i, j, sd)] = len(t_specs)
                t_specs.append({"agent": tour_cfg(rn, sd),
                                "opponent": tour_cfg(cn, 5000 + sd),
                                "env_err_agent": primary_noise,
                                "env_err_opponent": primary_noise,
                                "noise_seed": 8000 + sd * 41 + i * 5 + j})
    t_res = run_many(t_specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
    matrix = np.zeros((4, 4)); matrix_ci = np.zeros((4, 4, 2))
    for i in range(4):
        for j in range(4):
            v = [float(np.mean(t_res[t_reg[(i, j, sd)]]["hist"]["my_payoff"]))
                 for sd in range(seeds)]
            matrix[i, j] = np.mean(v)
            matrix_ci[i, j] = boot_mean_ci(v, seed=78)["ci"]
    supported = {"overall_payoff_advantage": bool(pri["mean_a"] > pri["mean_b"]),
                 "exploit_phase_defense": bool(px["mean_a"] > px["mean_b"]),
                 "lambda_responsive": lam_responsive,
                 "by_period": signs,
                 "period_dependent": bool(period_dependent)}
    LOGGER.info("[H7] → %s (총보수의 지평 의존성은 H7H·지평 비교에서 [확증] 검증)",
                supported)
    return {"cases": cases, "noise_levels": [str(n) for n in noise_levels],
            "primary_noise": primary_noise, "primary": pri,
            "sweep_focals": list(sweep_focals),
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
                         "slope": sl_a,
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
LARGE_MIX = {"tit_for_tat": 0.18, "generous_tft": 0.12, "wsls": 0.12,
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
    fillers = ["generous_tft", "allc"] if quick else \
        ["generous_tft", "tit_for_tat", "allc", "wsls"]
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
    "tit_for_tat": "#1f77b4", "generous_tft": "#2ca02c", "wsls": "#17becf",
    "allc": "#bcbd22", "alld": "#d62728", "random": "#7f7f7f",
    "capricious": "#9467bd", "adaptive": "#ff7f0e", "adaptive_imm": "#8c564b",
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
                ("tit_for_tat", "generous_tft", "wsls", "allc",
                 "adaptive", "adaptive_imm")]
    x0s = {}
    x_mix = resident_w * 0.9; x_mix[i_ad] = 0.1
    x0s["mix+10%ad"] = x_mix / x_mix.sum()
    x0s["uniform"] = np.ones(k) / k
    x_alld = np.ones(k) * 0.05
    x_alld[names.index("alld")] = 1.0 - 0.05 * (k - 1)
    x0s["alld_heavy"] = x_alld
    trajs, attractors, basins = {}, {}, {}
    n_basin = 40 if quick else 400
    for T in Ts:
        Pi = est[(primary_err, T)]["Pi"]
        trajs[T] = {lbl: evo.replicator_trajectory(Pi, x0, steps=500)
                    for lbl, x0 in x0s.items()}
        attractors[T] = evo.attractor_analysis(
            Pi, names, n_samples=n_basin, steps=800,
            seed=stable_seed("attr", T))
        top = attractors[T]["attractors"][:3]
        register_exploratory(
            "H8E", f"끌개 구조 (T={T})", 1.0,
            "; ".join(f"유역 {a['basin_frac']:.2f} → " + ", ".join(
                f"{names[i]}={c:.2f}" for i, c in enumerate(a["composition"])
                if c > 0.05) for a in top))
    for (err, T), e in est.items():
        Pi = e["Pi"]
        bw = evo.basin_analysis(Pi, names, n_samples=n_basin,
                                seed=802, coop_types=coop_idx)
        keep = [i for i in range(k) if i not in (i_ad, i_imm)]
        names_wo = [names[i] for i in keep]
        coop_wo = [names_wo.index(t) for t in
                   ("tit_for_tat", "generous_tft", "wsls", "allc")]
        bo = evo.basin_analysis(Pi[np.ix_(keep, keep)], names_wo,
                                n_samples=n_basin, seed=802,
                                coop_types=coop_wo)
        basins[(err, T)] = {"with": bw["coop_basin_frac"],
                            "without": bo["coop_basin_frac"],
                            "widening": float(bw["coop_basin_frac"]
                                              - bo["coop_basin_frac"])}
    for T in Ts:
        b = basins[(primary_err, T)]
        register_exploratory("H8E", f"협력 유역 확장 (T={T}, adaptive 유−무)",
                             1.0, f"Δbasin={b['widening']:+.3f} "
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
        "basin_widening_positive": {
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
        # H7 미니: adaptive vs GTFT, 3 case
        specs = []
        for nm, cfg in (("adaptive", ad), ("gtft", strat_spec("generous_tft", 0))):
            for case in cases:
                for sd in range(n_seed):
                    a = dict(cfg); a["seed"] = sd
                    specs.append({"agent": a,
                                  "opponent": capricious_case_spec(
                                      case, seed=700 + sd, n_rounds=rounds,
                                      error=0.10),
                                  "noise_seed": 7000 + sd, "game_ci": ci})
        res = run_many(specs, n_rounds=rounds, n_jobs=jobs, verbose=False)
        half = len(res) // 2
        pay_ad = np.mean([np.mean(r["hist"]["my_payoff"]) for r in res[:half]])
        pay_gt = np.mean([np.mean(r["hist"]["my_payoff"]) for r in res[half:]])
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
                        "h8_cc_frac0": float(cc0), "h8_cc_frac05": float(cc6),
                        "h8_direction_positive": bool(cc6 > cc0)}
        register_exploratory("GS", f"CI={ci} 강건성", 1.0,
                             f"착취Δ={np.mean(d_expl):+.3f}, "
                             f"H7 {pay_ad:.2f}vs{pay_gt:.2f}, "
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
          "H6: 잡음 하에서 GTFT/WSLS 가 TFT 를 앞선다 (라운드로빈 평균±부트 CI).",
          f"seeds={d['n_seed']}")


def fig_H7(d, tag=""):
    _kfont()
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    t = d["traces"]; rel = t["rel"]
    # (a) 전환 정렬 보수 — 시드 CI 밴드 + case 얇은선 (§5)
    for name, c in (("adaptive", "C0"), ("generous_tft", "C1")):
        mat = t["aligned_seed"][name]
        m = mat.mean(axis=0); lo, hi = np.percentile(mat, [2.5, 97.5], axis=0)
        ax[0, 0].plot(rel, m, color=c, lw=2, label=name)
        ax[0, 0].fill_between(rel, lo, hi, alpha=0.18, color=c)
    for case, arr in t["aligned_case"]["adaptive"].items():
        ax[0, 0].plot(rel, arr, color="C0", lw=0.5, alpha=0.25)
    ax[0, 0].axvline(0, color="0.5", ls="--"); ax[0, 0].legend(fontsize=8)
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
    order = ["adaptive", "generous_tft", "wsls", "tit_for_tat",
             "qlearner", "bayes_br"]
    order = [o for o in order if o in means]
    ags = t["agg_seed"]
    bar_ci(ax[0, 2], range(len(order)), [means[o] for o in order],
           [boot_mean_ci(ags[o])["ci"] for o in order], order,
           colors=["C0"] + ["0.6"] * (len(order) - 1))
    ax[0, 2].tick_params(axis="x", labelrotation=30)
    ax[0, 2].set_title(f"전 case 총보수 [확증] adaptive vs GTFT "
                       f"({fmt_es(d['primary']['es'], 'dz')})")
    ax[0, 2].set_ylabel("라운드당 평균 보수")
    # (d, 신규) 주기 의존성: Δ(P) — 지지 부호가 주기에 따라 다르면 색으로 구분
    ps = d["period_stats"]
    Ps = sorted(ps, key=lambda x: int(x))
    dm = [ps[P]["delta_mean"] for P in Ps]
    err = np.array([[ps[P]["delta_mean"] - ps[P]["ci"][0],
                     ps[P]["ci"][1] - ps[P]["delta_mean"]] for P in Ps]).T
    cols = ["C0" if ps[P]["supported"] else "C3" for P in Ps]
    ax[1, 0].bar(range(len(Ps)), dm, yerr=err, capsize=4, color=cols, alpha=0.85)
    ax[1, 0].axhline(0, color="0.4", lw=1)
    ax[1, 0].set_xticks(range(len(Ps)))
    ax[1, 0].set_xticklabels(
        [f"P={P}\n({ps[P]['switches_in_horizon']}회 전환"
         + ("; 무전환" if ps[P]['switches_in_horizon'] == 0 else "") + ")"
         for P in Ps], fontsize=8)
    dep = d["supported"].get("period_dependent", False)
    ax[1, 0].set_title("전환 주기별 Δ = adaptive − GTFT [탐색]\n"
                       f"(지지의 주기 의존성: {'있음 ⚠' if dep else '없음'}; "
                       "파랑=Δ>0, 빨강=Δ<0)")
    ax[1, 0].set_ylabel("Δ (라운드당)")
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
    # (f) 4×4 대전 행렬 (대칭 환경 잡음)
    mat = np.array(d["tournament"]["matrix"]); names = d["tournament"]["names"]
    im = ax[1, 2].imshow(mat, cmap="viridis")
    ax[1, 2].set_xticks(range(4)); ax[1, 2].set_xticklabels(names, rotation=30, fontsize=8)
    ax[1, 2].set_yticks(range(4)); ax[1, 2].set_yticklabels(names, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax[1, 2].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                          color="w", fontsize=8)
    ax[1, 2].set_title(f"4×4 대전 (대칭 잡음 {d['tournament']['symmetric_env_noise']})")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
    _save(fig, "h7_capricious_partners" + tag,
          "H7(v0.3): 주기 {10,30,60,120} × 순환족(AIF 국면 포함) case 격자. "
          "λ 는 전환에 반응(무작위 정렬 순열 대비). Δ(P) 패널은 지지 여부의 "
          "전환 주기 의존성을 명시(무전환 대조 병기). 대전은 환경 계층 대칭 잡음.",
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
    ax[0, 0].errorbar(Ts, m, yerr=err, fmt="o-", capsize=4, color="C0")
    ax[0, 0].axhline(0, color="r", ls="--", lw=1)
    ts = d["family_a"]["t_star"]
    if np.isfinite(ts["est"]):
        ax[0, 0].axvline(ts["est"], color="C2", ls=":",
                         label=f"T*={ts['est']:.0f} [{ts['ci'][0]:.0f},{ts['ci'][1]:.0f}]")
        ax[0, 0].legend(fontsize=8)
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
                f"협력 유역 지도 (adaptive–alld–TFT, T={Tmain})", mode="basin")
    _tern_panel(ax[1, 0], t["ternary"][(key_b, Tmain)],
                f"협력 유역 지도 (adaptive–alld–ALLC, T={Tmain})", mode="basin")
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
    ax[1, 1].set_title("협력 유역 비율 (9유형 전체, adaptive 포함)")
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
    ax[1, 2].set_title("협력 유역 확장 Δ (adaptive 유 − 무)")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
    _save(fig, "h8e_state_space_basins" + tag,
          "H8E-3: Replicator 상태 공간(3-유형 위상 초상 + 끌개), cooperation "
          "basin 지도, 협력 유역 비율·확장의 (err×T) 격자.",
          f"basin n={len(t['ternary'][(key_a, Tmain)]['map']['grid'])} 격자점")


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
    w = 0.35
    ax[1].bar(np.arange(len(cis)) - w / 2, [R[c]["h7_pay_adaptive"] for c in cis],
              w, label="adaptive", color="C0")
    ax[1].bar(np.arange(len(cis)) + w / 2, [R[c]["h7_pay_gtft"] for c in cis],
              w, label="GTFT", color="C1")
    ax[1].set_xticks(range(len(cis))); ax[1].set_xticklabels([f"CI={c}" for c in cis])
    ax[1].set_title("H7 총보수 (3 case)"); ax[1].legend(fontsize=8)
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
    "GS": (exp_GS, fig_GS),
}
GLOBAL_EXPERIMENTS = {"H7H", "H8E"}   # 내부 T 스윕 — rounds 목록을 통째로 전달


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
            d = {"전 case Δ(adaptive−GTFT)": float(m["adaptive"]
                                                   - m["generous_tft"])}
            for P, v in out["period_stats"].items():
                d[f"Δ(P={P})"] = float(v["delta_mean"])
            return d
        if name == "H8":
            return {"λ=0.4 조건부 frac 기울기": float(out["primary"]["est"])}
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
