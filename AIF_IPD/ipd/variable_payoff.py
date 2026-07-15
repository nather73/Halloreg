"""
ipd.variable_payoff
===================

**가변 페이오프 환경 (연속 협력–경쟁 트레이드오프).**

[동기 — 기존 한계]
H7(변덕 상대에 대한 단기 열세)·H8E(음의 침입 성장률)·H12(협력 유역 확장의 비유의)는
모두 **IPD 의 보수가 고정**이고 협력/경쟁이 **이분법적**이라는 데서 비롯할 수 있다.
고정전략(TFT/GTFT/WSLS/ALLC/ALLD)은 보수 구조를 읽지 않고 **행동 이력에만** 반응하는
반면, `AdaptiveAgent` 는 EFE 의 선호 C 를 **현재 보수**로 계산하므로, 보수 구조가 계속
변하는 환경에서는 맥락-의존적으로 자·타 효용을 재계산하는 이점을 가질 수 있다.

[이론적 근거 — Pisauro et al. (2022), Nat. Commun.]
Space Dilemma 는 PD 의 이분법을 **연속체**로 일반화하고, 재분배 파라미터로 사회적
맥락(협력/중간/경쟁)을 조작한다. 승리 모형 B6 은 (i) 상대의 **기대 위치**를 tit-for-tat
으로 되갚되, (ii) 그 되갚음 계수를 **맥락(사회적 위험)에 반비례**로 정규화하고, (iii)
사회적 편향과 (iv) 정밀도를 갖는다:

    Pos_i ~ N( TitxTatFactor(context)·⟨Pos_j⟩ + SocialBias , Precision )

핵심은 **맥락(보수 구조)에 따라 협력 정도를 연속적으로 조절**한다는 것이며, 상대 위치의
예측오차는 TPJ 와 연관된다.

[본 모듈의 조작화]
이산 IPD 를 유지하되 **협력지수 CI=(R−P)/(T−S)** 를 라운드마다 변동시킨다(가변 페이오프).
CI 는 극단 영역까지 포괄한다:
    CI<0   교착(deadlock; 상호협력<상호배신) — 협력이 손해
    0<CI<1 죄수의 딜레마(dilemma)
    CI≥1   조화/닭게임 영역(harmony; 협력이 우월)

라운드마다 `set_payoffs(ci_t)` 로 보수를 in-place 갱신하면, `AdaptiveAgent` 의 EFE 선호
C 가 **현재 맥락의 효용**을 반영한다(맥락-의존 효용). 고정전략은 CI 에 무감하다. 또한
CI 자체를 관측 보수로부터 **베이지안 추정**하는 맥락 추론기(`ContextEstimator`)를 두어
Pisauro B6 의 '맥락-의존 되갚음'을 재현한다(협력 편향을 추정 CI 로 변조).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from AIF_IPD.core import constants as C
from AIF_IPD.core.constants import (
    CC, CD, DC, DD, COOP, DEFECT, joint_index, mirror_state,
)
from AIF_IPD.ipd.agent import ToMEmpathicAgent


# ------------------------------------------------------------------ CI 스케줄
def ci_constant(ci: float) -> Callable[[int, int], float]:
    """모든 라운드에서 상수 CI (극단 포함)."""
    return lambda t, T: float(ci)


def ci_blocks(values: Sequence[float]) -> Callable[[int, int], float]:
    """지평을 등분해 블록마다 다른 CI (Pisauro 의 사회적 맥락 블록 대응)."""
    vals = list(values)

    def f(t: int, T: int) -> float:
        k = len(vals)
        b = min(k - 1, int(t * k / max(T, 1)))
        return float(vals[b])
    return f


def ci_oscillate(lo: float, hi: float, period: int) -> Callable[[int, int], float]:
    """lo↔hi 사이를 period 로 진동하는 CI (연속적 맥락 변동)."""
    def f(t: int, T: int) -> float:
        phase = 0.5 * (1 + np.sin(2 * np.pi * t / max(period, 1)))
        return float(lo + (hi - lo) * phase)
    return f


def ci_aba(a: float, b: float, T: int) -> Callable[[int, int], float]:
    """
    A→B→A 세 구간의 CI 스케줄(맥락 전환·복귀 검증용). 예: 협력적 맥락(A) →
    경쟁/교착 맥락(B) → 협력적 맥락(A). 세 등분.
    """
    def f(t: int, TT: int) -> float:
        third = max(TT // 3, 1)
        return float(a if (t < third or t >= 2 * third) else b)
    return f


# 사전 정의된 CI 레짐 카탈로그 (실험 VP 용). 극단 음수·1 이상을 모두 포함.
CI_REGIMES = {
    "coop_harmony": ci_constant(1.2),      # 조화(협력 우월)
    "mild_pd":      ci_constant(0.4),       # 현행 PD
    "harsh_pd":     ci_constant(0.25),      # 2R<T+S 근방(가혹한 PD)
    "deadlock":     ci_constant(-0.3),      # 교착(협력이 손해)
    "blocks_c_i_c": ci_blocks([1.2, 0.4, -0.3, 0.4, 1.2]),  # 조화→PD→교착→PD→조화
    "oscillate":    ci_oscillate(-0.3, 1.2, period=30),      # 연속 진동
    "aba_coop_deadlock": ci_aba(1.0, -0.3, 60),  # A(협력)→B(교착)→A(협력)
}


# ------------------------------------------------------- 맥락(CI) 베이지안 추정
@dataclass
class ContextEstimator:
    """
    관측된 라운드 보수로부터 현재 협력지수 CI 를 온라인 베이지안 추정 (Pisauro B6 의
    '맥락-의존 되갚음'의 근거 신호). 켤레 정규 갱신:

        prior  CI ~ N(m0, s0²)
        관측    ci_obs_t = (R̂ − P) / (T − S)  (관측 보수에서 역산; T,S,P 는 관측가능)
        posterior 평균을 정밀도 가중 이동평균으로 추적.

    반환 estimate() 는 현재 CI 사후평균(협력의 상대적 유리함)으로, 에이전트의 협력
    편향을 맥락에 맞춰 연속 조절하는 데 쓸 수 있다.
    """
    m0: float = 0.4
    s0: float = 0.6
    obs_noise: float = 0.3
    m: float = field(init=False)
    prec: float = field(init=False)

    def __post_init__(self):
        self.m = float(self.m0)
        self.prec = 1.0 / max(self.s0 ** 2, 1e-6)

    def observe_ci(self, ci_obs: float):
        obs_prec = 1.0 / max(self.obs_noise ** 2, 1e-6)
        new_prec = self.prec + obs_prec
        self.m = (self.prec * self.m + obs_prec * float(ci_obs)) / new_prec
        # 비정상 환경 추적을 위해 정밀도 상한(망각)
        self.prec = min(new_prec, 12.0)

    def estimate(self) -> float:
        return float(self.m)


# ---------------------------------------------------------------- 가변 다이애드
def _is_aif(a) -> bool:
    return isinstance(a, ToMEmpathicAgent)


def run_variable_dyad(agent, opponent, ci_fn: Callable[[int, int], float],
                      n_rounds: int = 60,
                      env_err_a: float = 0.0, env_err_b: float = 0.0,
                      noise_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    가변 CI 다이애드. 라운드마다 `set_payoffs(ci_fn(t,T))` 로 보수를 갱신한 뒤 두
    에이전트를 진행하고, **그 라운드의 실제 보수**(현재 CI 기준)를 기록한다.

    `AdaptiveAgent` 는 매 라운드 현재 보수로 EFE 를 계산하므로 맥락-의존 효용을
    자동 반영한다. 고정전략은 CI 에 무감(행동 이력만 사용).

    반환 dict 에 `ci`(라운드별 CI)와 `my_payoff/opp_payoff`(현재 CI 기준) 포함.
    """
    hist = {"my_act": [], "opp_act": [], "state": [],
            "my_payoff": [], "opp_payoff": [], "ci": []}
    prev_state = None
    prev_state_mirror = None
    if env_err_a > 0 or env_err_b > 0:
        nrng = np.random.default_rng(0 if noise_seed is None else int(noise_seed))
        flips_a = nrng.random(n_rounds) < env_err_a
        flips_b = nrng.random(n_rounds) < env_err_b
    else:
        flips_a = flips_b = None

    saved = (C.R, C.P)                       # 복원용
    try:
        for t in range(n_rounds):
            ci_t = float(ci_fn(t, n_rounds))
            C.set_payoffs(ci_t)              # 현재 맥락 보수 (극단 허용)

            my_a = agent.step(prev_state) if _is_aif(agent) else agent.act()
            opp_a = opponent.step(prev_state_mirror) if _is_aif(opponent) \
                else opponent.act()

            if flips_a is not None and flips_a[t]:
                my_a = 1 - my_a
            if flips_b is not None and flips_b[t]:
                opp_a = 1 - opp_a

            if not _is_aif(agent):
                agent.observe(opp_a)
            if not _is_aif(opponent):
                opponent.observe(my_a)

            state = joint_index(my_a, opp_a)
            hist["my_act"].append(my_a)
            hist["opp_act"].append(opp_a)
            hist["state"].append(state)
            hist["my_payoff"].append(float(C.PAYOFF_SELF[state]))
            hist["opp_payoff"].append(float(C.PAYOFF_OTHER[state]))
            hist["ci"].append(ci_t)
            prev_state = state
            prev_state_mirror = mirror_state(state)
    finally:
        C.R, C.P = saved
        C.set_payoffs((C.R - 1.0) / (C.T - C.S))   # 보수 배열 원복

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist


# --------------------------------------------------- 가변 CI 하 보수행렬 추정
def _var_worker(task):
    """
    spawn pickle 가능 top-level 워커. task=(idx, spec, regime_name, n_rounds).
    각 워커 프로세스는 **독립 메모리**이므로 라운드별 in-place 보수 갱신이
    다른 워커를 오염시키지 않는다(병렬 안전).
    """
    idx, spec, regime_name, n_rounds = task
    from AIF_IPD.ipd.variable_payoff import CI_REGIMES, run_variable_dyad as _rvd
    from AIF_IPD.ipd.sim import build_from_spec, _is_aif as _isa
    ci_fn = CI_REGIMES[regime_name]
    agent, opp = build_from_spec(spec)
    h = _rvd(agent, opp, ci_fn, n_rounds=n_rounds,
             env_err_a=float(spec.get("env_err_agent", 0.0)),
             env_err_b=float(spec.get("env_err_opponent", 0.0)),
             noise_seed=spec.get("noise_seed"))
    res = {"hist": h}
    if _isa(agent):
        res["agent_log"] = {k: np.asarray(v) for k, v in agent.log.items()}
    return idx, res


def run_variable_many(specs: List[dict], regime_name: str, n_rounds: int,
                      n_jobs: Optional[int] = None, verbose: bool = False) -> List[dict]:
    """가변 CI 다이애드 다수를 spawn 병렬 실행 (regime_name 은 CI_REGIMES 키)."""
    import multiprocessing as mp
    import os
    tasks = [(i, spec, regime_name, n_rounds) for i, spec in enumerate(specs)]
    results: List[Optional[dict]] = [None] * len(specs)
    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    if n_jobs is None or n_jobs <= 1:
        for t in tasks:
            idx, r = _var_worker(t)
            results[idx] = r
    else:
        n_jobs = min(n_jobs, len(tasks))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            for idx, r in pool.imap_unordered(_var_worker, tasks, chunksize=1):
                results[idx] = r
    return results


def estimate_variable_payoff_matrix(type_specs: Dict[str, dict],
                                    regime_name: str,
                                    n_rounds: int, seeds: int,
                                    env_error: float = 0.10, n_jobs: int = -1,
                                    seed_offset: int = 0) -> Dict:
    """
    가변 CI 환경(regime_name)에서 유형 쌍별 라운드당 평균 보수 행렬 Π 추정(병렬).
    각 다이애드가 동일 스케줄을 겪으므로 Π 는 '변동 맥락 전반의 평균 성능' 요약.
    RE/ORE 유역 분석의 입력.
    """
    names = list(type_specs)
    k = len(names)
    specs, registry = [], {}
    for i in range(k):
        for j in range(i, k):
            for sd in range(seeds):
                a = dict(type_specs[names[i]]); a["seed"] = 10000 + seed_offset + sd * 17 + i
                b = dict(type_specs[names[j]]); b["seed"] = 20000 + seed_offset + sd * 17 + j
                registry[(i, j, sd)] = len(specs)
                specs.append({"agent": a, "opponent": b,
                              "env_err_agent": env_error,
                              "env_err_opponent": env_error,
                              "noise_seed": 900000 + seed_offset * 7 + sd * 101 + i * 7 + j})
    res = run_variable_many(specs, regime_name, n_rounds, n_jobs=n_jobs)
    raw = np.zeros((k, k, seeds))
    ccm_raw = np.zeros((k, k, seeds))            # 행동적 CC율 행렬(대칭)
    for i in range(k):
        for j in range(i, k):
            for sd in range(seeds):
                h = res[registry[(i, j, sd)]]["hist"]
                my = float(np.mean(h["my_payoff"])); op = float(np.mean(h["opp_payoff"]))
                cc = float(np.mean(h["state"] == CC))
                if i == j:
                    raw[i, i, sd] = 0.5 * (my + op)
                else:
                    raw[i, j, sd] = my
                    raw[j, i, sd] = op
                ccm_raw[i, j, sd] = cc
                ccm_raw[j, i, sd] = cc
    Pi = raw.mean(axis=2)
    Pi_sd = raw.std(axis=2, ddof=1) if seeds > 1 else np.zeros((k, k))
    CCm = ccm_raw.mean(axis=2)
    return {"names": names, "Pi": Pi, "Pi_sd": Pi_sd, "raw": raw,
            "CC": CCm, "CC_raw": ccm_raw,
            "n_rounds": n_rounds, "seeds": seeds, "regime": regime_name}


# ---------------------------------------------------------- A-B-A 의도 전환 상대
def aba_schedule(intent_a: str, intent_b: str, n_rounds: int,
                 first_frac: float = 1 / 3, mid_frac: float = 1 / 3) -> list:
    """
    A→B→A 순서의 **의도(형질) 전환** 스케줄을 StrategyAgent schedule 로 반환.

    예: aba_schedule('tit_for_tat','alld',60) →
        [(0,'tit_for_tat'), (20,'alld'), (40,'tit_for_tat')]
    상대가 협력적 상호성(A) → 착취(B) → 다시 협력적 상호성(A) 로 돌아올 때,
    focal AdaptiveAgent 가 A 의도 추론을 **복구**하는지 검증하는 데 사용한다(§ABA).
    """
    c1 = int(round(first_frac * n_rounds))
    c2 = int(round((first_frac + mid_frac) * n_rounds))
    return [(0, intent_a), (c1, intent_b), (c2, intent_a)]
