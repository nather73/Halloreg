"""
ipd.sim
=======

시뮬레이션 실행기: 다이애드 루프 + 다중 스펙 병렬 실행.

[병렬화]
각 다이애드는 완전히 독립(embarrassingly parallel)이므로 multiprocessing 으로
분배한다. 'spawn' 컨텍스트를 강제하는 이유:
  · fork 는 numpy/BLAS 의 내부 스레드 상태를 자식에게 물려주어 교착이 날 수 있다.
  · Windows 기본이 spawn 이므로, 강제해 두면 OS 간 동작이 일관된다.
각 워커의 BLAS 스레드는 1개로 제한(모듈 상단, numpy import 이전)하여 과다구독
(oversubscription)을 막는다.

[가변 보수 환경]
`ci_schedule` 을 주면 매 라운드 `set_payoffs(ci)` 로 전역 보수를 갱신한다.
전역 배열이 in-place 로 바뀌므로 에이전트의 EFE·CoreAffect 가 **현재 맥락의
효용**을 자동으로 반영한다. 고정전략은 보수를 읽지 않으므로 무감하다 —
이 비대칭이 H4 의 이론적 근거다.
"""

from __future__ import annotations

# --- 워커의 스레드 과다생성 방지 (numpy/BLAS import 이전에 설정) ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from AIF_IPD.core.constants import (
    CC, COOP, PAYOFF_SELF, PAYOFF_OTHER, joint_index, mirror_state,
    reset_payoffs, set_payoffs,
)
from AIF_IPD.core.logging_utils import get_logger
from .agent import EmpathicAgent, HalloRegAgent
from .env import StrategyAgent, make_opponent

LOGGER = get_logger("HalloReg.sim")


def _is_aif(agent) -> bool:
    """능동추론 에이전트인지(step 인터페이스) 판별."""
    return isinstance(agent, EmpathicAgent)


# ==================================================================== 다이애드
def run_dyad(agent, opponent, n_rounds: int = 120,
             ci_schedule: Optional[Callable[[int, int], float]] = None,
             env_err_a: float = 0.0, env_err_b: float = 0.0,
             noise_seed: Optional[int] = None,
             partner_id_a: int = 1, partner_id_b: int = 2
             ) -> Dict[str, np.ndarray]:
    """
    focal `agent` vs `opponent` 동시행동 IPD 를 n_rounds 만큼 실행한다.

    Parameters
    ----------
    ci_schedule : (t, T) -> CI  |  None
        None 이면 고정 보수(정상 환경). 함수면 매 라운드 보수를 재설정한다.
    env_err_a, env_err_b : float
        **환경 계층** 실행오류율. 에이전트 종류와 무관하게 방출 행동을 뒤집으므로
        AIF/고정전략에 대칭 부과가 가능하다(StrategyAgent 내부 error 와는 별개).
    noise_seed : int | None
        뒤집힘 수열을 사전 생성하는 전용 시드. 조건 간 공유하면 잡음 실현이
        동일해져(공통난수, CRN) 짝지은 비교의 분산이 줄어든다.

    Returns
    -------
    dict of ndarray : my_act, opp_act, state, my_payoff, opp_payoff, ci
    """
    hist = {"my_act": [], "opp_act": [], "state": [],
            "my_payoff": [], "opp_payoff": [], "ci": []}

    # 상대 identity 통지 — HalloRegAgent 의 기억/사전 주입 훅.
    if hasattr(agent, "begin_partner"):
        agent.begin_partner(partner_id_a)
    if hasattr(opponent, "begin_partner"):
        opponent.begin_partner(partner_id_b)

    # 환경 잡음 수열 사전 생성 (경로와 독립 → CRN 성립)
    if env_err_a > 0 or env_err_b > 0:
        nrng = np.random.default_rng(0 if noise_seed is None else int(noise_seed))
        flips_a = nrng.random(n_rounds) < env_err_a
        flips_b = nrng.random(n_rounds) < env_err_b
    else:
        flips_a = flips_b = None

    prev_state = None            # focal 관점의 직전 joint outcome
    prev_state_mirror = None     # 상대 관점의 직전 joint outcome

    for t in range(n_rounds):
        # ---- 가변 보수: 이번 라운드의 게임 구조 확정 ----
        if ci_schedule is not None:
            ci_t = float(ci_schedule(t, n_rounds))
            set_payoffs(ci_t)
        else:
            ci_t = float("nan")

        # ---- 행동 방출 ----
        my_a = agent.step(prev_state) if _is_aif(agent) else agent.act()
        opp_a = (opponent.step(prev_state_mirror) if _is_aif(opponent)
                 else opponent.act())

        # ---- 환경 계층 실행오류 ----
        # 뒤집힘이 발생하면 AIF 에이전트에게 **실제 방출된 행동**을 통지한다.
        # 상대는 방출 행동에 반응하므로, 에이전트 내부 my_last 가 의도 행동에
        # 머물러 있으면 다음 라운드 우도의 호혜신호 f 가 상대가 본 것과 달라진다.
        if flips_a is not None and flips_a[t]:
            my_a = 1 - my_a
            if _is_aif(agent):
                agent.note_emitted(my_a)
        if flips_b is not None and flips_b[t]:
            opp_a = 1 - opp_a
            if _is_aif(opponent):
                opponent.note_emitted(opp_a)

        # ---- 상호 관측 통지 (고정전략 계열은 실제 방출 행동을 본다) ----
        if not _is_aif(agent):
            agent.observe(opp_a)
        if not _is_aif(opponent):
            opponent.observe(my_a)

        # ---- 결과 기록 ----
        state = joint_index(my_a, opp_a)
        hist["my_act"].append(my_a)
        hist["opp_act"].append(opp_a)
        hist["state"].append(state)
        hist["my_payoff"].append(float(PAYOFF_SELF[state]))
        hist["opp_payoff"].append(float(PAYOFF_OTHER[state]))
        hist["ci"].append(ci_t)

        prev_state = state
        prev_state_mirror = mirror_state(state)

    if ci_schedule is not None:
        reset_payoffs()          # 전역 상태 오염 방지

    return {k: np.asarray(v) for k, v in hist.items()}


# ==================================================================== 스펙 빌더
def build_agent(cfg: dict):
    """
    에이전트 스펙 dict → 인스턴스.

    cfg = {"type": "halloreg" | "empathic" | "strategy" | "likelihood", ...}
    """
    cfg = dict(cfg)
    typ = cfg.pop("type")
    if typ == "halloreg":
        return HalloRegAgent(**cfg)
    if typ == "empathic":
        return EmpathicAgent(**cfg)
    if typ == "strategy":
        kind = cfg.pop("kind")
        return make_opponent(kind, **cfg)
    if typ == "likelihood":
        # 우도 기저에서 직접 생성하는 상대 — 참 형질을 알므로 복원 검증에 쓴다.
        from .env import LikelihoodAgent
        return LikelihoodAgent(**cfg)
    raise ValueError(f"알 수 없는 에이전트 type: {typ}")


def build_from_spec(spec: dict):
    """다이애드 스펙 → (agent, opponent)."""
    return build_agent(spec["agent"]), build_agent(spec["opponent"])


# ==================================================================== 병렬 실행
def _worker(task):
    """
    top-level 워커 (spawn 에서 pickle 가능해야 하므로 모듈 최상위에 둔다).
    task = (idx, spec, n_rounds)
    """
    idx, spec, n_rounds = task
    from AIF_IPD.ipd.payoff_schedule import get_regime

    # 워커는 새 프로세스이므로 전역 보수를 기본 PD 로 초기화한 뒤 시작한다(멱등).
    reset_payoffs()
    ci_fn = get_regime(spec.get("regime")) if spec.get("regime") else None

    agent, opponent = build_from_spec(spec)
    hist = run_dyad(agent, opponent, n_rounds,
                    ci_schedule=ci_fn,
                    env_err_a=float(spec.get("env_err_agent", 0.0)),
                    env_err_b=float(spec.get("env_err_opponent", 0.0)),
                    noise_seed=spec.get("noise_seed"))
    out = {"hist": hist}
    # AIF 에이전트의 내부 로그(λ, 정서, θ̂ 궤적)를 수집한다.
    if _is_aif(agent):
        out["agent_log"] = {k: np.asarray(v) for k, v in agent.log.items()}
    if _is_aif(opponent):
        out["opponent_log"] = {k: np.asarray(v) for k, v in opponent.log.items()}
    return idx, out


def resolve_jobs(n_jobs: Optional[int]) -> int:
    """--jobs 인자 해석. -1 → (코어 수 − 1), None/0 → 1."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs == -1:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(n_jobs))


def run_many(specs: List[dict], n_rounds: int = 120,
             n_jobs: Optional[int] = None, verbose: bool = True,
             desc: str = "다이애드") -> List[dict]:
    """
    다수 다이애드 스펙을 순차/병렬 실행한다. 반환 순서는 specs 와 동일.
    """
    tasks = [(i, spec, n_rounds) for i, spec in enumerate(specs)]
    results: List[Optional[dict]] = [None] * len(specs)
    n_jobs = resolve_jobs(n_jobs)

    if n_jobs <= 1 or len(tasks) == 1:
        for i, spec, nr in tasks:
            _, res = _worker((i, spec, nr))
            results[i] = res
            if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                LOGGER.info("  진행 %d/%d %s", i + 1, len(tasks), desc)
    else:
        n_jobs = min(n_jobs, len(tasks))
        if verbose:
            LOGGER.info("  병렬 실행: %d %s / %d 워커", len(tasks), desc, n_jobs)
        ctx = mp.get_context("spawn")
        done = 0
        with ctx.Pool(processes=n_jobs) as pool:
            for idx, res in pool.imap_unordered(_worker, tasks, chunksize=1):
                results[idx] = res
                done += 1
                if verbose and done % max(1, len(tasks) // 10) == 0:
                    LOGGER.info("  진행 %d/%d %s", done, len(tasks), desc)
    return results


# ==================================================================== 요약지표
def coop_rate(acts: np.ndarray) -> float:
    """행동열의 협력률."""
    return float(np.mean(np.asarray(acts) == COOP))


def cc_rate(hist: dict) -> float:
    """상호협력(CC) 발생률."""
    return float(np.mean(hist["state"] == CC))


def mean_payoff(hist: dict, side: str = "my") -> float:
    """라운드당 평균 보수."""
    return float(np.mean(hist[f"{side}_payoff"]))
