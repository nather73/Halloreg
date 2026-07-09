"""
ipd.sim
=======

시뮬레이션 실행기: 다이애드 루프 + 다중 시드 병렬 실행 + 집단(population) 역학.

[병렬 실행]
각 다이애드는 서로 완전히 독립(embarrassingly parallel)이므로 multiprocessing 으로
CPU 멀티코어에서 분배한다. JAX 는 fork-unsafe 하므로 'spawn' 컨텍스트를 강제한다
(Windows 기본값과 동일; Linux/macOS 에서도 일관). 각 워커의 XLA/BLAS 내부 스레드는
1개로 제한(모듈 상단; import 이전 설정)하여 oversubscription 을 막는다.

로깅은 `core.logging_utils.get_logger` 를 사용한다.
"""

from __future__ import annotations

# --- 병렬 워커의 스레드 과다생성 방지 (JAX/BLAS import 이전에 설정) ---
import os
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from HalloReg.core.constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    PAYOFF_SELF, PAYOFF_OTHER, joint_index, mirror_state,
)
from HalloReg.core.logging_utils import get_logger
from .agent import ToMEmpathicAgent, AdaptiveAgent
from .env import StrategyAgent

LOGGER = get_logger("HalloReg.sim")


def _is_aif(agent) -> bool:
    return isinstance(agent, ToMEmpathicAgent)


def run_dyad(agent, opponent, n_rounds: int = 100,
             verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    focal `agent` vs `opponent` 동시행동 IPD.

    agent    : AIF 에이전트(step) 또는 StrategyAgent(act/observe).
    opponent : 위와 동일. AIF vs AIF, AIF vs Strategy, Strategy vs Strategy 모두 지원.
    반환      : per-round 배열 dict.
    """
    hist = {"my_act": [], "opp_act": [], "state": [],
            "my_payoff": [], "opp_payoff": []}
    prev_state = None
    prev_state_mirror = None

    for t in range(n_rounds):
        # focal 행동
        if _is_aif(agent):
            my_a = agent.step(prev_state)
        else:
            my_a = agent.act()

        # 상대 행동
        if _is_aif(opponent):
            opp_a = opponent.step(prev_state_mirror)
        else:
            opp_a = opponent.act()

        # 관측 상호 통지 (StrategyAgent)
        if not _is_aif(agent):
            agent.observe(opp_a)
        if not _is_aif(opponent):
            opponent.observe(my_a)

        state = joint_index(my_a, opp_a)
        hist["my_act"].append(my_a)
        hist["opp_act"].append(opp_a)
        hist["state"].append(state)
        hist["my_payoff"].append(PAYOFF_SELF[state])
        hist["opp_payoff"].append(PAYOFF_OTHER[state])
        prev_state = state
        prev_state_mirror = mirror_state(state)

        if verbose and (t % max(1, n_rounds // 5) == 0):
            LOGGER.info("t=%d my=%s opp=%s state=%d", t, my_a, opp_a, state)

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist


def coop_rate(acts: np.ndarray) -> float:
    return float(np.mean(np.asarray(acts) == COOP))


# --------------------------------------------------------------- 병렬 실행
def _worker(task):
    """
    top-level 워커 (spawn pickle 가능). task = (idx, factory_spec, n_rounds).
    factory_spec 는 (builder_name, kwargs) 로, `build_from_spec` 이 다이애드를 생성한다.
    반환 = (idx, result_dict).
    """
    idx, spec, n_rounds = task
    agent, opponent = build_from_spec(spec)
    hist = run_dyad(agent, opponent, n_rounds)
    result = {"hist": hist}
    if _is_aif(agent):
        result["agent_log"] = {k: np.asarray(v) for k, v in agent.log.items()}
    return idx, result


def build_from_spec(spec: dict):
    """
    다이애드 스펙 -> (agent, opponent).

    spec = {
        "agent":   {"type": "adaptive"|"tom_empathic"|"strategy", ...kwargs},
        "opponent":{"type": "strategy"|"adaptive"|..., ...kwargs},
    }
    seed 는 각 하위 dict 에 포함.
    """
    return _build_one(spec["agent"]), _build_one(spec["opponent"])


def _build_one(cfg: dict):
    cfg = dict(cfg)
    typ = cfg.pop("type")
    if typ == "adaptive":
        return AdaptiveAgent(**cfg)
    if typ == "tom_empathic":
        return ToMEmpathicAgent(**cfg)
    if typ == "strategy":
        from .env import make_opponent
        kind = cfg.pop("kind")
        return make_opponent(kind, **cfg)
    raise ValueError(f"알 수 없는 에이전트 type: {typ}")


def run_many(specs: List[dict], n_rounds: int = 100,
             n_jobs: Optional[int] = None, verbose: bool = True) -> List[dict]:
    """
    다수 다이애드 스펙을 순차/병렬 실행.

    n_jobs : None/1=순차, >=2=병렬, -1=(코어수-1).
    반환    : specs 순서와 동일한 결과 리스트.
    """
    tasks = [(i, spec, n_rounds) for i, spec in enumerate(specs)]
    results: List[Optional[dict]] = [None] * len(specs)

    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if n_jobs is None or n_jobs <= 1:
        for i, spec, nr in tasks:
            _, res = _worker((i, spec, nr))
            results[i] = res
            if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                LOGGER.info("진행 %d/%d 다이애드", i + 1, len(tasks))
    else:
        n_jobs = min(n_jobs, len(tasks))
        if verbose:
            LOGGER.info("병렬 실행: %d 다이애드 / %d 워커 (spawn)", len(tasks), n_jobs)
        ctx = mp.get_context("spawn")
        done = 0
        with ctx.Pool(processes=n_jobs) as pool:
            for idx, res in pool.imap_unordered(_worker, tasks, chunksize=1):
                results[idx] = res
                done += 1
                if verbose and done % max(1, len(tasks) // 10) == 0:
                    LOGGER.info("진행 %d/%d 다이애드", done, len(tasks))
    return results


# --------------------------------------------------------------- 집단 역학
def run_population(strategy_counts: Dict[str, int], n_rounds: int = 80,
                   n_adaptive: int = 0, adaptive_kwargs: Optional[dict] = None,
                   seed: int = 0, round_robin: bool = True) -> Dict:
    """
    혼합 전략 집단의 round-robin IPD (H8: 집단 상호협력률).

    strategy_counts : {'tit_for_tat': 3, 'alld': 2, ...} 고정전략 개체 수.
    n_adaptive : AdaptiveAgent 개체 수.
    반환 : {'coop_rate': 전체 상호협력률, 'by_pair': ..., 'agents': 라벨}.
    """
    adaptive_kwargs = adaptive_kwargs or {}
    rng = np.random.default_rng(seed)

    # 개체 생성 (라벨 유지)
    labels: List[str] = []
    for kind, n in strategy_counts.items():
        labels += [kind] * n
    labels += ["adaptive"] * n_adaptive

    def make(i, label):
        if label == "adaptive":
            return AdaptiveAgent(seed=seed * 100 + i, **adaptive_kwargs)
        from .env import make_opponent
        return make_opponent(label, seed=seed * 100 + i)

    n = len(labels)
    total_cc = 0
    total_rounds = 0
    pair_cc = {}

    for i in range(n):
        for j in range(i + 1, n):
            a = make(i, labels[i])
            b = make(j, labels[j])
            h = run_dyad(a, b, n_rounds)
            cc = float(np.mean(h["state"] == CC))
            total_cc += np.sum(h["state"] == CC)
            total_rounds += n_rounds
            pair_cc[(labels[i], labels[j])] = pair_cc.get((labels[i], labels[j]), []) + [cc]

    return {
        "cc_rate": float(total_cc / max(total_rounds, 1)),
        "by_pair": {k: float(np.mean(v)) for k, v in pair_cc.items()},
        "labels": labels,
        "n_agents": n,
    }
