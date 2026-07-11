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

from AIF_IPD.core.constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    PAYOFF_SELF, PAYOFF_OTHER, joint_index, mirror_state,
)
from AIF_IPD.core.logging_utils import get_logger
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


# --------------------------------------------------------- 스펙 기반 집단(H8)
def _sample_pairs(n: int, partners_per_agent: int, rng) -> list:
    """
    희소 상호작용 네트워크: 각 개체가 평균 `partners_per_agent` 명의 무작위
    파트너와 다이애드를 맺는다 (N≈100 규모에서 완전 라운드로빈의 O(N²) 비용을
    O(N·m) 로 낮추는 표본화; 무작위 표집이므로 비편향 상호협력률 추정).
    """
    pairs = set()
    for i in range(n):
        # i 자신을 제외한 파트너 표집 (중복 쌍은 set 으로 자연 제거)
        choices = rng.choice(np.delete(np.arange(n), i),
                             size=min(partners_per_agent, n - 1), replace=False)
        for j in choices:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def run_population_spec(spec: dict) -> Dict:
    """
    스펙 기반 혼합 집단 IPD (H8 전면수정판).

    spec = {
        "members":  [member_cfg, ...]          # _build_one 호환 개체 스펙 목록
                                               #   (strategy / adaptive / tom_empathic,
                                               #    schedule 을 가진 변덕 상대 포함)
        "labels":   [str, ...] (선택)          # 개체별 라벨 (없으면 type/kind 로 유도)
        "n_rounds": int,
        "seed":     int,                       # 짝짓기 표집 + 개체 seed offset
        "partners_per_agent": int | None,      # None → 완전 라운드로빈
    }

    각 다이애드는 독립적으로 생성된 개체 쌍으로 실행된다(다이애드 기억; 원 구현과
    동일한 설계). 반환 지표:
        cc_rate        : 전체 상호협력(CC)률
        mean_payoff    : 개체·라운드당 평균 보수 (집단 후생)
        group_payoff   : 집단 전체의 최종(누적) 보수 합
        payoff_trace   : 라운드별 평균 보수 (모든 다이애드 양측 평균)
        early_payoff / late_payoff : 초기/후기 1/3 구간 평균 보수
        payoff_growth  : (후기 − 초기) / max(초기, ε)  — 기대 payoff 증가율
        by_label_cc    : 라벨 조합별 CC률
    """
    members = spec["members"]
    n_rounds = int(spec.get("n_rounds", 60))
    seed = int(spec.get("seed", 0))
    ppa = spec.get("partners_per_agent")
    labels = spec.get("labels")
    if labels is None:
        labels = []
        for m in members:
            if m.get("type") == "strategy":
                labels.append(m.get("kind", "strategy")
                              + ("*" if m.get("schedule") else ""))
            else:
                labels.append(m.get("type", "agent"))

    n = len(members)
    rng = np.random.default_rng(seed)
    if ppa is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = _sample_pairs(n, int(ppa), rng)

    def build(idx: int, dyad_id: int):
        cfg = dict(members[idx])
        # 다이애드마다 독립 seed (개체 스펙 seed + 집단 seed + 다이애드 offset)
        cfg["seed"] = int(cfg.get("seed", 0)) + seed * 7919 + dyad_id * 131 + idx
        return _build_one(cfg)

    total_cc = 0
    total_rounds = 0
    payoff_sum = np.zeros(n_rounds)       # 라운드별 (양측 합) 보수 누적
    group_payoff = 0.0
    pair_cc: Dict[tuple, list] = {}

    for d, (i, j) in enumerate(pairs):
        a = build(i, d)
        b = build(j, d)
        h = run_dyad(a, b, n_rounds)
        total_cc += int(np.sum(h["state"] == CC))
        total_rounds += n_rounds
        payoff_sum += (h["my_payoff"] + h["opp_payoff"]) / 2.0
        group_payoff += float(np.sum(h["my_payoff"]) + np.sum(h["opp_payoff"]))
        key = tuple(sorted((labels[i], labels[j])))
        pair_cc.setdefault(key, []).append(float(np.mean(h["state"] == CC)))

    n_pairs = max(len(pairs), 1)
    payoff_trace = payoff_sum / n_pairs
    third = max(n_rounds // 3, 1)
    early = float(payoff_trace[:third].mean())
    late = float(payoff_trace[-third:].mean())
    return {
        "cc_rate": float(total_cc / max(total_rounds, 1)),
        "mean_payoff": float(payoff_trace.mean()),
        "group_payoff": group_payoff,
        "payoff_trace": payoff_trace,
        "early_payoff": early,
        "late_payoff": late,
        "payoff_growth": float((late - early) / max(early, 1e-9)),
        "by_label_cc": {" | ".join(k): float(np.mean(v))
                        for k, v in pair_cc.items()},
        "n_agents": n,
        "n_pairs": len(pairs),
        "labels": labels,
    }


def _pop_worker(task):
    """집단 시뮬레이션 병렬 워커 (spawn pickle 가능 top-level)."""
    idx, spec = task
    return idx, run_population_spec(spec)


def run_populations(specs: List[dict], n_jobs: Optional[int] = None,
                    verbose: bool = True) -> List[dict]:
    """
    다수의 집단 스펙을 순차/병렬 실행 (H8 병렬화).

    다이애드 병렬화(run_many)와 동일한 spawn 컨텍스트/스레드 제한 정책을 따르며,
    집단 replicate 단위(embarrassingly parallel)로 CPU 코어에 분배한다.
    """
    tasks = [(i, spec) for i, spec in enumerate(specs)]
    results: List[Optional[dict]] = [None] * len(specs)

    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if n_jobs is None or n_jobs <= 1:
        for i, spec in tasks:
            _, res = _pop_worker((i, spec))
            results[i] = res
            if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                LOGGER.info("진행 %d/%d 집단", i + 1, len(tasks))
    else:
        n_jobs = min(n_jobs, len(tasks))
        if verbose:
            LOGGER.info("병렬 실행: %d 집단 / %d 워커 (spawn)", len(tasks), n_jobs)
        ctx = mp.get_context("spawn")
        done = 0
        with ctx.Pool(processes=n_jobs) as pool:
            for idx, res in pool.imap_unordered(_pop_worker, tasks, chunksize=1):
                results[idx] = res
                done += 1
                if verbose and done % max(1, len(tasks) // 10) == 0:
                    LOGGER.info("진행 %d/%d 집단", done, len(tasks))
    return results
