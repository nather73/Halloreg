"""
AIF_IPD.ipd
===========

IPD 환경·에이전트·시뮬레이션·집단/진화 분석 계층.

  env             : IPD 환경과 고정전략 상대 (TFT/GTFT/WSLS/ALLC/ALLD)
  agent           : EmpathicAgent(고정 λ) 와 HalloRegAgent(내생 λ)
  sim             : 다이애드 루프와 병렬 실행기
  payoff_schedule : 비정상 보수 레짐 카탈로그
  population      : 혼합 집단의 정확한 분해식과 유형쌍 행렬 추정
  evolution       : 복제자(RE) / 최적 복제자(ORE) 동역학
  tom             : 조망수용 계층
  metrics         : 통계 인프라
"""

from .agent import EmpathicAgent, HalloRegAgent
from .env import (
    Environment, StrategyAgent, make_opponent, make_switching_opponent,
    ALL_TYPES, FIXED_STRATEGIES, TYPE_LABEL_KO, SWITCH_SCENARIOS,
    SWITCH_PERIOD, switch_rounds,
)
from .sim import (
    run_dyad, run_many, build_agent, build_from_spec,
    coop_rate, cc_rate, mean_payoff, resolve_jobs,
)
from .payoff_schedule import (
    CI_REGIMES, NONSTATIONARY_REGIMES, REGIME_LABEL_KO,
    get_regime, regime_trace,
)

__all__ = [
    "EmpathicAgent", "HalloRegAgent",
    "Environment", "StrategyAgent", "make_opponent", "make_switching_opponent",
    "ALL_TYPES", "FIXED_STRATEGIES", "TYPE_LABEL_KO", "SWITCH_SCENARIOS",
    "SWITCH_PERIOD", "switch_rounds",
    "run_dyad", "run_many", "build_agent", "build_from_spec",
    "coop_rate", "cc_rate", "mean_payoff", "resolve_jobs",
    "CI_REGIMES", "NONSTATIONARY_REGIMES", "REGIME_LABEL_KO",
    "get_regime", "regime_trace",
]
