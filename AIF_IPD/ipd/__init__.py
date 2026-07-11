"""HalloReg.ipd — 반복 죄수의 딜레마 에이전트/환경/시뮬레이션."""

from .agent import ToMEmpathicAgent, AdaptiveAgent
from .env import (
    Environment, StrategyAgent, make_opponent, make_capricious,
    CAPRICIOUS_CASES, make_capricious_case, capricious_case_spec,
    capricious_switch_rounds,
)
from .sim import (
    run_dyad, run_many, run_population, run_population_spec,
    run_populations, build_from_spec,
)

__all__ = [
    "ToMEmpathicAgent", "AdaptiveAgent",
    "Environment", "StrategyAgent", "make_opponent", "make_capricious",
    "CAPRICIOUS_CASES", "make_capricious_case", "capricious_case_spec",
    "capricious_switch_rounds",
    "run_dyad", "run_many", "run_population", "run_population_spec",
    "run_populations", "build_from_spec",
]

from .baselines import (  # noqa: F401
    QLearnerAgent, BayesBestResponse, FictitiousPlayAgent, make_baseline,
)
