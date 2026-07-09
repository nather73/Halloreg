"""HalloReg.ipd — 반복 죄수의 딜레마 에이전트/환경/시뮬레이션."""

from .agent import ToMEmpathicAgent, AdaptiveAgent
from .env import Environment, StrategyAgent, make_opponent, make_capricious
from .sim import run_dyad, run_many, run_population, build_from_spec

__all__ = [
    "ToMEmpathicAgent", "AdaptiveAgent",
    "Environment", "StrategyAgent", "make_opponent", "make_capricious",
    "run_dyad", "run_many", "run_population", "build_from_spec",
]
