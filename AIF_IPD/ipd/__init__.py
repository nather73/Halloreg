"""
AIF_IPD.ipd
===========

IPD environment, agents, simulation, and population/evolution layer.

  env             : IPD environment and fixed strategies
                    (TFT/GTFT/WSLS/ALLC/ALLD)
  agent           : EmpathicAgent (fixed lambda) and HalloRegAgent
                    (endogenous lambda)
  sim             : dyad loop and parallel runner
  payoff_schedule : catalogue of non-stationary payoff regimes
  population      : exact mixed-population decomposition and
                    type-pair matrix estimation
  evolution       : replicator (RE) / optimal replicator (ORE)
                    dynamics
  tom             : perspective-taking layer
  metrics         : statistical infrastructure
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
