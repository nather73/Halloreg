"""
ipd.tom — the perspective-taking (Theory of Mind) layer.

  inversion   : particle filter over opponent traits theta_j
  tom_core    : static/gated ToM and the recursive social EFE
  self_policy : trait-space policy — SMC over self traits (legacy)

The action-level planner was retired: the focal agent is represented
in the same trait space as the opponent, so planning evaluates trait
candidates rather than branching over actions.
"""

from .inversion import OpponentInversion, ObservationContext, THETA_AXES
from .self_policy import SELF_AXES, SELF_BOUNDS, SELF_PRIOR, SelfPolicy
from .tom_core import (
    TheoryOfMind, GatedToM, RecursiveSocialEFE, SocialEFEResult,
)

__all__ = [
    "OpponentInversion", "ObservationContext", "THETA_AXES",
    "SelfPolicy", "SELF_AXES", "SELF_PRIOR", "SELF_BOUNDS",
    "TheoryOfMind", "GatedToM", "RecursiveSocialEFE", "SocialEFEResult",
]
