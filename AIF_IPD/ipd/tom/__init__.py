"""HalloReg.ipd.tom — Theory of Mind: 추론(inversion), 예측/social EFE(tom_core), 계획."""

from .inversion import (
    OpponentInversion, ObservationContext, InversionState, THETA_AXES,
    THETA_AXES_FG, theta_axes,
)
from .tom_core import (
    TheoryOfMind, GatedToM, RecursiveSocialEFE, SocialEFEResult,
)
from .opponent_simulator import OpponentSimulator
from .sophisticated_planner import SophisticatedPlanner

__all__ = [
    "OpponentInversion", "ObservationContext", "InversionState", "THETA_AXES",
    "THETA_AXES_FG", "theta_axes",
    "TheoryOfMind", "GatedToM", "RecursiveSocialEFE", "SocialEFEResult",
    "OpponentSimulator", "SophisticatedPlanner",
]
