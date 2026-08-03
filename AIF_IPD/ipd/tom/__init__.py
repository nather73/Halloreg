"""
AIF_IPD.ipd.tom
===============

조망수용(perspective-taking) 계층.

  inversion : 상대 특성 θ_j 입자필터 (OpponentInversion)
  tom_core  : 정적/게이팅 ToM 과 재귀적 social EFE
  planner   : 다단계 rollout 계획기
"""

from .inversion import OpponentInversion, ObservationContext, THETA_AXES
from .tom_core import (
    TheoryOfMind, GatedToM, RecursiveSocialEFE, SocialEFEResult,
)
from .planner import OpponentSimulator, SophisticatedPlanner

__all__ = [
    "OpponentInversion", "ObservationContext", "THETA_AXES",
    "TheoryOfMind", "GatedToM", "RecursiveSocialEFE", "SocialEFEResult",
    "OpponentSimulator", "SophisticatedPlanner",
]
