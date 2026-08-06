"""
ipd.tom — 조망수용(Theory of Mind) 계층.

  inversion   : 상대 특성 θ_j 입자필터 (OpponentInversion)
  tom_core    : 정적/게이팅 ToM 과 재귀적 social EFE
  self_policy : **형질공간 정책** — 자기 형질 (ρ, ω, η) 의 SMC 선택

행동수준 계획기(planner)는 폐기되었다. focal 도 상대와 같은 형질공간에 표상
되므로, 계획은 행동 분기가 아니라 형질 후보 평가로 이루어진다.
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
