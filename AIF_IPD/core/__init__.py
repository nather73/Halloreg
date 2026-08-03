"""
AIF_IPD.core
============

생성모형·보수구조·위계적 이상성 조절(SelfModel / CoreAffect / Empathy)의 핵심 계층.

  constants     : PD 보수 구조와 joint-outcome 인덱스 규약, 가변 보수 API
  generative    : POMDP 생성모형(A,B,C,D)과 해석적 기대자유에너지(EFE)
  self_model    : identity 기억 + 사전 공급 + 할로스타틱 설정점
  core_affect   : valence(RPE) × arousal(KL) 의 2차원 핵심정서
  empathy       : λ_aff / λ_ctx 로부터 λ 를 산출하는 적분기
  logging_utils : 로거 및 한글 폰트 설정
"""

from .constants import (
    CC, CD, DC, DD, COOP, DEFECT,
    N_STATES, N_ACTIONS, PAYOFF_SELF, PAYOFF_OTHER, PD_PAYOFFS,
    set_payoffs, set_payoff_matrix, reset_payoffs, current_payoffs,
    empathy_shift, joint_index, split_joint, mirror_state,
    opponent_action_from_state, my_action_from_state,
)
from .generative import efe_terms, softmax, predicted_next_state
from .self_model import SelfModel, MemoryEntry, THETA_AXES
from .core_affect import CoreAffect
from .empathy import Empathy
from .logging_utils import get_logger, set_korean_font

__all__ = [
    "CC", "CD", "DC", "DD", "COOP", "DEFECT",
    "N_STATES", "N_ACTIONS", "PAYOFF_SELF", "PAYOFF_OTHER", "PD_PAYOFFS",
    "set_payoffs", "set_payoff_matrix", "reset_payoffs", "current_payoffs",
    "empathy_shift", "joint_index", "split_joint", "mirror_state",
    "opponent_action_from_state", "my_action_from_state",
    "efe_terms", "softmax", "predicted_next_state",
    "SelfModel", "MemoryEntry", "THETA_AXES", "CoreAffect", "Empathy",
    "get_logger", "set_korean_font",
]
