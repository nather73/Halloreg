"""HalloReg.core — 공유 유틸리티: 상수, 생성모형/EFE, 항상성 조절, 로깅."""

from .constants import (
    R, T, S, P, CC, CD, DC, DD, COOP, DEFECT,
    N_STATES, N_ACTIONS, PAYOFF_SELF, PAYOFF_OTHER, PD_PAYOFFS,
    ACTION_NAMES, STATE_NAMES,
    joint_index, split_joint, opponent_action_from_state,
    my_action_from_state, mirror_state,
)
from .generative import (
    build_A, build_B, build_C, build_D,
    efe_terms, G_self, G_other_perspective, predicted_next_state, softmax,
)
from .allostasis import (
    counterfactual_disposition,
    CoreAllostaticBeliefState, LambdaRegulator,
    CAUSE_AXES, DISPOSITIONAL_AXES, CONTEXTUAL_AXES,
)
from .logging_utils import get_logger, set_korean_font

__all__ = [
    "counterfactual_disposition",
    "R", "T", "S", "P", "CC", "CD", "DC", "DD", "COOP", "DEFECT",
    "N_STATES", "N_ACTIONS", "PAYOFF_SELF", "PAYOFF_OTHER", "PD_PAYOFFS",
    "ACTION_NAMES", "STATE_NAMES",
    "joint_index", "split_joint", "opponent_action_from_state",
    "my_action_from_state", "mirror_state",
    "build_A", "build_B", "build_C", "build_D",
    "efe_terms", "G_self", "G_other_perspective", "predicted_next_state", "softmax",
    "CoreAllostaticBeliefState", "LambdaRegulator",
    "CAUSE_AXES", "DISPOSITIONAL_AXES", "CONTEXTUAL_AXES",
    "get_logger", "set_korean_font",
]
