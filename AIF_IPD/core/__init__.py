"""
AIF_IPD.core
============

Core layer: generative model, payoff structure, and hierarchical
allostatic regulation (SelfModel / CoreAffect / Empathy).

  constants     : PD payoff structure, joint-outcome index
                  conventions, non-stationary payoff API
  generative    : POMDP generative model (A,B,C,D) and analytic EFE
  self_model    : identity memory, prior supply, allostatic setpoints
  core_affect   : two-dimensional core affect — valence (RPE) x
                  arousal (KL)
  empathy       : legacy lambda integrator (EmpathicAgent baseline)
  logging_utils : logger and font configuration
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
