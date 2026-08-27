"""
AIF_IPD — HalloReg
==================

**Adaptive Prosociality Through Hierarchical Allostatic Regulation in Social
Dynamics: A Simulation Study** (Choi, Albarracin, Pae, & Kim)

An active-inference agent whose prosociality is **endogenously
regulated** in the Iterated Prisoner's Dilemma.

  (i)   perspective taking — particle-filter Bayesian inference over
        the opponent's model parameters
  (ii)  core affect — two-dimensional (valence x arousal) affect
        constructed from expected-reward distributions
  (iii) empathy — regulates the prosociality weight lambda from core
        affect and the inferred context
  (iv)  self model — establishes allostatic setpoints from beliefs
        about the social environment
"""

__version__ = "1.0.0"
