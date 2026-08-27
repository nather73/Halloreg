"""
AIF_IPD.experiments
===================

Per-hypothesis experiment modules.

  common            : configuration, result registry, plotting utils
  arch_validation   : ARCH — architecture-implementation consistency
                      (mandatory before any hypothesis)
  h1_intent         : H1  — strategic intent inference
  h1a_tracking      : H1A — shifting-intent tracking and lambda
                      recovery
  h2_protection     : H2/H2A — exploiter defence and noise-vs-intent
                      discrimination
  h2b_counterfactual: H2B — regulated lambda vs lambda fixed at 0/1
                      (onset, withdrawal, withdrawal decomposition)
  h3_h4_population  : H3/H3A/H4/H4A — stationary/non-stationary
                      mixed populations
  h5_evolution      : H5  — survival under RE/ORE evolutionary
                      dynamics
  h6_recovery       : H6  — parameter recovery and self-projection
"""

from .common import Config, Registry

__all__ = ["Config", "Registry"]
