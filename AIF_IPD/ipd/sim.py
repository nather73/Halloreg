"""
ipd.sim
=======

Simulation runner: the dyad loop + parallel multi-spec execution.

[Parallelism]
Dyads are embarrassingly parallel, distributed via multiprocessing.
The 'spawn' context is forced because fork inherits numpy/BLAS thread
state into children (possible deadlocks) and spawn matches the
Windows default, keeping behaviour consistent across OSes. Each
worker's BLAS threads are capped at 1 (set at module top, before the
numpy import) to prevent oversubscription.

[Non-stationary payoffs]
With a `ci_schedule`, `set_payoffs(ci)` refreshes the global payoffs
every round. The globals change in place, so the agents' EFE and
CoreAffect automatically reflect the **current context's utilities**;
fixed strategies never read payoffs and are insensate — this
asymmetry is the theoretical basis of H4.
"""

from __future__ import annotations

# --- Prevent worker thread oversubscription (before numpy/BLAS) ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from typing import Callable, Dict, List, Optional

import numpy as np

from AIF_IPD.core.constants import (
    CC, COOP, PAYOFF_SELF, PAYOFF_OTHER, joint_index, mirror_state,
    reset_payoffs, set_payoffs,
)
from AIF_IPD.core.logging_utils import get_logger
from .agent import EmpathicAgent, HalloRegAgent
from .env import StrategyAgent, make_opponent

LOGGER = get_logger("HalloReg.sim")


def _is_aif(agent) -> bool:
    """Is this an active-inference agent (step interface)?"""
    return isinstance(agent, EmpathicAgent)


# ==================================================================== Dyad
def run_dyad(agent, opponent, n_rounds: int = 120,
             ci_schedule: Optional[Callable[[int, int], float]] = None,
             env_err_a: float = 0.0, env_err_b: float = 0.0,
             noise_seed: Optional[int] = None,
             partner_id_a: int = 1, partner_id_b: int = 2
             ) -> Dict[str, np.ndarray]:
    """
    Run the simultaneous IPD, focal `agent` vs `opponent`, for
    n_rounds.

    Parameters
    ----------
    ci_schedule : (t, T) -> CI | None
        None = fixed payoffs (stationary); a function resets payoffs
        each round.
    env_err_a, env_err_b : float
        **Environment-level** execution-error rates. They flip the
        emitted action regardless of agent type, so they can be
        imposed symmetrically on AIF and fixed strategies (distinct
        from StrategyAgent's internal error).
    noise_seed : int | None
        Dedicated seed for the pre-generated flip sequences. Sharing
        it across conditions realises identical noise (common random
        numbers), reducing paired-comparison variance.

    Returns
    -------
    dict of ndarray : my_act, opp_act, state, my_payoff, opp_payoff,
    ci
    """
    hist = {"my_act": [], "opp_act": [], "state": [],
            "my_payoff": [], "opp_payoff": [], "ci": []}

    # Partner-identity notification — HalloRegAgent's memory/prior
    # injection hook.
    if hasattr(agent, "begin_partner"):
        agent.begin_partner(partner_id_a)
    if hasattr(opponent, "begin_partner"):
        opponent.begin_partner(partner_id_b)

    # Pre-generate the noise sequences (path-independent -> CRN)
    if env_err_a > 0 or env_err_b > 0:
        nrng = np.random.default_rng(0 if noise_seed is None else int(noise_seed))
        flips_a = nrng.random(n_rounds) < env_err_a
        flips_b = nrng.random(n_rounds) < env_err_b
    else:
        flips_a = flips_b = None

    prev_state = None            # last joint outcome, focal view
    prev_state_mirror = None     # last joint outcome, opponent view

    for t in range(n_rounds):
        # ---- Non-stationary payoffs: fix this round's game ----
        if ci_schedule is not None:
            ci_t = float(ci_schedule(t, n_rounds))
            set_payoffs(ci_t)
        else:
            ci_t = float("nan")

        # ---- Emit actions ----
        my_a = agent.step(prev_state) if _is_aif(agent) else agent.act()
        opp_a = (opponent.step(prev_state_mirror) if _is_aif(opponent)
                 else opponent.act())

        # ---- Environment-level execution errors ----
        # On a flip, notify the AIF agent of the **actually emitted**
        # action: the opponent reacts to the emission, so if the
        # internal my_last stayed at the intent, next round's
        # reciprocity signal f would diverge from what the opponent
        # saw.
        if flips_a is not None and flips_a[t]:
            my_a = 1 - my_a
            if _is_aif(agent):
                agent.note_emitted(my_a)
        if flips_b is not None and flips_b[t]:
            opp_a = 1 - opp_a
            if _is_aif(opponent):
                opponent.note_emitted(opp_a)

        # ---- Mutual observation (fixed strategies see emissions) ----
        if not _is_aif(agent):
            agent.observe(opp_a)
        if not _is_aif(opponent):
            opponent.observe(my_a)

        # ---- Record ----
        state = joint_index(my_a, opp_a)
        hist["my_act"].append(my_a)
        hist["opp_act"].append(opp_a)
        hist["state"].append(state)
        hist["my_payoff"].append(float(PAYOFF_SELF[state]))
        hist["opp_payoff"].append(float(PAYOFF_OTHER[state]))
        hist["ci"].append(ci_t)

        prev_state = state
        prev_state_mirror = mirror_state(state)

    if ci_schedule is not None:
        reset_payoffs()          # avoid global-state contamination

    return {k: np.asarray(v) for k, v in hist.items()}


# ==================================================================== Spec builders
def build_agent(cfg: dict):
    """
    Agent spec dict -> instance.

    cfg = {"type": "halloreg" | "empathic" | "strategy" |
           "likelihood", ...}
    """
    cfg = dict(cfg)
    typ = cfg.pop("type")
    if typ == "halloreg":
        return HalloRegAgent(**cfg)
    if typ == "empathic":
        return EmpathicAgent(**cfg)
    if typ == "strategy":
        kind = cfg.pop("kind")
        return make_opponent(kind, **cfg)
    if typ == "likelihood":
        # Generates directly from the likelihood basis — true traits
        # known, used for recovery validation.
        from .env import LikelihoodAgent
        return LikelihoodAgent(**cfg)
    raise ValueError(f"unknown agent type: {typ}")


def build_from_spec(spec: dict):
    """Dyad spec -> (agent, opponent)."""
    return build_agent(spec["agent"]), build_agent(spec["opponent"])


# ==================================================================== Parallel run
def _worker(task):
    """
    Top-level worker (must be module-level to pickle under spawn).
    task = (idx, spec, n_rounds)
    """
    idx, spec, n_rounds = task
    from AIF_IPD.ipd.payoff_schedule import get_regime

    # Workers are fresh processes: reset globals to the default PD
    # before starting (idempotent).
    reset_payoffs()
    ci_fn = get_regime(spec.get("regime")) if spec.get("regime") else None

    agent, opponent = build_from_spec(spec)
    hist = run_dyad(agent, opponent, n_rounds,
                    ci_schedule=ci_fn,
                    env_err_a=float(spec.get("env_err_agent", 0.0)),
                    env_err_b=float(spec.get("env_err_opponent", 0.0)),
                    noise_seed=spec.get("noise_seed"))
    out = {"hist": hist}
    # Collect AIF-internal logs (lambda, affect, theta trajectories).
    if _is_aif(agent):
        out["agent_log"] = {k: np.asarray(v) for k, v in agent.log.items()}
    if _is_aif(opponent):
        out["opponent_log"] = {k: np.asarray(v) for k, v in opponent.log.items()}
    return idx, out


def resolve_jobs(n_jobs: Optional[int]) -> int:
    """Interpret --jobs: -1 -> (cores - 1), None/0 -> 1."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs == -1:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(n_jobs))


def run_many(specs: List[dict], n_rounds: int = 120,
             n_jobs: Optional[int] = None, verbose: bool = True,
             desc: str = "dyads") -> List[dict]:
    """
    Run many dyad specs sequentially or in parallel; results are
    returned in spec order.
    """
    tasks = [(i, spec, n_rounds) for i, spec in enumerate(specs)]
    results: List[Optional[dict]] = [None] * len(specs)
    n_jobs = resolve_jobs(n_jobs)

    if n_jobs <= 1 or len(tasks) == 1:
        for i, spec, nr in tasks:
            _, res = _worker((i, spec, nr))
            results[i] = res
            if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                LOGGER.info("  progress %d/%d %s", i + 1, len(tasks), desc)
    else:
        n_jobs = min(n_jobs, len(tasks))
        if verbose:
            LOGGER.info("  parallel: %d %s / %d workers", len(tasks), desc, n_jobs)
        ctx = mp.get_context("spawn")
        done = 0
        with ctx.Pool(processes=n_jobs) as pool:
            for idx, res in pool.imap_unordered(_worker, tasks, chunksize=1):
                results[idx] = res
                done += 1
                if verbose and done % max(1, len(tasks) // 10) == 0:
                    LOGGER.info("  progress %d/%d %s", done, len(tasks), desc)
    return results


# ==================================================================== Summary metrics
def coop_rate(acts: np.ndarray) -> float:
    """Cooperation rate of an action sequence."""
    return float(np.mean(np.asarray(acts) == COOP))


def cc_rate(hist: dict) -> float:
    """Mutual-cooperation (CC) rate."""
    return float(np.mean(hist["state"] == CC))


def mean_payoff(hist: dict, side: str = "my") -> float:
    """Mean per-round payoff."""
    return float(np.mean(hist[f"{side}_payoff"]))
