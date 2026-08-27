"""
ipd.population
==============

**Mixed-population analysis — the shared basis of
H3/H3A/H4/H4A/H5.**

The hypotheses require: {TFT=a, GTFT=b, WSLS=c, ALLC=d, ALLD=e,
HalloReg=f}, a+...+f = 30, "simulate as many combinations as
possible".

[Problem] Distributing 30 over 6 types gives C(35, 5) = 324,632
compositions. Directly simulating a 30-agent round robin per
composition (435 dyads x 120 rounds) yields 1.4e8 dyads — infeasible
on any budget.

[Solution — an **exact** decomposition, not mean-field]
In this design every round-robin dyad runs as an **independently
constructed pair** (agents are freshly built per dyad; no information
transfers across partners). Population-level metrics then equal the
composition-weighted average of type-pair metrics **exactly**:

    per-type mean payoff
        mu_i(n) = [sum_j n_j*Pi[i,j] - Pi[i,i]] / (N - 1)
    population mutual-cooperation rate
        CC(n) = [n' CCm n - sum_i n_i*CCm[i,i]] / [N*(N - 1)]

where Pi[i,j] is type i's mean per-round payoff against type j and
CCm[i,j] that dyad's CC rate; the diagonal subtraction removes
self-pairing. This identity is exact at the level of expectations,
not an approximation — so 21 type pairs (6*7/2) are simulated and
**all** 324,632 compositions are evaluated analytically. "As many
combinations as possible" is met literally at O(21 x seeds) cost.

[Verification duty] The identity rests on the fresh-agents-per-dyad
premise. HalloRegAgent carries identity memory, so persistent agents
would break it; the experiment scripts therefore run **direct
round-robin simulations** on representative compositions and compare
them to the analytic predictions (the H3 verification panel).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

from AIF_IPD.core.constants import CC
from AIF_IPD.core.logging_utils import get_logger
from .env import ALL_TYPES
from .sim import run_dyad, run_many

LOGGER = get_logger("HalloReg.population")


# ==================================================================== Type specs
def type_spec(name: str, **overrides) -> dict:
    """
    Type name -> agent spec dict (the caller fills the seed).
    Only HalloReg is an active-inference agent; the other five are
    fixed strategies.
    """
    if name == "halloreg":
        spec = {"type": "halloreg"}
    else:
        spec = {"type": "strategy", "kind": name}
    spec.update(overrides)
    return spec


def default_type_specs(halloreg_kwargs: Optional[dict] = None
                       ) -> Dict[str, dict]:
    """Default spec dict of the six types (ordered as ALL_TYPES)."""
    hk = halloreg_kwargs or {}
    return {n: type_spec(n, **(hk if n == "halloreg" else {}))
            for n in ALL_TYPES}


# ============================================================ Pair-matrix estimation
def estimate_pair_matrices(type_specs: Dict[str, dict], n_rounds: int,
                           seeds: int, n_jobs: int,
                           eval_from: int = 0,
                           regime: Optional[str] = None,
                           env_error: float = 0.05,
                           seed_offset: int = 0,
                           verbose: bool = True) -> Dict:
    """
    Estimate the pair payoff matrix Pi and mutual-cooperation
    matrix CCm from dyad simulations.

    Parameters
    ----------
    type_specs : {type name: spec}
        Key order becomes the row/column order of Pi.
    regime : str | None
        Payoff-regime name; None = fixed PD.
    env_error : float
        Environment-level execution error, imposed **symmetrically on
        every type** — asymmetric noise would bias comparisons.

    Returns
    -------
    dict
      names   : type names (k,)
      Pi      : (k, k) row type's mean per-round payoff
      Pi_sd   : (k, k) between-seed SD
      Pi_raw  : (k, k, seeds) per-seed raw values for seed-level
                statistics
      CCm     : (k, k) mutual-cooperation rates
      (CCm_raw carries per-seed raw values for seed-level tests.)
      CCm_raw : (k, k, seeds)
    """
    names = list(type_specs)
    k = len(names)
    # Symmetric game: run only i <= j and harvest both directions
    # from one dyad (my_payoff -> Pi[i,j], opp_payoff -> Pi[j,i]),
    # halving the dyad count.
    pairs = [(i, j) for i in range(k) for j in range(i, k)]

    specs, registry = [], {}
    for (i, j) in pairs:
        for sd in range(seeds):
            a = dict(type_specs[names[i]])
            a["seed"] = 10_000 + seed_offset + sd * 17 + i
            b = dict(type_specs[names[j]])
            b["seed"] = 20_000 + seed_offset + sd * 17 + j
            registry[(i, j, sd)] = len(specs)
            specs.append({
                "agent": a, "opponent": b,
                "regime": regime,
                "env_err_agent": env_error, "env_err_opponent": env_error,
                # Common random numbers across conditions: the same
                # (sd, i, j) realises identical noise
                "noise_seed": 900_000 + seed_offset * 7 + sd * 101 + i * 7 + j,
            })

    res = run_many(specs, n_rounds=n_rounds, n_jobs=n_jobs, verbose=verbose,
                   desc=f"type-pair dyads ({regime or 'stationary'})")

    Pi_raw = np.zeros((k, k, seeds))
    CC_raw = np.zeros((k, k, seeds))
    for (i, j) in pairs:
        # eval_from: the **post-training evaluation window** — the
        # warm-up (early learning span) is excluded from the mean;
        # 0 = full span (legacy behaviour).
        _e0 = max(0, min(int(eval_from), n_rounds - 1))
        my = np.array([float(np.mean(
            res[registry[(i, j, sd)]]["hist"]["my_payoff"][_e0:]))
            for sd in range(seeds)])
        op = np.array([float(np.mean(
            res[registry[(i, j, sd)]]["hist"]["opp_payoff"][_e0:]))
            for sd in range(seeds)])
        cc = np.array([float(np.mean(
            res[registry[(i, j, sd)]]["hist"]["state"][_e0:] == CC))
            for sd in range(seeds)])
        if i == j:
            # Self-pair: both sides are the same type — average the
            # two viewpoints
            Pi_raw[i, i] = 0.5 * (my + op)
        else:
            Pi_raw[i, j] = my
            Pi_raw[j, i] = op
        CC_raw[i, j] = cc
        CC_raw[j, i] = cc          # CC is viewpoint-symmetric

    return {
        "names": names,
        "Pi": Pi_raw.mean(axis=2),
        "Pi_sd": Pi_raw.std(axis=2, ddof=1) if seeds > 1 else np.zeros((k, k)),
        "Pi_raw": Pi_raw,
        "CCm": CC_raw.mean(axis=2),
        "CCm_raw": CC_raw,
        "n_rounds": n_rounds, "seeds": seeds, "regime": regime,
        "env_error": env_error, "eval_from": int(eval_from),
    }


# ============================================================ Composition enumeration
def enumerate_compositions(total: int = 30, k: int = 6,
                           min_each: int = 0) -> np.ndarray:
    """
    Enumerate **all** partitions of total into k non-negative
    integers ("stars and bars": choosing k - 1 bar positions among
    total + k - 1 slots corresponds one-to-one to compositions;
    count C(total + k - 1, k - 1), i.e. C(35, 5) = 324,632 for
    total=30, k=6).

    min_each : minimum count per type; e.g. 1 keeps only compositions
               where every type is present (well-defined
               comparisons).

    Returns an (M, k) int array.
    """
    if min_each > 0:
        # Assign min_each to every type first, then distribute the
        # remainder freely
        rest = total - min_each * k
        if rest < 0:
            return np.zeros((0, k), dtype=int)
        base = enumerate_compositions(rest, k, 0)
        return base + min_each

    n = total + k - 1
    out = []
    for bars in combinations(range(n), k - 1):
        prev = -1
        comp = []
        for b in bars:
            comp.append(b - prev - 1)
            prev = b
        comp.append(n - prev - 1)
        out.append(comp)
    return np.asarray(out, dtype=int)


def subsample_compositions(comps: np.ndarray, n_max: int,
                           seed: int = 0) -> np.ndarray:
    """Uniform random subsample when there are too many
    compositions (reproducible)."""
    if len(comps) <= n_max:
        return comps
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(comps), size=n_max, replace=False)
    return comps[np.sort(idx)]


# ============================================================ Analytic evaluation
def type_payoffs(Pi: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Per-composition, per-type mean payoff,

        mu_i(n) = [sum_j n_j*Pi[i,j] - Pi[i,i]] / (N - 1).

    counts : (M, k) or (k,); returns the same shape. Values for
    zero-count types are undefined — the caller must mask them (the
    formula is evaluated formally here).
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    N = c.sum(axis=1, keepdims=True)
    # c @ Pi.T -> [m, i] = sum_j c[m,j]*Pi[i,j]
    tot = c @ np.asarray(Pi, dtype=float).T
    out = (tot - np.diag(Pi)[None, :]) / np.maximum(N - 1.0, 1e-9)
    return out[0] if np.ndim(counts) == 1 else out


def population_cc(CCm: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Population mutual-cooperation rate per composition,

        CC(n) = [n' CCm n - sum_i n_i*CCm[i,i]] / [N*(N - 1)].

    Derivation: summing cross-type pairs sum_{i<j} n_i n_j CC[i,j]
    and within-type pairs sum_i n_i(n_i - 1)/2 * CC[i,i] yields the
    formula (self-pairs excluded).
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    M = np.asarray(CCm, dtype=float)
    N = c.sum(axis=1)
    quad = np.einsum("mi,ij,mj->m", c, M, c)
    diag = c @ np.diag(M)
    out = (quad - diag) / np.maximum(N * (N - 1.0), 1e-9)
    return out[0] if np.ndim(counts) == 1 else out


def population_mean_payoff(Pi: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Population-wide mean payoff per composition (agent-
    weighted)."""
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    mu = type_payoffs(Pi, c)
    N = c.sum(axis=1)
    out = np.sum(c * mu, axis=1) / np.maximum(N, 1e-9)
    return out[0] if np.ndim(counts) == 1 else out


def substitution_delta_cc(CCm: np.ndarray, counts: np.ndarray,
                          src: int, dst: int) -> np.ndarray:
    """
    **Substitution contrast** — the contribution measure of
    H3A/H4A: the change in population CC when one src agent in
    composition n is replaced by a dst agent,

        Delta = CC(n) - CC(n - e_src + e_dst).

    Delta > 0 means "having src in that slot lifts population
    cooperation more than having dst". With src = HalloReg and dst
    ranging over the five fixed strategies, HalloReg's **marginal
    contribution** is measured directly against each alternative.
    Compositions with n_src = 0 are undefined and return NaN.
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float)).copy()
    base = population_cc(CCm, c)
    alt = c.copy()
    alt[:, src] -= 1.0
    alt[:, dst] += 1.0
    delta = base - population_cc(CCm, alt)
    delta = np.where(c[:, src] >= 1.0, delta, np.nan)
    return delta[0] if np.ndim(counts) == 1 else delta


# ============================================================ Direct round robin
def run_round_robin(counts: Sequence[int], type_specs: Dict[str, dict],
                    n_rounds: int, seed: int = 0,
                    regime: Optional[str] = None,
                    env_error: float = 0.05,
                    persistent: bool = False,
                    n_jobs: int = 1, eval_from: int = 0) -> Dict:
    """
    **Direct** round-robin simulation — validation of the analytic
    decomposition.

    persistent=False (default)
        Agents are built fresh per dyad, matching the identity's
        premise, so both values must agree within sampling error.
        Dyads are independent and run through `run_many` in parallel.
    persistent=True
        Agents persist across the whole population. HalloReg's
        identity memory then accumulates across partners and the
        identity may fail — the size of that deviation quantifies the
        population-level effect of memory. State accumulates
        sequentially, so this path **cannot be parallelised**.

    Returns cc_rate, mean_payoff, by_type_payoff
    """
    from .payoff_schedule import get_regime
    from .sim import build_agent

    names = list(type_specs)
    labels: List[str] = []
    for i, nm in enumerate(names):
        labels += [nm] * int(counts[i])
    n = len(labels)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def cfg_for(idx: int, dyad_id: int) -> dict:
        cfg = dict(type_specs[labels[idx]])
        cfg["seed"] = seed * 7919 + dyad_id * 131 + idx
        return cfg

    total_cc = 0
    total_rounds = 0
    pay_sum = {nm: 0.0 for nm in names}
    pay_rounds = {nm: 0 for nm in names}

    if persistent:
        # ---- Sequential path (state persists across dyads) ----
        ci_fn = get_regime(regime) if regime else None
        agents = [build_agent(cfg_for(i, 0)) for i in range(n)]
        for d, (i, j) in enumerate(pairs):
            h = run_dyad(agents[i], agents[j], n_rounds, ci_schedule=ci_fn,
                         env_err_a=env_error, env_err_b=env_error,
                         noise_seed=seed * 31 + d,
                         partner_id_a=1000 + j, partner_id_b=1000 + i)
            _e0 = max(0, min(int(eval_from), n_rounds - 1))
            _nr = n_rounds - _e0
            total_cc += int(np.sum(h["state"][_e0:] == CC))
            total_rounds += _nr
            pay_sum[labels[i]] += float(np.sum(h["my_payoff"][_e0:]))
            pay_rounds[labels[i]] += _nr
            pay_sum[labels[j]] += float(np.sum(h["opp_payoff"][_e0:]))
            pay_rounds[labels[j]] += _nr
    else:
        # ---- Parallel path (independent dyads) ----
        specs = [{"agent": cfg_for(i, d), "opponent": cfg_for(j, d),
                  "regime": regime,
                  "env_err_agent": env_error, "env_err_opponent": env_error,
                  "noise_seed": seed * 31 + d}
                 for d, (i, j) in enumerate(pairs)]
        res = run_many(specs, n_rounds=n_rounds, n_jobs=n_jobs, verbose=False,
                       desc="round-robin validation")
        for d, (i, j) in enumerate(pairs):
            h = res[d]["hist"]
            _e0 = max(0, min(int(eval_from), n_rounds - 1))
            _nr = n_rounds - _e0
            total_cc += int(np.sum(h["state"][_e0:] == CC))
            total_rounds += _nr
            pay_sum[labels[i]] += float(np.sum(h["my_payoff"][_e0:]))
            pay_rounds[labels[i]] += _nr
            pay_sum[labels[j]] += float(np.sum(h["opp_payoff"][_e0:]))
            pay_rounds[labels[j]] += _nr

    by_type = {nm: (pay_sum[nm] / pay_rounds[nm]) if pay_rounds[nm] else np.nan
               for nm in names}
    all_pay = sum(pay_sum.values()) / max(sum(pay_rounds.values()), 1)
    return {"cc_rate": total_cc / max(total_rounds, 1),
            "mean_payoff": all_pay, "by_type_payoff": by_type,
            "n_agents": n, "n_pairs": len(pairs)}
