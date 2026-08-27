#!/usr/bin/env python
"""
custom_sim_example.py — template for your own HalloReg simulations
==================================================================

Runs HalloRegAgent against a chosen partner over an IPD payoff matrix
(R, T, S, P) that you specify, then writes a summary JSON and a
trajectory figure.

The partner can be either of two kinds:

  * a **fixed strategy** — TFT, GTFT, WSLS, ALLC, ALLD, exploiter,
    noisy_tft, random (``--partner tft``). Use these to ask what
    HalloReg does against a known, non-learning environment.
  * the **EmpathicAgent** of Albarracin et al. (2026), whose lambda is
    fixed (``--partner empathic --lam-fixed 0.4``). Use this to compare
    endogenous regulation against exogenous lambda in a like-for-like
    learning agent.

Run:
    python -B AIF_IPD/scripts/custom_sim_example.py
    python -B AIF_IPD/scripts/custom_sim_example.py --partner tft
    python -B AIF_IPD/scripts/custom_sim_example.py --partner alld --seeds 20
    python -B AIF_IPD/scripts/custom_sim_example.py --R 3 --T 5 --S 0 --P 1
    python -B AIF_IPD/scripts/custom_sim_example.py --partner halloreg \\
        --rounds 800 --eval-from 600 --particles 600 --out hr_vs_hr

Verified against v3.9.8 (commit 8a2edeb).
"""

from __future__ import annotations

# --- Limit BLAS threads before numpy is imported (single-threaded,
# --- CPU-bound work parallelises better across processes than threads).
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the package importable no matter where the script is run from.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AIF_IPD.core.constants import (           # noqa: E402
    CC, COOP, current_payoffs, reset_payoffs, set_payoff_matrix,
)
from AIF_IPD.ipd.agent import EmpathicAgent, HalloRegAgent  # noqa: E402
from AIF_IPD.ipd.env import make_opponent      # noqa: E402
from AIF_IPD.ipd.payoff_schedule import CI_REGIMES  # noqa: E402
from AIF_IPD.ipd.sim import run_dyad           # noqa: E402

#: Partner kinds that come from env.make_opponent (no learning).
FIXED_STRATEGIES = ("tft", "gtft", "wsls", "allc", "alld",
                    "exploiter", "noisy_tft", "random")
#: Partner kinds that are themselves inference agents.
LEARNING_PARTNERS = ("empathic", "halloreg")


# ======================================================= Partner factory
def build_partner(kind: str, seed: int, lam_fixed: float,
                  n_particles: int, opp_error: float):
    """
    Construct the partner.

    `opp_error` is the strategy's **own** execution error (e.g. a noisy
    TFT that sometimes mis-executes its own rule). It is distinct from
    `--env-error`, which run_dyad applies symmetrically to both sides
    at the environment level. Only fixed strategies take it.
    """
    if kind in FIXED_STRATEGIES:
        kw = {"error": opp_error} if opp_error > 0.0 else {}
        return make_opponent(kind, seed=seed, **kw)
    if kind == "empathic":
        return EmpathicAgent(lam=lam_fixed, seed=seed,
                             n_particles=n_particles, name="Empathic")
    if kind == "halloreg":
        return HalloRegAgent(seed=seed, n_particles=n_particles,
                             name="HalloReg-2")
    raise ValueError(f"unknown partner kind: {kind}")


# ============================================================== One dyad
def run_one(R: float, T: float, S: float, P: float,
            partner_kind: str, n_rounds: int, seed: int,
            lam_fixed: float, n_particles: int, env_error: float,
            opp_error: float, regime: str | None) -> dict:
    """
    Run one dyad and return the interaction history plus each agent's
    internal log.

    ORDER MATTERS. `set_payoff_matrix` must be called **before** the
    agents are constructed: they read the payoff support at
    construction time to set the Huber scale of QRTD / RewardModel, to
    seed the social history, and to place the flat Z-tilde
    initialisation at the support midpoint (2.5 for the default
    matrix). Constructing first and changing payoffs afterwards leaves
    those constants stale.
    """
    # 1. Install the payoff matrix (module-global, mutated in place).
    set_payoff_matrix(R, T, S, P)

    # 2. Build the agents. Every argument not named here stays at its
    #    v3.9.8 default; pass explicit values only for what you are
    #    actually manipulating. Useful knobs on HalloRegAgent:
    #      lam_fixed=x     hold lambda constant (regulation off)
    #      regulate=False  same, but keeps the Empathy module's lambda
    #      value_policy    "state" (default) | "last" (pre-v3.9.6)
    #      w_u, w_ig_r, w_ig_j, beta_g   social-EFE weights
    focal = HalloRegAgent(seed=seed, n_particles=n_particles,
                          name="HalloReg")
    partner = build_partner(partner_kind, seed + 500, lam_fixed,
                            n_particles, opp_error)

    # 3. Run the dyad.
    #    ci_schedule=None keeps the payoffs fixed for all rounds, so
    #    the matrix from step 1 stays in force. A schedule (--regime)
    #    overwrites the payoffs every round and restores the default
    #    at the end, which overrides --R/--T/--S/--P.
    #    env_err_* are environment-level execution errors applied
    #    symmetrically; noise_seed fixes the flip sequence so two
    #    conditions sharing it see identical noise (common random
    #    numbers).
    hist = run_dyad(focal, partner, n_rounds=n_rounds,
                    ci_schedule=(CI_REGIMES[regime] if regime else None),
                    env_err_a=env_error, env_err_b=env_error,
                    noise_seed=seed,
                    partner_id_a=1, partner_id_b=2)

    return {
        "hist": hist,
        "focal_log": focal.log,
        # Fixed strategies have no internal log; only learners do.
        "partner_log": getattr(partner, "log", None),
    }


# ============================================================== Analysis
def summarise(runs: list, eval_from: int) -> dict:
    """
    Reduce a list of runs to the statistics usually reported, all
    restricted to the evaluation window.
    """
    out = {}

    def _stat(vals):
        a = np.asarray(vals, dtype=float)
        return {"mean": float(np.nanmean(a)),
                "sd": float(np.nanstd(a, ddof=1)) if a.size > 1 else 0.0,
                "n": int(a.size)}

    w = slice(eval_from, None)
    out["focal_coop"] = _stat([np.mean(r["hist"]["my_act"][w] == COOP)
                               for r in runs])
    out["partner_coop"] = _stat([np.mean(r["hist"]["opp_act"][w] == COOP)
                                 for r in runs])
    out["focal_payoff"] = _stat([np.mean(r["hist"]["my_payoff"][w])
                                 for r in runs])
    out["partner_payoff"] = _stat([np.mean(r["hist"]["opp_payoff"][w])
                                   for r in runs])
    out["mutual_coop"] = _stat([np.mean(r["hist"]["state"][w] == CC)
                                for r in runs])
    # The endogenously regulated weight and its two components
    # (lambda = clip(lambda* + clip(phi, 0, 1) - 0.5, 0, 1)).
    out["focal_lambda"] = _stat([np.mean(r["focal_log"]["lam"][w])
                                 for r in runs])
    out["lambda_star"] = _stat([np.nanmean(
        np.asarray(r["focal_log"]["lam_l0"], dtype=float)[w]) for r in runs])
    out["phi"] = _stat([np.nanmean(
        np.asarray(r["focal_log"]["allo_phi"], dtype=float)[w])
        for r in runs])
    # Z-tilde advantages behind lambda* (v3.9.5+). A >= 0 means
    # cooperation already pays for the focal agent at that state, so
    # lambda* clips to 0; B < 0 means cooperating helps the partner
    # less than it helps the focal agent, and raising lambda then
    # lowers the cooperation probability.
    out["anchor_A"] = _stat([np.nanmean(
        np.asarray(r["focal_log"]["anchor_A"], dtype=float)[w])
        for r in runs])
    out["anchor_B"] = _stat([np.nanmean(
        np.asarray(r["focal_log"]["anchor_B"], dtype=float)[w])
        for r in runs])
    # The focal's inferred empathy of the partner. Against a
    # fixed-lambda partner this doubles as a recovery check.
    out["inferred_lambda_j"] = _stat([
        np.mean(r["focal_log"]["E_lambda_j"][w]) for r in runs])
    # Shares of the social-EFE logit carried by each term (v3.9.6+).
    # The epistemic terms should shrink as the partner becomes known.
    gp = np.array([np.nanmean(np.abs(
        np.asarray(r["focal_log"]["g_prag"], dtype=float)[w])) for r in runs])
    gr = np.array([np.nanmean(np.abs(
        np.asarray(r["focal_log"]["g_epi_r"], dtype=float)[w])) for r in runs])
    gj = np.array([np.nanmean(np.abs(
        np.asarray(r["focal_log"]["g_epi_j"], dtype=float)[w])) for r in runs])
    tot = gp + gr + gj + 1e-12
    out["share_pragmatic"] = _stat(gp / tot)
    out["share_epistemic_R"] = _stat(gr / tot)
    out["share_epistemic_theta"] = _stat(gj / tot)
    # State occupancy over the evaluation window.
    occ = np.zeros(4)
    for r in runs:
        st = np.asarray(r["hist"]["state"][w])
        for s in range(4):
            occ[s] += float(np.mean(st == s))
    out["state_occupancy"] = dict(
        zip(("CC", "CD", "DC", "DD"), (occ / max(len(runs), 1)).tolist()))
    return out


# ================================================================ Figure
def plot_runs(runs: list, out_dir: Path, title: str) -> None:
    """Three panels: lambda and its components, cooperation, payoff."""
    import matplotlib
    matplotlib.use("Agg")            # no display needed
    import matplotlib.pyplot as plt

    def _stack(key, source="focal_log"):
        return np.stack([np.asarray(r[source][key], dtype=float)
                         for r in runs])

    def _band(ax, arr, color, label, ls="-"):
        # lambda* is NaN before the regulator has enough observations
        # (the first few rounds), so an all-NaN column is expected;
        # suppress the warning rather than dropping the series.
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(arr, axis=0)
                lo = np.nanpercentile(arr, 25, axis=0)
                hi = np.nanpercentile(arr, 75, axis=0)
        ax.plot(m, lw=1.6, color=color, ls=ls, label=label)
        ax.fill_between(np.arange(arr.shape[1]), lo, hi,
                        color=color, alpha=0.18, lw=0)

    lam = _stack("lam")
    l0 = np.clip(_stack("lam_l0"), 0.0, 1.0)
    ljh = _stack("E_lambda_j")
    coop = np.stack([np.cumsum(r["hist"]["my_act"] == COOP)
                     / np.arange(1, len(r["hist"]["my_act"]) + 1)
                     for r in runs])
    pay = np.stack([np.cumsum(r["hist"]["my_payoff"])
                    / np.arange(1, len(r["hist"]["my_payoff"]) + 1)
                    for r in runs])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))

    ax = axes[0]
    _band(ax, lam, "#DA8BC3", r"$\lambda_t$ (regulated)")
    _band(ax, l0, "#937860", r"$\lambda^*$ (switch point)", ls="--")
    _band(ax, ljh, "#CCB974", r"$\hat{\lambda}_j$ (partner)", ls=":")
    ax.set(xlabel="Round", ylabel="Empathy weight", ylim=(-0.03, 1.03))
    ax.legend(fontsize=8)
    ax.set_title("(a) lambda and its components", fontsize=10)

    ax = axes[1]
    _band(ax, coop, "#4C72B0", "HalloReg")
    ax.set(xlabel="Round", ylabel="Cumulative cooperation rate",
           ylim=(-0.03, 1.03))
    ax.set_title("(b) cooperation", fontsize=10)

    ax = axes[2]
    _band(ax, pay, "#55A868", "HalloReg")
    ax.set(xlabel="Round", ylabel="Cumulative mean payoff / round")
    ax.set_title("(c) payoff", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "custom_sim.png", dpi=200)
    plt.close(fig)


# =========================================================== Entry point
def main() -> int:
    p = argparse.ArgumentParser(
        description="Custom HalloReg dyad against a fixed strategy or a "
                    "learning partner.")
    p.add_argument("--partner", default="empathic",
                   choices=list(FIXED_STRATEGIES) + list(LEARNING_PARTNERS),
                   help="opponent: a fixed strategy, the fixed-lambda "
                        "EmpathicAgent, or a second HalloReg")
    p.add_argument("--R", type=float, default=3.0, help="mutual cooperation")
    p.add_argument("--T", type=float, default=5.0, help="temptation")
    p.add_argument("--S", type=float, default=0.0, help="sucker")
    p.add_argument("--P", type=float, default=1.0, help="mutual defection")
    p.add_argument("--regime", default=None, choices=sorted(CI_REGIMES),
                   help="non-stationary payoff schedule; overrides "
                        "--R/--T/--S/--P")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seeds", type=int, default=10,
                   help="independent replicate dyads")
    p.add_argument("--eval-from", type=int, default=100, dest="ev",
                   help="first round of the evaluation window")
    p.add_argument("--lam-fixed", type=float, default=0.4,
                   help="fixed lambda of the EmpathicAgent partner")
    p.add_argument("--particles", type=int, default=400)
    p.add_argument("--env-error", type=float, default=0.05,
                   help="environment-level execution error, both sides")
    p.add_argument("--opp-error", type=float, default=0.0,
                   help="fixed strategy's own execution error "
                        "(e.g. 0.2 for a noisy TFT)")
    p.add_argument("--out", default="custom_results")
    args = p.parse_args()

    if args.ev >= args.rounds:
        print(f"--eval-from {args.ev} >= --rounds {args.rounds}; "
              f"using {args.rounds // 2}")
        args.ev = args.rounds // 2
    out_dir = Path(args.out)

    # A non-standard matrix is allowed on purpose (deadlock, harmony
    # and other non-PD structures are legitimate manipulations), but
    # say so, because the PD interpretation no longer holds.
    if args.regime:
        print(f"payoff regime: {args.regime} "
              f"(schedule overrides --R/--T/--S/--P)")
    else:
        is_pd = (args.T > args.R > args.P > args.S)
        print(f"payoffs (R, T, S, P) = "
              f"({args.R}, {args.T}, {args.S}, {args.P})"
              f"{'' if is_pd else '   [NOT a standard PD: T > R > P > S fails]'}")
        if is_pd and 2 * args.R <= args.T + args.S:
            print("   [warning: 2R <= T + S — alternating exploitation "
                  "beats mutual cooperation]")
    print(f"partner: {args.partner}"
          + (f" (lambda = {args.lam_fixed})" if args.partner == "empathic"
             else "")
          + (f", own error {args.opp_error}" if args.opp_error > 0 else ""))

    try:
        runs = []
        for sd in range(args.seeds):
            runs.append(run_one(args.R, args.T, args.S, args.P,
                                partner_kind=args.partner,
                                n_rounds=args.rounds, seed=sd,
                                lam_fixed=args.lam_fixed,
                                n_particles=args.particles,
                                env_error=args.env_error,
                                opp_error=args.opp_error,
                                regime=args.regime))
            print(f"  seed {sd + 1}/{args.seeds} done")
        # Confirm the payoffs actually in force during the runs.
        print(f"payoffs in force at end of runs: {current_payoffs()}")
        summary = summarise(runs, args.ev)
    finally:
        # ALWAYS restore the default PD. The payoff arrays are module
        # globals mutated in place, so leaving them modified would
        # silently contaminate anything else run in this process.
        reset_payoffs()

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"partner": args.partner, "regime": args.regime,
            "R": args.R, "T": args.T, "S": args.S, "P": args.P,
            "rounds": args.rounds, "seeds": args.seeds,
            "eval_from": args.ev, "lam_fixed": args.lam_fixed,
            "particles": args.particles, "env_error": args.env_error,
            "opp_error": args.opp_error}
    with open(out_dir / "custom_sim.json", "w", encoding="utf8") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)

    print(f"\nevaluation window: rounds {args.ev + 1}-{args.rounds}, "
          f"{args.seeds} seeds")
    for k, v in summary.items():
        if k == "state_occupancy":
            print("  state_occupancy    " + "  ".join(
                f"{s}={x:.3f}" for s, x in v.items()))
        else:
            print(f"  {k:<20} {v['mean']:+.4f} +/- {v['sd']:.4f}")

    title = (f"HalloReg vs {args.partner}"
             + (f"(lambda={args.lam_fixed})" if args.partner == "empathic"
                else "")
             + (f"  regime={args.regime}" if args.regime else
                f"  (R,T,S,P)=({args.R},{args.T},{args.S},{args.P})"))
    plot_runs(runs, out_dir, title)
    print(f"\nwritten: {out_dir / 'custom_sim.json'}, "
          f"{out_dir / 'custom_sim.png'}")
    return 0


if __name__ == "__main__":
    # Windows/spawn safety: if you later parallelise with a Pool, the
    # guard prevents child processes from re-running main() on import.
    raise SystemExit(main())
