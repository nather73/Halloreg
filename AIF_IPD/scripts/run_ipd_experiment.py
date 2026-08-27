#!/usr/bin/env python
"""
run_ipd_experiment.py
=====================

**HalloReg main entry point** — IPD simulation -> hypothesis testing
-> figures.

Paper: *Adaptive Prosociality Through Hierarchical Allostatic
Regulation in Social Dynamics: A Simulation Study*
(Choi, Albarracin, Pae, & Kim)

Hypotheses under test
---------------------
  ARCH  architecture implementation consistency (must pass before any
        hypothesis test)
  H1    does OpponentInversion robustly infer strategic intent
  H1A   does it track changing intent and recover lambda
  H2    does it protect its own payoff against exploiters
  H2A   can it tell an exploiter from a noisy TFT
  H2B   what does regulated lambda add over lambda fixed at 0 or 1
        (onset of cooperation / withdrawal from an exploiter)
  H3    in a stationary mixed population, does it out-earn the fixed
        strategies
  H3A   under the same conditions, does it contribute more to
        population cooperation
  H4    the same as H3 under non-stationary payoffs
  H4A   the same as H3A under non-stationary payoffs
  H5    does it survive across generations under RE/ORE dynamics
  H6    are parameter recovery and self-projection robust

Usage
-----
    python scripts/run_ipd_experiment.py            # everything
    python scripts/run_ipd_experiment.py --quick    # smoke (minutes)
    python scripts/run_ipd_experiment.py --experiments H1 H2
    python scripts/run_ipd_experiment.py --jobs 16  # physical cores

Run with no arguments it executes the full battery at the paper's
specification, so no opt-out flags need to be remembered.
"""

from __future__ import annotations

# --- Limit BLAS threads: must be set **before** numpy is imported ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

# Add the package root to the import path so the script runs from
# any working directory
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]              # .../Halloreg/AIF_IPD
sys.path.insert(0, str(_PKG_ROOT.parent))  # .../Halloreg

import numpy as np

from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.experiments import arch_validation, h1_intent, h1a_tracking
from AIF_IPD.experiments import h2_protection, h2b_counterfactual
from AIF_IPD.experiments import h3_h4_population
from AIF_IPD.experiments import h5_evolution, h6_recovery
from AIF_IPD.experiments.common import Config, Registry, save_json
from AIF_IPD.ipd.sim import resolve_jobs

LOGGER = get_logger("HalloReg.run")

#: Runnable experiment names (order = default execution order)
ALL_EXPERIMENTS = ["ARCH", "H1", "H1A", "H2", "H2B", "H3", "H4", "H5", "H6"]

#: User-facing aliases (H3A/H4A are tested inside the same
#: experiments as H3/H4)
ALIASES = {"H2A": "H2", "H3A": "H3", "H4A": "H4"}


# ==================================================================== CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HalloReg IPD experiments — simulation through "
                    "figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=None,
                   help=f"experiments to run (default: all). Choices: "
                        f"{' '.join(ALL_EXPERIMENTS)} "
                        f"(+ aliases H2A/H3A/H4A)")
    p.add_argument("--seeds", type=int, default=120,
                   help="number of seeds (replicates) per condition")
    p.add_argument("--rounds", type=int, default=120,
                   help="rounds per dyad")
    p.add_argument("--tom-es-mode", choices=["mirror", "analytic"],
                   default="mirror",
                   help="es of the ToM likelihood: mirror (default, "
                        "role-swapped Z alignment) | analytic (legacy "
                        "closed form)")
    p.add_argument("--eval-from", type=int, default=0,
                   help="first round of the evaluation window (0 = all "
                        "rounds; e.g. --rounds 800 --eval-from 600 tests "
                        "hypotheses on rounds 601-800)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="parallel workers; -1 = (logical cores - 1). "
                        "The work is CPU-bound and single-threaded, so "
                        "setting the physical core count is recommended")
    p.add_argument("--particles", type=int, default=400,
                   help="number of particles in the filter")
    p.add_argument("--payoff-access", choices=["oracle", "naive"],
                   default="naive",
                   help="payoff-matrix access (default naive). naive = "
                        "not observed, learned by QRTD; oracle = observed "
                        "directly (ceiling reference / ablation)")
    p.add_argument("--tau-risk", type=float, default=0.3,
                   help="lower-tail fraction of risk-sensitive "
                        "evaluation (1.0 = risk neutral)")
    p.add_argument("--policy-particles", type=int, default=64,
                   help="number of trait-space policy particles K")
    p.add_argument("--prop-sd", type=float, default=0.40,
                   help="trait proposal diffusion width sigma_prop")
    p.add_argument("--policy-gamma", type=float, default=8.0,
                   help="policy precision gamma (q(pi) ~ exp(-gamma*G))")
    p.add_argument("--w-epi-j", type=float, default=10.0,
                   help="weight of the epistemic term on the partner's "
                        "theta (0 = ablated)")
    p.add_argument("--w-epi-r", type=float, default=1.0,
                   help="weight of the epistemic term on the reward "
                        "model (0 = ablated)")
    p.add_argument("--w-cplx", type=float, default=0.15,
                   help="weight of the complexity term of the trait EFE "
                        "(KL from the prior)")
    p.add_argument("--horizon", type=int, default=6,
                   help="(retired in v1.6.0) meaningless since the "
                        "rollout was removed; affects only the terminal Z "
                        "discount")
    p.add_argument("--w-cd", type=float, default=0.5, dest="w_cd",
                   help="affective vs contextual channel weight w_cd of "
                        "Empathy")
    p.add_argument("--lam-gain", type=float, default=0.05, dest="lam_gain",
                   help="lambda integration gain eta")
    p.add_argument("--env-error", type=float, default=0.05, dest="env_error",
                   help="environment-layer execution error rate "
                        "(applied symmetrically to all types)")
    p.add_argument("--max-compositions", type=int, default=0,
                   help="cap on sampled population compositions "
                        "(0 = full enumeration)")
    p.add_argument("--results", type=str, default=None,
                   help="output directory (default: AIF_IPD/results)")
    p.add_argument("--quick", action="store_true",
                   help="smoke mode — greatly reduced seeds, rounds "
                        "and compositions for a fast check")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """CLI arguments -> Config; --quick shrinks every scale."""
    results = Path(args.results) if args.results else (_PKG_ROOT / "results")
    results.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        seeds=args.seeds, rounds=args.rounds, jobs=args.jobs,
        eval_from=args.eval_from,
        tom_es_mode=args.tom_es_mode,
        results=results, env_error=args.env_error,
        n_particles=args.particles, horizon=args.horizon, w_cplx=args.w_cplx, w_epi_j=args.w_epi_j, w_epi_r=args.w_epi_r, policy_gamma=args.policy_gamma, prop_sd=args.prop_sd, policy_particles=args.policy_particles,
        payoff_access=args.payoff_access,
        w_cd=args.w_cd, lam_gain=args.lam_gain, quick=args.quick,
        max_compositions=args.max_compositions)

    if args.quick:
        cfg.seeds = min(cfg.seeds, 8)
        cfg.rounds = min(cfg.rounds, 40)
        cfg.n_particles = min(cfg.n_particles, 200)
        cfg.max_compositions = 2000
        LOGGER.info("smoke mode — seeds=%d, rounds=%d, particles=%d, "
                    "compositions<=%d",
                    cfg.seeds, cfg.rounds, cfg.n_particles,
                    cfg.max_compositions)
    return cfg


def resolve_experiments(names) -> list:
    """Resolve aliases and return the experiment list in default
    order."""
    if not names:
        return list(ALL_EXPERIMENTS)
    wanted = set()
    for n in names:
        key = n.upper()
        key = ALIASES.get(key, key)
        if key not in ALL_EXPERIMENTS:
            raise SystemExit(f"unknown experiment: {n} "
                             f"(available: "
                             f"{', '.join(ALL_EXPERIMENTS)})")
        wanted.add(key)
    # ARCH always runs first when other experiments are present
    # (implementation consistency is a precondition)
    return [e for e in ALL_EXPERIMENTS if e in wanted]


# =============================================================== Reporting
def print_summary(final: dict, arch: dict, elapsed: float) -> None:
    """Final console summary."""
    line = "=" * 78
    print("\n" + line)
    print("HalloReg experiment summary")
    print(line)

    if arch is not None:
        print(f"\n[architecture validation]  "
              f"{arch['n_pass']}/{arch['n_total']} passed")
        for c in arch["checks"]:
            print(f"  {'✔' if c['passed'] else '✘'} [{c['id']}] "
                  f"{c['name']}\n      {c['detail']}")

    print("\n[confirmatory tests — Holm corrected, alpha=%.2f]"
          % final["alpha"])
    for r in final["primary"]:
        mark = "supported" if r["supported"] else "not supported"
        print(f"  [{r['hypothesis']:<4}] {mark:<13} "
              f"| p_holm={r['p_holm']:.4g} "
              f"| {r['label']}")
        if r["effect"]:
            print(f"             {r['effect']}")

    print("\n[final verdict per hypothesis]")
    for h in sorted(final["by_hypothesis"]):
        d = final["by_hypothesis"][h]
        print(f"  {h:<5} {d['verdict']:<14} "
              f"({d['n_ok']}/{d['n']} confirmatory tests passed)")

    n_sig = sum(1 for r in final["exploratory"] if r.get("significant"))
    print(f"\n[exploratory tests — BH-FDR] "
          f"{n_sig}/{len(final['exploratory'])} significant")
    print(f"\ntotal runtime: {elapsed / 60:.1f} min")
    print(line + "\n")


# ==================================================================== main
def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    experiments = resolve_experiments(args.experiments)

    n_jobs = resolve_jobs(cfg.jobs)
    LOGGER.info("=" * 70)
    LOGGER.info("HalloReg IPD experiments starting")
    LOGGER.info("  experiments : %s", ", ".join(experiments))
    LOGGER.info("  seeds       : %d | rounds: %d | particles: %d | "
                "horizon: %d",
                cfg.seeds, cfg.rounds, cfg.n_particles, cfg.horizon)
    LOGGER.info("  parallel    : %d workers (%d logical cores)",
                n_jobs, os.cpu_count() or 1)
    LOGGER.info("  lambda      : w_cd=%.2f, eta=%.3f", cfg.w_cd,
                cfg.lam_gain)
    LOGGER.info("  results     : %s", cfg.results)
    LOGGER.info("=" * 70)

    t0 = time.time()
    reg = Registry()
    arch = None
    h34 = None
    comps = None

    for name in experiments:
        t1 = time.time()
        LOGGER.info("")
        LOGGER.info("─" * 70)
        LOGGER.info("> %s starting", name)
        LOGGER.info("─" * 70)

        if name == "ARCH":
            arch = arch_validation.run(cfg)
            if arch["n_pass"] < arch["n_total"]:
                LOGGER.warning("architecture validation has failing "
                               "items — interpret the hypothesis results "
                               "below with caution.")
        elif name == "H1":
            h1_intent.run(cfg, reg)
        elif name == "H1A":
            h1a_tracking.run(cfg, reg)
        elif name == "H2":
            h2_protection.run(cfg, reg)
        elif name == "H2B":
            h2b_counterfactual.run(cfg, reg)
        elif name in ("H3", "H4"):
            # H3 and H4 share the composition space and the type-pair
            # matrices, so they run together.
            if h34 is None:
                comps = h3_h4_population._prepare_comps(cfg)
                h34 = h3_h4_population.run(cfg, reg)
        elif name == "H5":
            if h34 is None:
                # H5 needs H4's payoff matrices; compute them first if
                # they are missing.
                LOGGER.info("H5 requires H4's payoff matrices — running "
                            "H3/H4 first")
                comps = h3_h4_population._prepare_comps(cfg)
                h34 = h3_h4_population.run(cfg, reg)
            h5_evolution.run(cfg, reg, h34, comps)
        elif name == "H6":
            h6_recovery.run(cfg, reg)

        LOGGER.info("< %s done (%.1f min)", name, (time.time() - t1) / 60)

    final = reg.finalize()
    elapsed = time.time() - t0

    save_json({"config": {"seeds": cfg.seeds, "rounds": cfg.rounds,
                          "particles": cfg.n_particles, "horizon": cfg.horizon,
                          "w_cd": cfg.w_cd, "lam_gain": cfg.lam_gain,
                          "env_error": cfg.env_error,
                          "experiments": experiments,
                          "elapsed_min": elapsed / 60},
               "architecture": ({"n_pass": arch["n_pass"],
                                 "n_total": arch["n_total"],
                                 "checks": arch["checks"]} if arch else None),
               "verdicts": final},
              cfg.results / "SUMMARY.json")

    print_summary(final, arch, elapsed)
    LOGGER.info("results written: %s", cfg.results)
    LOGGER.info("figures written: %s", cfg.figdir)
    return 0


if __name__ == "__main__":
    # Windows/spawn compatibility: call main() only inside the
    # __main__ guard so child processes re-importing this module do not
    # re-run it.
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
