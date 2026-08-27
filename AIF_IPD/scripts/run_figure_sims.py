#!/usr/bin/env python3
"""
run_figure_sims — simulations for the poster figures (v3.9.2)
====================================================================
Produces data only, written to <out>/figdata/*.json; rendering is
handled by make_figures.py. Parallelism uses the same machinery as
run_ipd_experiment.py — `ipd.sim.run_many` (spec dicts + spawn Pool
+ imap_unordered) — via --jobs (-1 = cores - 1).

    python3 -B AIF_IPD/scripts/run_figure_sims.py --out results_figs \
        --seeds 120 --mat-seeds 12 --rounds 800 --eval-from 600 \
        --particles 600 --jobs 16

The discrimination block of fig1 matches the H2A verdict pipeline
**down to the seed convention** (agent 10_000 + 37*sd / opponent
11_000 + 37*sd + ci / noise 710_000 + 103*sd, env 0/0): run with the
same (seeds, rounds, particles) it reproduces the lam_traces of
H2.json bit-identically (a v3.9.2 verification item).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from AIF_IPD.ipd.agent import HalloRegAgent                      # noqa: E402
from AIF_IPD.ipd.sim import run_dyad, run_many, resolve_jobs     # noqa: E402
from AIF_IPD.ipd.env import (make_opponent, make_switching_opponent,
                             switch_rounds)                      # noqa: E402
from AIF_IPD.ipd.payoff_schedule import (get_regime,
                                         NONSTATIONARY_REGIMES)  # noqa: E402

FIXED = ("tft", "gtft", "wsls", "allc", "alld")
AGENTS6 = ("halloreg",) + FIXED
SCENARIOS = ("recip_expl_recon", "coop_trap", "wsls_flip")


# ------------------------------------------------------------ Spec helpers
def _aspec(kind: str, seed: int, particles: int) -> dict:
    if kind == "halloreg":
        return {"type": "halloreg", "seed": seed, "n_particles": particles}
    return {"type": "strategy", "kind": kind, "seed": seed}


def _pair_specs(i: str, j: str, seeds: int, particles: int,
                regime: str | None):
    """Per-seed specs for one dyad pair (same seed convention as the
    earlier sequential version)."""
    out = []
    for sd in range(seeds):
        out.append({"agent": _aspec(i, sd, particles),
                    "opponent": _aspec(j, 500 + sd, particles),
                    "env_err_agent": 0.05, "env_err_opponent": 0.05,
                    "noise_seed": sd,
                    **({"regime": regime} if regime else {})})
    return out


def _pair_stats(res_list, ev):
    mi = [float(r["hist"]["my_payoff"][ev:].mean()) for r in res_list]
    mj = [float(r["hist"]["opp_payoff"][ev:].mean()) for r in res_list]
    cc = [float(np.mean(r["hist"]["state"][ev:] == 0)) for r in res_list]
    return float(np.mean(mi)), float(np.mean(mj)), float(np.mean(cc))


def _matrix6(args, regime=None):
    """6x6 expected-payoff matrix — all pair specs parallelised in a
    single run_many call."""
    n = len(AGENTS6)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    specs, index = [], {}
    for (i, j) in pairs:
        index[(i, j)] = len(specs)
        specs.extend(_pair_specs(AGENTS6[i], AGENTS6[j], args.mat_seeds,
                                 args.particles, regime))
    res = run_many(specs, n_rounds=args.rounds, n_jobs=args.jobs,
                   desc=f"6x6 matrix ({regime or 'stationary'})")
    M = np.full((n, n), np.nan)
    for (i, j) in pairs:
        k = index[(i, j)]
        mi, mj, _ = _pair_stats(res[k:k + args.mat_seeds], args.ev)
        M[i, j] = mi
        M[j, i] = mj
    return M


# --------------------------------------------------------- Per-figure jobs
def fig1(args, out):
    """Lambda divergence by condition plus behavioural discrimination;
    the discrimination part follows the H2A seed convention."""
    opps = ("allc", "wsls", "gtft", "alld", "tft")
    specs = []
    for k in opps:
        for sd in range(args.seeds):
            specs.append({"agent": _aspec("halloreg", sd, args.particles),
                          "opponent": {"type": "strategy", "kind": k,
                                       "seed": 90 + sd},
                          "env_err_agent": 0.05, "env_err_opponent": 0.05,
                          "noise_seed": sd})
    # H2A discrimination protocol (same convention as
    # h2_protection.run_discrimination)
    conds = [("exploiter", {"type": "strategy", "kind": "alld",
                            "error": 0.0}),
             ("noisy_tft", {"type": "strategy", "kind": "tft",
                            "error": 0.20})]
    for ci, (_, ospec) in enumerate(conds):
        for sd in range(args.seeds):
            o = dict(ospec)
            o["seed"] = 11_000 + sd * 37 + ci
            specs.append({"agent": {"type": "halloreg",
                                    "seed": 10_000 + sd * 37,
                                    "n_particles": args.particles},
                          "opponent": o,
                          "env_err_agent": 0.0, "env_err_opponent": 0.0,
                          "noise_seed": 710_000 + sd * 103})
    res = run_many(specs, n_rounds=args.rounds, n_jobs=args.jobs,
                   desc="fig1 dyads")
    traces = {}
    p = 0
    for k in opps:
        tr = [np.asarray(res[p + sd]["agent_log"]["lam"], float)
              for sd in range(args.seeds)]
        traces[k] = np.nanmean(np.stack(tr), 0).tolist()
        p += args.seeds
    disc = {}
    for cname, _ in conds:
        tr = [np.asarray(res[p + sd]["agent_log"]["lam"], float)
              for sd in range(args.seeds)]
        disc[cname] = np.nanmean(np.stack(tr), 0).tolist()
        p += args.seeds
    json.dump({"opp": traces, "disc": disc},
              open(os.path.join(out, "fig1.json"), "w"))


def _fig2_worker(task):
    """Top-level worker (required for spawn pickling) — an instrumented
    TraceAgent dyad."""
    sc, sd, rounds, particles = task
    a = TraceAgent(seed=sd, n_particles=particles)
    opp = make_switching_opponent(sc, n_rounds=rounds, seed=90 + sd)
    run_dyad(a, opp, rounds, env_err_a=0.05, env_err_b=0.05, noise_seed=sd)
    return (sc, sd, np.asarray(a.log["lam"], float),
            np.asarray(a.tr["lam_j"], float), np.asarray(a.tr["rho"], float))


class TraceAgent(HalloRegAgent):
    """Instrumented agent recording the per-round ToM posterior means
    (lambda_j-hat, rho-hat) for Fig 2."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tr = {"lam_j": [], "rho": []}

    def step(self, prev_state):
        outv = super().step(prev_state)
        pm = self.inversion.posterior_means()
        self.tr["lam_j"].append(float(pm["lambda_j"]))
        self.tr["rho"].append(float(pm["rho"]))
        return outv


def fig2(args, out):
    """Trait-switch tracking — the same spawn Pool pattern as run_many.
    An instrumented class cannot travel through a spec dict, so only
    the worker lives at this module's top level; the machinery is
    identical."""
    tasks = [(sc, sd, args.rounds, args.particles)
             for sc in SCENARIOS for sd in range(args.seeds)]
    n_jobs = min(resolve_jobs(args.jobs), len(tasks))
    if n_jobs <= 1:
        results = [_fig2_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            results = list(pool.imap_unordered(_fig2_worker, tasks,
                                               chunksize=1))
    data = {"switches": switch_rounds(args.rounds)}
    for sc in SCENARIOS:
        rows = sorted([r for r in results if r[0] == sc], key=lambda r: r[1])
        data[sc] = {
            "lam": np.nanmean(np.stack([r[2] for r in rows]), 0).tolist(),
            "lam_j": np.nanmean(np.stack([r[3] for r in rows]), 0).tolist(),
            "rho": np.nanmean(np.stack([r[4] for r in rows]), 0).tolist()}
    json.dump(data, open(os.path.join(out, "fig2.json"), "w"))


def fig3(args, out):
    M = _matrix6(args)
    json.dump({"agents": list(AGENTS6), "M": M.tolist()},
              open(os.path.join(out, "fig3.json"), "w"))


def fig4(args, out):
    data = {}
    for r in NONSTATIONARY_REGIMES:
        f = get_regime(r)
        data[r] = [f(t, args.rounds) for t in range(args.rounds)]
    json.dump(data, open(os.path.join(out, "fig4.json"), "w"))


def fig5(args, out):
    Ms = {r: _matrix6(args, regime=r).tolist()
          for r in NONSTATIONARY_REGIMES}
    json.dump({"agents": list(AGENTS6), "regimes": Ms},
              open(os.path.join(out, "fig5.json"), "w"))


def fig6(args, out):
    """Change in population mutual cooperation when one fixed-strategy
    individual is replaced by HalloReg (parallel)."""
    regimes = ["stationary"] + list(NONSTATIONARY_REGIMES)
    specs, index = [], {}
    for ri, reg in enumerate(regimes):
        rg = None if reg == "stationary" else reg
        for i in range(len(FIXED)):
            for j in range(i + 1, len(FIXED)):
                index[(ri, "b", i, j)] = len(specs)
                specs.extend(_pair_specs(FIXED[i], FIXED[j], args.mat_seeds,
                                         args.particles, rg))
        for j in range(len(FIXED)):
            index[(ri, "h", j)] = len(specs)
            specs.extend(_pair_specs("halloreg", FIXED[j], args.mat_seeds,
                                     args.particles, rg))
    res = run_many(specs, n_rounds=args.rounds, n_jobs=args.jobs,
                   desc="fig6 population dyads")

    def _cc(key):
        k = index[key]
        return _pair_stats(res[k:k + args.mat_seeds], args.ev)[2]

    mat = np.full((len(regimes), len(FIXED)), np.nan)
    for ri in range(len(regimes)):
        base = {(i, j): _cc((ri, "b", i, j))
                for i in range(len(FIXED))
                for j in range(i + 1, len(FIXED))}
        hr = {j: _cc((ri, "h", j)) for j in range(len(FIXED))}
        base_group = float(np.mean(list(base.values())))
        for xi in range(len(FIXED)):
            vals = [cc for (i, j), cc in base.items() if xi not in (i, j)]
            vals += [hr[j] for j in range(len(FIXED)) if j != xi]
            mat[ri, xi] = (100.0 * (float(np.mean(vals)) - base_group)
                           / max(base_group, 1e-9))
    json.dump({"regimes": regimes, "replaced": list(FIXED),
               "delta_pct": mat.tolist()},
              open(os.path.join(out, "fig6.json"), "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_figs")
    p.add_argument("--figs", default="1,2,3,4,5,6")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--mat-seeds", type=int, default=3, dest="mat_seeds")
    p.add_argument("--rounds", type=int, default=800)
    p.add_argument("--eval-from", type=int, default=600, dest="ev")
    p.add_argument("--particles", type=int, default=130)
    p.add_argument("--jobs", type=int, default=1,
                   help="number of parallel workers (same meaning as in "
                        "run_ipd_experiment; -1 = cores - 1)")
    args = p.parse_args()
    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.mat_seeds = 1
        args.rounds = min(args.rounds, 240)
        args.ev = args.rounds * 2 // 3
        args.particles = min(args.particles, 100)
    if args.ev >= args.rounds:
        args.ev = args.rounds * 3 // 4
    out = os.path.join(args.out, "figdata")
    os.makedirs(out, exist_ok=True)
    for f in [s.strip() for s in args.figs.split(",") if s.strip()]:
        t0 = time.time()
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4,
         "5": fig5, "6": fig6}[f](args, out)
        print(f"[fig{f}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
