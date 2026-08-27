"""
experiments.arch_validation
===========================

**Architecture validation (ARCH) — the mechanism checks that must
pass before any hypothesis test.**

For hypothesis results to mean anything, the implemented hierarchy
must first be shown to **work as designed**. This module validates
**implementation consistency**, not hypotheses; a failure here voids
the interpretation of H1-H6, so it runs first.

Coverage
--------
V1  **Information flow** — does SelfModel supply priors and do the
    modules return their updates? After sequential encounters with
    several opponent types, memory should hold one entry per
    identity, each theta reflecting the opponent type.

V2  **Lambda directionality** — does lambda increase monotonically in
    the survival surplus rate phi?

V3  **Arousal and belief updating** — is arousal a monotone function
    of the distributional shift? By design
    arousal = 1 - exp(-W1/kappa), so the Spearman correlation must be
    essentially +1.

V4  **Exactness of the lambda mapping** — does the logged lambda_t
    equal the numerically reconstructed
    clip(lambda0 + clip(phi, 0, 1) - 0.5, 0, 1)?

V5  **Allostatic reference** — does the social reference move with
    experience while staying independent of the current partner (no
    self-chasing)?

V6  **Conditional divergence of lambda** — does lambda fall against
    exploiters and rise with cooperators? Without this the regulator
    has no directionality.

Every item is a **deterministic** check or a directional test, not a
statistical hypothesis, so nothing is registered in the
confirmatory/exploratory registry; results are reported in a separate
verdict table.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from AIF_IPD.core.constants import PAYOFF_SELF, COOP
from AIF_IPD.core.logging_utils import get_logger
from AIF_IPD.core.self_model import SelfModel
from AIF_IPD.ipd.agent import HalloRegAgent
from AIF_IPD.ipd.env import make_opponent
from AIF_IPD.ipd.sim import build_agent, run_dyad
from AIF_IPD.ipd.sim import run_dyad, run_many
from .common import Config, save_fig, save_json

LOGGER = get_logger("HalloReg.ARCH")

#: Opponent types used for validation (cooperative <-> exploitative)
PROBE_TYPES = ("allc", "gtft", "tft", "wsls", "alld")
PROBE_LABEL = {"allc": "ALLC", "gtft": "GTFT", "tft": "TFT",
               "wsls": "WSLS", "alld": "ALLD"}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (ties get the mean rank)."""
    def rank(v):
        o = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(1, len(v) + 1)
        uq, iv, ct = np.unique(v, return_inverse=True, return_counts=True)
        for u in np.where(ct > 1)[0]:
            m = iv == u
            r[m] = r[m].mean()
        return r
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def run(cfg: Config) -> dict:
    LOGGER.info("[ARCH] architecture consistency validation")
    checks: List[dict] = []
    n_probe = min(cfg.seeds, 30)

    # ===================================== V1. Information flow / memory
    # One agent encounters all probe types in sequence. run_dyad
    # announces the identity via begin_partner, so SelfModel.memory
    # must end up with one entry per type.
    agent = HalloRegAgent(seed=1, **cfg.halloreg_kwargs())
    theta_by_partner = {}
    reward_by_partner = {}
    for pid, kind in enumerate(PROBE_TYPES, start=1):
        opp = make_opponent(kind, seed=100 + pid)
        run_dyad(agent, opp, cfg.rounds, partner_id_a=pid, partner_id_b=999)
        ent = agent.self_model.memory[pid]
        theta_by_partner[kind] = dict(ent.theta)
        # v1.5.0: what is remembered is the **value distribution**
        # (a reduction of Z), not the observed-reward distribution.
        reward_by_partner[kind] = (
            float(np.median(ent.value_dist))
            if ent.value_dist is not None else float("nan"))
    # The seeded social history uses negative ids; count only the
    # experimental partners (positive ids).
    n_mem = sum(1 for k in agent.self_model.memory if k > 0)
    checks.append({
        "id": "V1",
        "name": "SelfModel accumulates memory (theta and value "
                "distribution per identity)",
        "passed": bool(n_mem == len(PROBE_TYPES)),
        "detail": f"{n_mem} memory entries / "
                  f"{len(PROBE_TYPES)} encounters",
    })
    # Is the remembered alpha-hat monotone in opponent
    # cooperativeness (highest for ALLC, lowest for ALLD)?
    alpha_order = [theta_by_partner[k]["alpha"] for k in PROBE_TYPES]
    checks.append({
        "id": "V1b",
        "name": "remembered alpha-hat reflects the cooperativeness "
                "ordering (ALLC > ALLD)",
        "passed": bool(alpha_order[0] > alpha_order[-1]),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={theta_by_partner[k]['alpha']:+.2f}"
                             for k in PROBE_TYPES),
    })

    # ================ V2-V4. Numerical consistency of affect and the
    # lambda mapping
    specs = []
    for ti, kind in enumerate(PROBE_TYPES):
        for sd in range(n_probe):
            specs.append({
                "agent": {"type": "halloreg", "seed": 200 + sd * 11 + ti,
                          **cfg.halloreg_kwargs()},
                "opponent": {"type": "strategy", "kind": kind,
                             "seed": 300 + sd * 11 + ti},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 400_000 + sd * 13 + ti})
    res = run_many(specs, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                   desc="ARCH validation dyads")

    sign_mismatch = 0
    sign_total = 0
    rho_ka: List[float] = []
    sp_rhos: List[float] = []
    allo_rhos: List[float] = []
    lam_max_dev = 0.0
    lam_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    base_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    set_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    dist_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    val_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}
    aro_traces = {k: np.zeros((n_probe, cfg.rounds)) for k in PROBE_TYPES}

    idx = 0
    for ti, kind in enumerate(PROBE_TYPES):
        for sd in range(n_probe):
            log = res[idx]["agent_log"]; idx += 1
            rpe = np.asarray(log["rpe"], float)
            val = np.asarray(log["valence"], float)
            aro = np.asarray(log["arousal"], float)
            surprise = np.asarray(log["surprise"], float)
            lam = np.asarray(log["lam"], float)
            l_aff = np.asarray(log["lambda_aff"], float)
            l_ctx = np.asarray(log["lambda_ctx"], float)

            lam_traces[kind][sd] = lam
            base_traces[kind][sd] = np.asarray(log["baseline_reward"], float)
            set_traces[kind][sd] = np.asarray(log["lambda_setpoint"], float)
            dist_traces[kind][sd] = np.asarray(log["social_distance"], float)
            val_traces[kind][sd] = val
            aro_traces[kind][sd] = aro

            # --- V2 material: sign agreement (round 0 has no
            # observation). Segments where the RPE is numerically
            # negligible (early learning, essentially tied with the
            # reference) are excluded — the check is about
            # directional consistency, not noise.
            m = np.abs(rpe[1:]) > 0.5
            sign_total += int(m.sum())
            lsp = np.asarray(log["lam_sp"], dtype=float)
            _fit = np.asarray(log["fitness"], dtype=float)
            _ok = np.isfinite(lsp) & np.isfinite(_fit)
            if int(_ok.sum()) > 20:
                _r = _spearman(_fit[_ok], lsp[_ok])
                if np.isfinite(_r):
                    sp_rhos.append(_r)
            _pa = np.asarray(log["allo_phi"], dtype=float)
            _ok2 = np.isfinite(_pa) & np.isfinite(lam)
            if int(_ok2.sum()) > 20:
                _r2 = _spearman(_pa[_ok2], lam[_ok2])
                if np.isfinite(_r2):
                    allo_rhos.append(_r2)
            mm = m & np.isfinite(lsp[1:])
            sign_mismatch += int(np.sum(
                np.sign(rpe[1:][mm]) != np.sign(lsp[1:][mm] - 0.4)))

            # --- V3 material: monotonicity of arousal in the
            # distributional shift (W1). Arousal has a floor, which
            # creates ties, so only the above-floor segment is used.
            _mk = aro[1:] > (np.min(aro[1:]) + 1e-9)
            r = (_spearman(surprise[1:][_mk], aro[1:][_mk])
                 if int(_mk.sum()) > 10 else np.nan)
            if np.isfinite(r):
                rho_ka.append(r)

            # --- V4: reconstruct the lambda mapping ---
            # The allostatic direct mapping is the only lambda path in
            # the current architecture:
            #     lambda_t = clip(lambda0 + clip(phi_t, 0, 1) - 0.5, 0, 1)
            # where lambda0 is the per-round switch point lambda*,
            # logged as lam_l0 — read directly rather than recomputed,
            # so this check tests the mapping, not a re-derivation of
            # its anchor.
            phi_a = np.asarray(log["allo_phi"], dtype=float)
            l0 = np.asarray(log.get("lam_l0", []), dtype=float)
            m_a = np.isfinite(phi_a) & np.isfinite(lam)
            if len(l0) >= len(m_a):
                m_a = m_a & np.isfinite(l0[: len(m_a)])
                if int(m_a.sum()) > 0:
                    ph = np.clip(phi_a[m_a], 0.0, 1.0)
                    l0m = l0[: len(phi_a)][m_a]
                    rec_a = np.clip(l0m + ph - 0.5, 0.0, 1.0)
                    lam_max_dev = max(
                        lam_max_dev,
                        float(np.max(np.abs(lam[m_a] - rec_a))))

    mean_rho = float(np.mean(rho_ka)) if rho_ka else np.nan
    rho_sp = float(np.mean(sp_rhos)) if sp_rhos else float("nan")
    _v2_allo = float(np.mean(allo_rhos)) if allo_rhos else None
    checks.append({
        # The allostatic mapping derives lambda from the survival
        # reference point, not from lam_sp, so what this check tests is
        # the **directionality of lambda**: does a larger resource
        # surplus phi yield a higher lambda? (lam_sp is still logged
        # for diagnostics of the retired integrator path.)
        "id": "V2",
        "name": "lambda increases monotonically in the survival "
                "surplus rate phi",
        "passed": bool(_v2_allo is not None and np.isfinite(_v2_allo)
                       and _v2_allo > 0.3),
        "detail": (f"Spearman(phi, lambda) = {_v2_allo:+.3f} "
                   f"(threshold 0.3)" if _v2_allo is not None
                   else "no allostatic samples"),
    })
    checks.append({
        "id": "V3",
        "name": "arousal is monotone increasing in the distributional "
                "shift W1 (rho ~ +1)",
        "passed": bool(np.isfinite(mean_rho) and mean_rho > 0.999),
        "detail": f"mean rho = {mean_rho:.6f} "
                  f"(n={len(rho_ka)} dyads)",
    })
    checks.append({
        "id": "V4",
        "name": "lambda mapping matches its numerical reconstruction "
                "(tolerance 1e-9)",
        "passed": bool(lam_max_dev < 1e-9),
        "detail": f"max deviation = {lam_max_dev:.3e}",
    })

    # ============================== V5. The reference does not chase
    # v1.5.1: the reference is learned from the seeded relationships'
    # characteristic rewards through **the same recursion** as the
    # current partner's Z (scale alignment). It therefore moves, but
    # its trajectory must be **identical whoever the current partner
    # is** — a reference that chases the current partner makes valence
    # extinguish or invert under chronic exploitation (the measured
    # failure of temporal self-comparison). Different seeds get
    # different social histories, so the comparison uses the **same
    # seed** against two opponents (confound control).
    _ref = {}
    for _kind in ("allc", "alld"):
        _ag = HalloRegAgent(seed=8801, **cfg.halloreg_kwargs())
        run_dyad(_ag, make_opponent(_kind, seed=8802), cfg.rounds,
                 partner_id_a=1, partner_id_b=999)
        _ref[_kind] = np.asarray(_ag.log["baseline_reward"], float)
    ref_allc, ref_alld = _ref["allc"], _ref["alld"]
    ref_gap = float(np.max(np.abs(ref_allc - ref_alld)))
    checks.append({
        "id": "V5",
        "name": "the reference trajectory is independent of the current "
                "partner (no self-chasing)",
        "passed": bool(ref_gap < 1e-6),
        "detail": f"max reference gap, ALLC vs ALLD partner = "
                  f"{ref_gap:.3e} (terminal reference "
                  f"{ref_allc[-1]:.3f})",
    })

    # V5b: valence separates opponent types (ALLC positive, ALLD
    # negative).
    v_allc = float(np.mean(val_traces["allc"][:, -cfg.rounds // 4:]))
    v_alld = float(np.mean(val_traces["alld"][:, -cfg.rounds // 4:]))
    checks.append({
        "id": "V5b",
        "name": "valence diverges by condition: ALLC > 0 > ALLD "
                "(social fitness)",
        "passed": bool(v_allc > 0.0 > v_alld),
        "detail": f"terminal mean valence: ALLC={v_allc:+.3f} vs "
                  f"ALLD={v_alld:+.3f}",
    })

    # V5c: **scale alignment** — if every opponent yields negative
    # valence early in learning, valence is measuring Z's learning
    # progress rather than the opponent's character. This guards
    # against the earlier defect of a theory-steady-state reference
    # (first round -0.96). [v3.5.0 re-operationalisation] n-step SARSA
    # needs n rounds before its first completed update, so the "early
    # learning" window moves to **the 3 rounds right after the first
    # value update**; before that there are no updates and the
    # percentile is filled by rank-statistic amplification of
    # near-ties, which is not a scale-alignment issue. Identical to
    # the old window when n_step = 1.
    _w0 = max(1, int(getattr(cfg, "n_step", 1)))
    early = {k: float(np.mean(val_traces[k][:, _w0:_w0 + 3]))
             for k in PROBE_TYPES}
    checks.append({
        # v1.7.0: Z starts from the prior of a typical relationship,
        # so the first rounds' valence must sit **near neutral**; a
        # collapse to -0.96 would mean valence tracks learning
        # progress rather than the opponent.
        "id": "V5c",
        "name": "early-learning valence is near neutral "
                "(scale alignment)",
        "passed": bool(abs(early["allc"]) < 0.6 and abs(early["alld"]) < 0.6),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={early[k]:+.3f}"
                             for k in PROBE_TYPES)
                  + f" (3R after the first update, w0={_w0})",
    })

    # ===================== V6. Conditional divergence of lambda
    lam_final = {k: lam_traces[k][:, -1] for k in PROBE_TYPES}
    checks.append({
        "id": "V6",
        "name": "directional divergence of lambda "
                "(lambda vs ALLC > lambda vs ALLD)",
        "passed": bool(lam_final["allc"].mean() > lam_final["alld"].mean()),
        "detail": " | ".join(f"{PROBE_LABEL[k]}={lam_final[k].mean():.3f}"
                             for k in PROBE_TYPES),
    })

    # ============================== V7. The single driver of lambda
    # lambda_ctx was removed in v1.5.0, so the lambda trajectory must
    # coincide exactly with the affective channel (V4 checks this
    # round by round). Here the terminal gap confirms that the
    # conditional divergence was produced **by affect alone**.
    lam_allc7 = float(np.mean(lam_traces["allc"][:, -1]))
    lam_alld7 = float(np.mean(lam_traces["alld"][:, -1]))
    checks.append({
        # Purely affective drive moves lambda little per round, so
        # divergence needs time; the threshold is relaxed at smoke
        # scale (measured gap 0.50 at 120R).
        "id": "V7",
        "name": "lambda diverges under purely affective drive",
        "passed": bool(lam_allc7 - lam_alld7
                       > (0.2 if cfg.rounds >= 80 else 0.1)),
        "detail": f"λ_final: ALLC={lam_allc7:.3f} vs ALLD={lam_alld7:.3f} "
                  f"(gap {lam_allc7 - lam_alld7:.3f})",
    })

    # ========================= V8. Distance-weighted setpoint
    # With a close (frequently met) cooperator and a distant
    # (once-met) exploiter, the setpoint must lean toward the close
    # one; an indifferent aggregate would make the two cases equal.
    sm_a = SelfModel()
    sm_a.set_payoff_scale(PAYOFF_SELF)
    for _ in range(40):                       # close cooperator
        sm_a.observe_identity(1)
        sm_a.commit_observation(1, opponent_cooperated=True)
    sm_a.observe_identity(2)                  # distant exploiter (once)
    sm_a.commit_observation(2, opponent_cooperated=False)

    sm_b = SelfModel()
    sm_b.set_payoff_scale(PAYOFF_SELF)
    for _ in range(40):                       # close exploiter
        sm_b.observe_identity(1)
        sm_b.commit_observation(1, opponent_cooperated=False)
    sm_b.observe_identity(2)                  # distant cooperator
    sm_b.commit_observation(2, opponent_cooperated=True)

    lam_a, lam_b = sm_a.lambda_setpoint(), sm_b.lambda_setpoint()
    d_close, d_far = sm_a.social_distance(1), sm_a.social_distance(2)
    checks.append({
        "id": "V8",
        "name": "the setpoint is weighted by social distance "
                "(the closer relationship dominates)",
        "passed": bool(lam_a > lam_b + 0.05 and d_close < d_far),
        "detail": f"lam0: close cooperator={lam_a:.3f} vs "
                  f"close exploiter={lam_b:.3f} | distance: "
                  f"close={d_close:.3f} < far={d_far:.3f}",
    })
    checks.append({
        "id": "V8b",
        "name": "a memoryless individual has a neutral setpoint "
                "(lam0 = 0.40)",
        "passed": bool(abs(SelfModel().lambda_setpoint() - 0.40) < 1e-9),
        "detail": f"lam0(no memory) = "
                  f"{SelfModel().lambda_setpoint():.6f}",
    })

    # ================================= V9. Seeded social history
    # Individuals must enter the experiment with an existing network,
    # not as blank slates.
    # (a) do the three distance bands actually form?
    # (b) does that produce **individual differences** in lam0? without
    #     a seeded history every individual sits at exactly 0.40 and
    #     social distance has no influence on lambda at all.
    lam0s, dmins, dmaxs = [], [], []
    for sd in range(24):
        sm = SelfModel()
        sm.seed_social_history(PAYOFF_SELF, rng=np.random.default_rng(sd))
        summ = sm.social_summary()
        lam0s.append(summ["lambda_setpoint"])
        ds = [r["distance"] for r in summ["rows"]]
        dmins.append(min(ds)); dmaxs.append(max(ds))
    lam0s = np.asarray(lam0s)
    checks.append({
        "id": "V9",
        "name": "the seeded history loads others across several "
                "distance bands",
        "passed": bool(np.mean(dmins) < 0.4 and np.mean(dmaxs) > 0.8
                       and summ["n_others"] == 5),
        "detail": f"mean distance range "
                  f"[{np.mean(dmins):.3f}, {np.mean(dmaxs):.3f}], "
                  f"{summ['n_others']} relationships",
    })
    checks.append({
        "id": "V9b",
        "name": "the seeded history creates individual differences in "
                "lam0 (SD > 0.01)",
        "passed": bool(lam0s.std(ddof=1) > 0.01),
        "detail": f"λ₀ = {lam0s.mean():.3f} ± {lam0s.std(ddof=1):.3f} "
                  f"[{lam0s.min():.3f}, {lam0s.max():.3f}]",
    })

    # (c) does the history's cooperativeness monotonically determine
    # lam0 — the causal check on the setpoint
    lam_by_coop = []
    for cm in (0.2, 0.35, 0.5, 0.65, 0.8):
        v = []
        for sd in range(12):
            sm = SelfModel()
            sm.seed_social_history(PAYOFF_SELF, coop_mean=cm,
                                   rng=np.random.default_rng(100 + sd))
            v.append(sm.lambda_setpoint())
        lam_by_coop.append(float(np.mean(v)))
    checks.append({
        "id": "V9c",
        "name": "history cooperativeness -> lam0 increases monotonically "
                "(causality of the setpoint)",
        "passed": bool(all(lam_by_coop[i] < lam_by_coop[i + 1]
                           for i in range(len(lam_by_coop) - 1))),
        "detail": " → ".join(f"{v:.3f}" for v in lam_by_coop)
                  + "  (history cooperation rate 0.2 -> 0.8)",
    })

    # (d) **Per-person marginal contribution** — the correct
    #     operationalisation of distance weighting. The earlier check
    #     compared "the whole close band cooperative" against "the
    #     whole distant band", without controlling band sizes
    #     (3 / 6 / 12), so it compared totals rather than per-person
    #     weights. Here **one** person is added to the same history
    #     and the resulting delta-lam0 is compared across distances.
    def _lam_with_extra(n_touch: int, seed: int = 7) -> tuple:
        sm = SelfModel()
        sm.seed_social_history(PAYOFF_SELF, coop_mean=0.5,
                               rng=np.random.default_rng(seed))
        base = sm.lambda_setpoint()
        extra = 999                       # one fully cooperative addition
        for _ in range(n_touch):
            sm._touch(extra)
        sm.memory[extra].n_obs += 20
        sm.memory[extra].coop_count += 20.0
        return base, sm.lambda_setpoint(), sm.social_distance(extra)

    b1, l_close, d_cl = _lam_with_extra(90)    # close (F = 3*F_scale)
    b2, l_far, d_fr = _lam_with_extra(3)       # far   (F = 0.1*F_scale)
    checks.append({
        "id": "V9d",
        "name": "per-person marginal contribution is inverse to distance "
                "(one close > one distant)",
        "passed": bool((l_close - b1) > (l_far - b2) + 0.01),
        "detail": f"delta-lam0: close (d={d_cl:.2f})="
                  f"{l_close - b1:+.4f} vs far (d={d_fr:.2f})="
                  f"{l_far - b2:+.4f}",
    })

    # ============================== V10-V12 retired (v3.9.3)
    # These three checks (V10 reciprocity-path adaptation, V11
    # two-path dissociation, V12 endogenous probing) all built agents
    # with policy_mode="traits" to switch on the preserved trait
    # policy path explicitly. Since v3.9.3 lambda-only action
    # selection is the sole policy path of HalloRegAgent and the
    # constructor no longer accepts policy_mode, so these checks could
    # only have been kept by re-exposing a policy path the
    # architecture no longer uses. They are retired rather than
    # rewritten: with the trait policy gone there is no "second path"
    # to dissociate from lambda, and the claims they supported are
    # withdrawn accordingly. The trait machinery itself remains in
    # agent.py for the EmpathicAgent baseline.
    #
    # Recorded results at retirement (not carried forward as claims):
    #   V10 failed under payoff_access="naive" (visit-frequency bias
    #       in R-hat starved the DC cell); it was reported as a
    #       failure, never threshold-tuned into a pass.
    #   V12 was an honest null — raising w_epi to 12 changed trait
    #       selection but did not improve trait recovery.

    # ============================== V13. Memory of one's own traits
    # Meeting the same partner again must restore the stance held
    # then.
    sm_agent = build_agent({"type": "halloreg", "seed": 4242,
                            **cfg.halloreg_kwargs()})
    run_dyad(sm_agent, make_opponent("tft", seed=4243), cfg.rounds,
             partner_id_a=77, partner_id_b=1)
    stored = dict(sm_agent.self_model.memory[77].self_theta)
    sm_agent.begin_partner(77)                     # re-encounter
    restored = sm_agent.self_policy.posterior_means()
    dev = max(abs(stored[a] - restored[a]) for a in stored)
    checks.append({
        "id": "V13",
        "name": "self-trait memory: the stance is restored on "
                "re-encounter",
        "passed": bool(dev < 0.35),
        "detail": "stored " + " ".join(f"{k}={v:+.2f}"
                                       for k, v in stored.items())
                  + " -> restored "
                  + " ".join(f"{k}={restored[k]:+.2f}" for k in stored)
                  + f" (max deviation {dev:.3f})",
    })

    # ==================================== V14. QRTD convergence
    # The learned 1-step expected-utility difference s-hat(lam, p)
    # must converge to the analytic empathy_shift(lam, p). If it does,
    # empathy_shift is not retired but reinterpreted as the
    # **closed-form oracle for the known-payoff case**.
    from AIF_IPD.core.constants import PAYOFF_OTHER, empathy_shift
    from AIF_IPD.core.qrtd import RewardModel

    rm = RewardModel(lr=0.10)
    rm.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min()))
    rng14 = np.random.default_rng(0)
    for _ in range(4000):
        jj = int(rng14.integers(0, 4))
        rm.update(jj, float(PAYOFF_SELF[jj]), float(PAYOFF_OTHER[jj]))
    errs14 = [abs(rm.shift(lm, pp) - empathy_shift(lm, pp))
              for lm in (0.0, 0.4, 1.0) for pp in (0.3, 0.5, 0.7)]
    checks.append({
        "id": "V14",
        "name": "QRTD: the learned s-hat(lam, p) converges to the "
                "analytic solution (error < 0.15)",
        "passed": bool(max(errs14) < 0.15),
        "detail": f"max |s-hat - s| = {max(errs14):.4f}, "
                  f"learned payoffs = "
                  f"{np.round(rm.payoff_vector('self'), 2).tolist()} "
                  f"(true {PAYOFF_SELF.tolist()})",
    })

    # =============================== V15. Payoff-access ablation
    # Does the cooperative relationship survive in naive mode (payoff
    # matrix unobserved, learned by QRTD)? Only if it does can the H4
    # advantage be claimed as a property of the model rather than an
    # information-access privilege. Some degradation against oracle is
    # acceptable; collapse is not.
    pay15 = {}
    for mode in ("oracle", "naive"):
        sp15 = []
        for sd in range(min(n_probe, 10)):
            kw = dict(cfg.halloreg_kwargs()); kw["payoff_access"] = mode
            sp15.append({
                "agent": {"type": "halloreg", "seed": 5100 + sd * 41, **kw},
                "opponent": {"type": "strategy", "kind": "tft",
                             "seed": 5200 + sd * 41},
                "env_err_agent": cfg.env_error,
                "env_err_opponent": cfg.env_error,
                "noise_seed": 490_000 + sd * 43})
        rr15 = run_many(sp15, n_rounds=cfg.rounds, n_jobs=cfg.jobs,
                        desc=f"ARCH V15 {mode}")
        pay15[mode] = float(np.mean([r["hist"]["my_payoff"].mean()
                                     for r in rr15]))
    checks.append({
        "id": "V15",
        "name": "the cooperative relationship survives without the "
                "payoff matrix (naive)",
        "passed": bool(pay15["naive"] > 0.85 * pay15["oracle"]),
        "detail": f"per-round payoff vs TFT: "
                  f"oracle={pay15['oracle']:.3f} vs "
                  f"naive={pay15['naive']:.3f} (retention "
                  f"{100 * pay15['naive'] / max(pay15['oracle'], 1e-9):.1f}%)",
    })

    n_pass = sum(c["passed"] for c in checks)
    LOGGER.info("[ARCH] checks passed %d/%d", n_pass, len(checks))
    for c in checks:
        LOGGER.info("  [%s] %s — %s (%s)", c["id"],
                    "PASS" if c["passed"] else "FAIL", c["name"],
                    c["detail"])

    out = {"checks": checks, "n_pass": n_pass, "n_total": len(checks),
           "theta_by_partner": theta_by_partner,
           "reward_by_partner": reward_by_partner,
           "lam_traces": lam_traces, "base_traces": base_traces,
           "set_traces": set_traces, "dist_traces": dist_traces,
           "val_traces": val_traces, "aro_traces": aro_traces,
           "n_probe": n_probe}
    _plot(cfg, out)
    save_json({"checks": checks, "n_pass": n_pass, "n_total": len(checks),
               "theta_by_partner": theta_by_partner,
               "reward_by_partner": reward_by_partner},
              cfg.results / "ARCH.json")
    return out


# ============================================================== Figures
def _plot(cfg: Config, out: dict) -> None:
    import matplotlib.pyplot as plt
    from .common import annotate_n, band_plot

    colors = {"allc": "#8172B2", "gtft": "#55A868", "tft": "#4C72B0",
              "wsls": "#C44E52", "alld": "#937860"}

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)

    # (a) Hierarchy schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10); ax.set_ylim(-1.8, 10.2); ax.axis("off")
    ax.set_title("(a) hierarchical allostatic regulation", loc="left")

    def box(x, y, w, h, text, fc):
        """Draw one rectangular node of the schematic."""
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           linewidth=1.1, edgecolor="#444444", facecolor=fc,
                           alpha=0.9)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.2)

    def arrow(x1, y1, x2, y2, rad=0.0):
        """Information-flow arrow between nodes."""
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#555555",
                                    connectionstyle=f"arc3,rad={rad}"))

    box(0.2, 8.4, 3.6, 1.4, "SelfModel\n(memory, prior supply)", "#F1E2F3")
    box(6.0, 8.4, 3.8, 1.4, "OpponentInversion\n(particle filter)",
        "#DCE6F5")
    box(6.0, 5.6, 3.8, 1.4, "CoreAffect\n(valence × arousal)", "#DDEEE0")
    box(2.6, 2.9, 4.4, 1.3, "Empathy — lambda carrier", "#FBE6D4")
    box(2.6, 0.7, 4.4, 1.3, "RecursiveSocialEFE -> action", "#EEEEEE")

    # SelfModel -> the two inference modules (prior supply)
    arrow(3.8, 9.4, 6.0, 9.4)
    ax.text(4.9, 9.62, "(id, theta) prior", fontsize=6.2, ha="center")
    arrow(3.8, 8.7, 6.0, 6.9)
    ax.text(4.9, 7.55, "(id, expected-reward dist.)", fontsize=6.2,
            ha="center")
    # SelfModel -> CoreAffect (allostatic setpoint)
    arrow(1.6, 8.4, 1.6, 6.3)
    arrow(1.6, 6.3, 6.0, 6.1)
    ax.text(1.75, 7.3, "baseline expected reward\n= setpoint",
            fontsize=6.2, ha="left",
            va="center")
    # Both modules -> Empathy (drive signals)
    arrow(7.2, 8.4, 6.6, 4.2, rad=0.22)
    ax.text(8.3, 6.6, "λ_ctx\n= f(α̂, λ̂ⱼ)", fontsize=6.2, ha="center")
    arrow(7.0, 5.6, 6.0, 4.2, rad=0.15)
    ax.text(6.2, 4.9, "λ_aff = V×A", fontsize=6.2, ha="center")
    # Empathy -> action
    arrow(4.8, 2.9, 4.8, 2.0)
    ax.text(5.0, 2.42, "λ_t", fontsize=6.6, ha="left")

    ax.text(5.0, -0.35, "λ_t = λ_{t−1} + η·[(1−w)·λ_aff + w·λ_ctx]",
            fontsize=7.2, ha="center", color="#222222")
    ax.text(5.0, -1.15,
            "updated (theta, dist) and expected-reward distributions "
            "are committed to SelfModel\n"
            "SelfModel does not infer; it only remembers",
            fontsize=6.2, ha="center", va="center", color="#666666")

    # (b) lambda trajectories by opponent type
    ax = fig.add_subplot(gs[0, 1])
    for k in PROBE_TYPES:
        band_plot(ax, out["lam_traces"][k], color=colors[k], label=PROBE_LABEL[k])
    ax.set_xlabel("round"); ax.set_ylabel("empathy weight lambda")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) V6 — conditional divergence of lambda")
    ax.legend(ncol=2, fontsize=7); annotate_n(ax, out["n_probe"])

    # (c) Drift of the allostatic setpoint (baseline expected reward)
    ax = fig.add_subplot(gs[0, 2])
    for k in PROBE_TYPES:
        band_plot(ax, out["base_traces"][k], color=colors[k],
                  label=PROBE_LABEL[k])
    ax.set_xlabel("round")
    ax.set_ylabel("baseline expected reward E_social[r]")
    ax.set_title("(c) V5 — slow drift of the allostatic setpoint")
    ax.legend(ncol=2, fontsize=7); annotate_n(ax, out["n_probe"])

    # (d) valence trajectories
    ax = fig.add_subplot(gs[1, 0])
    for k in PROBE_TYPES:
        band_plot(ax, out["val_traces"][k], color=colors[k], label=PROBE_LABEL[k])
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xlabel("round"); ax.set_ylabel("valence")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("(d) valence (social-fitness based)")
    ax.legend(ncol=2, fontsize=7)

    # (e) arousal trajectories
    ax = fig.add_subplot(gs[1, 1])
    for k in PROBE_TYPES:
        band_plot(ax, out["aro_traces"][k], color=colors[k], label=PROBE_LABEL[k])
    ax.set_xlabel("round"); ax.set_ylabel("arousal")
    ax.set_title("(e) V3 — arousal (distributional shift)")
    ax.legend(ncol=2, fontsize=7)

    # (f) Verdict table
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_title(f"(f) architecture validation verdicts "
                 f"({out['n_pass']}/{out['n_total']} passed)", loc="left")
    y = 0.94
    for c in out["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        col = "#2E7D32" if c["passed"] else "#C62828"
        ax.text(0.0, y, f"[{mark}] [{c['id']}] {c['name']}",
                fontsize=7.2, color=col, transform=ax.transAxes, va="top")
        ax.text(0.04, y - 0.045, c["detail"], fontsize=6.4, color="#555555",
                transform=ax.transAxes, va="top")
        y -= 0.115

    fig.suptitle("ARCH — implementation consistency of the hierarchical "
                 "allostatic regulation architecture",
                 fontsize=12, y=0.98)
    save_fig(fig, cfg, "ARCH_validation")
