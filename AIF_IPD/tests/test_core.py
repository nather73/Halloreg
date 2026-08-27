"""
tests.test_core
===============

**Deterministic unit checks** of the core mechanisms.

Confirms the implementation behaves as specified before any
hypothesis testing. Run:

    python -m AIF_IPD.tests.test_core          # or
    python AIF_IPD/tests/test_core.py

Every check asserts with the standard library only; no external test
runner is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AIF_IPD.core.constants import (
    CC, CD, DC, DD, COOP, DEFECT, PAYOFF_SELF, PAYOFF_OTHER,
    empathy_shift, joint_index, mirror_state, reset_payoffs, set_payoffs,
    split_joint,
)
from AIF_IPD.core.core_affect import CoreAffect
from AIF_IPD.core.empathy import Empathy
from AIF_IPD.core.generative import efe_terms, softmax
from AIF_IPD.core.self_model import SelfModel
from AIF_IPD.ipd.agent import EmpathicAgent, HalloRegAgent
from AIF_IPD.ipd.env import LikelihoodAgent, ProbeAgent, make_opponent
from AIF_IPD.ipd.evolution import ore_ends_batch, replicator_ends_batch
from AIF_IPD.ipd.metrics import bh_fdr, holm, perm_test
from AIF_IPD.ipd.population import (
    enumerate_compositions, population_cc, substitution_delta_cc, type_payoffs,
)
from AIF_IPD.ipd.sim import run_dyad
from AIF_IPD.ipd.tom.inversion import ObservationContext, OpponentInversion

_PASS, _FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Register a single check."""
    (_PASS if cond else _FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


# ========================================================= Constants/payoffs
def test_constants() -> None:
    print("\n[1] payoff structure and index conventions")
    reset_payoffs()
    check("joint_index / split_joint round trip",
          all(split_joint(joint_index(a, b)) == (a, b)
              for a in (0, 1) for b in (0, 1)))
    check("mirror_state is an involution",
          all(mirror_state(mirror_state(s)) == s for s in range(4)))
    check("mirror_state swaps CD and DC",
          mirror_state(CD) == DC and mirror_state(DC) == CD)
    check("PD ordering T > R > P > S",
          PAYOFF_SELF[DC] > PAYOFF_SELF[CC] > PAYOFF_SELF[DD] > PAYOFF_SELF[CD])
    check("payoff symmetry (only CD/DC swap)",
          PAYOFF_SELF[CC] == PAYOFF_OTHER[CC]
          and PAYOFF_SELF[CD] == PAYOFF_OTHER[DC])

    # Variable payoffs: is R recomputed exactly from CI?
    set_payoffs(1.2)
    check("set_payoffs(1.2) -> harmony game (R > T)",
          PAYOFF_SELF[CC] > PAYOFF_SELF[DC],
          f"R={PAYOFF_SELF[CC]:.1f}, T={PAYOFF_SELF[DC]:.1f}")
    set_payoffs(-0.3)
    check("set_payoffs(-0.3) -> deadlock game (R < P)",
          PAYOFF_SELF[CC] < PAYOFF_SELF[DD],
          f"R={PAYOFF_SELF[CC]:.1f}, P={PAYOFF_SELF[DD]:.1f}")
    reset_payoffs()
    check("reset_payoffs returns to the default PD", PAYOFF_SELF[CC] == 3.0)

    # Analytic coefficients of empathy_shift (5*lam - p - 1 in the
    # default PD)
    lam, p = 0.7, 0.3
    check("empathy_shift coefficients = (T-S, R-T+P-S, S-P)",
          abs(empathy_shift(lam, p) - (5 * lam - p - 1)) < 1e-12)


# ==================================================================== EFE
def test_efe() -> None:
    print("\n[2] analytic expected free energy")
    reset_payoffs()
    t = efe_terms(0.5, PAYOFF_SELF)
    check("neg_efe = pragmatic + epistemic",
          np.allclose(t["neg_efe"], t["pragmatic"] + t["epistemic"]))
    check("G = −neg_efe", np.allclose(t["G"], -t["neg_efe"]))
    # At pc=0.5 both actions must have equal state entropy (the
    # distributions are permutations of each other)
    check("equal entropy for both actions at pc=0.5",
          abs(t["epistemic"][0] - t["epistemic"][1]) < 1e-12)
    # If the opponent surely cooperates, defecting (T) has higher
    # pragmatic value than cooperating (R)
    t1 = efe_terms(0.999, PAYOFF_SELF)
    check("pragmatic value of D > C when cooperation is certain",
          t1["pragmatic"][DEFECT] > t1["pragmatic"][COOP])
    s = softmax(np.array([1.0, 2.0, 3.0]))
    check("softmax sums to 1", abs(s.sum() - 1.0) < 1e-12)


# ==================================================================== SelfModel
def test_self_model() -> None:
    print("\n[3] SelfModel — memory, social distance, setpoint, reference")
    reset_payoffs()
    sm = SelfModel()
    sm.set_payoff_scale(PAYOFF_SELF)
    pr = sm.theta_prior(None)
    check("new identity -> uninformative theta prior", abs(pr["lambda_j"][0] - 0.5) < 1e-9)
    lam0 = sm.lambda_setpoint()
    check("no memory -> lam0 at the interval midpoint", abs(lam0 - 0.40) < 1e-9, f"lam0={lam0:.6f}")

    sm.commit_theta(7, {"alpha": 1.5, "rho": 0.2, "omega": 0.0, "eta": 0.0,
                        "beta": 4.0, "lambda_j": 0.8},
                    {"alpha": 0.3, "rho": 0.3, "omega": 0.3, "eta": 0.3,
                     "beta": 0.3, "lambda_j": 0.05})
    pr2 = sm.theta_prior(7)
    check("prior mean reflects memory on re-encounter", abs(pr2["alpha"][0] - 1.5) < 1e-9)
    check("prior width has a floor", pr2["lambda_j"][1] >= 0.25 * 0.3 - 1e-12)

    # Distance: frequency and recency
    sm2 = SelfModel(); sm2.set_payoff_scale(PAYOFF_SELF)
    check("unknown partner has d = 1", abs(sm2.social_distance(99) - 1.0) < 1e-12)
    for _ in range(50):
        sm2.observe_identity(1)
    sm2.observe_identity(2)
    check("a frequently met partner is closer",
          sm2.social_distance(1) < sm2.social_distance(2))
    check("closer partners get larger inverse weights",
          sm2.distance_weight(1) > sm2.distance_weight(2))

    # ---- Seeded social history (5 people) + reference valence ----
    sm3 = SelfModel(); 
    sm3.seed_social_history(PAYOFF_SELF, gamma=0.9,
                            rng=np.random.default_rng(3))
    summ = sm3.social_summary()
    check("the seeded history holds 5 people", summ["n_others"] == 5, f"n={summ['n_others']}")
    ds = [r["distance"] for r in summ["rows"]]
    check("three distance bands", min(ds) < 0.4 and max(ds) > 0.8)
    check("value distributions assigned directly (no generated observations)",
          all(r["value_median"] is not None for r in summ["rows"]))

    # valence: a bad partner (low value) -> negative, a good one -> positive
    taus = sm3.taus
    # v1.7.0: the reference is **static memory** — placed directly by
    # cooperation rate, never rolled forward.
    ref_med = sm3.reference_median()
    # v2.7.0: with the Z-tilde normalisation the reference also lives on
    # the per-round mean-reward scale [1, 3].
    check("the reference lies on the normalised value scale", 1.0 < ref_med < 3.5,
          f"reference median={ref_med:.2f}")
    meds = sorted(float(e.value_dist[10]) for e in sm3.memory.values()
                  if e.value_dist is not None)
    check("the five values are spread apart (reference dispersion)",
          meds[-1] - meds[0] > 0.1, f"range [{meds[0]:.2f}, {meds[-1]:.2f}]")
    c0, sp0 = sm3.value_prior()
    check("value_prior returns the centre and spread of memory",
          1.0 < c0 < 3.5 and sp0 > 0.05, f"({c0:.2f}, {sp0:.2f})")
    lo = (ref_med - 3.0) + 0.0 * (taus - 0.5)     # a relationship below the reference
    hi = (ref_med + 3.0) + 0.0 * (taus - 0.5)     # a relationship above the reference
    sm3.update_partner_value(500, lo)
    check("bad relationship -> valence < 0", sm3.social_valence(500) < -0.3,
          f"v={sm3.social_valence(500):+.3f}")
    sm3.update_partner_value(501, hi)
    check("good relationship -> valence > 0", sm3.social_valence(501) > 0.3,
          f"v={sm3.social_valence(501):+.3f}")
    check("the reference excludes the current partner",
          abs(sm3.social_valence(500)
              - (2 * __import__("AIF_IPD.core.distributional",
                                fromlist=["vector_cdf"]).vector_cdf(
                    sm3.population_reference(exclude=500), taus,
                    float(lo[10])) - 1))
          < 1e-9)

    # Setpoint directionality (cooperation-rate based, independent of
    # the value distribution)
    sm4 = SelfModel()
    for _ in range(40):
        sm4.observe_identity(1); sm4.commit_observation(1, opponent_cooperated=True)
    sm5 = SelfModel()
    for _ in range(40):
        sm5.observe_identity(1); sm5.commit_observation(1, opponent_cooperated=False)
    check("a close cooperator raises lam0", sm4.lambda_setpoint() > lam0)
    check("a close exploiter lowers lam0", sm5.lambda_setpoint() < lam0)


def test_distributional() -> None:
    print("\n[3b] quantile-grid utilities")
    from AIF_IPD.core.distributional import (
        DEFAULT_TAUS, midpoint_taus, vector_cdf,
    )
    check("the default grid has 21 channels", len(DEFAULT_TAUS) == 21)
    check("a tau=0.5 knot exists", any(abs(x - 0.5) < 1e-12 for x in DEFAULT_TAUS))
    try:
        midpoint_taus(20); even_ok = False
    except ValueError:
        even_ok = True
    check("an even channel count is rejected", even_ok)

    taus = np.array(DEFAULT_TAUS)
    v = np.linspace(0.0, 10.0, 21)
    check("vector_cdf: median maps to 0.5", abs(vector_cdf(v, taus, 5.0) - 0.5) < 1e-9)
    check("vector_cdf is strictly increasing",
          all(vector_cdf(v, taus, x) < vector_cdf(v, taus, x + 0.5)
              for x in (-2.0, 2.0, 8.0, 12.0)))
    check("vector_cdf is bounded in (0,1)",
          0.0 < vector_cdf(v, taus, -1e6) and vector_cdf(v, taus, 1e6) < 1.0)
    check("degenerate distribution -> neutral 0.5",
          abs(vector_cdf(np.full(21, 3.0), taus, 7.0) - 0.5) < 1e-12)


def test_core_affect() -> None:
    print("\n[4] CoreAffect — state valence x update arousal, setpoint split")
    reset_payoffs()
    sm = SelfModel()
    sm.seed_social_history(PAYOFF_SELF, gamma=0.9,
                           rng=np.random.default_rng(1))
    ca = CoreAffect(sm, PAYOFF_SELF)
    ca.begin_partner(1)
    taus = sm.taus
    med = sm.reference_median()

    # --- valence is now between-state fitness (injected externally) ---
    o_bad = ca.step(CD, opponent_cooperated=False,
                    value_vector=(med - 3.0) + 0.5 * (taus - 0.5),
                    value_shift=0.02, state_valence=-0.6)
    check("valence uses the injected state fitness verbatim",
          abs(o_bad["valence"] + 0.6) < 1e-12, f"V={o_bad['valence']:+.3f}")
    check("λ_aff = valence × arousal",
          abs(o_bad["lambda_aff"]
              - o_bad["valence"] * o_bad["arousal"]) < 1e-12)

    # --- the setpoint derives from between-relationship fitness ---
    check("bad relationship -> low setpoint", o_bad["lam_sp"] < 0.4,
          f"λ_sp={o_bad['lam_sp']:.3f} (fitness={o_bad['fitness']:+.3f})")
    check("bad relationship -> fitness < 0", o_bad["fitness"] < 0)

    sm2 = SelfModel()
    sm2.seed_social_history(PAYOFF_SELF, gamma=0.9,
                            rng=np.random.default_rng(1))
    ca2 = CoreAffect(sm2, PAYOFF_SELF)
    ca2.begin_partner(1)
    good = (sm2.reference_median() + 3.0) + 0.5 * (sm2.taus - 0.5)
    o_good = ca2.step(CC, opponent_cooperated=True, value_vector=good,
                      value_shift=0.02, state_valence=+0.6)
    check("good relationship -> higher setpoint", o_good["lam_sp"] > o_bad["lam_sp"],
          f"{o_good['lam_sp']:.3f} vs {o_bad['lam_sp']:.3f}")
    check("good relationship -> fitness > 0", o_good["fitness"] > 0)

    # --- arousal: monotone in the update magnitude, with a floor ---
    a_small = ca2.step(CC, value_vector=good, value_shift=0.005,
                       state_valence=0.0)["arousal"]
    a_big = ca2.step(CC, value_vector=good, value_shift=0.20,
                     state_valence=0.0)["arousal"]
    check("arousal is monotone in the update magnitude", a_big > a_small,
          f"{a_small:.4f} < {a_big:.4f}")
    a_zero = ca2.step(CC, value_vector=good, value_shift=0.0,
                      state_valence=0.0)["arousal"]
    check("arousal floor > 0 (so affect keeps a channel to lambda)",
          abs(a_zero - ca2.arousal_floor) < 1e-12, f"a_min={a_zero:.4f}")

    # --- the setpoint does not extinguish under chronic exploitation ---
    bad_vec = (med - 3.0) + 0.5 * (taus - 0.5)
    sp0 = ca.step(CD, value_vector=bad_vec, value_shift=0.01,
                  state_valence=-0.5)["lam_sp"]
    for _ in range(200):
        ca.step(CD, value_vector=bad_vec, value_shift=0.01,
                state_valence=-0.5)
    sp1 = ca.last["lam_sp"]
    check("the setpoint stays low under chronic exploitation (no chasing)",
          sp1 < 0.4 and abs(sp1 - sp0) < 0.2,
          f"λ_sp[0]={sp0:.3f} → [200]={sp1:.3f}")

    # Degenerate path
    sm3 = SelfModel(); ca3 = CoreAffect(sm3, PAYOFF_SELF); ca3.begin_partner(1)
    o = ca3.step(CC)
    check("works without a value_vector (valence neutral)",
          o["valence"] == 0.0 and o["arousal"] == ca3.arousal_floor)


def test_empathy() -> None:
    print("\n[5] Empathy — the lambda integrator")
    em = Empathy(lam_init=0.4, w_cd=0.5, gain=0.05)
    check("neutral partner (alpha=0, lambda_j=0.5) -> lambda_ctx = 0",
          abs(em.contextual(0.0, 0.5)) < 1e-12)
    check("prosocial partner -> lambda_ctx > 0", em.contextual(2.0, 0.9) > 0)
    check("exploitative partner -> lambda_ctx < 0", em.contextual(-2.0, 0.1) < 0)
    check("λ_ctx ∈ [−1, +1]",
          abs(em.contextual(10.0, 1.0)) <= 1.0
          and abs(em.contextual(-10.0, 0.0)) <= 1.0)

    prev = em.lam
    new = em.step(lambda_aff=0.6, lambda_ctx=0.4, valence=-0.2, lam_sp=0.7)
    # v1.8.0: slow setpoint relaxation plus fast affect; lambda_ctx is
    # ignored.
    expected = min(prev + 0.05 * (0.7 - prev) + em.aff_gain * 0.6, em.lam_max)
    check("update = lam + eta_sp*(lam_sp - lam) + g_aff*lam_aff",
          abs(new - expected) < 1e-12, f"{new:.6f} vs {expected:.6f}")
    check("the lambda_ctx argument has no effect",
          abs(Empathy(lam_init=0.4, gain=0.05).step(0.6, 0.0, valence=0.1)
              - Empathy(lam_init=0.4, gain=0.05).step(0.6, 0.9, valence=0.1))
          < 1e-12)
    # Two timescales: the setpoint slowly, affect within a round.
    # v2.8.0: asymmetric relaxation — downward eta_down, upward eta_up.
    e_sp = Empathy(lam_init=0.5, gain=0.05, aff_gain=0.30, gain_down=0.45)
    e_sp.step(lambda_aff=0.0, lam_sp=0.0, threat=True)
    check("a low setpoint pulls lambda down fast (threat learning)",
          0.25 < e_sp.lam < 0.30, f"λ={e_sp.lam:.4f}")
    e_up = Empathy(lam_init=0.5, gain=0.05, aff_gain=0.30, gain_down=0.45)
    e_up.step(lambda_aff=0.0, lam_sp=1.0)
    check("a high setpoint raises lambda slowly (trust recovery)",
          0.52 < e_up.lam < 0.53, f"λ={e_up.lam:.4f}")
    e_af = Empathy(lam_init=0.5, gain=0.05, aff_gain=0.30)
    e_af.step(lambda_aff=-0.8, lam_sp=0.5)
    check("affect moves substantially within one round",
          e_af.lam < 0.30, f"λ={e_af.lam:.4f}")

    em2 = Empathy(lam_init=0.79, w_cd=0.0, gain=1.0, aff_gain=1.0)
    check("lambda is clipped at the ceiling (0.80)", em2.step(10.0, 0.0) <= 0.80 + 1e-12,
          f"λ={em2.lam:.4f}")
    em3 = Empathy(lam_init=0.01, w_cd=0.0, gain=1.0)
    check("lambda is clipped at the floor (0.0)", em3.step(-10.0, 0.0) >= 0.0)


# ========================================================== Particle filter
def test_inversion() -> None:
    print("\n[6] OpponentInversion — likelihood, tense, information gain")
    reset_payoffs()
    inv = OpponentInversion(n_particles=300, seed=0)
    check("the prior mean of lambda_j is near neutral (0.5)",
          abs(inv.posterior_means()["lambda_j"] - 0.5) < 0.08)

    # Sign conventions of the f/g feature extraction
    ctx = ObservationContext(my_last_action=COOP, their_last_action=DEFECT)
    check("f: I cooperated -> +1", inv._feature_f(ctx) == +1.0)
    check("g: they defected -> -1", inv._feature_g(ctx) == -1.0)
    check("no history -> 0", inv._feature_f(ObservationContext()) == 0.0)

    # Information gain is always non-negative (discrete mutual information)
    ig = inv.expected_infogain_exact(+1.0, 0.0, 0.0)
    check("expected information gain >= 0", ig >= 0.0, f"IG={ig:.4f}")

    # Reliability lies in [0,1]
    check("reliability in [0,1]", 0.0 <= inv.reliability() <= 1.0)

    # Recovery: rho-hat must grow against a deterministic reciprocator
    inv2 = OpponentInversion(n_particles=400, seed=1)
    opp = make_opponent("tft", seed=2)
    probe = ProbeAgent(0.5, seed=3)
    my_hist, opp_hist = [], []
    for t in range(150):
        a_j = opp.act()
        a_i = probe.act()
        opp.observe(a_i)
        c = ObservationContext(
            my_last_action=(my_hist[-1] if my_hist else None),
            their_last_action=(opp_hist[-1] if opp_hist else None))
        inv2.my_cooperation_rate = (float(np.mean(my_hist)) if my_hist else 0.5)
        inv2.update(a_j, c)
        my_hist.append(a_i); opp_hist.append(a_j)
    m = inv2.posterior_means()
    check("vs TFT -> strongly positive rho-hat", m["rho"] > 1.0, f"rho-hat={m['rho']:+.2f}")

    # WSLS must be expressed on the pure eta axis
    inv3 = OpponentInversion(n_particles=400, seed=4)
    opp3 = make_opponent("wsls", seed=5)
    probe3 = ProbeAgent(0.5, seed=6)
    my_hist, opp_hist = [], []
    for t in range(150):
        a_j = opp3.act()
        a_i = probe3.act()
        opp3.observe(a_i)
        c = ObservationContext(
            my_last_action=(my_hist[-1] if my_hist else None),
            their_last_action=(opp_hist[-1] if opp_hist else None))
        inv3.my_cooperation_rate = (float(np.mean(my_hist)) if my_hist else 0.5)
        inv3.update(a_j, c)
        my_hist.append(a_i); opp_hist.append(a_j)
    m3 = inv3.posterior_means()
    check("vs WSLS -> eta-hat is the dominant axis",
          m3["eta"] > abs(m3["rho"]) and m3["eta"] > 0.5,
          f"η̂={m3['eta']:+.2f}, ρ̂={m3['rho']:+.2f}")


# ================================================================== Agents
def test_agents() -> None:
    print("\n[7] agents — directionality and logging consistency")
    reset_payoffs()
    # Against an exploiter: lambda falls
    a = HalloRegAgent(seed=1, n_particles=200)
    run_dyad(a, make_opponent("alld", seed=2), 120)
    lam_alld = a.log["lam"]
    check("vs exploiter -> lambda falls", lam_alld[-1] < lam_alld[0],
          f"{lam_alld[0]:.3f} → {lam_alld[-1]:.3f}")

    # Against a cooperator: lambda rises
    b = HalloRegAgent(seed=1, n_particles=200)
    run_dyad(b, make_opponent("allc", seed=2), 120)
    lam_allc = b.log["lam"]
    check("vs cooperator -> lambda rises", lam_allc[-1] > lam_allc[0],
          f"{lam_allc[0]:.3f} → {lam_allc[-1]:.3f}")
    check("lambda diverges between the two conditions", lam_allc[-1] > lam_alld[-1])

    # Log-length consistency
    lens = {k: len(v) for k, v in b.log.items()}
    check("every log channel has length equal to the round count",
          set(lens.values()) == {120}, f"{sorted(set(lens.values()))}")

    # A fixed-lambda agent keeps lambda constant
    c = EmpathicAgent(lam=0.37, seed=1, n_particles=200)
    run_dyad(c, make_opponent("alld", seed=2), 40)
    check("EmpathicAgent keeps lambda fixed",
          all(abs(x - 0.37) < 1e-12 for x in c.log["lam"]))

    # Lambda schedule switching
    d = EmpathicAgent(lam=0.1, lam_schedule=[(0, 0.1), (20, 0.9)],
                      seed=1, n_particles=200)
    run_dyad(d, make_opponent("tft", seed=2), 40)
    check("lam_schedule switches at the specified round",
          abs(d.log["lam"][0] - 0.1) < 1e-12
          and abs(d.log["lam"][-1] - 0.9) < 1e-12)

    # Internal-state consistency under environment execution error
    e = HalloRegAgent(seed=3, n_particles=200)
    h = run_dyad(e, make_opponent("tft", seed=4), 60,
                 env_err_a=0.3, noise_seed=9)
    check("log and history lengths match under environment error",
          len(e.my_actions) == 60 and len(h["my_act"]) == 60)
    check("agent history matches the actually emitted actions",
          all(int(e.my_actions[i]) == int(h["my_act"][i]) for i in range(60)))


# ==================================================================== QRTD
def test_qrtd() -> None:
    print("\n[3c] QRTD — distributional value and reward learning")
    from AIF_IPD.core.constants import PAYOFF_OTHER
    from AIF_IPD.core.qrtd import (
        QuantileTD, RewardModel, mean_value, state_index,
    )
    reset_payoffs()

    check("state_index maps (my last, their last) to 4 states",
          state_index(0, 0) == 0 and state_index(1, 1) == 3
          and state_index(0, 1) == 1 and state_index(1, 0) == 2)

    q = np.arange(21, dtype=float)
    check("mean_value is the plain mean (midpoint integration)",
          abs(mean_value(q) - q.mean()) < 1e-12)
    # tau_risk retired — the risk-sensitive (CVaR) reduction is gone.
    check("value queries are expectations (no risk-sensitive reduction)",
          not hasattr(QuantileTD(seed=0), "tau_risk"))

    # ---- Does RewardModel converge to true payoffs and recover the
    # analytic solution? ----
    rm = RewardModel(lr=0.10)
    rm.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min()))
    rng = np.random.default_rng(0)
    for _ in range(4000):
        j = int(rng.integers(0, 4))
        rm.update(j, float(PAYOFF_SELF[j]), float(PAYOFF_OTHER[j]))
    err = float(np.max(np.abs(rm.payoff_vector("self") - PAYOFF_SELF)))
    check("R-hat converges to the true payoffs (error < 0.15)", err < 0.15,
          f"max error {err:.4f}")
    check("R-hat also learns the other's payoffs",
          float(np.max(np.abs(rm.payoff_vector("other") - PAYOFF_OTHER))) < 0.15)
    serr = max(abs(rm.shift(lm, pp) - empathy_shift(lm, pp))
               for lm in (0.0, 0.4, 1.0) for pp in (0.3, 0.7))
    check("the learned s-hat converges to empathy_shift (error < 0.15)", serr < 0.15,
          f"max |s-hat - s| = {serr:.4f}")

    # ---- QuantileTD bootstrapping ----
    z = QuantileTD(gamma=0.9, lr=0.05, seed=0)
    z.set_scale(5.0)
    check("Z is indexed by (state, action)",
          z.z_self.values.shape[:2] == (4, 2),
          f"shape={z.z_self.values.shape}")
    # An environment with only CC -> the raw return fixed point
    # The Huber step saturates, so convergence is linear — iterate enough.
    # v2.6.0: the last argument is the actual next action a', not a
    # cooperation probability (SARSA).
    for _ in range(20000):
        z.update(0, 0, float(PAYOFF_SELF[CC]), float(PAYOFF_OTHER[CC]), 0, COOP)
    v = z.value(0, 0, "self")
    # v2.7.0: with the normalised return the fixed point is the per-round
    # mean reward.
    check("QRTD bootstrap: Z-tilde converges to the mean reward", abs(v - 3.0) < 0.4,
          f"Z(s,C)={v:.2f} (theory 3.0)")
    check("Z-tilde lives on the per-round mean-reward scale",
          abs(v - float(PAYOFF_SELF[CC])) < 0.5)



# =========================================================== Trait policy
def test_self_policy() -> None:
    print("\n[7b] SelfPolicy — the trait-space policy")
    from AIF_IPD.ipd.tom.self_policy import SELF_AXES, SELF_BOUNDS, SelfPolicy
    reset_payoffs()

    tft = dict(alpha=0.0, rho=2.5, omega=0.0, eta=0.0, beta=4.0, lambda_j=0.5)
    alld = dict(alpha=-3.0, rho=0.0, omega=0.0, eta=0.0, beta=4.0, lambda_j=0.2)

    # ---- 1-step evaluation plus terminal Z (rollout retired) ----
    sp = SelfPolicy(n_particles=1, w_epi_j=0.0, w_epi_r=0.0, w_cplx=0.0,
                       seed=0)
    check("rollout is retired and replaced by _evaluate",
          hasattr(sp, "_evaluate") and not hasattr(sp, "_rollout"))
    sp.theta = {"rho": np.array([0.0]), "omega": np.array([0.0]),
                "eta": np.array([0.0])}
    v_hi = float(sp._evaluate(sp.theta, +1.0, +1.0, 0.5, 0.8, 1.0,
                              None, None, None, None, None, None)[0])
    v_lo = float(sp._evaluate(sp.theta, +1.0, +1.0, 0.5, 0.8, 0.0,
                              None, None, None, None, None, None)[0])
    check("higher opponent cooperation probability -> higher expected utility", v_hi > v_lo,
          f"p_j=1 → {v_hi:.2f} vs p_j=0 → {v_lo:.2f}")

    # ---- Epistemic term on environment structure ----
    from AIF_IPD.core.qrtd import RewardModel
    rm = RewardModel(lr=0.10)
    rm.set_scale(5.0)
    u0 = rm.uncertainty().copy()
    for _ in range(300):
        rm.update(0, 3.0, 3.0)          # repeatedly observe CC only
    u1 = rm.uncertainty()
    check("uncertainty of an observed cell decreases", u1[0] < u0[0],
          f"CC: {u0[0]:.3f} → {u1[0]:.3f}")
    check("uncertainty of an unobserved cell is preserved", abs(u1[2] - u0[2]) < 1e-9,
          f"DC: {u0[2]:.3f} -> {u1[2]:.3f} (source of exploration drive)")

    # ---- Complexity term ----
    sp2 = SelfPolicy(n_particles=3, w_cplx=1.0, seed=1)
    sp2.theta = {"rho": np.array([0.0, 1.0, 2.0]),
                 "omega": np.array([0.0, 0.0, 0.0]),
                 "eta": np.array([0.0, 0.0, 0.0])}
    c = sp2._complexity()
    check("complexity is 0 at the prior centre and grows with distance",
          abs(c[0]) < 1e-12 and c[0] < c[1] < c[2],
          f"{c.round(3).tolist()}")

    # ---- alpha=0, beta=4 fixed: lambda is the dynamic intercept ----
    from AIF_IPD.core.constants import empathy_shift
    lo = empathy_shift(0.0, 0.5); hi = empathy_shift(1.0, 0.5)
    check("lambda acts as the intercept (wide range)", hi - lo > 3.0,
          f"s(0,·)={lo:+.2f} → s(1,·)={hi:+.2f}")
    check("beta is fixed at 4.0", abs(SelfPolicy().beta - 4.0) < 1e-12)

    # ---- Particle properties ----
    sp3 = SelfPolicy(n_particles=64, seed=2)
    out = sp3.step(tft, 0.4, +1.0, +1.0, 0.8, 0.8, None)
    check("weights sum to 1", abs(sp3.weights.sum() - 1.0) < 1e-9)
    check("traits stay within range", all(
        SELF_BOUNDS[ax][0] - 1e-9 <= sp3.theta[ax].min()
        and sp3.theta[ax].max() <= SELF_BOUNDS[ax][1] + 1e-9
        for ax in SELF_AXES))
    check("ESS ∈ (0, K]", 0 < out["ess"] <= 64 + 1e-9)
    pc = sp3.coop_prob_mixture(+1.0, +1.0, 0.4, 0.8)
    check("mixture cooperation probability in (0,1)", 0.0 < pc < 1.0, f"P(C)={pc:.3f}")

    # ---- Prior restoration ----
    sp4 = SelfPolicy(n_particles=64, seed=3)
    sp4.set_prior({"rho": (1.5, 0.2), "omega": (0.0, 0.2), "eta": (0.0, 0.2)})
    check("set_prior reinitialises particles at the given location",
          abs(sp4.posterior_means()["rho"] - 1.5) < 0.15,
          f"ρ̄={sp4.posterior_means()['rho']:.3f}")

    # ---- Basis role swap: f is the opponent's last action ----
    sp5 = SelfPolicy(n_particles=1, seed=4)
    sp5.theta = {"rho": np.array([2.0]), "omega": np.array([0.0]),
                 "eta": np.array([0.0])}
    p_after_c = float(sp5._coop_prob(sp5.theta, +1.0, 0.0, 0.4, 0.8)[0])
    p_after_d = float(sp5._coop_prob(sp5.theta, -1.0, 0.0, 0.4, 0.8)[0])
    check("rho>0 gives cooperation after cooperation, defection after defection",
          p_after_c > 0.9 and p_after_d < 0.1,
          f"P(C|theirC)={p_after_c:.3f}, P(C|theirD)={p_after_d:.3f}")


# ============================================================== Population
def test_population() -> None:
    print("\n[8] population — composition enumeration and decomposition identity")
    comps = enumerate_compositions(30, 6, min_each=1)
    check("composition count = C(29,5) = 118,755", len(comps) == 118_755,
          f"{len(comps):,}")
    check("every composition sums to 30", bool(np.all(comps.sum(axis=1) == 30)))
    check("at least one of every type", bool(comps.min() >= 1))

    rng = np.random.default_rng(0)
    Pi = rng.uniform(0, 5, (6, 6))
    CCm = rng.uniform(0, 1, (6, 6))
    CCm = (CCm + CCm.T) / 2

    def brute_cc(n):
        labels = [i for i, v in enumerate(n) for _ in range(v)]
        tot = cnt = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                tot += CCm[labels[i], labels[j]]; cnt += 1
        return tot / cnt

    def brute_mu(n, t):
        labels = [i for i, v in enumerate(n) for _ in range(v)]
        idx = [k for k, l in enumerate(labels) if l == t]
        return float(np.mean([
            np.mean([Pi[t, labels[j]] for j in range(len(labels)) if j != k])
            for k in idx]))

    for n in (np.array([5, 5, 5, 5, 5, 5]), np.array([3, 7, 2, 9, 4, 5])):
        cc_a = float(population_cc(CCm, n)); cc_b = brute_cc(n)
        check(f"CC decomposition equals the exhaustive computation (n={n.tolist()})",
              abs(cc_a - cc_b) < 1e-10, f"{cc_a:.10f} vs {cc_b:.10f}")
        for t in (0, 3, 5):
            mu_a = float(type_payoffs(Pi, n)[t]); mu_b = brute_mu(n, t)
            check(f"  payoff decomposition equals the exhaustive computation (type {t})",
                  abs(mu_a - mu_b) < 1e-10)

    # Substitution contrast: replacing a type by itself gives 0
    d = substitution_delta_cc(CCm, np.array([5, 5, 5, 5, 5, 5]), 5, 5)
    check("substitution by the same type -> delta = 0", abs(float(d)) < 1e-12)
    # NaN when the type is absent
    d2 = substitution_delta_cc(CCm, np.array([30, 0, 0, 0, 0, 0]), 5, 0)
    check("substitution contrast is NaN when the type is absent", bool(np.isnan(float(d2))))


# =============================================================== Evolution
def test_evolution() -> None:
    print("\n[9] evolutionary dynamics — RE / ORE")
    # Two-type PD: defectors must dominate cooperators (a known RE
    # property)
    Pi = np.array([[3.0, 0.0], [5.0, 1.0]])       # rows: C, D
    X0 = np.array([[0.5, 0.5], [0.9, 0.1]])
    ends = replicator_ends_batch(Pi, X0, steps=400)
    check("RE: defectors dominate in a PD",
          bool(np.all(ends[:, 1] > 0.99)), f"x_D={ends[:, 1].round(4).tolist()}")

    ends_o = ore_ends_batch(Pi, X0, tau=1.0, steps=150, sweeps=20)
    check("the ORE terminal composition lies on the simplex",
          bool(np.allclose(ends_o.sum(axis=1), 1.0, atol=1e-6)))
    check("the ORE terminal composition is non-negative",
          bool(np.all(ends_o >= -1e-12)))
    check("ORE suppresses cooperation less than RE (group-level selection)",
          bool(np.mean(ends_o[:, 0]) >= np.mean(ends[:, 0]) - 1e-9),
          f"x_C: RE={np.mean(ends[:, 0]):.4f} → ORE={np.mean(ends_o[:, 0]):.4f}")

    # The replicator must be invariant to affine payoff transforms
    e1 = replicator_ends_batch(Pi, X0, steps=300)
    e2 = replicator_ends_batch(Pi + 10.0, X0, steps=300)
    check("RE is invariant to a constant payoff shift",
          bool(np.allclose(e1, e2, atol=1e-6)))


# ============================================================== Statistics
def test_stats() -> None:
    print("\n[10] statistical infrastructure")
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 60)
    b = rng.normal(0.0, 1.0, 60)
    t = perm_test(a, b, paired=False, n_perm=2000, alternative="greater")
    check("the permutation test detects a real difference", t["p"] < 0.05, f"p={t['p']:.4g}")
    t0 = perm_test(a, a.copy(), paired=True, n_perm=2000)
    check("identical samples -> p = 1", t0["p"] > 0.9, f"p={t0['p']:.3f}")

    ps = [0.001, 0.01, 0.04, 0.5]
    hp, bp = holm(ps), bh_fdr(ps)
    check("Holm correction is monotone non-decreasing",
          all(hp[i] <= hp[i + 1] + 1e-12 for i in range(len(hp) - 1)))
    check("Holm >= the raw p", all(h >= p - 1e-12 for h, p in zip(hp, ps)))
    check("BH <= Holm (FDR is more permissive than FWER)",
          all(x <= y + 1e-12 for x, y in zip(bp, hp)))
    check("corrected p <= 1", all(x <= 1.0 for x in hp + bp))


# ================================================================ Recovery
def test_recovery() -> None:
    print("\n[11] parameter recovery — round trip through the generative likelihood")
    reset_payoffs()
    truth = dict(alpha=0.8, rho=1.6, omega=-0.5, eta=0.9, beta=4.0,
                 lambda_j=0.3)
    gen = LikelihoodAgent(**truth, seed=11)
    probe = ProbeAgent(0.5, seed=12)
    inv = OpponentInversion(n_particles=600, seed=13)
    my_hist, opp_hist = [], []
    for t in range(400):
        a_j = gen.act()
        a_i = probe.act()
        gen.observe(a_i)
        c = ObservationContext(
            my_last_action=(my_hist[-1] if my_hist else None),
            their_last_action=(opp_hist[-1] if opp_hist else None))
        inv.my_cooperation_rate = (float(np.mean(my_hist)) if my_hist else 0.5)
        inv.update(a_j, c)
        my_hist.append(a_i); opp_hist.append(a_j)
    est = inv.posterior_means()
    for ax, tol in (("rho", 1.0), ("omega", 1.0), ("eta", 1.0)):
        check(f"{ax} recovery (tolerance +/-{tol})",
              abs(est[ax] - truth[ax]) < tol,
              f"true={truth[ax]:+.2f}, est={est[ax]:+.2f}")
    check("reciprocity sign recovered", np.sign(est["rho"]) == np.sign(truth["rho"]))
    check("inertia sign recovered", np.sign(est["omega"]) == np.sign(truth["omega"]))


# ==================================================================== main
def main() -> int:
    print("=" * 70)
    print("HalloReg core-mechanism unit checks")
    print("=" * 70)
    for fn in (test_constants, test_efe, test_self_model, test_distributional,
               test_core_affect, test_qrtd,
               test_empathy, test_inversion, test_agents, test_self_policy, test_population,
               test_evolution, test_stats, test_recovery):
        fn()
    print("\n" + "=" * 70)
    print(f"result: {len(_PASS)} passed / {len(_PASS) + len(_FAIL)} checks")
    if _FAIL:
        print("failed items:")
        for f in _FAIL:
            print(f"  - {f}")
    print("=" * 70)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
