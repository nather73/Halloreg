"""
tests.test_core
===============

핵심 기제의 **결정적(deterministic) 단위 검증**.

가설검증 이전에 구현이 사양대로 동작하는지 확인한다. 실행:

    python -m AIF_IPD.tests.test_core          # 또는
    python AIF_IPD/tests/test_core.py

모든 검사는 표준 라이브러리만으로 assert 하며 외부 테스트 러너가 필요 없다.
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
    """단일 검사 등록."""
    (_PASS if cond else _FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


# ==================================================================== 상수·보수
def test_constants() -> None:
    print("\n[1] 보수 구조와 인덱스 규약")
    reset_payoffs()
    check("joint_index / split_joint 왕복",
          all(split_joint(joint_index(a, b)) == (a, b)
              for a in (0, 1) for b in (0, 1)))
    check("mirror_state 는 대합(involution)",
          all(mirror_state(mirror_state(s)) == s for s in range(4)))
    check("mirror_state 가 CD↔DC 를 교환",
          mirror_state(CD) == DC and mirror_state(DC) == CD)
    check("PD 조건 T > R > P > S",
          PAYOFF_SELF[DC] > PAYOFF_SELF[CC] > PAYOFF_SELF[DD] > PAYOFF_SELF[CD])
    check("보수 대칭성 (CD/DC 만 교환)",
          PAYOFF_SELF[CC] == PAYOFF_OTHER[CC]
          and PAYOFF_SELF[CD] == PAYOFF_OTHER[DC])

    # 가변 보수: CI 로부터 R 이 정확히 재계산되는가
    set_payoffs(1.2)
    check("set_payoffs(1.2) → 조화 게임 (R > T)",
          PAYOFF_SELF[CC] > PAYOFF_SELF[DC],
          f"R={PAYOFF_SELF[CC]:.1f}, T={PAYOFF_SELF[DC]:.1f}")
    set_payoffs(-0.3)
    check("set_payoffs(-0.3) → 교착 게임 (R < P)",
          PAYOFF_SELF[CC] < PAYOFF_SELF[DD],
          f"R={PAYOFF_SELF[CC]:.1f}, P={PAYOFF_SELF[DD]:.1f}")
    reset_payoffs()
    check("reset_payoffs 로 기본 PD 복귀", PAYOFF_SELF[CC] == 3.0)

    # empathy_shift 의 해석적 계수 (기본 PD 에서 5λ − p − 1)
    lam, p = 0.7, 0.3
    check("empathy_shift 계수 = (T−S, R−T+P−S, S−P)",
          abs(empathy_shift(lam, p) - (5 * lam - p - 1)) < 1e-12)


# ==================================================================== EFE
def test_efe() -> None:
    print("\n[2] 해석적 기대자유에너지")
    reset_payoffs()
    t = efe_terms(0.5, PAYOFF_SELF)
    check("neg_efe = pragmatic + epistemic",
          np.allclose(t["neg_efe"], t["pragmatic"] + t["epistemic"]))
    check("G = −neg_efe", np.allclose(t["G"], -t["neg_efe"]))
    # pc=0.5 에서 두 행동의 상태 엔트로피가 같아야 한다 (분포가 서로 치환)
    check("pc=0.5 에서 두 행동의 엔트로피 동일",
          abs(t["epistemic"][0] - t["epistemic"][1]) < 1e-12)
    # 상대가 확실히 협력하면 배신(T)이 협력(R)보다 실용가치가 높다
    t1 = efe_terms(0.999, PAYOFF_SELF)
    check("상대 협력 확실 시 D 의 실용가치 > C",
          t1["pragmatic"][DEFECT] > t1["pragmatic"][COOP])
    s = softmax(np.array([1.0, 2.0, 3.0]))
    check("softmax 합 = 1", abs(s.sum() - 1.0) < 1e-12)


# ==================================================================== SelfModel
def test_self_model() -> None:
    print("\n[3] SelfModel — 기억·사회적 거리·설정점·모집단 참조")
    reset_payoffs()
    sm = SelfModel()
    sm.set_payoff_scale(PAYOFF_SELF)
    pr = sm.theta_prior(None)
    check("신규 identity → 무정보 θ 사전", abs(pr["lambda_j"][0] - 0.5) < 1e-9)
    lam0 = sm.lambda_setpoint()
    check("무기억 → λ_0 = 구간 중점", abs(lam0 - 0.40) < 1e-9, f"λ_0={lam0:.6f}")

    sm.commit_theta(7, {"alpha": 1.5, "rho": 0.2, "omega": 0.0, "eta": 0.0,
                        "beta": 4.0, "lambda_j": 0.8},
                    {"alpha": 0.3, "rho": 0.3, "omega": 0.3, "eta": 0.3,
                     "beta": 0.3, "lambda_j": 0.05})
    pr2 = sm.theta_prior(7)
    check("재조우 시 사전 평균이 기억을 반영", abs(pr2["alpha"][0] - 1.5) < 1e-9)
    check("사전 폭에 하한", pr2["lambda_j"][1] >= 0.25 * 0.3 - 1e-12)

    # 거리: 빈도·최근성
    sm2 = SelfModel(); sm2.set_payoff_scale(PAYOFF_SELF)
    check("미지의 상대는 d = 1", abs(sm2.social_distance(99) - 1.0) < 1e-12)
    for _ in range(50):
        sm2.observe_identity(1)
    sm2.observe_identity(2)
    check("자주 만난 상대가 더 가깝다",
          sm2.social_distance(1) < sm2.social_distance(2))
    check("가까울수록 역가중이 크다",
          sm2.distance_weight(1) > sm2.distance_weight(2))

    # ---- 사전 사회사 (5인) + 모집단 참조 valence ----
    sm3 = SelfModel(); 
    sm3.seed_social_history(PAYOFF_SELF, gamma=0.9,
                            rng=np.random.default_rng(3))
    summ = sm3.social_summary()
    check("사전 사회사는 5인", summ["n_others"] == 5, f"n={summ['n_others']}")
    ds = [r["distance"] for r in summ["rows"]]
    check("세 거리 대역", min(ds) < 0.4 and max(ds) > 0.8)
    check("가치분포가 직접 부여됨 (관측 생성 없음)",
          all(r["value_median"] is not None for r in summ["rows"]))

    # valence: 나쁜 상대(가치 낮음) → 음수, 좋은 상대 → 양수
    taus = sm3.taus
    # v1.7.0: 참조는 **정적 기억**이다. 협력률에 따라 직접 배치되며 굴리지 않는다.
    ref_med = sm3.reference_median()
    # v2.7.0: Z̃ = (1−γ)Z 정규화 → 참조도 **라운드당 평균 보상** 척도 [1, 3].
    check("참조가 정규화 가치 척도에 놓인다 (r̄)", 1.0 < ref_med < 3.5,
          f"참조 중앙값={ref_med:.2f}")
    meds = sorted(float(e.value_dist[10]) for e in sm3.memory.values()
                  if e.value_dist is not None)
    check("5인의 가치가 서로 벌어져 있다 (참조 산포)",
          meds[-1] - meds[0] > 0.1, f"범위 [{meds[0]:.2f}, {meds[-1]:.2f}]")
    c0, sp0 = sm3.value_prior()
    check("value_prior 가 기억의 중심·산포를 준다",
          1.0 < c0 < 3.5 and sp0 > 0.05, f"({c0:.2f}, {sp0:.2f})")
    lo = (ref_med - 3.0) + 0.0 * (taus - 0.5)     # 참조보다 낮은 관계
    hi = (ref_med + 3.0) + 0.0 * (taus - 0.5)     # 참조보다 높은 관계
    sm3.update_partner_value(500, lo)
    check("나쁜 관계 → valence < 0", sm3.social_valence(500) < -0.3,
          f"v={sm3.social_valence(500):+.3f}")
    sm3.update_partner_value(501, hi)
    check("좋은 관계 → valence > 0", sm3.social_valence(501) > 0.3,
          f"v={sm3.social_valence(501):+.3f}")
    check("참조는 현재 상대를 제외한다",
          abs(sm3.social_valence(500)
              - (2 * __import__("AIF_IPD.core.distributional",
                                fromlist=["vector_cdf"]).vector_cdf(
                    sm3.population_reference(exclude=500), taus,
                    float(lo[10])) - 1))
          < 1e-9)

    # 설정점 방향성 (협력률 기반 — 가치분포와 독립)
    sm4 = SelfModel()
    for _ in range(40):
        sm4.observe_identity(1); sm4.commit_observation(1, opponent_cooperated=True)
    sm5 = SelfModel()
    for _ in range(40):
        sm5.observe_identity(1); sm5.commit_observation(1, opponent_cooperated=False)
    check("가까운 협력자 → λ_0 상승", sm4.lambda_setpoint() > lam0)
    check("가까운 착취자 → λ_0 하강", sm5.lambda_setpoint() < lam0)


def test_distributional() -> None:
    print("\n[3b] 분위수 격자 유틸리티")
    from AIF_IPD.core.distributional import (
        DEFAULT_TAUS, midpoint_taus, vector_cdf,
    )
    check("기본 격자는 21채널", len(DEFAULT_TAUS) == 21)
    check("τ=0.5 매듭 존재", any(abs(x - 0.5) < 1e-12 for x in DEFAULT_TAUS))
    try:
        midpoint_taus(20); even_ok = False
    except ValueError:
        even_ok = True
    check("짝수 채널 거부", even_ok)

    taus = np.array(DEFAULT_TAUS)
    v = np.linspace(0.0, 10.0, 21)
    check("vector_cdf: 중앙값 → 0.5", abs(vector_cdf(v, taus, 5.0) - 0.5) < 1e-9)
    check("vector_cdf: 순증가",
          all(vector_cdf(v, taus, x) < vector_cdf(v, taus, x + 0.5)
              for x in (-2.0, 2.0, 8.0, 12.0)))
    check("vector_cdf ∈ (0,1) 유계",
          0.0 < vector_cdf(v, taus, -1e6) and vector_cdf(v, taus, 1e6) < 1.0)
    check("퇴화 분포 → 중립 0.5",
          abs(vector_cdf(np.full(21, 3.0), taus, 7.0) - 0.5) < 1e-12)


def test_core_affect() -> None:
    print("\n[4] CoreAffect — 상태 valence × 갱신량 arousal, λ_sp 분리")
    reset_payoffs()
    sm = SelfModel()
    sm.seed_social_history(PAYOFF_SELF, gamma=0.9,
                           rng=np.random.default_rng(1))
    ca = CoreAffect(sm, PAYOFF_SELF)
    ca.begin_partner(1)
    taus = sm.taus
    med = sm.reference_median()

    # --- valence 는 이제 **상태 간** 적합도 (외부에서 주입) ---
    o_bad = ca.step(CD, opponent_cooperated=False,
                    value_vector=(med - 3.0) + 0.5 * (taus - 0.5),
                    value_shift=0.02, state_valence=-0.6)
    check("valence 는 주입된 상태 적합도를 그대로 쓴다",
          abs(o_bad["valence"] + 0.6) < 1e-12, f"V={o_bad['valence']:+.3f}")
    check("λ_aff = valence × arousal",
          abs(o_bad["lambda_aff"]
              - o_bad["valence"] * o_bad["arousal"]) < 1e-12)

    # --- λ_sp 는 **관계 간** 적합도에서 나온다 ---
    check("나쁜 관계 → λ_sp 가 낮다", o_bad["lam_sp"] < 0.4,
          f"λ_sp={o_bad['lam_sp']:.3f} (fitness={o_bad['fitness']:+.3f})")
    check("나쁜 관계 → fitness < 0", o_bad["fitness"] < 0)

    sm2 = SelfModel()
    sm2.seed_social_history(PAYOFF_SELF, gamma=0.9,
                            rng=np.random.default_rng(1))
    ca2 = CoreAffect(sm2, PAYOFF_SELF)
    ca2.begin_partner(1)
    good = (sm2.reference_median() + 3.0) + 0.5 * (sm2.taus - 0.5)
    o_good = ca2.step(CC, opponent_cooperated=True, value_vector=good,
                      value_shift=0.02, state_valence=+0.6)
    check("좋은 관계 → λ_sp 가 높다", o_good["lam_sp"] > o_bad["lam_sp"],
          f"{o_good['lam_sp']:.3f} vs {o_bad['lam_sp']:.3f}")
    check("좋은 관계 → fitness > 0", o_good["fitness"] > 0)

    # --- arousal: 갱신량의 단조증가 + **하한** ---
    a_small = ca2.step(CC, value_vector=good, value_shift=0.005,
                       state_valence=0.0)["arousal"]
    a_big = ca2.step(CC, value_vector=good, value_shift=0.20,
                     state_valence=0.0)["arousal"]
    check("arousal 은 갱신량의 단조증가", a_big > a_small,
          f"{a_small:.4f} < {a_big:.4f}")
    a_zero = ca2.step(CC, value_vector=good, value_shift=0.0,
                      state_valence=0.0)["arousal"]
    check("arousal 하한 > 0 (정서가 λ 통로를 잃지 않도록)",
          abs(a_zero - ca2.arousal_floor) < 1e-12, f"a_min={a_zero:.4f}")

    # --- 만성 착취에서 λ_sp 가 소멸하지 않는다 (모집단 참조) ---
    bad_vec = (med - 3.0) + 0.5 * (taus - 0.5)
    sp0 = ca.step(CD, value_vector=bad_vec, value_shift=0.01,
                  state_valence=-0.5)["lam_sp"]
    for _ in range(200):
        ca.step(CD, value_vector=bad_vec, value_shift=0.01,
                state_valence=-0.5)
    sp1 = ca.last["lam_sp"]
    check("만성 착취에도 λ_sp 가 낮게 유지 (참조가 쫓아오지 않음)",
          sp1 < 0.4 and abs(sp1 - sp0) < 0.2,
          f"λ_sp[0]={sp0:.3f} → [200]={sp1:.3f}")

    # 퇴화 경로
    sm3 = SelfModel(); ca3 = CoreAffect(sm3, PAYOFF_SELF); ca3.begin_partner(1)
    o = ca3.step(CC)
    check("value_vector 없이도 동작 (valence 중립)",
          o["valence"] == 0.0 and o["arousal"] == ca3.arousal_floor)


def test_empathy() -> None:
    print("\n[5] Empathy — λ 적분기")
    em = Empathy(lam_init=0.4, w_cd=0.5, gain=0.05)
    check("중립 상대(α=0, λⱼ=0.5) → λ_ctx = 0",
          abs(em.contextual(0.0, 0.5)) < 1e-12)
    check("친사회적 상대 → λ_ctx > 0", em.contextual(2.0, 0.9) > 0)
    check("착취적 상대 → λ_ctx < 0", em.contextual(-2.0, 0.1) < 0)
    check("λ_ctx ∈ [−1, +1]",
          abs(em.contextual(10.0, 1.0)) <= 1.0
          and abs(em.contextual(-10.0, 0.0)) <= 1.0)

    prev = em.lam
    new = em.step(lambda_aff=0.6, lambda_ctx=0.4, valence=-0.2, lam_sp=0.7)
    # v1.8.0: 느린 설정점 이완 + 빠른 정서. λ_ctx 는 무시된다.
    expected = min(prev + 0.05 * (0.7 - prev) + em.aff_gain * 0.6, em.lam_max)
    check("갱신식 = λ + η_sp(λ_sp−λ) + g_aff·λ_aff",
          abs(new - expected) < 1e-12, f"{new:.6f} vs {expected:.6f}")
    check("lambda_ctx 인자가 결과에 영향 없음",
          abs(Empathy(lam_init=0.4, gain=0.05).step(0.6, 0.0, valence=0.1)
              - Empathy(lam_init=0.4, gain=0.05).step(0.6, 0.9, valence=0.1))
          < 1e-12)
    # 두 시간척도: 설정점은 느리게, 정서는 라운드 단위로.
    # v2.8.0: **비대칭 이완** — 하강은 η_down(0.45), 상승은 η_up(0.05).
    e_sp = Empathy(lam_init=0.5, gain=0.05, aff_gain=0.30, gain_down=0.45)
    e_sp.step(lambda_aff=0.0, lam_sp=0.0, threat=True)
    check("낮은 λ_sp 는 λ 를 빠르게 끌어내린다 (위협 학습)",
          0.25 < e_sp.lam < 0.30, f"λ={e_sp.lam:.4f}")
    e_up = Empathy(lam_init=0.5, gain=0.05, aff_gain=0.30, gain_down=0.45)
    e_up.step(lambda_aff=0.0, lam_sp=1.0)
    check("높은 λ_sp 는 λ 를 느리게 끌어올린다 (신뢰 회복)",
          0.52 < e_up.lam < 0.53, f"λ={e_up.lam:.4f}")
    e_af = Empathy(lam_init=0.5, gain=0.05, aff_gain=0.30)
    e_af.step(lambda_aff=-0.8, lam_sp=0.5)
    check("정서는 한 라운드에 크게 움직인다 (설정점보다 6배 이득)",
          e_af.lam < 0.30, f"λ={e_af.lam:.4f}")

    em2 = Empathy(lam_init=0.79, w_cd=0.0, gain=1.0, aff_gain=1.0)
    check("λ 상한 클리핑 (0.80)", em2.step(10.0, 0.0) <= 0.80 + 1e-12,
          f"λ={em2.lam:.4f}")
    em3 = Empathy(lam_init=0.01, w_cd=0.0, gain=1.0)
    check("λ 하한 클리핑 (0.0)", em3.step(-10.0, 0.0) >= 0.0)


# ==================================================================== 입자필터
def test_inversion() -> None:
    print("\n[6] OpponentInversion — 우도·시제·정보이득")
    reset_payoffs()
    inv = OpponentInversion(n_particles=300, seed=0)
    check("사전 λⱼ 평균이 중립(0.5) 근방",
          abs(inv.posterior_means()["lambda_j"] - 0.5) < 0.08)

    # f/g 특징 추출의 부호 규약
    ctx = ObservationContext(my_last_action=COOP, their_last_action=DEFECT)
    check("f: 내가 협력 → +1", inv._feature_f(ctx) == +1.0)
    check("g: 상대가 배신 → −1", inv._feature_g(ctx) == -1.0)
    check("이력 없음 → 0", inv._feature_f(ObservationContext()) == 0.0)

    # 정보이득은 항상 비음 (이산 상호정보)
    ig = inv.expected_infogain(+1.0, 0.0)
    check("기대 정보이득 ≥ 0", ig >= 0.0, f"IG={ig:.4f}")

    # 신뢰도는 [0,1]
    check("신뢰도 ∈ [0,1]", 0.0 <= inv.reliability() <= 1.0)

    # 복원: TFT 같은 결정론적 호혜 상대에게 ρ̂ 가 커져야 한다
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
    check("TFT 상대 → ρ̂ 가 크게 양수", m["rho"] > 1.0, f"ρ̂={m['rho']:+.2f}")

    # WSLS 는 순수 η 축으로 표현되어야 한다
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
    check("WSLS 상대 → η̂ 가 지배적 축",
          m3["eta"] > abs(m3["rho"]) and m3["eta"] > 0.5,
          f"η̂={m3['eta']:+.2f}, ρ̂={m3['rho']:+.2f}")


# ==================================================================== 에이전트
def test_agents() -> None:
    print("\n[7] 에이전트 — 방향성과 로깅 정합")
    reset_payoffs()
    # 착취자 상대: λ 하강
    a = HalloRegAgent(seed=1, n_particles=200)
    run_dyad(a, make_opponent("alld", seed=2), 120)
    lam_alld = a.log["lam"]
    check("착취자 상대 → λ 하강", lam_alld[-1] < lam_alld[0],
          f"{lam_alld[0]:.3f} → {lam_alld[-1]:.3f}")

    # 협력자 상대: λ 상승
    b = HalloRegAgent(seed=1, n_particles=200)
    run_dyad(b, make_opponent("allc", seed=2), 120)
    lam_allc = b.log["lam"]
    check("협력자 상대 → λ 상승", lam_allc[-1] > lam_allc[0],
          f"{lam_allc[0]:.3f} → {lam_allc[-1]:.3f}")
    check("두 조건의 λ 가 분기", lam_allc[-1] > lam_alld[-1])

    # 로그 길이 정합
    lens = {k: len(v) for k, v in b.log.items()}
    check("모든 로그 채널 길이 = 라운드 수",
          set(lens.values()) == {120}, f"{sorted(set(lens.values()))}")

    # 고정 λ 에이전트는 λ 가 불변
    c = EmpathicAgent(lam=0.37, seed=1, n_particles=200)
    run_dyad(c, make_opponent("alld", seed=2), 40)
    check("EmpathicAgent 의 λ 는 고정",
          all(abs(x - 0.37) < 1e-12 for x in c.log["lam"]))

    # λ 스케줄 전환
    d = EmpathicAgent(lam=0.1, lam_schedule=[(0, 0.1), (20, 0.9)],
                      seed=1, n_particles=200)
    run_dyad(d, make_opponent("tft", seed=2), 40)
    check("lam_schedule 이 지정 라운드에 전환",
          abs(d.log["lam"][0] - 0.1) < 1e-12
          and abs(d.log["lam"][-1] - 0.9) < 1e-12)

    # 환경 실행오류 시 내부 상태 정합 (note_emitted)
    e = HalloRegAgent(seed=3, n_particles=200)
    h = run_dyad(e, make_opponent("tft", seed=4), 60,
                 env_err_a=0.3, noise_seed=9)
    check("환경 오류 하에서도 로그·이력 길이 일치",
          len(e.my_actions) == 60 and len(h["my_act"]) == 60)
    check("에이전트 이력이 실제 방출 행동과 일치",
          all(int(e.my_actions[i]) == int(h["my_act"][i]) for i in range(60)))


# ==================================================================== QRTD
def test_qrtd() -> None:
    print("\n[3c] QRTD — 분포적 가치·보상 학습")
    from AIF_IPD.core.constants import PAYOFF_OTHER
    from AIF_IPD.core.qrtd import (
        QuantileTD, RewardModel, mean_value, state_index,
    )
    reset_payoffs()

    check("state_index 는 (내직전, 상대직전) 을 4상태로",
          state_index(0, 0) == 0 and state_index(1, 1) == 3
          and state_index(0, 1) == 1 and state_index(1, 0) == 2)

    q = np.arange(21, dtype=float)
    check("mean_value = 전체 평균 (중점법 적분)",
          abs(mean_value(q) - q.mean()) < 1e-12)
    # tau_risk 폐기 — 위험민감 축약(CVaR)은 더 이상 쓰지 않는다.
    check("가치 조회는 기대값 (위험민감 축약 없음)",
          not hasattr(QuantileTD(seed=0), "tau_risk"))

    # ---- RewardModel 이 참 보수로 수렴하고 해석해를 복원하는가 ----
    rm = RewardModel(lr=0.10)
    rm.set_scale(float(PAYOFF_SELF.max() - PAYOFF_SELF.min()))
    rng = np.random.default_rng(0)
    for _ in range(4000):
        j = int(rng.integers(0, 4))
        rm.update(j, float(PAYOFF_SELF[j]), float(PAYOFF_OTHER[j]))
    err = float(np.max(np.abs(rm.payoff_vector("self") - PAYOFF_SELF)))
    check("R̂ 가 참 보수로 수렴 (오차 < 0.15)", err < 0.15,
          f"최대 오차 {err:.4f}")
    check("R̂ 가 타자 보수도 학습",
          float(np.max(np.abs(rm.payoff_vector("other") - PAYOFF_OTHER))) < 0.15)
    serr = max(abs(rm.shift(lm, pp) - empathy_shift(lm, pp))
               for lm in (0.0, 0.4, 1.0) for pp in (0.3, 0.7))
    check("학습된 ŝ(λ,p) 가 empathy_shift 로 수렴 (오차 < 0.15)", serr < 0.15,
          f"최대 |ŝ − s| = {serr:.4f}")

    # ---- QuantileTD 부트스트랩 ----
    z = QuantileTD(gamma=0.9, lr=0.05, seed=0)
    z.set_scale(5.0)
    check("Z 는 (상태, 행위) 로 색인된다",
          z.z_self.values.shape[:2] == (4, 2),
          f"shape={z.z_self.values.shape}")
    # 항상 CC 만 일어나는 환경 → Z(s,C) → R/(1−γ) = 3/0.1 = 30
    # Huber 스텝이 포화하므로 수렴이 선형이다 — 충분히 반복한다.
    # v2.6.0: 마지막 인자는 협력확률이 아니라 **실제 다음 행위** a' (SARSA).
    for _ in range(20000):
        z.update(0, 0, float(PAYOFF_SELF[CC]), float(PAYOFF_OTHER[CC]), 0, COOP)
    v = z.value(0, 0, "self")
    # v2.7.0: 정규화 수익 Z̃ = (1−γ)Z 이므로 고정점은 **라운드당 평균 보상** r̄.
    check("QRTD 부트스트랩: Z̃ → r̄ 로 수렴", abs(v - 3.0) < 0.4,
          f"Z̃(s,C)={v:.2f} (이론값 3.0)")
    check("Z̃ 는 라운드당 평균 보상 척도다 (수익/지평)",
          abs(v - float(PAYOFF_SELF[CC])) < 0.5)



# ==================================================================== 형질정책
def test_self_policy() -> None:
    print("\n[7b] SelfPolicy — 형질공간 정책")
    from AIF_IPD.ipd.tom.self_policy import SELF_AXES, SELF_BOUNDS, SelfPolicy
    reset_payoffs()

    tft = dict(alpha=0.0, rho=2.5, omega=0.0, eta=0.0, beta=4.0, lambda_j=0.5)
    alld = dict(alpha=-3.0, rho=0.0, omega=0.0, eta=0.0, beta=4.0, lambda_j=0.2)

    # ---- 1-step 평가 + 종단 Z (rollout 폐기) ----
    sp = SelfPolicy(n_particles=1, w_epi_j=0.0, w_epi_r=0.0, w_cplx=0.0,
                       seed=0)
    check("rollout 이 폐기되고 _evaluate 로 대체됨",
          hasattr(sp, "_evaluate") and not hasattr(sp, "_rollout"))
    sp.theta = {"rho": np.array([0.0]), "omega": np.array([0.0]),
                "eta": np.array([0.0])}
    v_hi = float(sp._evaluate(sp.theta, +1.0, +1.0, 0.5, 0.8, 1.0,
                              None, None, None, None, None, None)[0])
    v_lo = float(sp._evaluate(sp.theta, +1.0, +1.0, 0.5, 0.8, 0.0,
                              None, None, None, None, None, None)[0])
    check("상대 협력확률이 높을수록 기대효용이 크다", v_hi > v_lo,
          f"p_j=1 → {v_hi:.2f} vs p_j=0 → {v_lo:.2f}")

    # ---- 환경 구조에 대한 인식항 (IG_R) ----
    from AIF_IPD.core.qrtd import RewardModel
    rm = RewardModel(lr=0.10)
    rm.set_scale(5.0)
    u0 = rm.uncertainty().copy()
    for _ in range(300):
        rm.update(0, 3.0, 3.0)          # CC 만 반복 관측
    u1 = rm.uncertainty()
    check("관측된 칸의 불확실성은 줄어든다", u1[0] < u0[0],
          f"CC: {u0[0]:.3f} → {u1[0]:.3f}")
    check("미관측 칸의 불확실성은 유지된다", abs(u1[2] - u0[2]) < 1e-9,
          f"DC: {u0[2]:.3f} → {u1[2]:.3f} (탐색 유인의 원천)")

    # ---- 복잡도 항 ----
    sp2 = SelfPolicy(n_particles=3, w_cplx=1.0, seed=1)
    sp2.theta = {"rho": np.array([0.0, 1.0, 2.0]),
                 "omega": np.array([0.0, 0.0, 0.0]),
                 "eta": np.array([0.0, 0.0, 0.0])}
    c = sp2._complexity()
    check("복잡도는 사전 중심에서 0, 멀수록 증가",
          abs(c[0]) < 1e-12 and c[0] < c[1] < c[2],
          f"{c.round(3).tolist()}")

    # ---- α=0, β=4 고정: λ 가 동적 절편 ----
    from AIF_IPD.core.constants import empathy_shift
    lo = empathy_shift(0.0, 0.5); hi = empathy_shift(1.0, 0.5)
    check("λ 가 절편 역할 (범위가 넓다)", hi - lo > 3.0,
          f"s(0,·)={lo:+.2f} → s(1,·)={hi:+.2f}")
    check("β 는 고정 4.0", abs(SelfPolicy().beta - 4.0) < 1e-12)

    # ---- 입자 성질 ----
    sp3 = SelfPolicy(n_particles=64, seed=2)
    out = sp3.step(tft, 0.4, +1.0, +1.0, 0.8, 0.8, None)
    check("가중치 합 = 1", abs(sp3.weights.sum() - 1.0) < 1e-9)
    check("형질이 범위 내", all(
        SELF_BOUNDS[ax][0] - 1e-9 <= sp3.theta[ax].min()
        and sp3.theta[ax].max() <= SELF_BOUNDS[ax][1] + 1e-9
        for ax in SELF_AXES))
    check("ESS ∈ (0, K]", 0 < out["ess"] <= 64 + 1e-9)
    pc = sp3.coop_prob_mixture(+1.0, +1.0, 0.4, 0.8)
    check("혼합 협력확률 ∈ (0,1)", 0.0 < pc < 1.0, f"P(C)={pc:.3f}")

    # ---- 사전 복원 ----
    sp4 = SelfPolicy(n_particles=64, seed=3)
    sp4.set_prior({"rho": (1.5, 0.2), "omega": (0.0, 0.2), "eta": (0.0, 0.2)})
    check("set_prior 가 입자를 지정 위치로 재초기화",
          abs(sp4.posterior_means()["rho"] - 1.5) < 0.15,
          f"ρ̄={sp4.posterior_means()['rho']:.3f}")

    # ---- 기저 역할 스왑: f 는 상대의 직전 행동 ----
    sp5 = SelfPolicy(n_particles=1, seed=4)
    sp5.theta = {"rho": np.array([2.0]), "omega": np.array([0.0]),
                 "eta": np.array([0.0])}
    p_after_c = float(sp5._coop_prob(sp5.theta, +1.0, 0.0, 0.4, 0.8)[0])
    p_after_d = float(sp5._coop_prob(sp5.theta, -1.0, 0.0, 0.4, 0.8)[0])
    check("ρ>0 이면 상대 협력 뒤 협력, 배신 뒤 배신 (호혜)",
          p_after_c > 0.9 and p_after_d < 0.1,
          f"P(C|그들C)={p_after_c:.3f}, P(C|그들D)={p_after_d:.3f}")


# ==================================================================== 집단
def test_population() -> None:
    print("\n[8] 집단 — 조합 열거와 분해식 항등성")
    comps = enumerate_compositions(30, 6, min_each=1)
    check("조합 수 = C(29,5) = 118,755", len(comps) == 118_755,
          f"{len(comps):,}")
    check("모든 조합의 합 = 30", bool(np.all(comps.sum(axis=1) == 30)))
    check("모든 유형 최소 1명", bool(comps.min() >= 1))

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
        check(f"CC 분해식 = 전수 계산 (n={n.tolist()})",
              abs(cc_a - cc_b) < 1e-10, f"{cc_a:.10f} vs {cc_b:.10f}")
        for t in (0, 3, 5):
            mu_a = float(type_payoffs(Pi, n)[t]); mu_b = brute_mu(n, t)
            check(f"  보수 분해식 = 전수 계산 (유형 {t})",
                  abs(mu_a - mu_b) < 1e-10)

    # 치환 대비: 자기 자신으로의 치환은 0
    d = substitution_delta_cc(CCm, np.array([5, 5, 5, 5, 5, 5]), 5, 5)
    check("자기 자신으로의 치환 → Δ = 0", abs(float(d)) < 1e-12)
    # 유형이 없으면 NaN
    d2 = substitution_delta_cc(CCm, np.array([30, 0, 0, 0, 0, 0]), 5, 0)
    check("해당 유형 부재 시 치환 대비 = NaN", bool(np.isnan(float(d2))))


# ==================================================================== 진화
def test_evolution() -> None:
    print("\n[9] 진화 동역학 — RE / ORE")
    # 2유형 PD: 배신자가 협력자를 지배해야 한다 (RE 의 알려진 성질)
    Pi = np.array([[3.0, 0.0], [5.0, 1.0]])       # 행: C, D
    X0 = np.array([[0.5, 0.5], [0.9, 0.1]])
    ends = replicator_ends_batch(Pi, X0, steps=400)
    check("RE: PD 에서 배신자가 지배",
          bool(np.all(ends[:, 1] > 0.99)), f"x_D={ends[:, 1].round(4).tolist()}")

    ends_o = ore_ends_batch(Pi, X0, tau=1.0, steps=150, sweeps=20)
    check("ORE 종착 조성이 심플렉스 위에 있음",
          bool(np.allclose(ends_o.sum(axis=1), 1.0, atol=1e-6)))
    check("ORE 종착 조성이 비음",
          bool(np.all(ends_o >= -1e-12)))
    check("ORE 가 RE 보다 협력을 덜 억제 (집단수준 선택)",
          bool(np.mean(ends_o[:, 0]) >= np.mean(ends[:, 0]) - 1e-9),
          f"x_C: RE={np.mean(ends[:, 0]):.4f} → ORE={np.mean(ends_o[:, 0]):.4f}")

    # 복제자는 보수의 아핀 변환에 불변이어야 한다
    e1 = replicator_ends_batch(Pi, X0, steps=300)
    e2 = replicator_ends_batch(Pi + 10.0, X0, steps=300)
    check("RE 가 보수의 상수 이동에 불변",
          bool(np.allclose(e1, e2, atol=1e-6)))


# ==================================================================== 통계
def test_stats() -> None:
    print("\n[10] 통계 인프라")
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 60)
    b = rng.normal(0.0, 1.0, 60)
    t = perm_test(a, b, paired=False, n_perm=2000, alternative="greater")
    check("순열검정이 실제 차이를 탐지", t["p"] < 0.05, f"p={t['p']:.4g}")
    t0 = perm_test(a, a.copy(), paired=True, n_perm=2000)
    check("동일 표본 → p = 1", t0["p"] > 0.9, f"p={t0['p']:.3f}")

    ps = [0.001, 0.01, 0.04, 0.5]
    hp, bp = holm(ps), bh_fdr(ps)
    check("Holm 보정이 단조 비감소",
          all(hp[i] <= hp[i + 1] + 1e-12 for i in range(len(hp) - 1)))
    check("Holm ≥ 원래 p", all(h >= p - 1e-12 for h, p in zip(hp, ps)))
    check("BH ≤ Holm (FDR 이 FWER 보다 관대)",
          all(x <= y + 1e-12 for x, y in zip(bp, hp)))
    check("보정 p ≤ 1", all(x <= 1.0 for x in hp + bp))


# ==================================================================== 복원
def test_recovery() -> None:
    print("\n[11] 파라미터 복원 — 생성 우도의 왕복")
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
        check(f"{ax} 복원 (허용 ±{tol})",
              abs(est[ax] - truth[ax]) < tol,
              f"참={truth[ax]:+.2f}, 추정={est[ax]:+.2f}")
    check("호혜성 부호 복원", np.sign(est["rho"]) == np.sign(truth["rho"]))
    check("관성 부호 복원", np.sign(est["omega"]) == np.sign(truth["omega"]))


# ==================================================================== main
def main() -> int:
    print("=" * 70)
    print("HalloReg 핵심 기제 단위 검증")
    print("=" * 70)
    for fn in (test_constants, test_efe, test_self_model, test_distributional,
               test_core_affect, test_qrtd,
               test_empathy, test_inversion, test_agents, test_self_policy, test_population,
               test_evolution, test_stats, test_recovery):
        fn()
    print("\n" + "=" * 70)
    print(f"결과: {len(_PASS)} 통과 / {len(_PASS) + len(_FAIL)} 검사")
    if _FAIL:
        print("실패 항목:")
        for f in _FAIL:
            print(f"  - {f}")
    print("=" * 70)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
