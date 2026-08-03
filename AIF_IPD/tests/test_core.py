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
    print("\n[3] SelfModel — 기억·사전·설정점")
    reset_payoffs()
    sm = SelfModel()
    # 신규 identity 는 무정보 사전
    pr = sm.theta_prior(None)
    check("신규 identity → 무정보 θ 사전",
          abs(pr["lambda_j"][0] - 0.5) < 1e-9)
    # 무정보 사회 기저에서 λ_0 = (floor + ceil)/2
    lam0 = sm.lambda_setpoint(PAYOFF_SELF)
    check("무정보 사회 기저 → λ_0 = 구간 중점",
          abs(lam0 - 0.5 * (sm.lam_floor + sm.lam_ceil)) < 1e-6,
          f"λ_0={lam0:.4f}")

    # 기억 commit 후 재조우 시 사전이 좁아지는가
    sm.commit_theta(7, {"alpha": 1.5, "rho": 0.2, "omega": 0.0, "eta": 0.0,
                        "beta": 4.0, "lambda_j": 0.8},
                    {"alpha": 0.3, "rho": 0.3, "omega": 0.3, "eta": 0.3,
                     "beta": 0.3, "lambda_j": 0.05})
    pr2 = sm.theta_prior(7)
    check("재조우 시 사전 평균이 기억을 반영",
          abs(pr2["alpha"][0] - 1.5) < 1e-9)
    check("재조우 사전의 폭이 무정보보다 좁음",
          pr2["alpha"][1] < pr["alpha"][1],
          f"{pr2['alpha'][1]:.3f} < {pr['alpha'][1]:.3f}")
    check("사전 폭에 하한이 걸림 (점질량 방지)",
          pr2["lambda_j"][1] >= 0.25 * sm.theta_prior_std["lambda_j"] - 1e-12)

    # 설정점의 방향성: 좋은 결과만 누적하면 λ_0 상승
    sm2 = SelfModel()
    for _ in range(200):
        sm2.commit_reward(1, np.ones(4), observed_state=CC)
    check("상호협력만 경험 → 설정점 상승 → λ_0 상승",
          sm2.lambda_setpoint(PAYOFF_SELF) > lam0)
    sm3 = SelfModel()
    for _ in range(200):
        sm3.commit_reward(1, np.ones(4), observed_state=CD)
    check("착취만 경험 → 설정점 하강 → λ_0 하강",
          sm3.lambda_setpoint(PAYOFF_SELF) < lam0)


# ==================================================================== CoreAffect
def test_core_affect() -> None:
    print("\n[4] CoreAffect — valence(RPE) × arousal(KL)")
    reset_payoffs()
    sm = SelfModel()
    ca = CoreAffect(sm, PAYOFF_SELF)
    ca.begin_partner(1)

    out_good = ca.step(CC)          # 최고 보수 R=3 > 기저 평균 2.25
    check("좋은 결과 → RPE > 0 그리고 valence > 0",
          out_good["rpe"] > 0 and out_good["valence"] > 0,
          f"RPE={out_good['rpe']:+.3f}, V={out_good['valence']:+.3f}")

    sm2 = SelfModel()
    ca2 = CoreAffect(sm2, PAYOFF_SELF)
    ca2.begin_partner(1)
    out_bad = ca2.step(CD)          # 최저 보수 S=0 < 기저 평균
    check("나쁜 결과 → RPE < 0 그리고 valence < 0",
          out_bad["rpe"] < 0 and out_bad["valence"] < 0,
          f"RPE={out_bad['rpe']:+.3f}, V={out_bad['valence']:+.3f}")

    check("valence 부호 = RPE 부호 (정의상 항등)",
          np.sign(out_good["rpe"]) == np.sign(out_good["valence"])
          and np.sign(out_bad["rpe"]) == np.sign(out_bad["valence"]))
    check("valence ∈ (−1, +1)", abs(out_good["valence"]) < 1.0)
    check("arousal ∈ [0, 1)",
          0.0 <= out_good["arousal"] < 1.0 and 0.0 <= out_bad["arousal"] < 1.0)
    check("λ_aff = valence × arousal",
          abs(out_good["lambda_aff"]
              - out_good["valence"] * out_good["arousal"]) < 1e-12)

    # 각성의 감쇠: 같은 관측을 반복하면 믿음 갱신(KL)이 줄어 arousal 이 하강
    sm3 = SelfModel()
    ca3 = CoreAffect(sm3, PAYOFF_SELF)
    ca3.begin_partner(1)
    aro = [ca3.step(CC)["arousal"] for _ in range(20)]
    check("반복 관측 → arousal 단조 감쇠 (놀람 소멸)",
          all(aro[i] >= aro[i + 1] - 1e-9 for i in range(len(aro) - 1)),
          f"{aro[0]:.4f} → {aro[-1]:.4f}")


# ==================================================================== Empathy
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
    new = em.step(lambda_aff=0.6, lambda_ctx=0.4)
    expected = prev + 0.05 * (0.5 * 0.6 + 0.5 * 0.4)
    check("갱신식 수치 일치", abs(new - expected) < 1e-12,
          f"{new:.6f} vs {expected:.6f}")

    # 클리핑
    em2 = Empathy(lam_init=0.99, w_cd=0.0, gain=1.0)
    check("λ 상한 클리핑", em2.step(10.0, 0.0) <= 1.0)
    em3 = Empathy(lam_init=0.01, w_cd=0.0, gain=1.0)
    check("λ 하한 클리핑", em3.step(-10.0, 0.0) >= 0.0)

    # 채널 가중
    ea = Empathy(lam_init=0.5, w_cd=0.0, gain=0.1)     # 순수 정서 구동
    ec = Empathy(lam_init=0.5, w_cd=1.0, gain=0.1)     # 순수 맥락 구동
    check("w_cd=0 이면 λ_ctx 무시",
          abs(ea.step(0.0, 1.0) - 0.5) < 1e-12)
    check("w_cd=1 이면 λ_aff 무시",
          abs(ec.step(1.0, 0.0) - 0.5) < 1e-12)


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
    for fn in (test_constants, test_efe, test_self_model, test_core_affect,
               test_empathy, test_inversion, test_agents, test_population,
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
