"""
ipd.tom.self_policy
===================

**SelfPolicy — 형질공간(trait space)에서의 정책 선택.**

focal 이 매 라운드 고르는 것은 행동 a_i ∈ {C, D} 가 아니라 **자기 형질**
θ_i = (ρ, ω, η) 다. 행동은 그 형질의 우도에서 표집된다.

    P(a_i = C | h, θ_i, λ) = σ( β·( ρ·f + ω·g + η·f·g + s(λ, p_j) ) )

  · α_i ≡ 0 로 **고정**한다. α 는 무조건적 협력 편향인데, 이를 0 으로 묶으면
    "HalloReg 의 친사회적 행위에는 무조건적 성분이 없다" 는 명제가 구조에
    새겨진다. 모든 협력은 호혜적이거나 공감적이어야 한다.
    절편이 사라지는 것은 아니다 — s(λ, p) 가 λ 에 따라 절편을 움직이므로
    **λ 가 동적 절편**의 역할을 한다(기본 PD 에서 −1.5 ~ +3.5).
  · β_i ≡ 4.0 로 고정한다. 결정 정밀도는 친사회성 구성개념이 아니므로
    조절 대상에서 제외해 혼입을 막는다.

────────────────────────────────────────────────────────────────────────
왜 행동공간이 아니라 형질공간인가
────────────────────────────────────────────────────────────────────────
Albarracin et al. (2026) 이 θ 를 도입한 이유는, 지평에 따라 폭발하는 잠재상태를
Theory of Mind 로 압축하기 위해서였다. **그 압축은 자기 정책에도 똑같이 적용
되어야 한다.** 자기만 행동수준에 남겨두면 자·타 표상이 비대칭해지고, 조망수용
("상대 입장이 되어 본다")이 잘 정의되지 않는다.

계산상으로도 형질공간이 유리하다. 행동수준 계획은 지평 H 에서 2^H 분기를 타지만,
형질공간에서는 (f, g) 결합상태가 4개뿐이라 **분포를 그대로 전방전파**할 수 있다.
후보당 비용이 O(4·H) 로 분기가 사라진다.

────────────────────────────────────────────────────────────────────────
형질 EFE — 무엇을 계산하는가
────────────────────────────────────────────────────────────────────────
    G(θ_i | θ̂_j, λ, ctx) = − Σ_t γ^t [ E[u_λ]_t + w_epi · IG_j,t ]

  · **pragmatic** E[u_λ] = (1−λ)·u_self + λ·u_other 의 기댓값.
    s(λ, p) 가 이 양의 **1-step 해석해**임을 확인했다(전 조합 수치 일치):
        s(λ, p) = (1−λ)·Δ_self + λ·Δ_other,   Δ_x = E[u_x|C] − E[u_x|D]
    즉 우도의 s(λ,p) 는 이미 λ-가중 pragmatic 항을 담고 있고, rollout 은 그것의
    **다단계 확장**이다. 따라서 별도의 (1−λ)self + λother 합성을 EFE 안에서
    다시 할 필요가 없다 — λ 는 두 곳(우도의 절편, rollout 의 효용가중)에서
    일관되게 같은 것을 뜻한다.
  · **epistemic** IG_j = θ̂_j 에 대한 기대 정보이득. 이 항이 θ_i 에 의존하는
    이유는, θ_i 가 (f, g) 상태분포를 바꾸어 **어떤 관측이 실현될지**를 바꾸기
    때문이다. 그 결과 불확실할 때 상대를 분간해 주는 형질이 낮은 G 를 받는다 —
    **내생적 탐침(endogenous probing)** 이 모형 안에서 나온다.
  · 상대의 인식항(IG_other, 자기-사영 필터의 정보이득)은 **포함하지 않는다.**
    상대의 pragmatic 기여는 이미 λ 를 통해 들어와 있고, "상대가 나를 얼마나
    알게 되는가" 까지 목적함수에 넣으면 λ 의 구성개념이 흐려진다.

────────────────────────────────────────────────────────────────────────
후보 생성 — 순차 몬테카를로 (SMC)
────────────────────────────────────────────────────────────────────────
원형 격자나 θ̂_j 근방 격자는 결국 후보를 손으로 고르는 자유도다. 대신 자기
형질에 대한 **입자집합**을 유지한다 — `OpponentInversion` 과 정확히 대칭이며,
사후가 다음 라운드의 사전이 되는 구조가 자동으로 성립한다.

    1. 제안   θ^(k) ← θ^(k) + N(0, σ_prop²)        (확산 커널)
    2. 평가   G^(k) = G(θ^(k) | θ̂_j, λ, ctx)
    3. 가중   w^(k) ∝ exp(−γ · G^(k))              ← q(π) ∝ exp(−G(π))
    4. 재표집 ESS < frac·K 이면 systematic resampling
    5. 행동   P(C) = Σ_k w^(k) · σ(β(ρ^(k)f + ω^(k)g + η^(k)fg + s(λ,p)))

3단계는 능동추론의 정책 사후 q(π) ∝ exp(−G(π)) 를 입자로 근사한 것이다. 임의로
도입한 형태가 아니라 표준 정식화다. 4단계의 재표집된 집합이 곧 다음 라운드의
사전이므로 "사후를 사전으로" 가 자동이다.

MH 수용단계(min(1, exp(−γΔG)))로 바꿀 수도 있으나, 행동확률을 매끄럽게 하려면
표본 하나가 아니라 **분포**를 주변화해야 하므로 SMC 를 쓴다.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, PAYOFF_OTHER, PAYOFF_SELF, empathy_shift, joint_index,
)

_EPS = 1e-12

#: 자기 형질의 축. α 와 β 는 고정이므로 제외된다.
SELF_AXES = ("rho", "omega", "eta")

#: 축별 사전 (평균, 표준편차). 중립(0) 에서 출발한다 — 초기 호혜 성향을
#: 심어두면 "호혜성이 추론에서 나온다" 는 주장이 초기화에서 나오게 된다.
SELF_PRIOR = {"rho": (0.0, 1.0), "omega": (0.0, 0.6), "eta": (0.0, 0.6)}

#: 형질 범위. 우도가 σ(β··) 이고 β=4 이므로 |ρ| ≳ 3 이면 이미 포화한다.
SELF_BOUNDS = {"rho": (-3.0, 3.0), "omega": (-2.0, 2.0), "eta": (-2.0, 2.0)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class SelfPolicy:
    """
    자기 형질 (ρ, ω, η) 에 대한 순차 몬테카를로 정책.

    Parameters
    ----------
    n_particles : int
        입자 수 K.
    prop_sd : float
        제안 확산폭 σ_prop. 크면 태세가 불안정하고 작으면 적응이 느리다.
    gamma : float
        정책 정밀도 γ. 클수록 EFE 차이를 날카롭게 반영한다.
    horizon : int
        형질공간 rollout 지평 H.
    discount : float
        시간 할인 γ_t.
    w_epi : float
        인식항 가중. 0 이면 순수 pragmatic (절제 조건).
    beta_self : float
        행동 우도의 정밀도 β (고정).
    resample_frac : float
        ESS 가 이 비율 아래로 떨어지면 재표집.
    """

    def __init__(self, n_particles: int = 64, prop_sd: float = 0.40,
                 gamma: float = 8.0, horizon: int = 6, discount: float = 0.9,
                 w_epi_j: float = 10.0, w_epi_r: float = 1.0,
                 w_cplx: float = 0.15,
                 beta_self: float = 4.0, resample_frac: float = 0.5,
                 seed: Optional[int] = None):
        self.K = int(n_particles)
        self.prop_sd = float(prop_sd)
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.discount = float(discount)
        # 두 인식항의 가중을 **분리**한다 (v1.6.1). IG_j 는 상대 θ̂ 수렴과 함께
        # 20R 만에 실질 소멸(0.14→0.0008)하는 반면 IG_R 은 지속(0.38→0.31)하여
        # 규모가 400배까지 벌어진다. 단일 가중으로는 하나를 조율하면 다른
        # 하나가 무의미해진다.
        self.w_epi_j = float(w_epi_j)
        self.w_epi_r = float(w_epi_r)
        self.w_cplx = float(w_cplx)
        self.beta = float(beta_self)
        self.resample_frac = float(resample_frac)
        self.rng = np.random.default_rng(seed)

        self.theta: Dict[str, np.ndarray] = {}
        self._init_particles()
        self.weights = np.ones(self.K) / self.K
        self.last_G = np.zeros(self.K)

    # ============================================================ 초기화
    def _init_particles(self) -> None:
        for ax in SELF_AXES:
            mu, sd = SELF_PRIOR[ax]
            lo, hi = SELF_BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.rng.normal(mu, sd, self.K), lo, hi)

    def set_prior(self, prior: Optional[Dict[str, tuple]]) -> None:
        """
        SelfModel 이 공급한 사전으로 입자를 재초기화한다(재조우 시).
        "이 사람에게는 이런 태세로 대했었다" 가 복원된다.
        """
        if not prior:
            return
        for ax in SELF_AXES:
            if ax not in prior:
                continue
            mu, sd = prior[ax]
            lo, hi = SELF_BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.rng.normal(float(mu), max(float(sd), 1e-3), self.K),
                lo, hi)
        self.weights = np.ones(self.K) / self.K

    # ============================================================ 우도
    def _coop_prob(self, th: Dict[str, np.ndarray], f: float, g: float,
                   lam: float, p_other: float,
                   shift: Optional[float] = None) -> np.ndarray:
        """
        자기 행동 우도 P(a_i = C | f, g, θ_i, λ).

        **기저의 시제·역할 규약 (상대 모형과 뒤바뀐다는 점에 주의)**
          · f = 상대의 직전 행동 → 내가 되갚을 대상 = **호혜 신호**
          · g = 나 자신의 직전 행동 → **자기 관성**
        상대 우도에서는 f 가 '내 직전 행동' 이었다. 이 스왑을 놓치면 호혜성이
        조용히 뒤집힌다.

        p_other 는 **상대의** 협력률이다(상대 모형에서는 내 협력률이었다).
        """
        # shift 가 주어지면 그것을 쓴다(naive 모드의 학습된 ŝ). 주어지지 않으면
        # 보수행렬을 아는 것으로 보고 해석해를 쓴다(oracle 모드).
        if shift is None:
            shift = empathy_shift(lam, p_other)
        z = self.beta * (th["rho"] * f + th["omega"] * g
                         + th["eta"] * f * g + shift)
        return _sigmoid(z)

    def coop_prob_mixture(self, f: float, g: float, lam: float,
                          p_other: float,
                          shift: Optional[float] = None) -> float:
        """입자 가중 평균 협력확률 — 실제 행동 표집에 쓴다."""
        pc = self._coop_prob(self.theta, f, g, lam, p_other, shift)
        return float(np.sum(self.weights * pc))

    # ============================================================ 형질 EFE
    def _evaluate(self, th: Dict[str, np.ndarray], f: float, g: float,
                  lam: float, p_other: float, p_coop_j: float,
                  shift_i: Optional[float],
                  u_self: Optional[np.ndarray],
                  u_other: Optional[np.ndarray],
                  terminal, ig_j: Optional[np.ndarray],
                  ig_r: Optional[np.ndarray]) -> np.ndarray:
        """
        **형질 EFE — 1-step pragmatic + 종단 Z + 두 인식항.**

            G(θ_i) = −[ E[u_λ] + w_epi·(IG_j + IG_R) ] + w_cplx·KL(θ_i‖prior)

        [rollout 폐기 — v1.6.0]
        이전 판은 지평 H 까지 (f,g) 상태분포를 전방전파했다. 폐기하는 이유는
        절편에 λ-가중 수익 Z 를 쓰기로 했기 때문이다. Z 는 부트스트랩된 수익이라
        **이미 미래를 담고 있으므로**, rollout 이 다시 미래를 누적하면 같은 항이
        H+1 번 세어진다. 둘 중 하나만 남겨야 하고, Z 쪽이 (i) 학습된 양이고
        (ii) 지평을 임의로 고르지 않아도 되므로 그쪽을 택한다.

        따라서 평가는 1-step 이다: 현재 (f,g) 에서 각 행위의 즉각 효용을 상대
        행동확률로 주변화하고, 그 이후는 종단 Z 가 요약한다.

        [자·타 대칭]
        타자 효용도 자신의 학습된 기대 보상 분포 R̂_other 로 계산하며, 상대의
        행위는 OpponentInversion 이 준 협력확률 p_coop_j 로 주변화한다. 종단
        가치도 Z_other 를 함께 쓰므로 자·타의 시제가 대칭이다.

        [두 인식항]
        ig_j : 상대의 숨겨진 의도(θ̂_j)에 대한 정보이득 — 행위별 (2,)
        ig_r : **환경의 잠재 구조(R̂)에 대한 정보이득** — 행위별 (2,)
        후자가 v1.6.0 의 핵심이다. R̂ 은 결합결과로 색인되므로 내 행위가 어떤
        칸을 관측할지의 분포를 바꾼다. 겪어보지 않은 결과(예: 협력이 정착한
        관계에서의 DC)는 폭이 넓게 남아 높은 IG 를 받고, 한 번 겪으면 폭이
        줄어 IG 가 사라진다. **탐색이 사전(낙관적 초기화)이 아니라 목적함수
        에서 나온다** — 안 해본 것이 '좋다'가 아니라 '모른다'이기 때문이다.
        """
        US = PAYOFF_SELF if u_self is None else np.asarray(u_self, float)
        UO = PAYOFF_OTHER if u_other is None else np.asarray(u_other, float)
        pc_i = self._coop_prob(th, f, g, lam, p_other, shift_i)   # (K,)
        pj = float(np.clip(p_coop_j, 0.0, 1.0))

        total = np.zeros(self.K)
        # 현재 상태 인덱스 (f = 상대 직전, g = 내 직전)
        o_prev = int(round((1.0 - f) / 2.0)) if f != 0.0 else 0
        m_prev = int(round((1.0 - g) / 2.0)) if g != 0.0 else 0
        s_cur = 2 * m_prev + o_prev

        for a_i in (COOP, DEFECT):
            p_i = pc_i if a_i == COOP else (1.0 - pc_i)
            # --- 1-step pragmatic: 상대 행동으로 주변화 ---
            u = 0.0
            for a_j, p_j in ((COOP, pj), (DEFECT, 1.0 - pj)):
                idx = joint_index(a_i, a_j)
                u += p_j * ((1.0 - lam) * US[idx] + lam * UO[idx])
            # --- 종단 가치: 이후는 Z 가 요약 (자·타 대칭) ---
            if terminal is not None:
                s_next = 2 * a_i + (COOP if pj >= 0.5 else DEFECT)
                u += self.discount * terminal(s_next, pj, lam)
            # --- 인식항 두 갈래 ---
            epi = 0.0
            if ig_j is not None:
                epi += self.w_epi_j * float(ig_j[a_i])
            if ig_r is not None:
                epi += self.w_epi_r * float(ig_r[a_i])
            total = total + p_i * (u + epi)
        return total

    def step(self, theta_j: Dict[str, float], lam: float,
             f: float, g: float, p_other: float, p_self: float,
             ig_j: Optional[np.ndarray] = None,
             u_self: Optional[np.ndarray] = None,
             u_other: Optional[np.ndarray] = None,
             terminal=None, shift_i: Optional[float] = None,
             p_coop_j: float = 0.5,
             ig_r: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        한 라운드의 SMC 갱신: 제안 → 평가 → 가중 → 재표집.

        반환: 사후 평균 형질과 진단값.
        """
        # --- 1. 제안 (확산 커널) ---
        for ax in SELF_AXES:
            lo, hi = SELF_BOUNDS[ax]
            self.theta[ax] = np.clip(
                self.theta[ax] + self.rng.normal(0.0, self.prop_sd, self.K),
                lo, hi)

        # --- 2. 평가 ---
        value = self._evaluate(self.theta, f, g, lam, p_other, p_coop_j,
                               shift_i, u_self, u_other, terminal,
                               ig_j, ig_r)
        G = -value + self.w_cplx * self._complexity()
        self.last_G = G

        # --- 3. 가중: q(π) ∝ exp(−γ G) ---
        logw = -self.gamma * G
        logw -= logw.max()
        w = np.exp(logw)
        s = float(w.sum())
        self.weights = (w / s) if s > _EPS else np.ones(self.K) / self.K

        # --- 4. 재표집 ---
        ess = 1.0 / float(np.sum(self.weights ** 2))
        if ess < self.resample_frac * self.K:
            self._resample()

        out = {ax: float(np.sum(self.weights * self.theta[ax]))
               for ax in SELF_AXES}
        out["ess"] = ess
        out["G_mean"] = float(np.sum(self.weights * G))
        return out

    def _complexity(self) -> np.ndarray:
        """
        **복잡도 항** — 사전으로부터의 KL 비용 (가우시안 사전에서 마할라노비스²/2).

            C(θ_i) = Σ_ax (θ_ax − μ_ax)² / (2 σ_ax²)

        변분자유에너지의 표준 구성요소이며, 여기서는 필수적이다. 이것이 없으면
        **증거 없이 형질이 표류**한다. 구체적으로: ALLD 처럼 행동이 상수인 상대
        앞에서는 f 가 항상 −1 로 고정되어 ρ·f = −ρ 가 사실상 절편이 된다. 즉
        ρ 와 절편이 완전 공선이 되어 ρ 가 '호혜성' 이 아니라 **배신 장치**로
        선택된다(실측: 복잡도 항 없이 ρ(ALLD) > ρ(TFT), 해리 −0.22).

        복잡도 항을 넣으면 호혜성이 실제로 값을 만들 때만(상대가 내 행동에
        반응할 때만) 그 비용을 상쇄하고 살아남는다. 따라서 이 항은 계산상의
        정칙화가 아니라 **"호혜는 반응하는 상대에게만 의미가 있다"** 는 구성
        개념을 강제하는 장치다.
        """
        c = np.zeros(self.K)
        for ax in SELF_AXES:
            mu, sd = SELF_PRIOR[ax]
            c = c + (self.theta[ax] - mu) ** 2 / (2.0 * max(sd, 1e-6) ** 2)
        return c

    def _resample(self) -> None:
        """Systematic resampling. 재표집 후 가중치는 균등으로 되돌린다."""
        pos = (self.rng.random() + np.arange(self.K)) / self.K
        idx = np.searchsorted(np.cumsum(self.weights), pos)
        idx = np.clip(idx, 0, self.K - 1)
        for ax in SELF_AXES:
            self.theta[ax] = self.theta[ax][idx]
        self.weights = np.ones(self.K) / self.K

    # ============================================================ 진단
    def posterior_means(self) -> Dict[str, float]:
        return {ax: float(np.sum(self.weights * self.theta[ax]))
                for ax in SELF_AXES}

    def posterior_stds(self) -> Dict[str, float]:
        out = {}
        for ax in SELF_AXES:
            mu = float(np.sum(self.weights * self.theta[ax]))
            var = float(np.sum(self.weights * (self.theta[ax] - mu) ** 2))
            out[ax] = float(np.sqrt(max(var, 0.0)))
        return out
