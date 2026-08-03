"""
core.self_model
===============

**SelfModel — 기억(memory) 저장소이자 사전(prior) 공급자.**

[역할 재정의 — v1.0.0 아키텍처]
이전 판에서 SelfModel 은 λ 의 항상성 설정점(λ_sp)을 직접 산출했다. 본 판에서는
그 역할을 완전히 폐기하고, 다음 두 가지로만 남긴다.

  (1) **기억**: 상대 identity 별로 (id, dist, theta, expected reward distribution)
      을 보관한다. 여기서
        · theta : 상대 특성 파라미터의 사후 평균 (α, ρ, ω, η, β, λ_j)
        · dist  : 그 사후의 축별 표준편차 — 즉 "θ 에 대한 믿음의 분포 폭"
        · expected reward distribution : 그 상대와의 상호작용에서 기대되는
          보상의 분포. 4-범주(CC/CD/DC/DD) Dirichlet 농도로 표현한다.
  (2) **사전 공급**: 다음 라운드/다음 조우 시
        · OpponentInversion 에 (id, theta) 를 θ 사전으로,
        · CoreAffect 에 (id, expected reward distribution) 을 보상분포 사전으로
      각각 내려보낸다. 두 모듈은 t−1 관측으로 갱신한 결과를 다시 SelfModel 에
      commit 한다. 즉 SelfModel 자체는 **추론하지 않는다.**

[할로스타틱 설정점(allostatic setpoint)의 재정의]
설정점은 더 이상 λ 가 아니다. SelfModel 은 자기가 접해온 **사회적 환경 전반**에
대한 믿음으로부터 *기저 기대보상 분포* q_social(r) 를 유지하고, 이를 CoreAffect 에
추가로 공급한다. CoreAffect 의 valence 는 이 기저 분포를 기준으로 한 보상예측오차
(RPE)로 생성된다. 즉 설정점은 "정서가(valence)의 영점"이다.

기저 분포는 **느리게** 갱신된다(social_lr ≪ 1). 이것이 항상성이 아니라 이상성
(allostasis)인 이유다: 개체는 국소 사건에 즉각 순응하지 않고, 누적된 사회적
기대를 기준으로 현재를 평가하되, 그 기준 자체도 장기적으로 이동한다.

[왜 Dirichlet–Categorical 인가]
"기대 보상 분포"는 보상 값 위의 분포다. IPD 에서 보상은 항상 4개 joint outcome
(CC/CD/DC/DD)에 대응하는 4개 값 중 하나이므로, 보상 분포는 4-범주 categorical 로
**정확히** 표현된다. 켤레사전인 Dirichlet 을 쓰면
  · 사후 예측분포가 닫힌 형태(농도 정규화)로 나오고,
  · CoreAffect 가 요구하는 갱신 전/후 KL-divergence 가 해석적으로 계산되며,
  · 가변 보수 환경에서 범주는 불변인 채 각 범주에 붙는 **값**만 변하므로
    비정상 보수 구조를 자연스럽게 흡수한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

_EPS = 1e-12

# 상대 특성 θ 의 축 순서. inversion 모듈의 fg 기저와 일치시킨다.
THETA_AXES = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")


@dataclass
class MemoryEntry:
    """상대 identity 하나에 대한 기억 항목."""
    identity: int
    # theta : 축별 사후 평균 (상대 특성의 점추정)
    theta: Dict[str, float] = field(default_factory=dict)
    # dist : 축별 사후 표준편차 (그 점추정을 얼마나 믿는지 = 믿음의 폭)
    dist: Dict[str, float] = field(default_factory=dict)
    # reward : 기대 보상 분포의 Dirichlet 농도 (4,) — 순서 [CC, CD, DC, DD]
    reward: np.ndarray = field(default_factory=lambda: np.ones(4))
    # 이 상대와 실제로 상호작용한 라운드 수
    n_obs: int = 0


class SelfModel:
    """
    identity 기반 기억 + 사회적 기저분포 보유자.

    Parameters
    ----------
    theta_prior_mean, theta_prior_std : dict
        신규 identity 에 대한 θ 사전(무정보 사전). OpponentInversion 의 기본
        사전과 동일한 값을 쓰며, 재조우 시에는 저장된 기억이 이를 대체한다.
    dirichlet_prior : float
        신규 identity 의 보상분포 Dirichlet 농도 초깃값(범주당). 값이 클수록
        새 관측 하나가 분포를 덜 움직인다(= 강한 사전).
    social_prior_strength : float
        사회 전반 기저분포의 초기 농도 총합. 크면 기저(설정점)가 더 느리게 이동.
    social_lr : float
        관측 하나가 사회 기저분포에 기여하는 가중. ≪1 이어야 기저가 "느린 시간
        척도"로 움직인다(이상성). 1.0 이면 identity 별 갱신과 같은 속도가 되어
        RPE 가 즉시 소멸해버린다.
    lam_floor, lam_ceil : float
        λ 초깃값(λ_{t=0})의 하한/상한. 기저 기대보상이 낮을수록 λ_floor 에,
        높을수록 λ_ceil 에 접근한다.
    """

    def __init__(self,
                 theta_prior_mean: Optional[Dict[str, float]] = None,
                 theta_prior_std: Optional[Dict[str, float]] = None,
                 dirichlet_prior: float = 1.0,
                 social_prior_strength: float = 8.0,
                 social_lr: float = 0.06,
                 lam_floor: float = 0.10,
                 lam_ceil: float = 0.70):
        # ---- θ 무정보 사전 (신규 상대) ----
        self.theta_prior_mean = dict(
            alpha=0.0, rho=0.5, omega=0.0, eta=0.0, beta=3.0, lambda_j=0.5)
        self.theta_prior_std = dict(
            alpha=2.0, rho=1.2, omega=1.0, eta=1.0, beta=1.8, lambda_j=0.3)
        if theta_prior_mean:
            self.theta_prior_mean.update(theta_prior_mean)
        if theta_prior_std:
            self.theta_prior_std.update(theta_prior_std)

        self.dirichlet_prior = float(dirichlet_prior)
        self.social_lr = float(social_lr)
        self.lam_floor = float(lam_floor)
        self.lam_ceil = float(lam_ceil)

        # ---- 사회적 기저 기대보상 분포 (할로스타틱 설정점) ----
        # 초기에는 4 결과가 동등하게 가능하다는 무정보 믿음.
        self.social_reward = np.full(4, social_prior_strength / 4.0)

        # ---- identity → MemoryEntry ----
        self.memory: Dict[int, MemoryEntry] = {}

    # ============================================================ 사전 공급
    def theta_prior(self, identity: Optional[int]) -> Dict[str, tuple]:
        """
        OpponentInversion 에 내려보낼 (id, theta) 사전.

        반환: {축: (평균, 표준편차)}.
        재조우(identity 가 기억에 있음)면 저장된 (theta, dist) 를, 신규면 무정보
        사전을 돌려준다. 재조우 시 사전이 좁아지므로 입자필터가 즉시 과거 관계
        지점에서 출발한다 — "아, 그때 그 사람" 에 해당하는 조기 보정.
        """
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or not ent.theta:
            return {ax: (self.theta_prior_mean[ax], self.theta_prior_std[ax])
                    for ax in THETA_AXES}
        out = {}
        for ax in THETA_AXES:
            mu = float(ent.theta.get(ax, self.theta_prior_mean[ax]))
            # 기억된 사후 폭을 그대로 쓰되, 완전 점질량이 되지 않도록 하한을 둔다.
            # 하한이 없으면 상대가 변했을 때 입자필터가 새 가설을 만들지 못한다.
            sd_floor = 0.25 * self.theta_prior_std[ax]
            sd = max(float(ent.dist.get(ax, self.theta_prior_std[ax])), sd_floor)
            out[ax] = (mu, sd)
        return out

    def reward_prior(self, identity: Optional[int]) -> np.ndarray:
        """
        CoreAffect 에 내려보낼 (id, expected reward distribution) 사전.
        반환: (4,) Dirichlet 농도. 신규 identity 면 균등 사전.
        """
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None:
            return np.full(4, self.dirichlet_prior)
        return ent.reward.copy()

    def social_reward_prior(self) -> np.ndarray:
        """
        **기저(baseline) 기대보상 분포** — 할로스타틱 설정점.
        주변 사회적 환경 전반에 대한 믿음이며, CoreAffect 의 valence 영점이 된다.
        반환: (4,) Dirichlet 농도.
        """
        return self.social_reward.copy()

    # ============================================================ 기억 갱신
    def commit_theta(self, identity: Optional[int],
                     theta: Dict[str, float], dist: Dict[str, float]) -> None:
        """OpponentInversion 이 갱신한 (theta, dist) 를 기억에 반영."""
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(
            identity=identity, reward=np.full(4, self.dirichlet_prior)))
        ent.theta = {ax: float(theta.get(ax, 0.0)) for ax in THETA_AXES}
        ent.dist = {ax: float(dist.get(ax, 0.0)) for ax in THETA_AXES}

    def commit_reward(self, identity: Optional[int],
                      reward_dirichlet: np.ndarray,
                      observed_state: Optional[int] = None) -> None:
        """
        CoreAffect 가 갱신한 기대보상 분포를 기억에 반영하고, 동시에 **사회적
        기저분포**를 느린 학습률로 갱신한다.

        사회 기저는 identity 별 사후 전체가 아니라 *이번 관측 하나*만 social_lr
        가중으로 흡수한다. 이렇게 해야
          · 한 상대에게 오래 노출되어도 기저가 그 상대로 완전히 수렴하지 않고,
          · 여러 상대를 겪을수록 기저가 사회 전반의 평균으로 이동한다.
        """
        if identity is not None:
            ent = self.memory.setdefault(identity, MemoryEntry(
                identity=identity, reward=np.full(4, self.dirichlet_prior)))
            ent.reward = np.asarray(reward_dirichlet, dtype=float).copy()
            ent.n_obs += 1
        if observed_state is not None:
            self.social_reward[int(observed_state)] += self.social_lr

    def observe_identity(self, identity: int) -> None:
        """상대 identity 를 관측했다는 통지(신규면 빈 항목 생성)."""
        self.memory.setdefault(identity, MemoryEntry(
            identity=int(identity), reward=np.full(4, self.dirichlet_prior)))

    # ============================================================ 설정점 유도
    @staticmethod
    def expected_reward(dirichlet: np.ndarray, payoffs: np.ndarray) -> float:
        """Dirichlet 농도 → 사후예측 categorical → 기대보상 E[r]."""
        a = np.asarray(dirichlet, dtype=float)
        p = a / max(a.sum(), _EPS)
        return float(p @ np.asarray(payoffs, dtype=float))

    @staticmethod
    def reward_std(dirichlet: np.ndarray, payoffs: np.ndarray) -> float:
        """사후예측 하의 보상 표준편차 — RPE 의 정밀도 가중(스케일)에 쓴다."""
        a = np.asarray(dirichlet, dtype=float)
        p = a / max(a.sum(), _EPS)
        u = np.asarray(payoffs, dtype=float)
        m = float(p @ u)
        return float(np.sqrt(max(float(p @ (u - m) ** 2), 0.0)))

    def lambda_setpoint(self, payoffs: np.ndarray) -> float:
        """
        **λ_{t=0} — Empathy 모듈의 초깃값.**

        폐기된 λ_sp 를 대체한다. 사회적 기저 기대보상 E_social[r] 이
          · 보상 지지집합의 중간값보다 높으면 → 우호적 사회 환경 → 높은 초기 공감
          · 낮으면 → 적대적 사회 환경 → 낮은 초기 공감
        이 되도록 로지스틱으로 사상한다.

            λ_0 = floor + (ceil − floor) · σ( (E_social[r] − r_mid) / scale )

        r_mid 는 보상 지지집합의 산술평균(무정보 기준점), scale 은 지지집합의
        표준편차. 무정보 사전(균등 Dirichlet)에서는 E_social[r] = r_mid 이므로
        σ(0) = 0.5 → λ_0 = (floor + ceil)/2 = 0.40 (기본값에서). 즉 기억이 없는
        개체는 중립적 공감에서 출발한다.
        """
        u = np.asarray(payoffs, dtype=float)
        r_mid = float(np.mean(u))
        scale = max(float(np.std(u)), 1e-3)
        e_soc = self.expected_reward(self.social_reward, u)
        sig = 1.0 / (1.0 + np.exp(-(e_soc - r_mid) / scale))
        return float(self.lam_floor + (self.lam_ceil - self.lam_floor) * sig)

    # ============================================================ 진단
    def snapshot(self, payoffs: np.ndarray) -> dict:
        """로깅/시각화용 상태 요약."""
        return {
            "n_identities": len(self.memory),
            "social_expected_reward": self.expected_reward(
                self.social_reward, payoffs),
            "social_strength": float(self.social_reward.sum()),
            "lambda_setpoint": self.lambda_setpoint(payoffs),
        }
