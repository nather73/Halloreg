"""
core.core_affect
================

**CoreAffect — 2차원 핵심정서(core affect): valence × arousal.**

Russell 의 핵심정서 원환과 Barrett 의 구성된 정서 이론(theory of constructed
emotion)을 따라, 정서를 범주가 아니라 **내수용 예측부호화의 두 축**으로 구성한다.

  · valence (쾌–불쾌)  ← 보상 예측오차(RPE)의 부호와 크기
  · arousal (각성)     ← 믿음 갱신의 크기(베이지안 놀람, Bayesian surprise)

[본 모듈이 구현하는 정확한 사양]

  (1) SelfModel 은 identity 별 기대보상 분포를 사전으로 내려준다.
      CoreAffect 는 t−1 관측으로 이를 갱신하고 SelfModel 에 되돌려준다.

  (2) valence — SelfModel 이 추가로 공급하는 **기저(사회 전반) 기대보상 분포**를
      기준으로 한 예측오차:

          RPE = r_obs − E_{q_social}[r]

      RPE 의 기댓값이 클수록(양수) 긍정적, 작을수록(음수) 부정적 valence.
      스케일은 기저 분포의 보상 표준편차로 정규화한다(정밀도 가중):

          valence = tanh( RPE / (σ_social + σ_floor) )   ∈ (−1, +1)

      정규화의 근거: 사회 환경 자체가 변동이 큰(σ 큰) 곳이라면 같은 크기의
      RPE 라도 덜 놀랍다. 예측부호화에서 예측오차는 항상 기대 정밀도로 가중된다.

  (3) arousal — **이번 상대 identity** 의 기대보상 분포가 이번 관측으로 얼마나
      움직였는가:

          arousal = 1 − exp( − KL( q_post ‖ q_prior ) / κ )   ∈ [0, 1)

      q_prior : 갱신 이전 사후예측 categorical
      q_post  : 갱신 이후 사후예측 categorical
      KL 이 클수록 각성이 높다. 포화 사상(1 − e^{−x})을 쓰는 이유는 KL 이
      무계인 반면 각성은 유계여야 λ_aff = V × A 가 λ 와 같은 스케일에 머물기
      때문이다.

  (4) λ_aff = valence × arousal   (정서적 동기, affective motivation)

      곱셈 결합의 의미:
        · 각성이 0 이면(예측대로였다) 정서는 λ 를 움직이지 않는다 — 놀람이 없으면
          재조정할 이유가 없다.
        · 각성이 높고 valence 가 음수면 λ_aff ≪ 0 → 강한 자기보호 방향.
        · 각성이 높고 valence 가 양수면 λ_aff ≫ 0 → 관계 투자 방향.
      이것이 곧 이상성(allostatic) 예측: **예측된 손실은 정서를 만들지 않고,
      예측되지 않은 손실만 정서를 만든다.**

[중요 — 이상성이 항상성으로 붕괴하지 않는 이유]
기저 분포는 SelfModel 에서 social_lr ≪ 1 로 느리게만 이동한다. 따라서 착취자를
계속 만나더라도 RPE 가 즉시 0 이 되지 않고, 기저가 이동하는 만큼만 서서히
둔감해진다. 이는 "만성 스트레스 하에서 설정점이 이동한다"는 이상성의 예측과
일치하며, 순수 항상성(즉시 순응) 모형이 갖는 역설 — 착취자를 정확히 예측하면
자기보호 압력이 사라진다 — 을 회피한다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .self_model import SelfModel

_EPS = 1e-12


def _dirichlet_predictive(alpha: np.ndarray) -> np.ndarray:
    """Dirichlet 농도 → 사후예측 categorical 확률벡터."""
    a = np.asarray(alpha, dtype=float)
    return a / max(a.sum(), _EPS)


def _kl_categorical(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q). 두 categorical 모두 양수 지지(Dirichlet 유래)라 안전."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=float), _EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


class CoreAffect:
    """
    핵심정서 생성기.

    Parameters
    ----------
    self_model : SelfModel
        사전 공급자이자 기억 저장소. 매 스텝 (i) 기저 기대보상 분포와
        (ii) identity 별 보상분포 사전을 받아오고, 갱신 결과를 되돌려준다.
    payoffs : (4,) array
        현재 보수 벡터 PAYOFF_SELF (가변 보수 환경에서는 매 라운드 갱신).
    kl_scale : float
        arousal 포화 상수 κ. 작을수록 작은 믿음갱신에도 쉽게 각성한다.
    sigma_floor : float
        valence 정규화의 분모 하한. 기저 분포가 한 범주로 붕괴해 σ→0 이 될 때
        tanh 인자가 발산하는 것을 막는다.
    obs_weight : float
        identity 별 Dirichlet 이 관측 하나로부터 받는 가중(기본 1.0 = 표준 계수).
    """

    def __init__(self, self_model: SelfModel,
                 payoffs: np.ndarray,
                 kl_scale: float = 0.05,
                 sigma_floor: float = 0.5,
                 obs_weight: float = 1.0):
        self.self_model = self_model
        self.payoffs = np.asarray(payoffs, dtype=float).copy()
        self.kl_scale = float(kl_scale)
        self.sigma_floor = float(sigma_floor)
        self.obs_weight = float(obs_weight)

        # 현재 상대의 기대보상 분포(Dirichlet 농도). begin_partner 에서 주입된다.
        self.identity: Optional[int] = None
        self.alpha = self_model.reward_prior(None)

        # 최근 스텝 진단값 (로깅용)
        self.last = {"valence": 0.0, "arousal": 0.0, "lambda_aff": 0.0,
                     "rpe": 0.0, "kl": 0.0}

    # ------------------------------------------------------------ 보수 갱신
    def set_payoffs(self, payoffs: np.ndarray) -> None:
        """
        가변 보수 환경 정합. 범주(4 joint outcome)는 불변이고 각 범주에 붙는
        **값**만 바뀌므로, Dirichlet 믿음은 유지한 채 값 벡터만 교체하면 된다.
        """
        self.payoffs = np.asarray(payoffs, dtype=float).copy()

    # ------------------------------------------------------------ 상대 전환
    def begin_partner(self, identity: Optional[int]) -> None:
        """
        새 상대와의 상호작용 시작. SelfModel 에서 그 identity 의 기대보상 분포
        사전을 받아 현재 상태로 삼는다(재조우면 과거 기억, 신규면 무정보 사전).
        """
        self.identity = identity
        self.alpha = self.self_model.reward_prior(identity)

    # ------------------------------------------------------------ 한 스텝
    def step(self, observed_state: int) -> dict:
        """
        t−1 의 joint outcome 관측으로 정서를 구성한다.

        절차
        ----
        1. q_prior  ← 갱신 이전 identity 별 사후예측 분포
        2. α ← α + obs_weight · onehot(observed_state)      (Dirichlet 갱신)
        3. q_post   ← 갱신 이후 사후예측 분포
        4. RPE      = r_obs − E_{q_social}[r]                (기저는 SelfModel)
        5. valence  = tanh(RPE / (σ_social + σ_floor))
        6. arousal  = 1 − exp(−KL(q_post‖q_prior) / κ)
        7. λ_aff    = valence × arousal
        8. 갱신 결과를 SelfModel 에 commit (identity 기억 + 사회 기저 느린 갱신)

        반환: 진단 dict.
        """
        s = int(observed_state)

        # --- 1. 갱신 이전 사후예측 (arousal 의 기준점) ---
        q_prior = _dirichlet_predictive(self.alpha)

        # --- 2. Dirichlet 켤레 갱신 ---
        self.alpha = self.alpha.copy()
        self.alpha[s] += self.obs_weight

        # --- 3. 갱신 이후 사후예측 ---
        q_post = _dirichlet_predictive(self.alpha)

        # --- 4. 기저(사회 전반) 기대보상 대비 RPE ---
        social = self.self_model.social_reward_prior()
        r_base = SelfModel.expected_reward(social, self.payoffs)
        sigma_base = SelfModel.reward_std(social, self.payoffs)
        r_obs = float(self.payoffs[s])
        rpe = r_obs - r_base

        # --- 5. valence: 정밀도 가중된 RPE 를 유계로 압착 ---
        valence = float(np.tanh(rpe / (sigma_base + self.sigma_floor)))

        # --- 6. arousal: 베이지안 놀람의 포화 사상 ---
        kl = _kl_categorical(q_post, q_prior)
        arousal = float(1.0 - np.exp(-kl / max(self.kl_scale, _EPS)))

        # --- 7. 정서적 동기 ---
        lambda_aff = valence * arousal

        # --- 8. 기억 commit (identity 사후 + 사회 기저의 느린 이동) ---
        self.self_model.commit_reward(self.identity, self.alpha,
                                      observed_state=s)

        self.last = {"valence": valence, "arousal": arousal,
                     "lambda_aff": lambda_aff, "rpe": float(rpe),
                     "kl": float(kl), "r_base": float(r_base),
                     "r_obs": r_obs}
        return dict(self.last)

    # ------------------------------------------------------------ 진단
    def expected_reward(self) -> float:
        """현재 상대에 대한 기대보상 E[r] (진단·시각화용)."""
        return SelfModel.expected_reward(self.alpha, self.payoffs)
