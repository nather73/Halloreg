"""
core.core_affect
================

**CoreAffect — 2차원 핵심정서: valence × arousal.**

전제: 일차적 보상 획득이 내부 모델에 대한 항상성으로 이어지며, **기대 보상
분포는 내수용감각 인식에 대한 생성 모델로 단순화**되어 간주된다. CoreAffect 는
그 생성 모델과 관측 사이의 관계를 저차원 정동 두 축으로 축소한다.

────────────────────────────────────────────────────────────────────────
valence — 사회적 모집단에서의 적합도 (수준 신호)
────────────────────────────────────────────────────────────────────────
    valence = 2 · F̂_ref( median(Q_현재상대) ) − 1

Q_현재상대 : 현재 상대에 대한 기대 보상(가치) 분포 — QRTD 의 Z(s,a) 를 경험
상황에 대해 EMA 축약한 분위수 벡터 (SelfModel.update_partner_value).
F̂_ref     : 자기경계 안 **다른 타인들**의 기대 보상 분포를 거리반비례 가중
평균한 참조분포(Wasserstein 무게중심)에서의 누적확률.

valence 는 **예측 오류가 아니다.** 현재의 생성 모델(기대 보상 분포)이 SelfModel
이 지닌 사회적 환경에 대한 믿음 안에서 좋은 방향에 있는지 나쁜 방향에 있는지의
**주관적 적합도**다. ALLD 를 상대하면 그에 대한 기대 보상 분포의 기댓값이 기억
속 다른 이들의 것보다 낮게 형성되고, 그것이 부정적 valence 다.

참조가 시간이 아니라 **타자들**에게 있으므로 만성 착취에서도 소멸하지 않는다
(시간 자기비교는 참조가 쫓아와 valence 가 소멸·역전됨을 실측으로 확인했다).
현재 상대는 모집단에서 제외한다.

────────────────────────────────────────────────────────────────────────
arousal — 생성 모델의 갱신량 (오차 신호)
────────────────────────────────────────────────────────────────────────
    arousal = 1 − exp( − W₁(Z_{t−1}, Z_t) / span_V / κ )

t−1 의 기대 보상 분포와 t 의 기대 보상 분포의 불일치 — **모형 갱신량**이다.
놀람의 대상은 즉각 보상이 아니라 상황의 가치이며, 오랜 협력 뒤의 한 번의
배신은 보상 기대를 조금 바꾸지만 전망(Z)을 크게 무너뜨린다.

두 축의 분업: valence 는 **수준**(지속적 — 관계의 질), arousal 은 **오차**
(자기소멸적 — 예측된 것은 놀랍지 않다). λ_aff = valence × arousal 이므로
예측되지 않은 사건만이, 그 관계의 질의 방향으로, λ 를 움직인다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .self_model import SelfModel

_EPS = 1e-12


class CoreAffect:
    """
    핵심정서 생성기 (v1.5.0 — QRTD 기반).

    Parameters
    ----------
    self_model : SelfModel
        기억 저장소. 현재 상대의 가치분포 갱신과 모집단 참조를 담당한다.
    payoffs : (4,) array
        현재 보수 벡터 (정규화 상수 공급).
    kl_scale : float
        arousal 포화 상수 κ.
    """

    def __init__(self, self_model: SelfModel, payoffs: np.ndarray,
                 kl_scale: float = 0.05, sigma_floor: float = 0.5,
                 obs_weight: float = 1.0):
        self.self_model = self_model
        self.payoffs = np.asarray(payoffs, dtype=float).copy()
        self.kl_scale = float(kl_scale)
        self.sigma_floor = float(sigma_floor)   # 서명 호환용 (미사용)
        self.obs_weight = float(obs_weight)     # 서명 호환용 (미사용)
        self.self_model.set_payoff_scale(self.payoffs)
        self.identity: Optional[int] = None
        self.last = {"valence": 0.0, "arousal": 0.0, "lambda_aff": 0.0,
                     "rpe": 0.0, "surprise": 0.0, "value": float("nan")}

    def set_payoffs(self, payoffs: np.ndarray) -> None:
        """비정상 보수 정합 — 정규화 상수 갱신."""
        self.payoffs = np.asarray(payoffs, dtype=float).copy()
        self.self_model.set_payoff_scale(self.payoffs)

    def begin_partner(self, identity: Optional[int]) -> None:
        self.identity = identity

    # ------------------------------------------------------------ 한 스텝
    def step(self, observed_state: int,
             opponent_cooperated: Optional[bool] = None,
             value_vector: Optional[np.ndarray] = None,
             value_shift: Optional[float] = None,
             z_snapshot: Optional[np.ndarray] = None) -> dict:
        """
        t−1 의 joint outcome 관측으로 정서를 구성한다.

        value_vector : 방금 겪은 (상태, 행위) 의 Z 분위수 벡터 — 현재 상대의
                       기대 보상 분포를 EMA 갱신하는 재료.
        value_shift  : W₁(Z_before, Z_after)/span_V — 모형 갱신량 (arousal).
        z_snapshot   : Z_self 전체 스냅숏 — 재조우 복원용으로 기억에 보관.

        절차: (1) 가치분포 EMA 갱신 → (2) valence = 모집단 순위 →
              (3) arousal = 갱신량 → (4) λ_aff = V×A → (5) 협력 여부 commit.
        """
        s = int(observed_state)
        r_obs = float(self.payoffs[s])

        # --- 1. 현재 상대의 기대 보상 분포 갱신 ---
        if value_vector is not None:
            self.self_model.update_partner_value(
                self.identity, value_vector, z_snapshot=z_snapshot)

        # --- 2. valence: 사회적 모집단에서의 적합도 (수준) ---
        valence = self.self_model.social_valence(self.identity)
        ent = self.self_model.memory.get(self.identity)
        my_med = (float(ent.value_dist[len(ent.value_dist) // 2])
                  if ent is not None and ent.value_dist is not None
                  else float("nan"))
        rpe = (my_med - self.self_model.reference_median(exclude=self.identity)
               if np.isfinite(my_med) else 0.0)

        # --- 3. arousal: 생성 모델의 갱신량 (오차) ---
        surprise = float(value_shift) if value_shift is not None else 0.0
        arousal = float(1.0 - np.exp(-surprise / max(self.kl_scale, _EPS)))

        # --- 4. 정서적 동기 ---
        lambda_aff = valence * arousal

        # --- 5. 기억 commit (협력률 → λ₀ 설정점) ---
        self.self_model.commit_observation(
            self.identity, opponent_cooperated=opponent_cooperated)

        self.last = {
            "valence": float(valence), "arousal": arousal,
            "lambda_aff": float(lambda_aff), "rpe": float(rpe),
            "surprise": surprise, "tau_hat": float((valence + 1.0) / 2.0),
            "r_base": self.self_model.reference_median(exclude=self.identity),
            "r_mean": self.self_model.reference_median(exclude=self.identity),
            "r_obs": r_obs, "value": my_med,
            "pessimism": 0.0,
        }
        return dict(self.last)

    # ------------------------------------------------------------ 진단
    def expected_reward(self) -> float:
        ent = self.self_model.memory.get(self.identity)
        if ent is None or ent.value_dist is None:
            return 0.0
        return float(np.mean(ent.value_dist))

    def pessimism(self) -> float:
        return 0.0
