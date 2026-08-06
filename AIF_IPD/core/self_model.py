"""
core.self_model
===============

**SelfModel — 기억(memory) 저장소이자 사전(prior) 공급자.**

identity 별로 **(id, dist, theta, 기대 보상 분포)** 를 보관한다. v1.5.0 에서
기대 보상 분포는 보상 관측의 주변분포(구 QuantileCode)가 아니라 **QRTD 가치
Z(s,a) 로부터 유도된 분위수 벡터**다 — "이 사람과 있을 때 내가 놓이는 상황들의
장기 가치 분포".

────────────────────────────────────────────────────────────────────────
valence 의 사회적 참조 — 모집단 횡단비교
────────────────────────────────────────────────────────────────────────
본 연구의 전제: 일차적 보상 획득이 내부 모델에 대한 항상성으로 이어지며, 기대
보상 분포는 내수용감각 인식에 대한 생성 모델로 단순화되어 간주된다. valence 는
그 생성 모델(현재 상대에 대한 기대 보상 분포)이 **SelfModel 이 지닌 사회적
환경에 대한 믿음** — 즉 자기경계(self boundary) 안의 다른 타인들의 기대 보상
분포 — 에 대해 갖는 주관적 적합도다.

    참조분포  ref = Σ_k w_k · Q_k / Σ_k w_k     (k ≠ 현재 상대)
    w_k ∝ 사회적 거리의 역가중 (쌍곡 할인)
    valence = 2 · F̂_ref( median(Q_현재) ) − 1

  · 분위수 벡터의 가중 평균은 분포들의 **Wasserstein 무게중심**이므로, ref 는
    "내 사회적 관계들의 전형적 가치 분포" 라는 정확한 의미를 갖는다.
  · 비교는 **중앙값 기준 순위**다: 현재 상대의 분포 중앙값이 참조분포에서
    차지하는 누적확률.
  · **참조가 시간이 아니라 타자들에게 있다**는 점이 결정적이다. 시간에 대한
    자기비교(느린 이동평균)는 참조가 현재 상대를 쫓아가 만성 착취에서 valence
    가 소멸·역전된다(실측 확인). 모집단 비교는 쫓아가지 않는다 — 착취자와
    아무리 오래 있어도 기억 속 다른 관계들의 분포는 그대로이므로 "이 관계는
    내가 아는 다른 관계들에 비해 나쁘다" 가 유지된다. 이것이 이상성이다.
  · 현재 상대는 모집단에서 제외한다 (자기비교 혼입 방지).

────────────────────────────────────────────────────────────────────────
사전 사회사 — 5인
────────────────────────────────────────────────────────────────────────
자기경계 안의 관계는 **5명 정도**면 충분하다 (21 은 분위수 채널 수였을 뿐,
사회사 인원과 무관하다). 각 인물에는 (id, 거리, 협력률, **가치 분포**)를
부여한다. 협력률로 보상 관측을 생성하지 않는다 — 가치 분포는 직접 부여하되,
협력적 관계일수록 높은 가치를 갖도록 사상한다(r̄ = 1 + 2c ∈ [P, R] → V = r̄/(1−γ)).
협력률 자체는 λ₀ 설정점(거리가중 협력률)의 원자료로만 쓰인다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .distributional import DEFAULT_TAUS, vector_cdf

_EPS = 1e-12

THETA_AXES = ("alpha", "rho", "omega", "eta", "beta", "lambda_j")


@dataclass
class MemoryEntry:
    """상대 identity 하나에 대한 기억 항목: (id, dist, theta, 가치분포)."""
    identity: int
    theta: Dict[str, float] = field(default_factory=dict)
    theta_sd: Dict[str, float] = field(default_factory=dict)
    self_theta: Dict[str, float] = field(default_factory=dict)
    self_theta_sd: Dict[str, float] = field(default_factory=dict)
    # value_dist : **기대 보상(가치) 분포** — 분위수 벡터 (21,).
    # 실험 상대는 Z(s,a) 의 경험점유 EMA, 사전 인물은 아래 r_bar 로부터
    # **같은 QR-Huber TD 재귀**로 나란히 학습된다(_tick_reference).
    value_dist: Optional[np.ndarray] = None
    # r_bar : 사전 인물의 특성 보상(라운드당). 참조 가치분포를 현재 상대의 Z 와
    # 같은 추정 단계로 굴리기 위한 원자료. 실험 상대는 None.
    r_bar: Optional[float] = None
    # z_snapshot : 마지막 commit 시점의 Z_self(s,a) 전체 (4,2,21) — 재조우 복원용.
    z_snapshot: Optional[np.ndarray] = None
    familiarity: float = 0.0
    last_seen: int = 0
    n_obs: int = 0
    coop_count: float = 0.0


class SelfModel:
    """identity 기억 + 사회적 거리 + 설정점 + 모집단 가치 참조."""

    def __init__(self,
                 theta_prior_mean: Optional[Dict[str, float]] = None,
                 theta_prior_std: Optional[Dict[str, float]] = None,
                 recency_tau: float = 200.0,
                 familiarity_scale: float = 30.0,
                 k_disc: float = 1.0,
                 prior_coop_weight: float = 1.0,
                 social_lr: float = 0.02,
                 identity_lr: float = 0.15,
                 lam_floor: float = 0.10,
                 lam_ceil: float = 0.70,
                 taus=DEFAULT_TAUS):
        self.theta_prior_mean = dict(
            alpha=0.0, rho=0.5, omega=0.0, eta=0.0, beta=3.0, lambda_j=0.5)
        self.theta_prior_std = dict(
            alpha=2.0, rho=1.2, omega=1.0, eta=1.0, beta=1.8, lambda_j=0.3)
        if theta_prior_mean:
            self.theta_prior_mean.update(theta_prior_mean)
        if theta_prior_std:
            self.theta_prior_std.update(theta_prior_std)

        self.recency_tau = float(recency_tau)
        self.familiarity_scale = float(familiarity_scale)
        self.k_disc = float(k_disc)
        self.prior_coop_weight = float(prior_coop_weight)
        self.social_lr = float(social_lr)      # (보존: 향후 참조 이동에 사용 가능)
        self.identity_lr = float(identity_lr)  # value_dist 온라인 EMA 율
        self.lam_floor = float(lam_floor)
        self.lam_ceil = float(lam_ceil)
        self.taus = np.asarray(taus, dtype=float)

        self._payoff_span = 1.0
        self.memory: Dict[int, MemoryEntry] = {}
        self.clock: int = 0

    # ============================================================ 보수 스케일
    def set_payoff_scale(self, payoffs: np.ndarray) -> None:
        """보수 지지범위 기록 — 정규화 상수로만 쓴다."""
        u = np.asarray(payoffs, dtype=float)
        self._payoff_span = max(float(u.max() - u.min()), 1e-6)

    @property
    def payoff_span(self) -> float:
        return self._payoff_span

    # ============================================================ 사회적 거리
    def social_distance(self, identity: Optional[int]) -> float:
        """d = 1/(1 + F/F_scale) ∈ (0, 1]. 미지의 상대는 1."""
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None:
            return 1.0
        return float(1.0 / (1.0 + ent.familiarity / self.familiarity_scale))

    def distance_rank(self, identity: Optional[int]) -> float:
        """무계 서열거리 N = F_scale / F (쌍곡 할인은 여기에 적용)."""
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or ent.familiarity <= _EPS:
            return float("inf")
        return float(self.familiarity_scale / ent.familiarity)

    def distance_weight(self, identity: Optional[int]) -> float:
        """거리 역가중 w = 1/(1 + k·N) — 가까울수록 크다 (Jones & Rachlin)."""
        n = self.distance_rank(identity)
        if not np.isfinite(n):
            return 0.0
        return float(1.0 / (1.0 + self.k_disc * n))

    def _touch(self, identity: int) -> MemoryEntry:
        """조우 통지: F ← F·exp(−Δt/T) + 1 (빈도 누적 + 최근성 망각)."""
        ent = self.memory.setdefault(identity, MemoryEntry(
            identity=int(identity), last_seen=self.clock))
        dt = max(self.clock - ent.last_seen, 0)
        ent.familiarity = ent.familiarity * float(
            np.exp(-dt / max(self.recency_tau, _EPS))) + 1.0
        ent.last_seen = self.clock
        self.clock += 1
        return ent

    def observe_identity(self, identity: int) -> None:
        self._touch(int(identity))

    # ============================================================ 사전 공급
    def theta_prior(self, identity: Optional[int]) -> Dict[str, tuple]:
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or not ent.theta:
            return {ax: (self.theta_prior_mean[ax], self.theta_prior_std[ax])
                    for ax in THETA_AXES}
        out = {}
        for ax in THETA_AXES:
            mu = float(ent.theta.get(ax, self.theta_prior_mean[ax]))
            sd_floor = 0.25 * self.theta_prior_std[ax]
            sd = max(float(ent.theta_sd.get(ax, self.theta_prior_std[ax])),
                     sd_floor)
            out[ax] = (mu, sd)
        return out

    def self_theta_prior(self, identity: Optional[int]):
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or not ent.self_theta:
            return None
        return {ax: (float(ent.self_theta[ax]),
                     max(float(ent.self_theta_sd.get(ax, 0.3)), 0.15))
                for ax in ent.self_theta}

    def z_prior(self, identity: Optional[int]) -> Optional[np.ndarray]:
        """재조우 시 복원할 Z_self(s,a) 스냅숏."""
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or ent.z_snapshot is None:
            return None
        return ent.z_snapshot.copy()

    # ============================================================ 기억 갱신
    def commit_theta(self, identity, theta, theta_sd) -> None:
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        ent.theta = {ax: float(theta.get(ax, 0.0)) for ax in THETA_AXES}
        ent.theta_sd = {ax: float(theta_sd.get(ax, 0.0)) for ax in THETA_AXES}

    def commit_self_theta(self, identity, theta, theta_sd) -> None:
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        ent.self_theta = {k: float(v) for k, v in theta.items()}
        ent.self_theta_sd = {k: float(v) for k, v in theta_sd.items()}

    def commit_observation(self, identity: Optional[int],
                           opponent_cooperated: Optional[bool] = None) -> None:
        """상대의 협력 여부 누적 — 거리가중 협력률(λ₀ 설정점)의 원자료."""
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        ent.n_obs += 1
        if opponent_cooperated is not None:
            ent.coop_count += 1.0 if opponent_cooperated else 0.0

    def update_partner_value(self, identity: Optional[int],
                             value_vector: np.ndarray,
                             z_snapshot: Optional[np.ndarray] = None) -> None:
        """
        현재 상대의 **기대 보상(가치) 분포**를 온라인 갱신한다.

        value_vector 는 방금 겪은 (상태, 행위) 의 Z 분위수 벡터다. EMA 로
        누적하면 "이 사람과 있을 때 내가 실제로 놓이는 상황들의 가치 분포" —
        경험 점유율 가중 축약 — 가 된다. 분위수 벡터의 EMA 는 Wasserstein
        기하에서의 이동평균이므로 분포로서의 의미가 보존된다.
        """
        if identity is None:
            return
        ent = self.memory.setdefault(identity, MemoryEntry(identity=identity))
        v = np.asarray(value_vector, dtype=float)
        if ent.value_dist is None:
            ent.value_dist = v.copy()
        else:
            a = self.identity_lr
            ent.value_dist = (1.0 - a) * ent.value_dist + a * v
        if z_snapshot is not None:
            ent.z_snapshot = np.asarray(z_snapshot, dtype=float).copy()

    # ============================================================ 참조의 동보 갱신
    def tick_reference(self, gamma: float, lr: float, kappa: float) -> None:
        """
        사전 관계들의 가치분포를 **현재 상대의 Z 와 동일한 재귀**로 한 스텝
        갱신한다. 매 라운드 QRTD 갱신 직후에 호출된다.

        [왜 필요한가 — 척도 정합]
        이전 판은 사전 인물의 가치분포를 **이론적 정상상태** V = r̄/(1−γ) 로
        직접 부여했다. 그런데 현재 상대의 Z 는 TD 부트스트랩이라 참값에 닿는 데
        수백 라운드가 걸린다(CC 에서 이론 30, 600R 후 17). 그 결과 학습 초기에는
        **모든 상대가 참조보다 낮게** 보였고, valence 가 상대의 성질이 아니라
        **Z 의 학습 진행도**를 재는 지표가 되어 버렸다(실측: ALLC 상대에서
        valence 가 −0.959 → +0.978 로 단조 상승, value 2.20 → 7.74 와 동행).

        참조를 같은 추정기·같은 학습률·같은 라운드 수로 굴리면 미수렴 편향이
        **양변에서 상쇄**된다. 해석도 자연스럽다 — 기억 속 관계들에 대한 가치
        모형도 현재 진행 중인 생성 모형이며, 완결된 사실이 아니다.

        갱신은 분포적 TD 다: 목표 = r̄ + γ·(현재 분위수 벡터), 자기 부트스트랩
        (사전 관계는 정상적 요약이므로 상태 전이가 없다).
        """
        g = float(gamma)
        a = float(lr)
        kap = max(float(kappa), 1e-6)
        for ent in self.memory.values():
            if ent.r_bar is None or ent.value_dist is None:
                continue
            cur = ent.value_dist
            target = float(ent.r_bar) + g * cur          # (n,) 분포 목표
            delta = target[None, :] - cur[:, None]
            w = np.where(delta > 0.0, self.taus[:, None],
                         1.0 - self.taus[:, None])
            step = np.clip(delta, -kap, kap)
            ent.value_dist = np.maximum.accumulate(
                cur + a * np.mean(w * step, axis=1))

    # ============================================================ 모집단 참조
    def population_reference(self, exclude: Optional[int] = None
                             ) -> Optional[np.ndarray]:
        """
        **자기경계 안 타인들의 기대 보상 분포의 거리반비례 가중 평균.**

        분위수 벡터의 가중 평균 = Wasserstein 무게중심. 현재 상대(exclude)는
        모집단에서 제외한다. 아무도 없으면 None (valence 는 중립 0).
        """
        num = None
        den = 0.0
        for pid, ent in self.memory.items():
            if pid == exclude or ent.value_dist is None:
                continue
            w = self.distance_weight(pid)
            if w <= _EPS:
                continue
            num = w * ent.value_dist if num is None else num + w * ent.value_dist
            den += w
        if num is None or den <= _EPS:
            return None
        return num / den

    def social_valence(self, identity: Optional[int]) -> float:
        """
        valence = 2·F̂_ref( median(Q_현재) ) − 1.

        현재 상대의 가치분포 **중앙값**이, 다른 타인들의 참조분포에서 차지하는
        순위. 참조가 없거나 현재 분포가 없으면 중립 0.
        """
        ent = self.memory.get(identity) if identity is not None else None
        if ent is None or ent.value_dist is None:
            return 0.0
        ref = self.population_reference(exclude=identity)
        if ref is None:
            return 0.0
        med = float(ent.value_dist[len(ent.value_dist) // 2])
        return float(2.0 * vector_cdf(ref, self.taus, med) - 1.0)

    def reference_median(self, exclude: Optional[int] = None) -> float:
        ref = self.population_reference(exclude=exclude)
        return float(ref[len(ref) // 2]) if ref is not None else 0.0

    # ============================================================ 사전 사회사
    def seed_social_history(self, payoffs: np.ndarray,
                            gamma: float = 0.9,
                            value_init: float = 2.0,
                            value_spread: float = 2.0,
                            n_close: int = 1, n_middle: int = 2,
                            n_far: int = 2,
                            coop_mean: float = 0.55, coop_sd: float = 0.18,
                            distance_coop_slope: float = 0.0,
                            rng: Optional[np.random.Generator] = None) -> None:
        """
        실험 이전의 관계망 **5명**을 적재한다: (id, 거리, 협력률, 가치분포).

        · 협력률 c_k ~ Beta(coop_mean, coop_sd) — λ₀ 설정점의 원자료로만 쓴다.
          **보상 관측을 생성하지 않는다.**
        · 가치분포는 직접 부여한다: r̄_k = 1 + 2c_k ∈ [P, R] (배신쌍→상호협력),
          중심 V_k = r̄_k/(1−γ), 폭은 가치 척도의 30%. 협력적 관계일수록 높은
          가치라는 사상이며 임의 부여의 한 규약이다.
        · distance_coop_slope 기본 0 — 거리·협력 상관을 초기화에 심지 않는다.
        """
        rng = rng or np.random.default_rng(0)
        u = np.asarray(payoffs, dtype=float)
        self.set_payoff_scale(u)

        bands = [(n_close, 3.0), (n_middle, 0.6), (n_far, 0.1)]
        pid = -1
        m = float(np.clip(coop_mean, 0.02, 0.98))
        s2 = float(max(coop_sd, 1e-3)) ** 2
        nu = max(m * (1.0 - m) / s2 - 1.0, 0.1)

        for n_band, fam_mult in bands:
            for _ in range(int(n_band)):
                pid -= 1
                n_touch = max(int(round(fam_mult * self.familiarity_scale)), 1)
                for _ in range(n_touch):
                    self._touch(pid)
                d_k = self.social_distance(pid)
                c_k = float(rng.beta(m * nu, (1.0 - m) * nu))
                c_k = float(np.clip(
                    c_k + distance_coop_slope * (0.5 - d_k), 0.02, 0.98))

                ent = self.memory[pid]
                n_obs = max(int(round(0.5 * n_touch)), 2)
                ent.n_obs += n_obs
                ent.coop_count += float(round(c_k * n_obs))

                # 가치분포는 **직접 부여하지 않는다.** 현재 상대의 Z 와 같은
                # 초깃값에서 출발해, 매 라운드 tick_reference 로 나란히 학습
                # 된다. 부여하는 것은 특성 보상 r̄ 뿐이다.
                #   r̄ = 1 + 2c ∈ [P, R] — 협력적 관계일수록 높은 보상.
                ent.r_bar = 1.0 + 2.0 * c_k
                ent.value_dist = (float(value_init)
                                  + float(value_spread) * (self.taus - 0.5))

    # ============================================================ 설정점
    def weighted_cooperation(self) -> float:
        num = self.prior_coop_weight * 0.5
        den = self.prior_coop_weight
        for pid, ent in self.memory.items():
            if ent.n_obs <= 0:
                continue
            w = self.distance_weight(pid)
            num += w * (ent.coop_count / ent.n_obs)
            den += w
        return float(num / max(den, _EPS))

    def lambda_setpoint(self, payoffs: Optional[np.ndarray] = None,
                        scale: float = 0.25) -> float:
        """λ₀ = floor + (ceil − floor)·σ((c̄ − 0.5)/scale). 무기억 → 0.40."""
        c_bar = self.weighted_cooperation()
        sig = 1.0 / (1.0 + np.exp(-(c_bar - 0.5) / max(scale, 1e-6)))
        return float(self.lam_floor + (self.lam_ceil - self.lam_floor) * sig)

    # ============================================================ 진단
    def social_summary(self) -> dict:
        rows = []
        for pid, ent in sorted(self.memory.items()):
            if ent.n_obs <= 0:
                continue
            rows.append({"id": pid, "distance": self.social_distance(pid),
                         "rank": self.distance_rank(pid),
                         "weight": self.distance_weight(pid),
                         "coop": ent.coop_count / max(ent.n_obs, 1),
                         "n_obs": ent.n_obs,
                         "value_median": (float(ent.value_dist[
                             len(ent.value_dist) // 2])
                             if ent.value_dist is not None else None)})
        return {"n_others": len(rows), "rows": rows,
                "weighted_cooperation": self.weighted_cooperation(),
                "lambda_setpoint": self.lambda_setpoint()}

    def snapshot(self, payoffs=None) -> dict:
        return {"n_identities": len(self.memory),
                "weighted_cooperation": self.weighted_cooperation(),
                "lambda_setpoint": self.lambda_setpoint(),
                "distances": {pid: self.social_distance(pid)
                              for pid in self.memory}}
