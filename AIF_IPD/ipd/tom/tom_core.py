"""
ipd.tom.tom_core
================

Theory-of-Mind 핵심: 상대 행동 예측 q(a_j|h_t) 과 **재귀적 social EFE**.

Albarracin et al. (2026) eq.(4,7):
    G_social(a_i) = (1-λ)·G_self(a_i) + λ·E_{q(a_j|h_t)}[G_other(a_j)] + G_epistemic

[HalloReg 확장 — 두 가지 재귀성]
  (R1) 재귀적 ToM: 원 논문은 상대 EFE 계산 시 static ToM 만 가정했으나, 본 모형은
       상대 또한 static ⊕ learned ToM 을 모두 사용한다고 가정한다. 즉 상대의 행동선택
       q(a_j|h_t) 를 구할 때, 상대가 '나'를 예측하는 믿음 π_i 자체를 best-response 한
       단계(depth-2)로 정련한다.
  (R2) 재귀적 epistemic: 원 논문은 social EFE 에 상대의 expected *pragmatic* value 만
       포함했으나, 본 모형은 상대의 expected *epistemic* value(상대가 나를 학습하며
       얻는 정보이득)도 λ 가중으로 추가한다.

이 모든 EFE 는 `core.generative` 의 해석적 numpy EFE(=pymdp EFE 와 등가)로 계산된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

from AIF_IPD.core.constants import (
    COOP, DEFECT, PD_PAYOFFS, PAYOFF_SELF, PAYOFF_OTHER,
)
from AIF_IPD.core.generative import efe_terms, softmax
from .inversion import OpponentInversion, ObservationContext


# --------------------------------------------------------------- static ToM
class TheoryOfMind:
    """
    상대를 '나와 구조적으로 동형인 합리적 행위자'로 보는 static ToM.

    상대의 행동선택:  q(a_j|h_t) ∝ exp( β_j · negEFE_j(a_j) )
    상대의 negEFE_j(a_j) = 상대가 자기 보수 하에서 얻는 기대가치. 상대는 내 정책 π_i
    (내 협력율에 대한 상대의 믿음)에 대해 기대한다.
    """

    def __init__(self, beta_other: float = 4.0,
                 use_pragmatic: bool = True, use_epistemic: bool = True):
        self.beta_other = beta_other
        self.use_pragmatic = use_pragmatic
        self.use_epistemic = use_epistemic
        self._believed_my_policy = np.array([0.5, 0.5])  # 상대가 믿는 내 (C,D) 확률

    def update_my_policy_belief(self, my_coop_rate: float):
        self._believed_my_policy = np.array([my_coop_rate, 1.0 - my_coop_rate])

    def opponent_efe(self, believed_my_policy: Optional[np.ndarray] = None) -> np.ndarray:
        """상대의 각 행동에 대한 negEFE (상대 보수 관점). shape (2,) = [C,D]."""
        pi = self._believed_my_policy if believed_my_policy is None else believed_my_policy
        negG = np.zeros(2)
        for a_j in (COOP, DEFECT):
            val = 0.0
            for a_i in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += pi[a_i] * other_payoff
            negG[a_j] = val
        return negG

    def predict_opponent_action(self,
                                believed_my_policy: Optional[np.ndarray] = None
                                ) -> np.ndarray:
        """static ToM 상대 행동 분포 q(a_j|h_t) = softmax(β_j·negEFE_j)."""
        negG = self.opponent_efe(believed_my_policy)
        return softmax(negG, temperature=1.0 / self.beta_other)


# --------------------------------------------------------------- gated ToM
class GatedToM:
    """
    신뢰도 게이팅된 ToM: static prior 와 learned posterior(입자필터)를
    입자필터 신뢰도 r 로 부드럽게 보간.

        q_gated = r · q_learned + (1 - r) · q_static
    """

    def __init__(self, tom: TheoryOfMind, inversion: OpponentInversion):
        self.tom = tom
        self.inversion = inversion

    def predict_opponent_action(self, ctx: Optional[ObservationContext]) -> np.ndarray:
        r = self.inversion.reliability()
        q_static = self.tom.predict_opponent_action()
        q_learned = self.inversion.predict_action(ctx)
        q = r * q_learned + (1 - r) * q_static
        return q / q.sum()


# --------------------------------------------------------------- social EFE
@dataclass
class SocialEFEResult:
    G_social: np.ndarray            # (2,) 각 내 행동
    G_self: np.ndarray
    G_other_pragmatic: np.ndarray
    G_epistemic_self: np.ndarray
    G_epistemic_other: np.ndarray
    q_response: np.ndarray          # 상대 행동 예측
    info: dict


class RecursiveSocialEFE:
    """
    재귀적 social EFE 계산기 — **v0.9.0 엄밀화(§4~7)**.

    최종형(§7):
        G_social(a_i) = (1−λ)·( prag_self(a_i) − IG_self(a_i) )
                        +  λ ·( prag_other(a_i) − IG_other(a_i) )

      · prag_* : **EFE pragmatic 키만**(§4). 상태 엔트로피 H[dist] 는 제외(→ §5 로
                 일원화). 비용형이라 −기대효용으로 부호정합(작을수록 선호).
      · IG_self : 전축 histogram, θ̂(상대) (§5).
      · IG_other: 전축 histogram, θ̂_self(자기-사영) (§6.1). 손튜닝 [0.5r,0.2r] 폐기.
      · 가중은 (1−λ), λ 뿐 — 별도 epistemic 가중 없음(§7, w_epi 삭제).
      · q(a_j) 는 depth-2 자연내장된 GatedToM 예측(§6.2). 1/2 후혼합 폐기.

    하위호환: legacy_efe=True 면 v0.8.2 형(w_epi·[0.5r,0.2r] IG_other·1/2 혼합)을
    복원한다. 기본은 v0.9.0.
    """

    def __init__(self, gated_tom: GatedToM, inversion: OpponentInversion,
                 empathy_factor: float = 0.4, beta_self: float = 4.0,
                 w_epi_self: float = 0.6, w_epi_other: float = 0.3,
                 recursive_depth: int = 2,
                 self_inversion: Optional[OpponentInversion] = None,
                 legacy_efe: bool = False):
        self.gated = gated_tom
        self.inversion = inversion
        # [v0.9.0 §6.1] 자기-사영 필터 θ̂_self ("상대가 나를 이렇게 추론할 것이다").
        self.self_inversion = self_inversion
        self.lam = empathy_factor
        self.beta_self = beta_self
        self.w_epi_self = w_epi_self          # legacy 경로에서만 사용
        self.w_epi_other = w_epi_other        # legacy 경로에서만 사용
        self.recursive_depth = recursive_depth
        self.legacy_efe = bool(legacy_efe)
        self.my_coop_rate = 0.5

    # ---- 상대 예측: depth-2 자연내장 (§6.2) ----
    def _opponent_prediction(self, ctx: Optional[ObservationContext],
                             my_last_action: int) -> np.ndarray:
        """
        GatedToM 예측에 depth-2 를 **자연 내장**한다(§6.2). 상대가 '나를 어떻게
        볼까'를 θ̂_self 사영으로 계산해 상대의 believed_my_policy 로 직접 주입하고,
        그 조건 하의 상대 우도를 q 로 쓴다. 신뢰도 r 은 learned/static 보간(GatedToM
        본연)에만 쓰이고, depth 혼합용 0.5r 은 사라진다.
        """
        r = self.inversion.reliability()
        q_static = self.gated.tom.predict_opponent_action()
        q_learned = self.inversion.predict_action(ctx)

        if self.legacy_efe or self.recursive_depth < 2:
            q = r * q_learned + (1 - r) * q_static
            if not self.legacy_efe:
                return q / q.sum()
            # legacy: 1/2 후혼합 재현
            tom = self.gated.tom
            tom.update_my_policy_belief(self.my_coop_rate)
            pc_learned = float(q[COOP])
            g_self = efe_terms(pc_learned, PAYOFF_SELF)["G"]
            my_br = softmax(-g_self, temperature=1.0 / self.beta_self)
            q2 = tom.predict_opponent_action(believed_my_policy=my_br)
            q = (1 - 0.5 * r) * q + 0.5 * r * q2
            return q / q.sum()

        # ---- v0.9.0 §6.2: θ̂_self 사영을 believed_my_policy 로 주입 ----
        # '상대가 믿는 내 정책'을 자기-사영 필터의 예측 협력확률로 구성한다.
        if self.self_inversion is not None:
            # 상대가 관측한 나의 직전 호혜신호 = 상대의 직전 행동(their_last).
            f_me = (0.0 if ctx is None or ctx.their_last_action is None
                    else 1.0 - 2.0 * float(ctx.their_last_action))
            g_me = (0.0 if ctx is None or ctx.my_last_action is None
                    else 1.0 - 2.0 * float(ctx.my_last_action))
            my_pc = self.self_inversion.predict_coop(f_me, g=g_me)
        else:
            my_pc = float(self.my_coop_rate)
        believed_my = np.array([my_pc, 1.0 - my_pc])
        tom = self.gated.tom
        # 그 조건 하의 상대 우도 자체가 q_learned 를 정련 → depth-2 자연내장.
        q_static_cond = tom.predict_opponent_action(believed_my_policy=believed_my)
        q = r * q_learned + (1 - r) * q_static_cond
        return q / q.sum()

    # ---- 공유 per-step EFE 항 (planner 와 depth-EFE 경로 일치, §5.3) ----
    def step_terms(self, ctx: Optional[ObservationContext],
                   q: np.ndarray, lam: float,
                   my_action_for_igother: Optional[int] = None) -> dict:
        """
        상대 예측 q 하에서 각 내 행동의 v0.9.0 social EFE 분해.
        planner 의 각 rollout 스텝과 depth-EFE 경로가 **같은 항 구성**을 쓰도록
        일원화한다(§5.3: 순수 깊이 대조).
        """
        pc = float(q[COOP])
        # prag_self: EFE pragmatic 키만(§4). 비용형 = −기대효용.
        prag_self = -efe_terms(pc, PAYOFF_SELF)["pragmatic"]         # (2,)
        # prag_other: 상대 기대보수의 비용형(§4, 대칭화).
        prag_other = np.zeros(2)
        for a_i in (COOP, DEFECT):
            val = 0.0
            for a_j in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += q[a_j] * other_payoff
            prag_other[a_i] = -val

        # IG_self: 전축 histogram, θ̂(상대) (§5).
        f_next = np.array([+1.0, -1.0])
        g_next = (0.0 if ctx is None or ctx.their_last_action is None
                  else 1.0 - 2.0 * float(ctx.their_last_action))
        if self.legacy_efe:
            IG_self = np.array([
                self.w_epi_self * self.inversion.expected_infogain(
                    COOP, f_next[COOP], g_next),
                self.w_epi_self * self.inversion.expected_infogain(
                    DEFECT, f_next[DEFECT], g_next)])
            r = self.inversion.reliability()
            IG_other = lam * self.w_epi_other * np.array([r * 0.5, r * 0.2])
        else:
            IG_self = np.array([
                self.inversion.expected_infogain_allaxis(
                    COOP, f_next[COOP], g_next),
                self.inversion.expected_infogain_allaxis(
                    DEFECT, f_next[DEFECT], g_next)])
            # IG_other: 전축 histogram, θ̂_self 자기-사영 (§6.1).
            IG_other = np.zeros(2)
            if self.self_inversion is not None:
                f_me = (0.0 if ctx is None or ctx.their_last_action is None
                        else 1.0 - 2.0 * float(ctx.their_last_action))
                g_me = (0.0 if ctx is None or ctx.my_last_action is None
                        else 1.0 - 2.0 * float(ctx.my_last_action))
                for a_i in (COOP, DEFECT):
                    IG_other[a_i] = self.self_inversion.observed_infogain_allaxis(
                        a_i, f_me, g_me)
        return {"prag_self": prag_self, "prag_other": prag_other,
                "IG_self": IG_self, "IG_other": IG_other, "pc": pc}

    def compute(self, ctx: Optional[ObservationContext],
                my_last_action: int, lam: Optional[float] = None) -> SocialEFEResult:
        lam = self.lam if lam is None else lam
        q = self._opponent_prediction(ctx, my_last_action)
        terms = self.step_terms(ctx, q, lam)
        pc = terms["pc"]

        G_self_full = efe_terms(pc, PAYOFF_SELF)["G"]   # 진단·로깅용(H 포함)
        G_other_full = terms["prag_other"]

        if self.legacy_efe:
            G_social = ((1 - lam) * G_self_full + lam * terms["prag_other"]
                        - terms["IG_self"] - terms["IG_other"])
            G_epi_self = -terms["IG_self"]
            G_epi_other = -terms["IG_other"]
        else:
            # v0.9.0 §7 최종형 — 가중은 (1−λ), λ 뿐.
            self_branch = terms["prag_self"] - terms["IG_self"]
            other_branch = terms["prag_other"] - terms["IG_other"]
            G_social = (1 - lam) * self_branch + lam * other_branch
            G_epi_self = -terms["IG_self"]
            G_epi_other = -terms["IG_other"]

        return SocialEFEResult(
            G_social=G_social,
            G_self=G_self_full,
            G_other_pragmatic=G_other_full,
            G_epistemic_self=G_epi_self,
            G_epistemic_other=G_epi_other,
            q_response=q,
            info={"pc": pc, "lam": lam,
                  "reliability": self.inversion.reliability(),
                  # [v0.9.0 §3.1] 단일 r_pred 계약: 선택 행동의 pragmatic 키.
                  "pragmatic_self": efe_terms(pc, PAYOFF_SELF)["pragmatic"]},
        )

    def select_action(self, ctx: Optional[ObservationContext],
                      my_last_action: int, lam: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[int, SocialEFEResult]:
        res = self.compute(ctx, my_last_action, lam)
        q_pi = softmax(-res.G_social, temperature=1.0 / self.beta_self)
        rng = rng or np.random.default_rng()
        action = COOP if rng.random() < q_pi[COOP] else DEFECT
        res.info["q_pi"] = q_pi
        res.info["action"] = action
        # [v0.9.0 §3.1] 단일 r_pred 계약 — 선택 행동의 pragmatic 기대보수.
        res.info["r_pred"] = float(res.info["pragmatic_self"][action])
        return action, res
