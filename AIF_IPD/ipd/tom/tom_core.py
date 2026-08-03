"""
ipd.tom.tom_core
================

Theory-of-Mind 핵심: 상대 행동 예측 q(a_j | h_t) 과 **재귀적 social EFE**.

Albarracin et al. (2026) eq. (4):

    G_social(a_i) = (1 − λ)·G_self(a_i) + λ·E_{q(a_j)}[ G_other(a_j) ] + G_epistemic

본 판의 최종형:

    G_social(a_i) = (1 − λ)·( prag_self(a_i)  − IG_self(a_i) )
                   +    λ ·( prag_other(a_i) − IG_other(a_i) )

  · prag_self  : 내 보수 기준 기대효용의 **비용형**(−E[U]). 작을수록 선호.
  · prag_other : 상대 보수 기준 기대효용의 비용형 (조망수용 — C 만 교체).
  · IG_self    : 내가 상대에 대해 얻는 기대 정보이득 (θ̂ 필터).
  · IG_other   : 내 행동이 **상대가 나에 대해 갖는 믿음**을 얼마나 좁히는가
                 (자기-사영 필터 θ̂_self). λ 로 가중되므로, 공감이 클수록
                 "상대가 나를 이해하도록 돕는" 행동에 가치가 생긴다.
  · 가중은 (1 − λ), λ 뿐이다. 별도의 인식항 가중을 두지 않아, λ 가 자·타
    분기의 유일한 볼록결합 계수라는 구성개념이 훼손되지 않는다.

[재귀성 두 가지]
 (R1) depth-2 조망수용: 상대가 '나'를 어떻게 볼지를 자기-사영 필터로 계산해
      상대 모형의 `believed_my_policy` 에 직접 주입한다. 손으로 섞는 혼합계수
      없이 재귀 깊이가 자연 내장된다.
 (R2) 재귀적 인식가치: 상대가 나를 학습하며 얻는 정보이득(IG_other)도 λ 가중으로
      포함한다. 원 논문은 상대의 실용가치만 포함했다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from AIF_IPD.core.constants import COOP, DEFECT, PD_PAYOFFS, PAYOFF_SELF
from AIF_IPD.core.generative import efe_terms, softmax
from .inversion import OpponentInversion, ObservationContext


# --------------------------------------------------------------- static ToM
class TheoryOfMind:
    """
    상대를 '나와 구조적으로 동형인 합리적 행위자'로 보는 정적(static) ToM.

    상대의 행동선택:  q(a_j) ∝ exp( β_j · negEFE_j(a_j) )
    상대의 negEFE_j 는 상대 보수 하의 기대가치이며, 상대가 믿는 내 정책 π_i 에
    대해 기댓값을 취한다.

    입자필터가 아직 데이터를 못 본 초기 라운드의 **사전 예측**을 담당한다.
    """

    def __init__(self, beta_other: float = 4.0):
        self.beta_other = float(beta_other)
        # 상대가 믿는 내 (C, D) 확률. 기본은 무정보.
        self._believed_my_policy = np.array([0.5, 0.5])

    def update_my_policy_belief(self, my_coop_rate: float) -> None:
        """내 실현 협력률로 '상대가 믿는 내 정책' 을 갱신."""
        p = float(np.clip(my_coop_rate, 0.0, 1.0))
        self._believed_my_policy = np.array([p, 1.0 - p])

    def opponent_efe(self, believed_my_policy: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        """상대의 각 행동에 대한 negEFE (상대 보수 관점). shape (2,) = [C, D]."""
        pi = (self._believed_my_policy if believed_my_policy is None
              else believed_my_policy)
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
        """정적 ToM 상대 행동 분포 q(a_j) = softmax(β_j · negEFE_j)."""
        return softmax(self.opponent_efe(believed_my_policy),
                       temperature=1.0 / self.beta_other)


# --------------------------------------------------------------- gated ToM
class GatedToM:
    """
    신뢰도 게이팅된 ToM: 정적 사전과 학습된 사후(입자필터)를 필터 신뢰도 r 로
    부드럽게 보간한다.

        q_gated = r · q_learned + (1 − r) · q_static

    데이터가 없을 때는 구조적 사전에, 쌓일수록 관측된 개인 특성에 의존한다.
    """

    def __init__(self, tom: TheoryOfMind, inversion: OpponentInversion):
        self.tom = tom
        self.inversion = inversion

    def predict_opponent_action(self, ctx: Optional[ObservationContext]
                                ) -> np.ndarray:
        r = self.inversion.reliability()
        q = r * self.inversion.predict_action(ctx) \
            + (1.0 - r) * self.tom.predict_opponent_action()
        return q / q.sum()


# --------------------------------------------------------------- social EFE
@dataclass
class SocialEFEResult:
    """한 라운드 social EFE 계산 결과."""
    G_social: np.ndarray          # (2,) 내 각 행동의 social EFE (작을수록 선호)
    q_response: np.ndarray        # (2,) 상대 행동 예측
    info: dict


class RecursiveSocialEFE:
    """
    재귀적 social EFE 계산기.

    Parameters
    ----------
    gated_tom : GatedToM
    inversion : OpponentInversion            — 상대 θ̂ 필터
    self_inversion : OpponentInversion|None  — 자기-사영 θ̂_self 필터
    empathy_factor : float                   — λ 초깃값(에이전트가 매 라운드 갱신)
    beta_self : float                        — 내 행동선택 정밀도
    recursive_depth : int                    — 2 면 depth-2 조망수용 활성
    """

    def __init__(self, gated_tom: GatedToM, inversion: OpponentInversion,
                 empathy_factor: float = 0.4, beta_self: float = 4.0,
                 recursive_depth: int = 2,
                 self_inversion: Optional[OpponentInversion] = None):
        self.gated = gated_tom
        self.inversion = inversion
        self.self_inversion = self_inversion
        self.lam = float(empathy_factor)
        self.beta_self = float(beta_self)
        self.recursive_depth = int(recursive_depth)
        self.my_coop_rate = 0.5

    # ------------------------------------------------- 상대 예측 (depth-2)
    def opponent_prediction(self, ctx: Optional[ObservationContext]
                            ) -> np.ndarray:
        """
        상대 행동 예측 q(a_j).

        depth ≥ 2 이면 '상대가 믿는 내 정책'을 자기-사영 필터의 예측 협력확률로
        구성해 정적 ToM 에 주입한다. 이때 자기-사영 필터의 관점에서:
            f_me = 상대가 본 나의 호혜 자극 = 상대 자신의 직전 행동
            g_me = 나 자신의 직전 행동
        (즉 focal 필터와 f/g 의 역할이 정확히 뒤바뀐다.)
        """
        r = self.inversion.reliability()
        q_learned = self.inversion.predict_action(ctx)

        if self.recursive_depth < 2 or self.self_inversion is None:
            q = r * q_learned + (1.0 - r) * self.gated.tom.predict_opponent_action()
            return q / q.sum()

        f_me = (0.0 if ctx is None or ctx.their_last_action is None
                else 1.0 - 2.0 * float(ctx.their_last_action))
        g_me = (0.0 if ctx is None or ctx.my_last_action is None
                else 1.0 - 2.0 * float(ctx.my_last_action))
        my_pc = self.self_inversion.predict_coop(f_me, g_me)
        believed_my = np.array([my_pc, 1.0 - my_pc])

        q_static_cond = self.gated.tom.predict_opponent_action(
            believed_my_policy=believed_my)
        q = r * q_learned + (1.0 - r) * q_static_cond
        return q / q.sum()

    # ------------------------------------------------- 스텝 항 분해
    def step_terms(self, ctx: Optional[ObservationContext],
                   q: np.ndarray) -> dict:
        """
        상대 예측 q 하에서 내 각 행동의 social EFE 구성항.
        플래너의 rollout 스텝과 단일스텝 경로가 **같은 항 구성**을 쓰도록 일원화.
        """
        pc = float(q[COOP])

        # --- 실용가치: 내 관점 (비용형) ---
        prag = efe_terms(pc, PAYOFF_SELF)["pragmatic"]     # (2,) 기대효용
        prag_self = -prag

        # --- 실용가치: 상대 관점 (조망수용, 비용형) ---
        prag_other = np.zeros(2)
        for a_i in (COOP, DEFECT):
            val = 0.0
            for a_j in (COOP, DEFECT):
                _, other_payoff = PD_PAYOFFS[(a_i, a_j)]
                val += q[a_j] * other_payoff
            prag_other[a_i] = -val

        # --- 인식가치: 내가 상대를 알아가는 이득 ---
        # 내가 C 를 두면 다음 라운드 호혜신호 f_next = +1, D 면 −1.
        g_next = (0.0 if ctx is None or ctx.their_last_action is None
                  else 1.0 - 2.0 * float(ctx.their_last_action))
        IG_self = np.array([
            self.inversion.expected_infogain(+1.0, g_next),
            self.inversion.expected_infogain(-1.0, g_next),
        ])

        # --- 인식가치: 상대가 나를 알아가는 이득 (λ 로 가중됨) ---
        IG_other = np.zeros(2)
        if self.self_inversion is not None:
            f_me = (0.0 if ctx is None or ctx.their_last_action is None
                    else 1.0 - 2.0 * float(ctx.their_last_action))
            g_me = (0.0 if ctx is None or ctx.my_last_action is None
                    else 1.0 - 2.0 * float(ctx.my_last_action))
            for a_i in (COOP, DEFECT):
                IG_other[a_i] = self.self_inversion.observed_infogain(
                    a_i, f_me, g_me)

        return {"prag_self": prag_self, "prag_other": prag_other,
                "IG_self": IG_self, "IG_other": IG_other,
                "pragmatic_utility": prag, "pc": pc}

    # ------------------------------------------------- 단일스텝 계산
    def compute(self, ctx: Optional[ObservationContext],
                lam: Optional[float] = None) -> SocialEFEResult:
        lam = self.lam if lam is None else float(lam)
        q = self.opponent_prediction(ctx)
        t = self.step_terms(ctx, q)

        self_branch = t["prag_self"] - t["IG_self"]
        other_branch = t["prag_other"] - t["IG_other"]
        G_social = (1.0 - lam) * self_branch + lam * other_branch

        return SocialEFEResult(
            G_social=G_social, q_response=q,
            info={"pc": t["pc"], "lam": lam,
                  "reliability": self.inversion.reliability(),
                  # 선택 행동의 기대보수 — 진단용 (RPE 는 CoreAffect 가 기저
                  # 기대보상 분포로부터 별도 계산하므로 여기 값은 쓰지 않는다).
                  "pragmatic_utility": t["pragmatic_utility"],
                  "IG_self": t["IG_self"], "IG_other": t["IG_other"]})

    def select_action(self, ctx: Optional[ObservationContext],
                      lam: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[int, SocialEFEResult]:
        """softmax(−G_social) 에서 행동을 표집."""
        res = self.compute(ctx, lam)
        q_pi = softmax(-res.G_social, temperature=1.0 / self.beta_self)
        rng = rng or np.random.default_rng()
        action = COOP if rng.random() < q_pi[COOP] else DEFECT
        res.info["q_pi"] = q_pi
        res.info["action"] = action
        return action, res
