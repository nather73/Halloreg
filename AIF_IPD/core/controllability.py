"""
core.controllability
=====================

**자기 대 타인 귀인(self-vs-other causal attribution)과 통제권 추론.**

[동기 — 기존 한계]
기존 `AdaptiveAgent` 의 core allostatic belief 는 예측오차를 상대의 원인 축
(transient/α/λ_j/ρ/β)에만 귀인한다. 즉 "관측된 불균형이 **타인의** 무엇 때문인가"
만 묻고, "그것이 **나 자신** 때문(내 행동/내 통제권)인가"를 분리하지 못했다. 착취적
상대와 잡음 상대를 구분하는 데에는 β 축이 쓰였으나, **결과의 자기-기인(self-caused)
성분** — 예컨대 내 행동이 환경 실행오류로 뒤집혀 상호배신이 되었거나, 내가 먼저
방어(배신)하여 유발된 상호배신 — 은 여전히 타인의 기질로 오귀인될 수 있다.

[이론적 근거 — Spiering et al. (2025), Nat. Commun.]
사회적 신용할당(credit assignment)에서 사람은 총 예측오차(tPE)를 지각된 **통제권
(control)** 에 비례하여 자기(sPE)와 타인(oPE)에 분할한다:

    tPE  = 관측된 결과 − 기대된 결과
    sPE  = control       · tPE      (내 통제권이 클수록 내 탓)
    oPE  = (1 − control) · tPE      (내 통제권이 작을수록 남 탓)

그리고 통제권 자체를 (능동적 애매성 해소 active disambiguation 를 통해) 추론한다.
이 계산은 상연변이랑(supramarginal gyrus, SMG)의 활동과 연결된다. 통제권이 애매할
때 사람은 자기 기여를 의도적으로 제거(AD)하여 결과가 얼마나 타인 때문인지를
드러낸다.

[본 모듈의 IPD 조작화]
IPD 의 joint outcome 은 (내 행동, 상대 행동)의 함수다. 결과에 대한 나의 통제권
c_t ∈ [0,1] 은 "내 **의도한** 행동이 실제 방출·결과에 반영된 정도"로 조작화한다.
환경 실행오류(env_err)나 내 자신의 방어 전환이 결과를 좌우할수록 통제권 추정이
낮아진다. 통제권은 다음 두 신호의 공변으로 온라인 추론된다:

  (1) 자기-효능 증거 : 내 의도 행동과 실제 방출 행동의 일치(내가 통제) vs 불일치
      (환경/잡음이 통제) — 환경오류로 뒤집힌 라운드는 통제권을 낮춘다.
  (2) 능동적 애매성 해소(AD) : 내가 방금 배신(D)을 두어 상호배신(DD)이 되었다면,
      그 DD 는 **내가 유발한** 것이므로 상대 기질의 증거가 아니다(자기-기인).
      내가 협력(C)했는데 배신당함(CD)은 온전히 상대에게 귀속된다(타인-기인).

산출:
  · control : 현재 통제권 추정 c_t.
  · w_other : 예측오차를 **타인**에게 귀인할 가중치 = (1 − c_self_for_outcome).
             grievance 충전과 core belief 의 dispositional 증거를 이 가중치로 게이팅한다.
  · sPE / oPE : 통제권으로 분할된 자기/타인 예측오차(로깅·fMRI regressor 대용).

이로써 "결과가 자신 때문인지 타인 때문인지"를 명시적으로 귀인하고, 자기-기인
불균형이 타인의 기질로 새는 것을 막는다(자기보호의 정밀도 향상).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import CC, CD, DC, DD, COOP, DEFECT


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class ControllabilityAttribution:
    """
    통제권(control) 누출적분기 + 자기/타인 예측오차 분할기 (SMG 유사).

    파라미터
    --------
    c0 : 초기 통제권 사전(불확실 → 0.5).
    lr_control : 통제권 갱신률(자기-효능 증거의 통합 속도).
    decay : 통제권의 사전(0.5)으로의 완만한 회귀(누출) — 비정상 환경 대비.
    ad_gain : 능동적 애매성 해소(AD) 신호가 통제권/귀인에 주는 이득.
    floor_other : w_other 하한 — 타인 귀인을 완전히 0 으로 만들지 않음(수치 안정).
    """

    c0: float = 0.5
    lr_control: float = 0.20
    decay: float = 0.98
    ad_gain: float = 0.6
    floor_other: float = 0.05

    control: float = field(init=False)
    _s_pe: float = field(default=0.0, init=False)
    _o_pe: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.control = _clip01(self.c0)

    # ------------------------------------------------------------------ 갱신
    def update(self, intended_action: int, emitted_action: int,
               observed_state: int, total_pe: float) -> dict:
        """
        한 라운드 통제권·자기/타인 귀인 갱신.

        intended_action : 내가 **두려던** 행동(정책 표본).
        emitted_action  : 실제 **방출된** 행동(환경 실행오류 반영 후).
        observed_state  : 관측된 joint outcome (CC/CD/DC/DD).
        total_pe        : 총 예측오차 tPE (상대 행동 서프라이즈 등 현저성 스칼라, ≥0).

        반환: {control, w_other, sPE, oPE, self_caused}.
        """
        # (1) 자기-효능 증거: 내 의도가 결과에 그대로 반영되었는가.
        #     의도≠방출(환경/잡음이 뒤집음) → 이 라운드 통제권 증거는 낮음.
        acted_as_intended = 1.0 if int(intended_action) == int(emitted_action) else 0.0

        # (2) 능동적 애매성 해소(AD): 내 방출 행동이 결과의 '배신 성분'을 유발했는가.
        #     - 내가 D 를 방출 → 상호배신(DD)·유혹(DC) 은 내가 유발(자기-기인).
        #     - 내가 C 를 방출했는데 배신당함(CD) → 온전히 타인-기인.
        my_emitted_defect = (int(emitted_action) == DEFECT)
        outcome_self_caused = 1.0 if my_emitted_defect else 0.0    # 내 배신이 결과를 좌우
        # 내가 협력했는데 상대가 배신(CD): 결과는 순수 타인-기인(자기 기여 0 → AD 성립)
        clean_other_evidence = 1.0 if observed_state == CD else 0.0

        # (3) 통제권 갱신: 의도대로 행동했고(내가 통제) 결과가 내 행동으로 설명될수록 ↑;
        #     의도와 다르게 방출되었으면(환경이 통제) ↓.
        control_evidence = 0.5 * acted_as_intended + 0.5 * (
            outcome_self_caused if acted_as_intended > 0 else 0.0)
        c = self.decay * self.control + (1 - self.decay) * 0.5   # 사전으로 누출
        c = c + self.lr_control * (control_evidence - c)
        self.control = _clip01(c)

        # (4) 이 결과에 대한 '자기-기인 비중': 통제권과 AD 신호의 결합.
        #     내가 배신을 방출해 생긴 결과는 자기-기인(통제권 무관하게 높음),
        #     내가 협력했는데 배신당한 결과는 타인-기인.
        self_share = _clip01(
            outcome_self_caused * (0.5 + 0.5 * self.control)
            - self.ad_gain * clean_other_evidence * 0.0)          # CD 는 자기몫 0
        if clean_other_evidence > 0:
            self_share = 0.0

        w_other = _clip01(1.0 - self_share)
        w_other = float(np.clip(w_other, self.floor_other, 1.0))

        # (5) 통제권으로 분할된 예측오차(Spiering 신용할당).
        pe = float(max(total_pe, 0.0))
        self._s_pe = self_share * pe
        self._o_pe = (1.0 - self_share) * pe

        return {
            "control": self.control,
            "w_other": w_other,
            "sPE": self._s_pe,
            "oPE": self._o_pe,
            "self_caused": self_share,
        }
