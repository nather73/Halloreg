"""
core.empathy
============

**Empathy — 공감 가중치 λ 의 내생적 조절기.**

λ 는 행동선택 시 상대의 기대자유에너지에 두는 가중이다(Albarracin et al. 2026,
eq. 4). 원 논문에서 λ 는 **외생 고정 상수**였다. 본 모듈은 λ 를 두 개의 동기
신호로부터 **내생적으로** 산출한다.

    λ_aff  (정서적 동기, affective motivation)
        CoreAffect 로부터. valence × arousal.
        "예측되지 않은 손실/이득이 나를 어느 방향으로 밀어붙이는가."
        내수용(interoceptive) 경로.

    λ_ctx  (관측적·맥락적 동기, observational/contextual motivation)
        OpponentInversion 으로부터. 추론된 상대의 α 와 λ_j 의 합.
        "이 상대는 어떤 사람이라고 내가 추론했는가."
        외수용(exteroceptive) 경로.

    λ_t = λ_{t−1} + (1 − w_cd)·λ_aff + w_cd·λ_ctx

즉 λ 는 **적분기(integrator)** 다. 두 동기 신호가 매 라운드 λ 를 밀고, λ 자신은
누적된 관계사(relationship history)를 담는 상태변수가 된다. λ_{t=0} 은 SelfModel 의
할로스타틱 설정점(폐기된 λ_sp 를 대체)이 준다.

[구현상 필수 보완 두 가지 — 학술적 근거 포함]

  (A) 이득 상수 η (gain / time constant)
      λ_aff, λ_ctx 는 모두 O(1) 스케일이다. 사양의 갱신식을 그대로 쓰면 한 라운드
      만에 λ 가 [0,1] 경계로 포화해버려 120 라운드의 동역학이 사라진다. 따라서

          λ_t = clip( λ_{t−1} + η · [ (1 − w_cd)·λ_aff + w_cd·λ_ctx ] , λ_min, λ_max )

      로 두고 η 를 시간상수로 노출한다. η 는 갱신식의 **함수형을 바꾸지 않으며**
      (신호의 선형 결합 그대로), 단지 적분의 시간척도를 설정한다. 신경생물학적으로
      이는 조절계(neuromodulatory) 이득에 해당한다. 기본 η = 0.05 → 시상수 약 20
      라운드로, 120 라운드 지평에서 형성·붕괴·회복을 모두 관측할 수 있다.

  (B) λ_ctx 의 표준화
      사양은 "추론한 타인의 α 와 λ 의 합"이다. 그러나 두 양은 척도가 다르다.
        · α : 로짓 스케일의 협력 편향, 대략 (−4, +4). 무계.
        · λ_j : 확률 스케일의 공감 가중, [0, 1]. 중립점은 0.5.
      원값을 그대로 더하면 α 가 합을 지배하고 λ_j 항은 사실상 무시되며, 중립
      상대(α=0, λ_j=0.5)가 λ_ctx = 0.5 ≠ 0 이라는 부호 편향까지 생긴다. 따라서
      두 신호를 각각 **부호 있는 [−1, +1] 친사회성 척도**로 사상한 뒤 더한다.

          λ_ctx = ½ · [ tanh(α̂ / a_scale) + (2·λ̂_j − 1) ]

      · tanh(α̂/a_scale) : 협력 편향의 부호 있는 압착. a_scale=2.0 은 α 사전
        표준편차와 같아, 사전 1 SD 가 약 0.76 에 대응한다.
      · (2λ̂_j − 1)      : λ_j 의 중립점 0.5 를 0 으로 옮긴 부호화.
      · ½                : 두 항의 합을 [−1, +1] 로 되돌려 λ_aff 와 같은 스케일에
                           둔다(가중 w_cd 가 두 채널을 공정하게 섞도록).

      이 변환은 단조(monotone)이므로 "α 와 λ_j 가 클수록 λ_ctx 가 크다"는 사양의
      정성적 내용을 정확히 보존한다.

[두 채널의 비대칭적 성질 — 해석에 중요]
  · λ_aff 는 **오차 신호**다. 기저 기대가 적응하면 RPE → 0 → λ_aff → 0.
    즉 자기소멸적이며, 놀람이 있을 때만 λ 를 민다.
  · λ_ctx 는 **수준 신호**다. 상대가 일관되게 착취적이면 λ_ctx 는 계속 음수로
    남아 λ 를 하한까지 끌어내린다. 이것이 의도된 동작이다: 확신을 갖고 착취자로
    판정한 상대에게 공감 가중을 유지할 이유가 없다.
  두 채널의 결합이 곧 "놀람 기반 즉시 반응 + 추론 기반 지속 유지"의 이중 시간척도다.
"""

from __future__ import annotations

import numpy as np


class Empathy:
    """
    λ 적분기.

    Parameters
    ----------
    lam_init : float
        λ_{t=0}. SelfModel.lambda_setpoint() 이 공급한다(폐기된 λ_sp 대체).
    w_cd : float ∈ [0, 1]
        정서 채널 대 맥락 채널의 가중. 0 이면 순수 정서 구동, 1 이면 순수 추론
        구동. 기본 0.5(동등 가중).
    gain : float
        적분 이득 η (위 (A) 참조).
    lam_min, lam_max : float
        λ 의 허용 구간. λ 는 EFE 의 볼록결합 가중이므로 [0, 1] 을 벗어나면 안 된다.
    alpha_scale : float
        λ_ctx 표준화의 a_scale (위 (B) 참조).
    """

    def __init__(self, lam_init: float, w_cd: float = 0.5,
                 gain: float = 0.05, lam_min: float = 0.0,
                 lam_max: float = 1.0, alpha_scale: float = 2.0):
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.lam_init = float(np.clip(lam_init, lam_min, lam_max))
        self.lam = self.lam_init
        self.w_cd = float(np.clip(w_cd, 0.0, 1.0))
        self.gain = float(gain)
        self.alpha_scale = float(alpha_scale)

        # 최근 스텝 진단값
        self.last = {"lambda": self.lam, "lambda_aff": 0.0,
                     "lambda_ctx": 0.0, "drive": 0.0}

    # ------------------------------------------------------------ λ_ctx
    def contextual(self, alpha_hat: float, lambda_j_hat: float) -> float:
        """
        추론된 상대 특성 → 맥락적 동기 λ_ctx ∈ [−1, +1].
        (위 문서 (B) 의 표준화된 합.)
        """
        a = float(np.tanh(float(alpha_hat) / max(self.alpha_scale, 1e-6)))
        l = 2.0 * float(np.clip(lambda_j_hat, 0.0, 1.0)) - 1.0
        return 0.5 * (a + l)

    # ------------------------------------------------------------ 한 스텝
    def step(self, lambda_aff: float, lambda_ctx: float = 0.0) -> float:
        """
        λ_t = clip( λ_{t−1} + η · λ_aff ).

        **v1.5.0 — λ 는 오직 CoreAffect(정서)만이 움직인다.** λ_ctx(추론된
        α̂·λ̂_j 수준신호)는 제거되었다. OpponentInversion 의 영향은 호혜 경로
        (ρ, ω, η — SelfPolicy)로 이관되었으므로, 같은 정보원이 두 경로를 모두
        구동하면 두 친사회 경로의 해리 주장이 무너진다. 실측으로도 λ_ctx 가 λ
        궤적을 지배하고 있었다(정서 비중 16~21%) — 공감 경로라 이름 붙인 것을
        추론이 굴리고 있던 셈이다. lambda_ctx 인자는 서명 호환용이며 무시된다.
        """
        drive = float(lambda_aff)
        self.lam = float(np.clip(self.lam + self.gain * drive,
                                 self.lam_min, self.lam_max))
        self.last = {"lambda": self.lam, "lambda_aff": float(lambda_aff),
                     "lambda_ctx": 0.0, "drive": drive}
        return self.lam

    def reset(self, lam_init: float = None) -> float:
        """
        새 상대와의 관계 시작 시 λ 를 설정점으로 되돌린다.
        lam_init 을 주면 그 값(=갱신된 SelfModel 설정점)으로, 아니면 최초 설정점으로.
        """
        if lam_init is not None:
            self.lam_init = float(np.clip(lam_init, self.lam_min, self.lam_max))
        self.lam = self.lam_init
        return self.lam
