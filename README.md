# HalloReg — Adaptive Prosociality Through Hierarchical Allostatic Regulation

반복 죄수의 딜레마(Iterated Prisoner's Dilemma, IPD)에서 **친사회성이 내생적으로
조절되는** 능동추론(Active Inference) 에이전트의 구현.

> *Adaptive Prosociality Through Hierarchical Allostatic Regulation in Social
> Dynamics: A Simulation Study.*

---

## 1. 무엇이 문제인가

친사회적 행위자는 **자기보호**와 **상호협력으로의 수렴**을 어떻게 화해시키는가?

- TFT·WSLS 같은 고전적 반응 전략은 제한적 조건(예: 정상 보수구조)에서만 잘 작동한다.
- Albarracin et al. (2026) 의 공감적 능동추론 에이전트는 상대의 기대자유에너지에
  가중치 λ 를 두어 조망수용을 구현했으나, **λ 가 외생 고정 상수**다. 따라서 λ 가
  높으면 착취자에게 자기보호를 하지 못하고, 낮으면 협력 관계를 구축하지 못한다.

본 프로젝트는 λ 를 **위계적 이상성 조절(hierarchical allostatic regulation)** 로
내생화한다.

---

## 2. 아키텍처

```
SelfModel ──(id, θ)──────────────▶ OpponentInversion (입자필터)
    │                                        │
    ├──(id, 기대보상분포)─────────▶ CoreAffect       │
    │                                  │             │
    └──(기저 기대보상 = 할로스타틱 설정점)           │
                                       │             │
                          λ_aff = valence × arousal  λ_ctx = f(α̂, λ̂ⱼ)
                                       ╲             ╱
                                        ╲           ╱
                                        Empathy (λ 적분기)
                            λ_t = λ_{t−1} + η·[(1−w_cd)·λ_aff + w_cd·λ_ctx]
                                             │
                                    RecursiveSocialEFE → 행동
```

갱신된 `(θ, dist)` 와 기대보상 분포는 다시 `SelfModel` 로 commit 된다.
**`SelfModel` 은 추론하지 않고 기억과 사전 공급만 담당한다.**

### 2.1 `core/self_model.py` — SelfModel

- identity 별로 `(id, dist, theta, expected reward distribution)` 를 보관한다.
  - `theta` : 상대 특성의 사후 평균 (α, ρ, ω, η, β, λ_j)
  - `dist`  : 그 사후의 축별 표준편차 (믿음의 폭)
  - 기대보상 분포 : 4-범주(CC/CD/DC/DD) Dirichlet 농도
- `OpponentInversion` 에 `(id, θ)` 사전을, `CoreAffect` 에 `(id, 기대보상분포)` 사전을
  공급한다. 재조우 시에는 저장된 기억이 무정보 사전을 대체한다.
- **할로스타틱 설정점**은 λ 가 아니라, 사회적 환경 전반에 대한 믿음인
  *기저 기대보상 분포* `q_social(r)` 다. 이 분포는 학습률 `social_lr ≪ 1` 로
  느리게만 이동한다 — 이것이 항상성이 아니라 이상성인 이유다.
- λ 의 초깃값 λ_{t=0} 은 이 기저 분포에서 유도된다(폐기된 λ_sp 를 대체).

### 2.2 `core/core_affect.py` — CoreAffect

2차원 핵심정서(core affect)를 내수용 예측부호화로 구성한다.

```
RPE     = r_obs − E_{q_social}[r]                      (기저 = SelfModel 설정점)
valence = tanh( RPE / (σ_social + σ_floor) )           ∈ (−1, +1)
arousal = 1 − exp( − KL(q_post ‖ q_prior) / κ )        ∈ [0, 1)
λ_aff   = valence × arousal
```

- `q_prior` / `q_post` 는 **이번 상대 identity** 의 기대보상 분포를 이번 관측으로
  갱신하기 전/후의 사후예측 categorical 이다.
- 곱셈 결합의 의미: 각성이 0 이면(예측대로였다) 정서가 λ 를 움직이지 않는다.
  **예측된 손실은 정서를 만들지 않고, 예측되지 않은 손실만 정서를 만든다.**

### 2.3 `core/empathy.py` — Empathy

```
λ_ctx = ½·[ tanh(α̂ / a_scale) + (2·λ̂ⱼ − 1) ]          ∈ [−1, +1]
λ_t   = clip( λ_{t−1} + η·[ (1 − w_cd)·λ_aff + w_cd·λ_ctx ] , 0, 1 )
```

두 채널의 성질이 다르다.

| 채널 | 성질 | 시간척도 |
|---|---|---|
| `λ_aff` | **오차 신호**. 기저 기대가 적응하면 RPE→0 → 자기소멸적 | 빠름 (놀람 기반) |
| `λ_ctx` | **수준 신호**. 착취적 상대로 판정하면 계속 음수로 남음 | 느림 (추론 기반) |

### 2.4 구현상 두 가지 보완 (사양 대비)

| 항목 | 왜 필요한가 |
|---|---|
| 적분 이득 η | λ_aff·λ_ctx 는 O(1) 스케일이라, 사양의 갱신식을 그대로 쓰면 한 라운드에 λ 가 경계로 포화해 120 라운드 동역학이 사라진다. η 는 갱신식의 **함수형을 바꾸지 않고** 시간척도만 설정한다(η=0.05 → 시상수 약 20 라운드). |
| λ_ctx 의 표준화 | α 는 로짓 스케일(대략 −4~+4, 무계), λ_j 는 확률 스케일([0,1], 중립 0.5)이다. 원값을 더하면 α 가 합을 지배하고, 중립 상대(α=0, λ_j=0.5)가 λ_ctx=0.5≠0 이라는 부호 편향이 생긴다. 변환은 단조이므로 "α·λ_j 가 클수록 λ_ctx 가 크다"는 사양의 정성적 내용을 정확히 보존한다. |

---

## 3. 조망수용 계층 (`ipd/tom/`)

### 우도 기저 (1, f, g, f·g)

```
P(a_j = C | h_t, θ) = σ( β·( α + ρ·f + ω·g + η·f·g + s(λ_j, p) ) )
```

- `f` = 내 직전 행동의 호혜신호 (+1 협력 / −1 배신)
- `g` = 상대 **자신의** 직전 행동 신호
- 이 4개 자유도가 기억-1(memory-one) 시그니처 전 공간을 스팬한다.
  특히 **WSLS 는 순수 η 축**으로 표현된다 — (1, f) 기저만 쓰면 WSLS 는 필연적으로
  오분류된다.

### 시제(tense) 규약 — off-by-one 재발 방지

| 경로 | 대상 | f | g |
|---|---|---|---|
| 갱신용 | 관측 opp_{k−1} | my_{k−2} | opp_{k−2} |
| 결정용 | 예측 opp_k | my_{k−1} (최신) | opp_{k−1} (최신) |

갱신 경로의 f·g 는 **둘 다 한 시점 더 과거**를 가리킨다. `agent.py` 가
`ObservationContext` 에 올바른 시제를 담아 넘길 책임을 진다.

### social EFE

```
G_social(a_i) = (1 − λ)·( prag_self(a_i)  − IG_self(a_i) )
              +    λ ·( prag_other(a_i) − IG_other(a_i) )
```

- `IG_self`  : 상대 θ̂ 필터의 전축 히스토그램 기대 정보이득
- `IG_other` : **자기-사영 필터** θ̂_self 의 실현 정보이득 — "내 행동이 상대가
  나에 대해 갖는 믿음을 얼마나 좁히는가". λ 로 가중되므로, 공감이 클수록
  "상대가 나를 이해하도록 돕는" 행동에 가치가 생긴다.
- 가중은 (1−λ), λ 뿐이다 — λ 가 자·타 분기의 유일한 볼록결합 계수라는
  구성개념이 훼손되지 않는다.
- **호혜의 도구적 가치는 λ 가 아니라 rollout 이 담당한다.** planner 는 step>0 에서
  ρ̂ 로 "내가 협력하면 다음 라운드에 되돌아온다"를 계산한다. 이를 λ 로 처리하면
  λ 가 공감이 아니라 전략 파라미터가 되어 구성개념이 붕괴한다.

---

## 4. 검증하는 가설

| | 내용 |
|---|---|
| **ARCH** | 아키텍처 구현 정합성 (가설검증 이전 필수 통과) |
| **H1** | `OpponentInversion` 은 타인의 전략적 의도(TFT/GTFT/WSLS/ALLC/ALLD/HalloReg)를 강건하게 추론하는가 |
| **H1A** | 변동하는 의도를 잘 추적하며 λ 도 적절히 복원되는가 |
| **H2** | 착취자로부터 보수를 잘 보호할 수 있는가 |
| **H2A** | 착취자와 noisy TFT 를 구분할 수 있는가 |
| **H3** | 정상 보수구조 혼합 집단(30명)에서 고정전략 대비 높은 보상을 얻는가 |
| **H3A** | 같은 조건에서 집단 상호협력률 상승에 유의하게 더 많이 기여하는가 |
| **H4** | 비정상 보수구조에서 H3 와 같은가 |
| **H4A** | 비정상 보수구조에서 H3A 와 같은가 |
| **H5** | RE·ORE 시뮬레이션에서 세대에 걸쳐 생존하는가 |
| **H6** | 파라미터 복원과 자기-사영(projection)이 강건하게 이루어지는가 |

### 4.1 집단 조합 문제와 그 해법 (H3~H5)

가설은 `{TFT=a, GTFT=b, WSLS=c, ALLC=d, ALLD=e, HalloReg=f}, Σ=30` 의
"최대한 많은 조합" 을 요구한다. 각 유형 최소 1명 조건에서 조합 수는
**C(29, 5) = 118,755** 개다. 직접 라운드로빈으로 돌리면 5,100만 다이애드가 되어
불가능하다.

**해법 — 근사가 아니라 정확한 분해.** 각 다이애드가 독립 생성된 개체 쌍으로
실행되면 집단 지표는 유형쌍 지표의 조합가중 평균과 **항등적으로 일치**한다.

```
μ_i(n) = [ Σ_j n_j·Π[i,j] − Π[i,i] ] / (N − 1)
CC(n)  = [ nᵀ·CCm·n − Σ_i n_i·CCm[i,i] ] / [ N·(N − 1) ]
```

따라서 **유형쌍 21개만 시뮬레이션하고 118,755개 조합 전부를 해석적으로 평가**한다.
이 항등식은 `ipd/population.py` 에 유도와 함께 문서화되어 있고, 실험 스크립트가
대표 조합에 대해 **직접 라운드로빈**을 수행해 예측과 대조한다(H3 의 검증 패널).

### 4.2 협력 기여의 조작화 (H3A / H4A)

"더 많이 기여" 는 **치환 대비(substitution contrast)** 로 정의한다.

```
Δ^CC_i(n) = CC(n) − CC(n − e_HalloReg + e_i)
```

집단 크기 30 을 고정한 채 한 자리만 바꾸므로, "HalloReg 가 많은 조합의 CC 가 높다"
는 단순 상관이 갖는 구성·크기 교란이 제거된다.

### 4.3 비정상 보수 레짐 (H4/H4A/H5)

협력지수 `CI = (R − P)/(T − S)` 를 라운드마다 변동시킨다 (T=5, S=0, P=1 고정,
R = 1 + 5·CI).

| 레짐 | 성격 |
|---|---|
| `stationary` | 대조군 (CI = 0.4, 기본 PD) |
| `blocks` | 계단형 전환: 조화 → PD → 교착 → PD → 조화 |
| `oscillate` | 정현파 진동 (CI: −0.3 ↔ 1.2, 주기 30) |
| `aba` | A → B → A 복귀 (맥락 복귀 학습) |
| `drift` | 선형 표류 (1.2 → −0.2) |
| `shock` | 희소·급성 교착 충격 |

고정전략은 보수행렬을 읽지 않고 행동 이력에만 반응한다. 능동추론 에이전트는 EFE 의
선호 C 를 현재 보수로 계산한다 — **이 비대칭이 H4 의 이론적 근거다.**

### 4.4 확증 / 탐색 구분

- 확증(confirmatory): 사전 지정한 주 검정. **Holm** 보정(FWER 통제).
- 탐색(exploratory): 나머지. **BH-FDR** 보정.
- 확증은 보정 p < α **그리고** 방향성립을 모두 만족해야 '지지' 로 기록된다.
- 지지되지 않으면 억지로 긍정 서술을 만들지 않고 그대로 '미지지' 로 남긴다.

---

## 5. 실행

### 설치

```bash
pip install numpy matplotlib
```

`scipy` 는 필요하지 않다 — 모든 통계는 순수 numpy 순열/부트스트랩이다.
한글 폰트는 OS별로 자동 선택된다(Windows: 맑은 고딕, macOS: AppleGothic,
Linux: NanumGothic / Noto Sans CJK KR).

### 기본 실행

```bash
python AIF_IPD/scripts/run_ipd_experiment.py
```

인자 없이 실행하면 논문 사양(**시드 120, 라운드 120**)으로 ARCH → H1 → H1A → H2
→ H3/H4 → H5 → H6 전체가 순서대로 돌아간다. 옵트아웃 플래그를 외울 필요가 없다.

### 자주 쓰는 옵션

```bash
# 스모크 (수 분) — 구현 점검용
python AIF_IPD/scripts/run_ipd_experiment.py --quick

# 특정 가설만 (H2A/H3A/H4A 는 각각 H2/H3/H4 별칭)
python AIF_IPD/scripts/run_ipd_experiment.py --experiments H1 H2

# 병렬 워커 — CPU 바운드 단일스레드 작업이므로 물리 코어 수 지정을 권장
python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16

# 조합 공간 부분표집 (0 = 전수 118,755개)
python AIF_IPD/scripts/run_ipd_experiment.py --max-compositions 20000
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--seeds` | 120 | 조건당 반복 수 |
| `--rounds` | 120 | 다이애드당 라운드 수 |
| `--jobs` | −1 | 병렬 워커 (−1 = 논리 코어 − 1) |
| `--particles` | 400 | 입자필터 입자 수 |
| `--horizon` | 2 | 계획 지평 (1 = myopic) |
| `--w-cd` | 0.5 | Empathy 의 정서–맥락 가중 |
| `--lam-gain` | 0.05 | λ 적분 이득 η |
| `--env-error` | 0.05 | 환경 계층 실행오류 (모든 유형 대칭) |

### 단위 검증

```bash
python AIF_IPD/tests/test_core.py
```

보수 구조·해석적 EFE·SelfModel 사전과 설정점·CoreAffect 의 부호와 유계성·
Empathy 갱신식·입자필터 시제 규약과 WSLS 의 η 복원·에이전트 λ 방향성과 로그
정합·집단 분해식 항등성·RE/ORE·통계 인프라·파라미터 복원을 **결정적으로**
검사한다(외부 테스트 러너 불필요, 82개 검사).

### 출력

```
AIF_IPD/results/
├── SUMMARY.json              전체 판정표 (확증 Holm / 탐색 BH-FDR)
├── ARCH.json  H1.json  H1A.json  H2.json  H3_H4.json  H5.json  H6.json
└── figures/
    ├── ARCH_validation.png
    ├── H1_intent_inference.png
    ├── H1A_tracking_recovery.png
    ├── H2_protection_discrimination.png
    ├── H3_stationary_population.png
    ├── H4_nonstationary_population.png
    ├── H5_evolution.png
    └── H6_recovery_projection.png
```

### 예상 실행시간

16 물리 코어 기준 전체 실행은 대략 **4~7시간**이다. 대부분은 H3/H4 의 유형쌍 행렬
추정(6 레짐 × 21 쌍 × 120 시드 = 15,120 다이애드)과 H3 의 직접 라운드로빈 검증
(435 다이애드 × 3회)에 쓰인다. 계산 비용을 줄이려면 `--seeds` 를 낮추기보다
`--max-compositions` 로 조합 표집을 줄이는 편이 통계적 손실이 적다(조합은 서로
독립이 아니지만 시드는 독립 replicate 다).

---

## 6. 디렉터리 구조

```
AIF_IPD/
├── core/
│   ├── constants.py       PD 보수 구조, joint-outcome 규약, 가변 보수 API
│   ├── generative.py      POMDP 생성모형과 해석적 EFE
│   ├── self_model.py      SelfModel — 기억·사전 공급·할로스타틱 설정점
│   ├── core_affect.py     CoreAffect — valence(RPE) × arousal(KL)
│   ├── empathy.py         Empathy — λ 적분기
│   └── logging_utils.py   로거·한글 폰트
├── ipd/
│   ├── tom/
│   │   ├── inversion.py   OpponentInversion (입자필터)
│   │   ├── tom_core.py    정적/게이팅 ToM, 재귀적 social EFE
│   │   └── planner.py     다단계 rollout 계획기
│   ├── agent.py           EmpathicAgent (고정 λ) / HalloRegAgent
│   ├── env.py             고정전략, 형질전환 상대, 복원용 생성 상대
│   ├── sim.py             다이애드 루프, 병렬 실행기
│   ├── payoff_schedule.py 비정상 보수 레짐
│   ├── population.py      집단 정확 분해식, 유형쌍 행렬 추정
│   ├── evolution.py       복제자(RE) / 최적 복제자(ORE)
│   └── metrics/stats.py   순열검정·효과크기·다중비교 보정
├── experiments/
│   ├── common.py          설정·결과 등록기·시각화 유틸
│   ├── arch_validation.py ARCH
│   ├── h1_intent.py       H1
│   ├── h1a_tracking.py    H1A
│   ├── h2_protection.py   H2 / H2A
│   ├── h3_h4_population.py H3 / H3A / H4 / H4A
│   ├── h5_evolution.py    H5
│   └── h6_recovery.py     H6
├── scripts/run_ipd_experiment.py   메인 엔트리포인트
├── tests/test_core.py              단위 검증 (82개 검사)
└── results/
```

---

## 7. 알려진 한계와 정직한 보고

이 프로젝트는 **결과를 보기 전에 검정을 고정하고, 지지되지 않은 결과를 그대로
보고한다**는 원칙으로 작성되었다. 다음은 코드에 문서화된 주요 한계다.

### 7.1 λ = 0 대조군이 착취자 맥락에서 정의상 최적이다

λ = 0 은 공감 가중이 0 인 **순수 자기이익** 능동추론 에이전트로, Albarracin et al.
이 제안한 공감 모형이 아니다. 착취자 맥락에서는 배신을 억제할 유일한 힘이 없으므로
정의상 최적이다. 따라서 확증 대조군에서 제외하고 **탐색으로 명시 보고**한다
(감추지 않는다).

관련해서, 본 구현의 λ=0 에이전트는 협력자 맥락에서도 높은 협력을 달성한다. 이는
**rollout 안의 호혜성 전파** 때문이며(§3 참조), "공감 없이는 협력이 불가능하다"는
식의 과잉 주장을 금지한다.

### 7.2 행동적 등가류 — TFT / GTFT / ALLC

focal 이 배신하지 않으면 GTFT 의 관대함은 발현되지 않는다. 즉 이 세 유형은
on-policy 관측만으로는 부분적으로 구별 불가능하다. 이는 추론기의 결함이 아니라
**능동 자극(active sampling)의 부재**다. H1 은 on-policy 조건과 probe 조건
(focal 실행오류 ε=0.20)을 나란히 돌려 이 진단을 직접 검정한다.

### 7.3 α 와 λ_j 의 공선성

λ_j 는 우도에 `s(λ_j, p) = (T−S)·λ_j + …` 라는 **절편** 형태로만 들어간다. 고정
보수에서는 (T−S)=5 가 상수이므로 λ_j 항이 α 와 완전 공선이 되어 개별 식별이
불가능하다. H6 은 블록마다 (T, S) 를 변조해 이 공선성을 깨는 조건을 대조로 두고,
복원이 회복되는지를 확증 검정으로 확인한다.

### 7.4 β 의 약한 식별성

β 는 선형예측자 전체에 곱해지는 정밀도이므로 선택확률의 **극단성**만 조절한다.
다른 축과 곱셈적으로 얽혀 있고 중간 크기의 β 에서는 σ(·) 가 이미 포화에 가까워
관측 우도가 β 변화에 둔감하다. **β 의 복원 상관은 다른 축보다 낮을 것으로 사전
예측한다.**

### 7.5 ALLD 와의 보수 비교

ALLD 는 협력자가 많은 조합에서 착취로 높은 보수를 얻는다. 이는 '강건한 성과' 가
아니라 타인 보수의 **이전(transfer)** 이다. 따라서 H3/H4 의 확증 검정은 협력 계열
4종(TFT/GTFT/WSLS/ALLC)에 대해서만 수행하고, ALLD 비교는 해석 주의와 함께 탐색으로
등록한다. 이 결정은 결과를 보기 전에 고정했다.

### 7.6 identity 기억의 집단 수준 효과

해석적 분해식(§4.1)은 '다이애드마다 새 개체' 라는 전제에 의존한다. HalloReg 는
identity 기억을 가지므로, 개체가 파트너를 넘나들며 지속되면 항등식이 깨질 수 있다.
`run_round_robin(persistent=True)` 로 그 편차를 직접 측정할 수 있다.

---

## 8. 이론적 배경

- Albarracin, M. et al. (2026). Empathy modeling in active inference agents for
  perspective-taking and alignment. — 공감적 능동추론 IPD 의 원형
- Barrett, L. F. (2017). The theory of constructed emotion. — 핵심정서의 내수용
  예측부호화 구성
- Sterling, P. / Barrett & Simmons. Allostasis. — 설정점 자체가 이동하는 예측적 조절
- Bravetti, A. & Padilla, P. (2018). An optimal strategy to solve the Prisoner's
  Dilemma. *Sci. Rep.* 8:1948. — 최적 복제자 방정식(ORE)
- Smith, R. et al. (2022). A step-by-step tutorial on active inference. — EFE 정식화
