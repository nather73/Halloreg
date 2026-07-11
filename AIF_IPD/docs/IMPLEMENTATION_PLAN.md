# HalloReg — Implementation Plan

**H**ierarchical **Allo**static **Reg**ulation of empathy in Active Inference agents
for the Iterated Prisoner's Dilemma.

> **문서 상태 (v0.2 — 연구 보완판).** 본 문서는 리팩터링·가설 재설계(경로 이전
> `HalloReg/` → `Halloreg/AIF_IPD/`, 메인 엔트리포인트 단수화, H5/H7/H8 재설계,
> 가설별 개별 PNG)에 더해, **학술 비판에 대한 6축 보완**(통계·설계·단순성·해석·
> 시각화·지평 검증)을 반영한 **2차 수정본**이다. 새 절 §10(통계 인프라·사전등록),
> §11(H7H 지평 스윕), §12(H8E 진화 역학·베이스라인·게임구조), §13(주장 감사)이
> 추가되었다. 이론적 근거(§1, §4)는 유지된다.

---

## 0. 한 줄 요약

Albarracin et al. (2026) 의 공감 능동추론 에이전트는 공감 파라미터 **λ 가 고정**이다.
그 결과 λ 가 높은 에이전트는 착취자에게 구조적으로 무방비다. HalloReg 는 λ 를
**core allostatic belief state** — "내 항상성 불균형의 원인은 무엇인가"에 대한
누적 분포 — 로부터 **내생적으로 조절**한다. 조절의 개인차는 그 분포의 **귀인 범위**로
구현된다.

---

## 1. 이론적 근거

| 근거 | 본 구현에서의 역할 |
|---|---|
| **Albarracin et al. (2026)** | 사회적 EFE `G_social = (1−λ)·G_self + λ·G_other`, 관점취하기(perspective-taking), 계획지평 |
| **Kim (2020), mPFC 위계** | vmPFC = 내적/즉각/안정성, rmPFC = 중재/누적/allostatic, dmPFC = 외적/가소성/mentalizing |
| **Yoon & Lee (2018)** | vmPFC 는 즉각 피드백에, rmPFC 는 누적 증거에 반응 → `sophisticated` 토글의 신경적 대응 |
| **Sul et al. (2015)** | 친사회성의 vmPFC↔dmPFC 표상 구배 → `kappa`(기질귀인 성향) 개인차 |
| **Barrett / Katsumi (allostasis-first)** | 항상성 조절은 **반응적이 아니라 예기적**이다 → 불만 누적기의 tonic·anticipatory 항 |
| **Buergi et al. (2026), CHASE** | 믿음갱신 크기(KL, rTPJ) → `inversion.belief_update_magnitude()` 를 모델기반 fMRI 회귀자로 노출 |

### 왜 '예기적' 불만인가 (핵심 설계 결정)

순수 반응적(homeostatic) 설계에서는 불만 충전이 놀람(예측오차)에 비례한다:

```
grievance ← decay·grievance + gain · P(예측했던 협력) · [배신 발생]
```

이 경우 상대의 배신이 **예측 가능**해지는 순간 `P(예측 협력)→0` 이므로 충전이 꺼지고,
불만은 누출되어 λ 가 기저값으로 되돌아간다. 즉 **착취자를 정확히 알아볼수록 자기보호를
포기하는** 역설이 발생한다 (초기 구현에서 실제로 관측: λ 0.40 → 0.36 에서 정지, 협력률 92%).

Allostasis 는 항상성 **예측 오차의 사후 교정**이 아니라 **미래 필요의 선제적 충족**이다.
따라서 구동을 두 성분으로 분해한다:

```
betrayal_drive = tonic_weight + acute_weight · P(예측 협력)             # 배신 관측 시
anticipatory   = κ · disposition · disp_credence · (1 − P(예측 협력))   # 관측 없어도
grievance ← decay·g⁻ + protective_gain·(κ·disp·cred·betrayal_drive)
                     + anticipatory_gain·anticipatory
```

`anticipatory` 항이 없으면 자기보호 전환 이후 DD 상태에서 관측 배신이 사라져 불만이
누출되고 → λ 회복 → 재착취 → 진동한다. 이 항이 곧 **vmPFC(즉각) 위의 rmPFC(예기) 층**이며,
`sophisticated=False` 일 때 정확히 이 항이 꺼진다.

---

## 2. 아키텍처

리팩터링으로 최상위 패키지가 `HalloReg/` 에서 `AIF_IPD/` 로 바뀌었다. 리포지토리 루트는
`Halloreg/` 이며, 그 아래 파이썬 패키지 `AIF_IPD/` 가 놓인다 (`from AIF_IPD. …` 로 import).

```
Halloreg/                         ← 리포지토리 루트
├── .gitignore                    ★ 신규: __pycache__/·산출물 png·json·zip·log·에디터 잡파일 무시
├── README.md
└── AIF_IPD/                      ← 파이썬 패키지 루트
   ├── core/                      공유 유틸 + HalloReg 의 심장
   │  ├── constants.py            PD 상수, joint-outcome 인덱싱 (CC/CD/DC/DD)
   │  ├── generative.py           POMDP (A,B,C,D) + 해석적 EFE (pymdp 와 동치)
   │  ├── pymdp_backend.py        pymdp 1.0.x JAX 백엔드 (선택) + 동치성 검증
   │  ├── allostasis.py           ★ CoreAllostaticBeliefState, LambdaRegulator
   │  └── logging_utils.py        logging 설정, OS 적응형 한글 폰트
   ├── ipd/
   │  ├── agent.py                ToMEmpathicAgent (고정 λ), AdaptiveAgent (조절 λ)
   │  ├── env.py                  Environment, StrategyAgent(TFT/ALLC/ALLD/WSLS/GTFT/…),
   │  │                           형질전환 스케줄, make_opponent,
   │  │                           ★ CAPRICIOUS_CASES 카탈로그 + make_capricious_case /
   │  │                             capricious_case_spec / capricious_switch_rounds
   │  ├── sim.py                  run_dyad / run_many(멀티코어) / run_population
   │  │                           ★ run_population_spec / run_populations(병렬) / _pop_worker
   │  ├── tom/
   │  │  ├── tom_core.py          TheoryOfMind(정적 최적반응), GatedToM, RecursiveSocialEFE
   │  │  ├── opponent_simulator.py 계획 롤아웃용 상대 반응 시뮬레이터
   │  │  ├── sophisticated_planner.py 계획지평 H 정책 열거·롤아웃 (Albarracin 보존)
   │  │  └── inversion.py         입자필터 기반 상대 θ=(α,ρ,β,λ_j) 역추론
   │  └── metrics/exploitability.py 착취가능성·CC율·방어특이성·Welch t
   │                              ★ first_defection_round · payoff_growth 추가
   ├── scripts/
   │  ├── run_ipd_experiment.py   ★ 메인 엔트리포인트 (단수형; H1–H10 + 가설별 시각화)
   │  └── run_ipd_experiments.py  ★ 하위호환 셔틀 (복수형 → 단수형으로 위임)
   ├── docs/IMPLEMENTATION_PLAN.md
   └── results/                   (gitignored)
```

### 명세/이전 판 대비 변경 사항

* **패키지 이전**: 전 모듈의 `from HalloReg.` → `from AIF_IPD.` 로 일괄 수정.
  스크립트는 `sys.path` 에 리포지토리 루트(`Halloreg/`)를 삽입해 `AIF_IPD` 를 최상위
  패키지로 import 한다.
* **엔트리포인트 단수화**: 메인은 `scripts/run_ipd_experiment.py` **(단수형)** 이다.
  기존 복수형 `run_ipd_experiments.py` 는 동일 인자를 단수형에 위임하는 셔틀로 남겨
  기존 명령이 그대로 동작한다.
* **가설별 개별 PNG**: 3×3 통합 그림 하나(`halloreg_ipd.png`)를 폐기하고, 가설마다
  개별 파일(`h1_*.png` … `h10_*.png`)을 저장한다. 같은 실험에서 나오는 H2/H3, H9/H10 도
  **각각 별도 파일**이다. 여러 지표를 함께 보는 것이 의미 있는 가설(H5/H7/H8)만 한
  파일 안에서 다중 패널로 묶는다.
* **집단 시뮬레이션 병렬화**: H8 을 위해 `run_populations`(spawn 풀)를 추가했다.
* **제거한 파일 없음.** 명세된 모든 파일이 실제로 쓰인다.

---

## 3. 계산 백엔드: numpy 해석해 vs pymdp 1.0.x

요구사항은 pymdp 1.0.x(JAX)이되, "pymdp 없이 훨씬 간결하면 생략 가능"이었다.
본 설정에서 다음이 **해석적으로 성립**한다.

* 관측모형 `A = I₄` (joint outcome 완전관측) → `q(s) = o`, VMP 가 자명하게 수렴
* `policy_len = 1`

이때 pymdp 의 `infer_policies` 가 반환하는 `neg_efe(a)` 는 정확히

```
neg_EFE(a) = E_{s'~B(·|a,p_c)}[ C[s'] ]  +  H[ B(·|a,p_c) ]
             └── 실용적(pragmatic) 가치 ──┘   └─ 인식적(epistemic) ─┘
```

와 같다. `core/pymdp_backend.py::PymdpEFE.check_equivalence()` 가 이를 **런타임에 assert**
한다 (`--check-equivalence` 로 실행).

따라서:

* **기본 경로 = numpy 해석해** (`core/generative.py`). 수천 배 빠르고, 재귀적 확장(R1/R2)의
  수식이 그대로 드러난다.
* **검증/확장 경로 = pymdp 1.0.x JAX** (`--backend pymdp`, `AdaptiveAgent(use_pymdp=True)`).
  `Agent(A,B,C,D, batch_size=1, policy_len=1)`, 관점취하기는 `eqx.tree_at` 으로 선호(C)를
  교체해 구현한다. `A/B/C/D` 는 batch 선행축을 가진 jax 배열 리스트.

> JAX 는 `JAX_PLATFORMS=cpu` 로 강제한다 (Windows + AMD 에서 ROCm 미지원). 이 모델
> 규모에서는 CPU 로 충분하다.

---

## 4. HalloReg 의 핵심 기제

### 4.1 Core Allostatic Belief State

원인 축 `CAUSE_AXES = (transient, alpha, lambda_j, rho, beta)` 위의 분포.

* **기질적(dispositional)**: `alpha`(협력편향), `lambda_j`(상대의 공감), `rho`(상호성)
* **맥락적(contextual)**: `transient`(일시적 손실), `beta`(행동 정밀도 = 잡음/의도성)

배신(CD) 관측 시 증거 벡터는 추론된 정밀도 `precision = clip(E[β]/4, 0, 1)` 로 갈린다:

```
evidence[transient] = 1 − precision           # 낮은 β → 잡음
evidence[beta]      = 1 − precision
evidence[alpha]     = σ(−E[α]) · precision    # 높은 β → 의도
evidence[lambda_j]  = (1 − E[λ_j]) · precision
evidence[rho]       = σ(−E[ρ]) · precision
```

**β 가 의도성의 결정적 신호다.** 의도적 착취자(ALLD, 실행잡음 0) → 높은 β 로 추론되고,
잡음 파트너(TFT + 20% 실행오류) → 낮은 β 로 추론된다.

DD(상호배신) 관측도 증거로 반영하되, `update(betrayal, opp_defected)` 가 두 경로를 구분한다.
`CoreAllostaticBeliefState.update` 는 `defect_evidence = betrayal or opp_defected` 로 배신
증거를 켜되, DD 는 불균형이 작으므로 약한 가중으로 반영한다. 이것이 없으면 자기보호로
전환한 순간 CD 관측이 사라져 core belief 이 동결된다.

### 4.2 귀인 개인차: 두 겹의 마스크

| | 무엇을 제한하나 | 어디에 작용하나 |
|---|---|---|
| **θ 마스크** | 입자필터가 갱신할 수 있는 상대 형질 축 | `inversion._spread(axis) = 0.08 + 1.2·w` → `w=0` 이면 사전이 점질량으로 붕괴하여 그 축을 **설명 자원으로 쓸 수 없다** |
| **cause 마스크** | core belief 이 귀인할 수 있는 원인 축 | 증거 벡터를 마스킹 후 재정규화 → 잔여 증거가 허용축으로 **재분배** |

`attribution_target` 이 두 마스크를 동시에 결정한다:

| target | 의미 | 예측되는 실패 양상 |
|---|---|---|
| `all` | 의도 + 맥락 | — (참조) |
| `intent_only` | 맥락 귀인 불가 | 잡음을 의도로 오귀인 → **과잉처벌** |
| `alpha_only` | α 로만 상대를 읽음 | ρ·λ_j 전환 추적 실패 |
| `lambda_only` | λ_j 로만 | 모호한 배신을 공감결여로 오귀인 |
| `beta_context` | 의도 귀인 불가 | **과소방어**, 착취자에게 당함 |
| `rho_only` | ρ 로만 | (예비) |

`LambdaRegulator` 의 `disposition` 도 **허용된 축만으로** 구성된다 —
`lambda_only` 에이전트는 상대를 오직 E[λ_j] 로만 읽는다.

### 4.3 λ 조절기 (두 개의 누출적분기)

`LambdaRegulator.step(betrayal, opp_cooperated, …, opp_defected)`:

```
g⁻ (grievance) ← decay·g⁻ + protective_gain·attributed_disp
                          + anticipatory_gain·anticipatory
                          − forgiveness·[상대 협력]
g⁺ (trust)     ← trust_decay·g⁺ + trust_gain·coop_credence·[협력]
                          − trust_decay_on_betrayal·[배신]

λ_eff = clip( λ_base·(1 − g⁻) + (λ_max − λ_base)·g⁺ ,  0 , λ_max )
```

배신 신호 자체가 정교/즉각을 가른다:

```
defect_signal = betrayal  if sophisticated  else  (betrayal or opp_defected)
```

* `sophisticated=True` (rmPFC): CD(내가 협력했는데 당함)만을 배신 신호로 삼고,
  `disposition` 을 `precision_conf = E[β]/4` 로 게이팅하며, 누적된 `disp_credence` 로
  변조하고, 예기적 항을 켠다. → 맥락(내가 방어 중인지, 잡음인지)을 함께 고려.
* `sophisticated=False` (vmPFC 즉각): 정밀도도, 누적 귀인 신뢰도도 무시하고
  (`disposition = disp_credence = 1`), **자신이 방어 중(DD)인지 여부를 구분하지 못해
  상대의 모든 배신(CD ∪ DD)이 grievance 를 충전**한다. 즉 상대의 배신을 전적으로
  타인의 내재된 의도에 귀인한다. 예기적 항은 꺼진다. → **H5 즉각형의 신경적 조작화**.
* `kappa` (κ∈[0,1]): 기질귀인 성향의 개인차 — 전략적(κ≈1) vs 친사회적(κ≈0).

### 4.4 Albarracin 대비 두 재귀적 확장

* **R1 — 재귀적 ToM.** 상대도 나를 모형화한다고 가정한다.
  `RecursiveSocialEFE._recursive_opponent_prediction(depth=2)` 는 상대의 EFE 를
  "상대가 믿는 나의 정책"(= 내 협력률) 하에서 계산하고, 한 단계 더 나의 최적반응으로 정련한다.
* **R2 — 상대의 인식적 가치.** Albarracin 은 `G_other` 의 실용항만 쓴다. 여기서는

```
G_social(a) = (1−λ)·G_self(a) + λ·E[G_other(a)]
              − w_epi_self  · IG_self(a)
              − λ · w_epi_other · IG_other(a)
```

`IG_other` 는 상대가 얻을 기대 정보이득이다 (`inversion.expected_infogain`).

### 4.5 ToM 신뢰도 게이팅

`GatedToM`: `q_gated = r·q_learned + (1−r)·q_static`, `r = ESS/N` (입자필터 신뢰도).
학습된 ToM 이 아직 불확실하면 정적 최적반응 모형으로 후퇴한다.

---

## 5. 가설 ↔ 실험 매핑

기본 실행 규모는 **seeds=120, rounds=60, jobs=−1** 이다.

| 가설 | 실험 함수 | 출력 PNG | 주 지표 |
|---|---|---|---|
| H1 자기보호 | `exp_H1` | `h1_self_protection` | λ_final, 착취가능성 |
| H2 의도/맥락 구분 | `exp_H2_H3` | `h2_lambda_recovery` | λ 회복 |
| H3 형질 추정 구분 | `exp_H2_H3` | `h3_trait_estimates` | E[α], E[β] |
| H4 협력 복원 | `exp_H4` | `h4_cooperation_restoration` | CC율 |
| H5 즉각 vs 정교 | `exp_H5` | `h5_immediate_vs_sophisticated` | 방어량·누적보수·첫배신 |
| H6 잡음과 고정전략 | `exp_H6` | `h6_noise_fixed_strategies` | 평균 보수 |
| H7 변덕 상대 | `exp_H7` | `h7_capricious_partners` | 평균 보수·λ 반응성·국면분해·4×4 |
| H8 집단 역학 | `exp_H8` | `h8_population_dynamics` | CC율·집단후생·성장률 |
| H9 α 전용 귀인 | `exp_H9_H10` | `h9_alpha_only_attribution` | 누적 보수 |
| H10 λ 전용 귀인 | `exp_H9_H10` | `h10_lambda_only_attribution` | CC율 + 이중해리 |

### 5.1 H5 재설계 — 즉각형(vmPFC) vs 정교형(rmPFC)

**이전 판**은 `sophisticated` 토글만 켜고 껐다. **수정본**은 H5 를 두 유형의
*자기보호 계산*의 대비로 재정의한다:

* **정교형** (`sophisticated=True`, `attribution_target="all"`; rmPFC): 의도(α, ρ, λ_j)와
  맥락(β)을 함께 귀인. β 게이팅으로 증거가 쌓일 때까지 초기 관대함을 유지.
* **즉각형** (`sophisticated=False`, `attribution_target="intent_only"`; vmPFC): 맥락을
  귀인할 수 없고, 상대의 모든 배신(자신이 방어 중인 DD 포함, §4.3)을 기질 증거로 삼음.

**검정 (전부 Welch t):**

1. **착취자(ALLD) 방어량**(−착취가능성): 즉각 > 정교 — 즉각형은 grievance 가 조기
   포화되어 더 빨리·지속적으로 방어하므로 자원 방어가 많다.
2. **noisy TFT 누적보수**: 즉각 < 정교 — 이른 배신으로 보복 나선(DD)에 갇혀 손해.
3. **(보조) 첫 배신 라운드**: 즉각 < 정교 — 조기 배신의 직접 증거.

`supported = (1) ∧ (2)`.

**실증 결과(20 시드 프로브).** 착취자 방어량 즉각 −0.035 > 정교 −0.066 (p<0.0001);
noisy TFT 누적보수 즉각 114.2 < 정교 143.4 (p<0.0001), CC율 0.12 vs 0.92. → **지지**.
DD 충전(`opp_defected`)이 즉각형의 핵심 조작화이며, 이것이 없으면 즉각형은 자기보호
전환 후 grievance 가 누출되어 조작이 무력화된다.

### 5.2 H7 보완 — 다양한 형질전환(변덕) 상대

**보완 3축:** (1) `CAPRICIOUS_CASES` 카탈로그 8종(주기 10/12/15/20/30, 순환 5종:
상호성→착취→화해, 착취-선행, 교대, WSLS 혼합 등)을 case별·합산 비교. (2) 최종 보수 외에
**전환시점 정렬(switch-aligned)** per-round 보수·λ 추이와 |Δλ| 반응성. (3) TFT/GTFT/WSLS/
Adaptive **4×4 상호 대전**.

**정직한 보고 — 이득의 지평 의존성.** 24 시드·60 라운드 검증에서 AdaptiveAgent 는 전
case 합산 총보수에서 GTFT 에 **열세**다(1.775 vs 1.928, p<0.0001; 8/8 case 열세).
국면 분해가 그 기제를 드러낸다:

* **착취(ALLD) 국면**: adaptive 0.80 < GTFT 1.00 — 공감 prior(λ_base=0.4)로 방어가
  **느려** 자기보호 이득이 발생하지 않는다.
* **협력 국면**: adaptive 2.43 < GTFT 2.55 (p<0.001) — grievance 히스테리시스로 인한
  **화해 지연 비용**.
* **유의하게 성립하는 것**: λ 가 의도 전환을 추적한다(|Δλ|≈0.08 > 0). 즉 ToM 기제는
  '작동'하나 이 지평·이 보수 구조에서 순이득으로 전환되지 않는다.

진단상 원 저장소식 단순 3국면 스케줄 **120 라운드**에서는 adaptive 총보수가 GTFT 를
근소 상회한다(2.12 vs 2.07) — 이득이 지평 의존적임을 확인한다. 따라서 `supported` 를
단일 불리언이 아니라 dict 로 보고한다:

```
supported = {"overall_payoff_advantage": False,   # 60라운드
             "exploit_phase_defense":     False,
             "lambda_responsive":         True}
```

H7 는 억지로 지지시키지 않고, 국면 분해와 지평 의존성을 정직하게 시각화한다
(패널 F = 착취/협력 국면 보수 분해).

### 5.3 H8 전면 수정 — 집단 역학

**병렬화**: 모든 집단 replicate 를 `run_populations`(spawn 풀)로 실행.
**지표 확장**: CC율 + 집단 전체 평균 payoff + 초기→후기 기대보수 증가율(`payoff_growth`).
**네 한계 극복 + 통합:**

* (a) λ_base ∈ {0.1, 0.2, 0.22, **0.24**, 0.26, 0.28, 0.3, 0.4} 스윕 (Albarracin 경계 0.24 포함).
* (b) 소집단에서 고정전략(TFT/GTFT/WSLS/ALLC/ALLD) 동수 투입 시 CC 기울기 대조.
* (c) 즉각형:정교형 비율 ∈ {0, 20, 40, 60, 80, 100}%.
* (d) 대규모 N∈{30, 100} 확장(희소 무작위 짝짓기 표본화), **변덕 상대 포함 base**
  (`LARGE_MIX`: TFT/GTFT/WSLS/ALLC/ALLD/random/capricious), Adaptive 비율에 따른 CC.
* (f) 대규모 공정비교: 동일 N=100·변덕 포함 환경에서 filler ∈ {adaptive, 각 고정전략} 를
  동일 비율로 투입해 기울기 비교.
* (e) **통합 요인 시뮬레이션** (N≈100): λ × 즉각형 비율 × Adaptive 비율 격자 + CC 에 대한
  다중회귀(OLS)로 Adaptive 비율(frac)의 독립 효과 검정.

**정직한 보고 — 지지 기준과 ALLC 착시.** 핵심 주장(집단 규모에서 Adaptive 비율↑ → 협력↑)의
판정은 **규모 확장 증거**로 한다:

```
supported = (slope[λ=0.4] > 0)  ∧  (대규모 CC 기울기 > 0, 모든 N)  ∧  (통합 OLS: frac 계수 > 0)
```

고정전략 대조(b/f)는 지지/미지지 **게이트가 아니라 한정 분석**이다. 얕은 CC 지표에서는
무조건협력(ALLC)이 기계적으로 우세할 수 있고, 심지어 집단 평균 payoff 도 앞설 수 있으나,
이는 base 의 착취자에게 보수를 상납해 총량을 부풀린 **취약한 협력**이다. 바로 이 착시를
드러내기 위해 CC 외에 후생·성장 지표를 병기했다.

**추가 통찰 — λ 경계.** λ 에 대한 CC 기울기의 구배가 양수이고, **λ=0.24 부근에서 기울기
부호가 전환**된다(λ=0.24 에선 음수, λ=0.4 에선 양수). 이는 Albarracin 의 공감 경계를
집단 수준에서 재현한 것으로, 결함이 아니라 예측이다.

### 5.4 H9/H10 — 귀인 범위 절제 + 이중해리

`attribution_target ∈ {all, intent_only, alpha_only, lambda_only, beta_context}` 절제.
H9 는 변덕 상대(의도 변동)에서 `alpha_only` < `all`(누적보수), H10 은 정적 잡음 TFT 에서
`lambda_only` < `all`(CC율)을 검정한다. 추가로 **이중해리**를 보고한다:
`intent_only` 는 잡음을 의도로 오귀인해 **과잉처벌**(정적 잡음 상대 CC 붕괴),
`beta_context` 는 의도 귀인 불가로 **과소방어**(착취자에게 보수 상실).

---

## 6. 병렬 실행

다이애드·집단 replicate 는 완전 독립이다.
`ipd/sim.py::run_many` 와 `run_populations` 가
`multiprocessing.get_context("spawn").Pool` 로 코어에 분배한다.

* **spawn 필수**: JAX 는 fork-unsafe.
* 워커 oversubscription 방지를 위해 `sim.py` 최상단(= jax import 이전)에서
  `XLA_FLAGS`(intra-op 1), `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `JAX_PLATFORMS=cpu` 를 설정한다.
  메인 스크립트도 import 이전에 동일 환경변수를 이중 설정한다.
* `--jobs -1` → `cpu_count() − 1`. `--jobs 1` → 순차. 시드로 결정되므로 워커 수와
  결과가 무관하다.
* `mp.freeze_support()` 로 Windows/spawn 안전.

---

## 7. 실행법

리포지토리 루트(`Halloreg/`)에서 실행한다.

```bash
# 전체 실험 (기본값: seeds=120, rounds=60, jobs=-1)
python AIF_IPD/scripts/run_ipd_experiment.py

# 빠른 스모크 테스트 (seeds=3, rounds=30)
python AIF_IPD/scripts/run_ipd_experiment.py --quick

# 일부 가설만 / 순차 실행
python AIF_IPD/scripts/run_ipd_experiment.py --experiments H5 H7 H8 --jobs 1

# pymdp 1.0.x ↔ numpy 해석적 EFE 동치성 검증
python AIF_IPD/scripts/run_ipd_experiment.py --check-equivalence --experiments H1

# (하위호환) 기존 복수형 이름도 동일하게 동작
python AIF_IPD/scripts/run_ipd_experiments.py --quick
```

인자: `--seeds`(기본 120), `--rounds`(기본 60), `--jobs`(기본 −1),
`--experiments`(기본 전체: `H1 H2H3 H4 H5 H6 H7 H8 H9H10`), `--backend {numpy,pymdp}`,
`--quick`, `--tag`.

산출물: `results/` 에 가설별 PNG 10 개(`h1_*` … `h10_*`)와
`results/halloreg_results{tag}.json`(수치 요약).

### 그림의 불확실성 표기 (공통)

음영대와 오차막대는 **시드/replicate 간 표본표준편차(±1 SD, `ddof=1`)** 이다.

* **꺾은선 패널** — 평균 곡선 + 평균±1SD 반투명 음영대(`band()`). 라운드별로 시드 축에
  대해 SD 를 계산한다.
* **막대 패널** — 평균 막대 + 1SD 오차막대(capsize). 일부 패널은 개별 시드값을 지터
  산점(`scatter_seeds`)으로 겹쳐 분포 형태까지 드러낸다.
* **H6 의 SD**: 시드마다 라운드로빈 전체 평균을 하나의 관측치로 삼아, 상대 전략 간
  변동이 시드 간 변동과 섞이지 않게 한다.
* **H8 의 SD**: replicate 간. 대규모 집단은 희소 짝짓기 표본화의 분산도 포함한다.

의존성: `numpy`, `matplotlib`. 선택: `inferactively-pymdp>=1.0`, `jax`, `equinox`.
한글 폰트는 OS 적응형으로 자동 설정된다(Windows = Malgun Gothic).

---

## 8. 확장 훅 (fMRI / 향후 과제)

* `inversion.belief_update_magnitude(prev_means)` → Buergi et al. (2026) 의 믿음갱신 회귀자
  (rTPJ). 라운드별로 `agent.log["belief_update"]` 에 기록된다.
* `agent.log["disp_credence"] / ["ctx_credence"]` → rmPFC 중재 신호 후보.
* `agent.log["grievance"] / ["trust"]` → vmPFC 즉각가치 / 누적가치 후보.
* `CoreAllostaticBeliefState.prior_reliability` → 사전학습된 core belief 을 주입하는 자리
  (Sul et al. 2015 이타성 과제 등으로 개인별 사전 추정 후 주입).
* `StrategyAgent.schedule` / `CAPRICIOUS_CASES` → 임의의 형질 전환 스케줄(변덕 상대 일반화).

---

## 9. 알려진 한계

1. `policy_len=1` 이 기본이다. `planning_horizon>1` 이면 `SophisticatedPlanner` 가
   2^H 정책을 열거하므로 H≤4 를 권한다.
2. 완전관측(A=I₄) 가정 하에서는 pymdp 의 상태추론이 자명하다. 지각 잡음을 도입하려면
   `generative.build_A(noise)` 를 쓰고, 이 경우 numpy 해석해 경로는 무효가 된다(pymdp 경로 사용).
3. **H7 의 지평 의존성**: 60 라운드 예산에서는 화해 지연 비용이 방어 이득을 상회해
   AdaptiveAgent 가 GTFT 에 총보수로 열세다. ToM 이득은 더 긴 지평에서 발현된다(§5.2).
   이는 모형의 결함이 아니라 정직하게 보고되는 경계조건이다.
4. **H8 의 CC 착시**: 무조건협력자(ALLC)는 CC율(및 착취자 상납으로 부풀린 집단 payoff)을
   기계적으로 올릴 수 있다. 강건한 협력과 취약한 협력을 구분하려면 CC 만이 아니라
   payoff·성장률·강건성을 함께 보아야 한다(§5.3).
5. λ_base 는 구조적 사전으로 남아, 용서가 보답되지 않는 환경(완전 무작위 상대)에서는
   정교형이 계속 협력해 손해를 볼 수 있다. 의도적 설계다.

---

## 10. 통계적 추론 인프라 (v0.2 §1)

`ipd/metrics/stats.py` 를 신설해 모든 가설이 공통 통계 인프라만 쓰도록 강제한다.
효과크기 보고와 다중비교 보정이 구조적으로 담보된다.

**검정 원칙.** 모든 p 값은 순열/부트스트랩 기반(분포 무가정)이다 — 유계·바닥/천장
지표(CC율·λ)에서 정규근사보다 안전하다. 판정 문장은 p 가 아니라 **효과크기+CI**
중심으로 보고한다(`g=2.1 [1.8, 2.4]` 식). seeds=120 에서는 거의 모든 검정이
유의하므로, 유의성보다 크기·정밀도가 정보를 담는다.

**제공 함수.** `effect_size`(Hedges' g + 부트 CI), `effect_size_paired`(dz),
`perm_test`(순열/부호뒤집기), `one_sample_perm`(상수 대비 — 퇴화 두표본 검정 대체),
`holm`/`bh_fdr`, `wilson_ci`, `prop_perm_test`, `hazard_perm_test`(로그랭크형),
`boot_ci`/`boot_mean_ci`, `ols_boot`(군집 부트스트랩), `slope_boot`.

**다중비교 (두 층위).** 각 가설은 확증적(confirmatory) 1차 지표를 **하나만** 갖는다.
1차 지표들의 p 벡터에는 Holm(가족오류율)을, 그 외 모든 탐색 지표에는 BH-FDR(q=0.05)을
적용한다. JSON 에 `p_raw / p_holm / q_fdr / g / ci` 를 함께 기록한다.

**사전등록.** `experiments/hypotheses.yaml` 에 가설별 1차 지표·검정·방향·판정 상수·
근거를 코드 밖에 고정한다. 스크립트(`load_prereg`)는 이를 읽어 판정만 수행하며,
`cc>0.3`·`0.95×` 같은 상수는 yaml 에 근거와 함께 명시된다(사후 조정 방지).

**퇴화·절단 검정 교체.** (i) H1 의 λ_final 두표본 검정은 고정군이 상수(0.400)라
퇴화 — adaptive 군의 λ_final < λ_base 단일표본 순열로 대체. (ii) `first_defection_round`
는 우측 절단 — 평균 비교를 버리고 "T 내 배신 비율"(Wilson CI + 비율 순열)과 이산시간
위험곡선(로그랭크형 순열)으로 대체.

**공통난수(CRN) 강화.** 실행오류 뒤집힘 수열을 시드별로 사전 생성(`noise_seed`)해
조건 간 동일 공급한다. 모든 조건 간 비교는 짝지은 순열(시드별 차분 부호뒤집기)로
통일된다 — H5 방어 특이성의 짝지음이 예외가 아니라 기본이다.

---

## 11. H7H — 지평 의존성의 체계적 검증 (v0.2 §6)

v0.1.1 에서 관찰된 "60라운드에서 adaptive < GTFT, 120라운드에서 역전"은 사후
관찰이었다. H7H 는 이를 **모형의 파생 예측**으로 격상한다.

Δ(T) = payoff_adaptive − payoff_GTFT (CRN 짝지음) 를 두 스케줄 족에서 분리 추정한다:

- **족 a (전환수 고정).** 국면 길이 ∝ T (3국면). 전환당 비용이 상수이므로
  Δ(T) ≈ b·T − c·k₀ → **T 에 선형 개선**. [확증] dΔ/dT > 0 (시드 원자료 회귀 +
  부트스트랩 CI 가 0 배제).
- **족 b (주기 고정).** 전환수 k(T) ∝ T. 비용·이득 모두 T 에 비례 → **Δ(T) 근사
  T-불변**, 부호는 주기가 결정. [탐색] 주기별 기울기 ≈ 0.

두 족이 **다른 함수형**을 예측한다는 점이 검정력의 원천이다 — "길면 이긴다"가 아니라
전환당 비용 c 와 라운드당 이득 b 의 분해 모형을 검증한다. Δ(T)=0 의 교차 지평 T\* 를
선형 근사 근으로 추정하고 시드 부트스트랩으로 CI 를 붙인다.

**기제 조작 검증.** 화해 지연의 원천인 히스테리시스(forgiveness↑, grievance decay↓)를
조작하면 "forgiveness↑ → c↓ → Δ 개선 → T\*↓"의 방향성 예측이 나온다. 이것이 확인되면
지평 의존성은 모형의 파생 예측이 된다(탐색으로 병기).

---

## 12. H8E·베이스라인·게임구조 (v0.2 §3)

**H8E — 실제 '집단 역학'.** 기존 H8 은 전략 갱신이 없는 정적 매칭이었다. `ipd/evolution.py`
로 2단 역학을 도입한다: (1) 유형 쌍별 평균 보수 행렬 Π(9유형: 고정 7 + adaptive 2)를
다이애드로 추정한 뒤 **복제자 동역학**으로 침입 성장률·고정점·유역(basin)을 분석한다.
(2) 검증용 **경험보수 Moran 과정**(N=100, 적합도 비례 출생 + 돌연변이 ε=0.01)을 소규모로
돌려 평균장 예측과 대조한다. 가설이 "AdaptiveAgent 는 혼합 집단을 침입하고(침입 성장률>0)
협력 균형의 유역을 넓힌다"로 재정식화되어 비로소 '집단 역학'이 된다. [확증] 기준 혼합
상주집단에 대한 adaptive 침입 성장률 > 0 (Π 시드 부트스트랩 CI 가 0 배제).

**매칭 분산 분해.** `run_population_spec` 에 `match_resamples` 를 추가해 replicate 내에서
매칭 그래프를 M 회 재표집, 분산을 "매칭 내 vs replicate 간"으로 분해 보고한다. N=30 은
완전 라운드로빈으로 전환해 표본화 오차 자체를 제거한 기준선을 둔다.

**구성 민감도.** LARGE_MIX 를 (i) ALLD 비중 {10,20,30}% 스윕, (ii) Dirichlet(α=10·기준)
무작위 구성 20개, (iii) capricious case 개체별 무작위 배정 — 세 방식으로 흔들어 결론
(frac 기울기 부호)의 강건성 영역을 지도화한다.

**게임구조 일반화 (GS).** 보수행렬을 협력지수 CI=(R−P)/(T−S) 로 매개화(`set_coop_index`)
해 {0.4(현행), 0.5, 0.6} 을 스윕한다. 핵심 결론(H1 부호, H7 열세, H8 방향)만 이 축을
태워 비용을 제한한다. **주의:** 현행 게임(R=3,P=1,T=5,S=0)은 CI=0.4 이다(보완안 초안의
"0.6 현행"은 착오). 2R>T+S 를 만족하는 상향 스윕으로 조정했다.

**비-ToM 학습 베이스라인.** `ipd/baselines.py` 에 memory-1 Q-러너(ε-greedy), 베이지안
최적반응(Thompson, 4-상태 가치반복), fictitious play 를 추가한다. H7/H9 비교군에 포함해
**"잠재 형질 추론(ToM)의 이득"을 "아무 학습이나 하면 되는 이득"과 분리 귀속**한다.

**H3 복원 연구.** `scripts/validate_tom_recovery.py` — 입자필터와 동일한 생성 모형으로
알려진 θ=(α,ρ,β) 합성 상대를 만들고, 사후 편향·RMSE, 90% 신용구간 커버리지, 사후 수축을
보고한다. 낮은 β 에서 α·ρ 가 혼동되는 식별 불가능 영역을 히트맵으로 드러낸다. 이로써
H3 는 판별(discrimination)에서 복원(recovery)으로 격상된다.

---

## 13. 주장 감사 (Claims Audit) — 확증/탐색 구분 (v0.2 §4)

로그·JSON·문서의 모든 결과 문장에 `[확증]`/`[탐색]` 태그를 단다. 확증은 사전등록된
1차 지표(Holm 보정), 탐색은 그 외 전부(BH-FDR).

| 가설 | [확증] 1차 지표 | 증거 강도 | 주요 [탐색] |
|------|----------------|-----------|-------------|
| H1 | 착취가능성 adaptive<fixed (짝지은 순열) | 강 | λ_final<λ_base, 누적보수 |
| H2 | λ_final noisy_tft>exploiter | 강 | **β-경로 절제**(intent_only/beta_clamp) |
| H3 | E[β] exploiter>noisy_tft | 중 | E[α] 판별, **복원 연구**(별도 스크립트) |
| H4 | CC 비열등성(margin 0.95, floor 0.30) | 중 | — |
| H5 | 결합: 방어(즉각>정교)∧보수(즉각<정교) | 중 | 2×2×2 요인, 절단 대응 첫 배신 |
| H6 | 잡음 하 GTFT∨WSLS>TFT | 강(재현) | — |
| H7 | 전 case 총보수 adaptive>GTFT | **지평 의존**(H7H 참조) | λ 반응성(순열 영가설), 국면분해, 베이스라인 |
| H7H | 족a dΔ/dT>0 | 강 | T\*, 족b, **히스테리시스 조작** |
| H8 | λ=0.4 조건부 frac 기울기>0 | 중 | λ 부호구조(정성적 일치), filler 대조, **ALLC 이전 회계** |
| H8E | adaptive 침입 성장률>0 | 중 | 협력 유역 확장, Moran 교차검증 |
| H9 | 변덕 보수 all>α-only | 중 | λ 반응성, 비-ToM 베이스라인 |
| H10 | 정적 CC all>λ-only | 중 | — |
| GS | 없음(전면 탐색) | — | CI 스윕 강건성 지도 |

**해석 재보정 두 건.** (1) **λ=0.24 주장.** "재현(reproduction)"을 "정성적 일치
(qualitative consistency)"로 강등한다. λ별 frac 기울기에 부트스트랩 CI 를 붙여 0 을
배제하는 λ 구간만 부호를 주장한다. Albarracin 원 설정의 직접 대응은 선택 과제로 분리.
(2) **ALLC 착시.** "ALLC 투입으로 후생이 증가"라는 해석을 측정으로 바꾼다 —
`run_population_spec` 의 착취 흐름 회계(`alld_exploit_gain`)로 "집단 후생 중 x% 가
ALLD 로의 이전(transfer)"을 정량화하고, 강건 협력 지표로 "착취자 제외 부분집단 평균
보수"(`nonexploiter_mean_payoff`)를 병기한다.

## 14. 시각화 개정 (v0.2 §5)

불확실성 표현을 목적별로 이원화한다: **검정 주석 패널 = 평균의 부트스트랩 95% CI
오차막대**, **궤적 밴드 = 시드 16–84 백분위**(유계 지표에서 [0,1] 경계 초과 없음).
모든 패널에 n(시드/replicate) 표기. H8 의 이중 y축은 상하 분리 소패널로 교체하고,
통합 히트맵은 회귀의 **조건부 기울기 플롯**(λ별 frac 효과 + CI)으로 대체했다 —
상호작용 모형과 정합적이다. H7 전환정렬 밴드는 계산 순서를 뒤집어(시드별 창 평균 →
시드 간 CI) 불확실성과 case 이질성(얇은 case별 선)을 분리한다. 그림마다 벡터 PDF 와
캡션 메타데이터(JSON)를 함께 내보낸다.

## 15. 실행법 (v0.2 추가)

```bash
# 전체 배터리 (H1–H10 + H7H/H8E/GS)
python scripts/run_ipd_experiment.py

# 보완 실험만
python scripts/run_ipd_experiment.py --experiments H7H H8E GS

# ToM 파라미터 복원 연구 (H3 격상)
python scripts/validate_tom_recovery.py

# 황금 시드 회귀 테스트 (엔지니어링 안전망)
python tests/test_golden.py            # 검증
python tests/test_golden.py --update   # 스냅숏 재생성
pytest tests/                          # (pytest 설치 시)
```

의존성: `experiments/hypotheses.yaml` 사전등록 로딩에 PyYAML 이 있으면 사용하고,
없으면 내장 사본으로 폴백한다(선택 의존성).
