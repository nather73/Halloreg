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

## 16. v0.3 확장 — 지평 이중화·변덕 격자·진화 동역학 재정식화

### 16.1 지평(T) 이중화와 정직한 지평 비교
`main()` 을 재구성해 `--rounds` 를 목록으로 받고(기본 `60 240`), 각 지평에서
전 실험을 반복 실행한다. 지평별 실험 그림에는 `_T60`/`_T240` 접미가 붙는다.
H7H·H8E 는 함수 내부에서 여러 T 를 함께 스윕하므로 지평 루프에서 제외하고
(GLOBAL_EXPERIMENTS) 1회만 실행한다.

`interpret_horizons()` 는 두 지평의 `supported`(딕셔너리로 분해된 하위기준
포함)를 비교해 **지지 여부가 뒤집히는 하위기준만** `support_flips` 로 추린다.
뒤집힘이 있으면 H7H 의 Δ(T) ≈ b·T − c·k 기제(학습·화해의 고정비용이 긴
지평에서 상환; T*≈170 부근 교차)와 정합하는지 지표 방향으로 확인하라는
해석 텍스트를 자동 생성한다. 이는 "짧은 지평의 결론을 긴 지평으로 외삽하지
않는다"는 방법론적 약속을 코드 수준에서 강제하는 장치다. `horizon_overview.png`
는 확증 지표별 (방향 성립 ∧ Holm 유의) 판정을 지평 축으로 나열해, 지평
의존적 가설을 한눈에 드러낸다.

### 16.2 H7 — 변덕 상대의 (주기 × 순환족) 격자와 SwitchingAgent
기존 H7 은 고정전략 사이의 전환만 다뤘다. v0.3 은 case 카탈로그를
전환 주기 {10,30,60,120} × 순환족 8종으로 재정의한다(총 32 case). 순환족
중 셋(coop_trap, adaptive_expl, adaptive_mix)은 국면에 **정교형 adaptive
에이전트**를 포함한다.

`SwitchingAgent` 는 비-AIF 경로(act/observe 인터페이스)로 구현하되, 국면이
전략일 때는 내부 `StrategyAgent` 의 kind 를 교체해 memory-1 연속성을
유지하고, 국면이 adaptive 일 때는 **지속되는 단일 AdaptiveAgent 를 매 라운드
step** 시킨다. 방출 행동은 관측 상태에 인코딩되어, 국면 전환을 "실행 오류와
동일한 self-model 정합" 방식으로 처리한다(온난 전환 — cold reset 이 아니라
행동 계층의 연속적 재조정). 실행 잡음은 방출 계층에서 대칭으로 부과한다.

주기 의존성은 두 층위로 보고한다: (a) 주기별 Δ(P)=adaptive−GTFT 를
one-sample 순열 + CI 로 검정하고 `switches_in_horizon` 을 병기하며, 지평 내
전환이 0회인 퇴화 case(예: T=60 에서 P=60,120)를 무전환 대조로 명시한다.
(b) 순환족 분해(fixed_cycles vs aif_cycles). 지지 부호가 주기에 따라 갈리면
`period_dependent=True` 로 표시하고 그림의 Δ(P) 패널을 색으로 구분한다.
확증 지표는 종전대로 전 case 합산 보수 하나로 유지한다(다중검정 통제).

### 16.3 H8E — 침입·복제자·상태공간의 재정식화
`estimate_payoff_matrix` 에 대칭 재사용(`symmetric=True`: i≤j 다이애드만
실행하고 Π[i,j]/Π[j,i] 를 동시 기록 — 다이애드 절반화)과 seed_offset 을
추가했다. 격자는 env_error {0,.05,.10,.15,.20} × T {60,240}, Π seeds=50.

확증 지표는 **err=0.10 의 지평별 침입 성장률** 두 개(T=60, T=240)로, LARGE_MIX
상주집단에 대한 adaptive 침입 g>0 를 부트스트랩 CI 로 판정한다. 지지가 지평
의존적이면 `horizon_dependent` 로 정직 보고한다(v0.1.2 에서 T=60 은 g<0:
학습·화해 고정비용 미상환, H7H T*≈170 과 정합).

탐색 축은 다섯 갈래다. (a) 전 격자 침입 g(즉각형 adaptive 대조 포함).
(b) **역방향 침입**: 순수 adaptive 및 혼합(85% adaptive+15% LARGE_MIX)
상주집단에 각 고정전략+즉각형이 침입하는 g 를 전 격자에서 부트스트랩하고,
ALLD 격퇴 여부를 확인한다. (c) 상주 구성 민감도: Dirichlet 무작위 구성
100개의 침입가능 비율과 g~ALLD비중 기울기(slope_boot) + ALLD 비중 구조적
스윕. (d) **Replicator 상태공간**: attractor_analysis(종착점 0.02 양자화
군집→끌개 목록+유역 비율), 3-유형 부분계(ad-alld-TFT, ad-alld-ALLC)의
심플렉스 위상 초상(궤적+끌개)과 cooperation basin 지도(basin_map_3), 9유형
협력 유역 비율의 adaptive 유/무 확장 Δ. (e) Moran 평균장 교차검증(지평별).

복제자 궤적 그림은 9개 유형 전체를 고정 색 매핑(TYPE_COLORS)으로 그리고
legend 를 명시한다. 세 그림으로 분할: `h8e_invasion_structure`(보수 구조·
침입 격자·역침입 forest·구성 민감도), `h8e_replicator_dynamics`(궤적·역침입
동역학·끌개 조성·Moran), `h8e_state_space_basins`(위상 초상·유역 지도·유역
확장 격자).

### 16.4 예상 로컬 런타임
seeds=240, rounds={60,240} 전 배터리는 16코어(Ryzen AI MAX+ 395)에서
대략 1.5–3시간으로 추정된다(H8 population 실행과 H8E 격자가 지배적).
컨테이너(1 CPU)에서는 `--quick` 스모크만 검증하고, 본 실행은 로컬 권장.

## 17. v0.4 확장 — GTFT 이중화·H7 상대이점 명시·H11 Keystone

### 17.1 GTFT 의 두 용서 기제 (확률론적 vs 횟수 기반)

기존 구현은 GTFT 의 용서를 확률(generosity=0.3)로만 정의했다. 그러나 관용은
**인내 임계(patience threshold)** 로도 정식화할 수 있다: 상대의 배신이 N회
누적되기 전까지는 용서하고, 임계를 넘어선 뒤에야 보복하는 방식. v0.4 는 두
기제를 별개 전략 kind 로 분리 구현한다 (`ipd/env.py`):

* `generous_tft` — **확률론적 용서.** 각 배신을 독립 확률 `generosity` 로
  용서(협력). Nowak & Sigmund (1992) 의 고전적 정식화. 배신 이력과 무관.
* `generous_tft_count` — **횟수 기반 용서.** 관측한 focal 의 **연속** 배신
  카운터(`_defect_streak`, observe 에서 갱신; 협력 1회 관측 시 초기화)가
  `forgive_streak`(기본 2) 이하이면 용서, 초과하면 보복(TFT). 결정적이라
  잡음-용서 상호작용이 확률형과 질적으로 다르다: 확률형은 배신 폭주에도
  주기적으로 문을 열어 주지만, 횟수형은 임계 초과 후 상대가 협력으로 복귀할
  때까지 문을 닫는다.

두 유형은 GTFT 가 등장하는 **모든** 지점에 병기된다: H6 잡음 강건성(전략
목록+탐색 지표 2개: 횟수>TFT, 횟수 vs 확률 직접 비교), H7(focal·rival·순환족
`recip_expl_reconC`/`wsls_mixC` 8 case 추가로 총 40 case·5×5 대전), H7H(족a
Δ_count(T) 탐색 곡선+기울기), H8(LARGE_MIX 0.12→0.06/0.06 분할, filler 목록),
H8E(유형계 10종으로 확장, 협력 유역 coop_idx), H11(조건부 협력자 기저
TFT 4/GTFT확률 2/GTFT횟수 2/WSLS 4), GS(H7 미니 3-focal). H7 의 확증 지표는
사전등록 정합성을 위해 GTFT(확률) 대비로 유지하고, 횟수형 대비는 탐색으로
병기한다.

### 17.2 H7 — adaptive 상대이점의 주기 × 지평 명시

v0.4 요구: capricious 상대에 대해 GTFT 뿐 아니라 **WSLS 와의 비교**를 수행하고,
전환 간격(주기 P)과 총 라운드 수(지평 T)에 따라 adaptive 가 payoff 를 더 많이
획득하는 조건을 명시할 것.

구현 (`exp_H7` 의 `adaptive_advantage`): rival ∈ {GTFT확률, GTFT횟수, WSLS}
각각에 대해 (i) 전 case 합산 짝지은 순열 + dz, (ii) 주기 P∈{10,30,60,120} 별
Δ(P) 평균·부트 CI·양측 p·유의성(0 배제)·`adaptive_wins` 부호, (iii)
`win_periods` (adaptive 우위 주기 목록). 지평 T 의존성은 main 의 지평 이중화
(T=60/240 반복 + `interpret_horizons`)와 `_horizon_metrics` 의 rival 별 Δ 로
자동 비교된다 — "어느 전환 간격·어느 지평에서 adaptive 가 이기는가" 가
`summary.json` 의 `adaptive_advantage`/`horizon_comparison` 과 그림 (d) 패널
(주기 × rival 그룹 막대, *=CI 0 배제)에 직접 나타난다.

### 17.3 H11 — Keystone(핵심종) 검정

**프레임.** H7(경쟁 열세)·H8E(g<0 침입 실패)는 "adaptive 가 승자인가" 를
기각했을 뿐, "adaptive 가 남(협력자)을 이롭게 하는가" 는 별개 질문이다.
생태학의 keystone 논리처럼 본인은 번성하지 못해도 생태계 전체의 협력을
떠받치는 개체가 존재할 수 있다. 가설은 두 주장으로 분해된다:
* **주장 A** (집단 내·유형 간): 협력자에게 ALLD 대비 상대이점.
  DV = gap = (기저 조건부 협력자 가중평균 보수) − (ALLD 보수).
* **주장 B** (집단 간): 집단 상호협력률(CC) 증진.

**구성 인공물 차단 — 치환 설계.** "adaptive 를 넣었더니 CC 가 올랐다" 는 그
자체로 아무것도 증명하지 못한다 — ALLD 를 빼고 넣었다면 CC 상승은 착취자
감소의 산물이다. 그래서 비-swap 구성(특히 ALLD 9/30=30% 착취 압력)을 고정한
치환 설계를 쓴다: N=30 완전 라운드로빈(매칭 잡음 제거), 기저 = ALLD 9 +
TFT 4 + GTFT확률 2 + GTFT횟수 2 + WSLS 4, swap dose ∈ {0,3,6,9}, 잔여는
random filler 패딩. dose 0 은 swap 이 없어 arm 무관 동일 → **전 arm 공유
셀**(Δ≡0 앵커). CRN 짝지음은 population seed 를 `stable_seed("H11pop",
dose, rep)` 로 arm 간 공유하여 달성 (공유 기저 구성원의 다이애드 시드가
arm 간 일치).

**arm 설계 — 각 대조가 다른 교란을 분리.**

| arm | swap 유형 | 분리하는 것 |
|---|---|---|
| treatment | 정교형 AdaptiveAgent (all 귀인) | — |
| control-1 (핵심) | fixed-λ ToMEmpathic (λ=0.4) | λ 자기조절 자체의 효과 |
| control-2 | 즉각형 adaptive (sophisticated=False) | 정교함(rmPFC)의 효과 |
| control-3 | 추가 TFT | 조건부 협력자 증량 대비 우위 |
| control-4 | ALLC | 순진한 협력자 대비 우위 |

control-1 이 과학적으로 가장 중요하다: ToM·공감 기계를 동일하게 두고 오직
λ 조절만 뺀 대조라서, treatment−ctrl1 차이가 곧 "λ 위계적 조절이 남을
돕는다" 는 가설의 순수 효과다.

**확증 지표** (`experiments/hypotheses.yaml` H11): ΔCC = CC(treat)−CC(ctrl1)
와 Δgap 각각의 dose 기울기 > 0 (rep 군집 slope_boot, CI 0 배제).

**탐색·기제.** (i) ALLD 억제: 최대 dose 에서 payoff_ALLD(treat) <
payoff_ALLD(ctrl1) — 정적 라운드로빈에서 adaptive 가 gap 을 벌리는 직접
경로는 ALLD 를 만나 방어(D)해 ALLD 평균 보수를 T→P 로 끌어내리는 것뿐이다.
(ii) 협력자 보호: 유형별 payoff(treat) > payoff(ctrl1). (iii) **이전 회계
가드**: `alld_exploit_gain_total / group_payoff` 로 CC 상승이 실질인지
ALLD→피착취자 후생 이전의 착시인지 구분 (H8 §4 회계 재사용). (iv) **다이애드
기제**: `run_population_spec` 에 신설된 방향성 라벨쌍 행동 회계
(`dyad_behavior`: "li→lj" 별 CC/DD 율)로 adaptive→ALLD 의 DD 방어율(선택적
방어)과 adaptive→협력자의 CC 유지율(협력 지탱)을 직접 측정 — 두 값이 동시에
크면 H1·H5 의 선택적 방어가 집단 수준 keystone 효과로 발현됨을 보인다.
fixed-λ(tom_fixed)·즉각형(adaptive_imm) 비교군 병기.

**진화 프레임** (Π 추정 후 대수 연산 — 거의 무비용). 유형계 9종
(TFT/GTFT확률/GTFT횟수/WSLS/ALLC/ALLD/random/adaptive/tom_fixed)의 Π 를
`estimate_payoff_matrix`(대칭 재사용, env_error=0.10, seeds=30)로 추정한 뒤:
* **ALLD 침입장벽**: 협력자-only 상주(TFT .30/GTFT확률 .15/GTFT횟수 .15/
  WSLS .30/ALLC .10) vs 협력자+adaptive(20%) vs 협력자+tom_fixed(20%) 상주에
  대한 g_ALLD 를 Π-시드 부트로 비교. **장벽 심화** Δg = g(only) − g(+adaptive)
  > 0 이면 adaptive 가 협력 균형을 ALLD 침입으로부터 지킨다 — 상대이점의
  진화적 정의. fixed-λ 대조로 λ 조절 귀속.
* **협력자 클러스터 침입성**: ALLD-heavy 상주(ALLD .8/random .2)에 혼합 조성
  클러스터가 침입하는 성장률 — 단일 유형 침입의 일반화
  `cluster_invasion_growth(Pi, resident, cluster)` = (w_cluster·f) − f̄_res
  (`ipd/evolution.py` 신설). adaptive 포함/제외 클러스터 비교.
* **후보 유형별 협력 유역 확장**: widening(X) = coop_basin(S) −
  coop_basin(S∖X), X ∈ {adaptive, tom_fixed, TFT, GTFT확률, GTFT횟수, WSLS}.
  adaptive 의 widening 이 각 고정전략보다 큰지 **짝지은 Π-시드 부트**(동일
  boot 인덱스)로 검정. 유역 재적분 비용은 신설 `replicator_ends_batch`
  (초기점 배치를 (n,k)@(k,k) 행렬곱 수열로 벡터화)와 `coop_basin_frac`
  (공통 초기점 X0 공유로 짝지은 비교)로 실용화 — per-초기점 파이썬 루프 대비
  ~100배 가속.

주의(문서화된 구분): H8E 의 기존 역침입은 **순수 adaptive 상주**에 대한
것이라 이 분석과 다르다. H11 은 **협력자(+adaptive) 혼합 상주** 케이스를
계산한다.

**시각화** (`h11_keystone.png`, 3×3): (a/c) dose→CC·gap arm 곡선(dose0 공유
명시), (b/d) Δ 기울기 forest(확증=vs fixed-λ 강조), (e) 최대 dose 라벨별
보수(ALLD 억제+협력자 보호), (f) keystone 기제 막대(→ALLD DD율, →협력자
CC율; actor 3종), (g) ALLD 침입장벽 forest, (h) 클러스터 침입성, (i) 후보별
유역 확장 forest.

### 17.4 기존 결과와의 정합

H8 filler 회귀의 frac×TFT<0 (adaptive 가 TFT 추가보다 CC 를 더 올림), H8
민감도의 ALLD 비중 0.1–0.3 전 구간 양의 frac 기울기, H8E 의 전 격자 양수
basin widening 과 T=60 ALLD 역침입 격퇴는 모두 keystone 가설의 약한 지지
증거였다. H11 은 이를 (1) ALLD·비-swap 구성 고정 치환 설계, (2) fixed-λ
대조, (3) cooperator−ALLD gap 확증 지표, (4) 혼합 상주 ALLD 침입장벽으로
정면 검증한다. 결과가 H7/H8E "기각" 과 공존하면 — adaptive 는 지지만 남을
돕는다 — 두 기각의 재해석 서사가 완성된다.

================================================================================
v0.6 — 네 구조적 한계 보완 (가변 페이오프·A-B-A·최적복제자·ANN-ToM)
================================================================================

기존 실험군의 네 한계와 각 대응:

  (1) 고정 payoff·이분법 협력/경쟁  → §VP 가변 페이오프(연속 CI, 극단 포함)
  (2) 의도 전환 후 복구 미검증        → §ABA A→B→A 의도추론 복구(용서 게이팅)
  (3) 단순 복제자 동역학              → §ORE 최적복제자(집단 수준 경쟁)
  (4) ToM 의 신경 구현·학습 미검증    → ANN-ToM(입자필터 사후 증류)

추가로 자기/타인 통제권 귀인(Spiering 2025)을 선택적으로 도입(기본 off).

--------------------------------------------------------------------------------
§VP  가변 페이오프 (ipd/variable_payoff.py, exp_VP/fig_VP)
--------------------------------------------------------------------------------
근거: Pisauro et al. (2022) Space Dilemma B6 — 협력/경쟁을 연속체로 일반화하고
재분배(맥락)로 되갚음 계수를 정규화. 조작화: 이산 IPD 를 유지하되 CI=(R−P)/(T−S)를
라운드별 변동(set_payoffs 로 in-place; 음수 교착·1 이상 조화까지). adaptive 의 EFE
선호 C 가 현재 맥락 효용을 반영 → 맥락-의존 효용. 고정전략은 CI 에 무감.

핵심 설계 결정(정직 보고 원칙):
 · '최고 고정전략' = 시드별 max(승자의 저주 편향) 아님 → 평균 최고 전략 1개 짝지음.
 · 확증은 시변 레짐(oscillate·blocks·aba)만 — 고정-극단 레짐(harsh/harmony/deadlock)
   에서는 최적 고정전략과 대등할 수 있고(우위 없음이 정상), 이는 adaptive 의 이점이
   '맥락 변동 강건성' 임을 보임(대조군으로 병기).
 · 협력 유역은 제거가 아니라 중립 filler(random) 대치로 측정(H12 원칙 B), RE·ORE 병기.
확증: C-VP1 시변 payoff 우위>0, C-VP2 시변 RE sub_widening>0.
병렬: run_variable_dyad 는 라운드별 전역 보수를 in-place 변경 → 스폰 프로세스 간
분리 메모리에서만 안전(_var_worker/run_variable_many). CI 레짐은 이름(문자열)으로
워커에 전달(클로저 비직렬화 회피).

--------------------------------------------------------------------------------
§ABA  A-B-A 의도 복구 (exp_ABA/fig_ABA, ipd/variable_payoff.aba_schedule)
--------------------------------------------------------------------------------
상대: TFT(A)→ALLD(B)→TFT(A) 형질 전환. 사전 예상은 'adaptive 가 복구' 였으나
실측 기제는 상반: 완전한 allostatic 에이전트는 배신 후 자기보호 고착으로 A 복귀 시
자동 복구하지 않음(히스테리시스). 자기 방어적 배신이 상대 개심 관측을 스스로 차단하는
'배신 후 선택적 관측' 함정. 용서(재탐색)가 복구를 게이팅함을 용량-반응으로 확증:
 · C-ABA1 高용서(0.30): 복구오차<0.15 & CC 복원비 CC(A3)/CC(A1)>0.70.
 · C-ABA2 복구오차∼용서 기울기<0 (replicate 부트 CI 상한<0).
확증은 최장 지평에서 판정. 탐색: β-절제·통제권·비-ToM·자기충족 함정(A3 배신율→
복구실패). 이는 프로젝트의 selective-observation-after-betrayal 문헌과 정합.

--------------------------------------------------------------------------------
§ORE  최적복제자방정식 (ipd/evolution.py: ore_trajectory/ore_*_basin/ore_two_type)
--------------------------------------------------------------------------------
Bravetti & Padilla (2018). RE 는 개체 선택만 → T>R>P>S 하 배신 지배. ORE 는 집단
수준 경쟁 가정으로 최종 평균적합도 g(x(τ))=xᵀΠx 최대화 최적제어:
    forward   ẋ_a = x_a(p_a−⟨p⟩)                x(0)=x0
    backward  ṗ_a = ⟨p⟩p_a − ½p_a²              p(τ)=∇g(x(τ))=(Π+Πᵀ)x(τ)
공상태 p 가 '집단 최종 보상' 을 역방향 전달 → 이기적 개체도 협력. FBSM 으로 BVP
해결, 공상태 Riccati 폭발은 보수 규모 배수 클램프(유역 판정은 종착 부호에만 의존 →
정성 결론 불변), 초기점 전체 벡터화(N× 가속). 확증: C-ORE1 ORE 유역>RE 유역(레짐
짝지음), C-ORE2 ORE 하 adaptive sub_widening>0. 재현: Bravetti Fig.1(2-유형).
정직 사례: deadlock(CI<0) 에서 RE 유역=0 이나 ORE 유역>0 — 집단 경쟁이 교착 구제.

--------------------------------------------------------------------------------
ANN-ToM  (ipd/ann_tom.py, scripts/validate_ann_tom.py) — 독립 스크립트
--------------------------------------------------------------------------------
입자필터 사후(교사) 지식 증류로 ANN 의 베이지안 ToM 재현 검증. 매개적 상대
(ParametricOpponent, 참 θ 보유)가 입자필터와 동일 생성모형으로 행동 → 탐색적 focal
(ProbingFocal)이 상호작용 → 교사가 θ 사후평균·특성별 정밀도 산출. 두 ANN:
 · SchwarczGRU: GRU → 매 스텝 (θ평균4, 로그정밀도4).
 · KimGCNRNN: GRU 맥락에 4개 특성 질의가 주의 → 간선 엔트로피=특성별 reliability,
   특성 노드 간 학습 인접(관계구조) 1-hop 메시지 전달.
학습: 표준화 사후평균 가우시안 NLL(정밀도 헤드로 불확실성). 평가: 교사 재현 R²
(≈0.77–0.80), 참 θ — α↔λ_j 가법 축퇴로 개별 회복 제한, 식별가능 합성 α+5λ_j 회복
(≈0.56), 보정(로그정밀도↔|오차| 음상관), A-B-A 일반화(θ 전환 복구). JAX/Equinox.

--------------------------------------------------------------------------------
통제권 귀인 (core/controllability.py, core/allostasis.py attr_gate)
--------------------------------------------------------------------------------
Spiering et al. (2025) 자기/타인 귀인. tPE=관측−예측, sPE=control·tPE(자기),
oPE=(1−control)·tPE(타인). AdaptiveAgent(controllability=True) 시 intended(직전 선택)
vs emitted(환경오류 반영 실제 방출) 불일치와 결과 유형(CD=순수 타인기인, 내 DEFECT의
DD/DC=자기기인)으로 통제권·w_other 산출. w_other(=attr_gate)로 dispositional grievance
충전과 예기 항, core 의 dispositional 증거를 게이팅(자기기인 배신은 상대 기질 불만을
덜 충전). 기본 off(attr_gate=1.0)로 H1–H12 결과 보존.

================================================================================
v0.6.1 — 협력 지표 개정 (이분 유역 → 행동적 CC율)
================================================================================

동기: 이분 협력 유역(coop_basin_frac; 협력-라벨 종착 점유>0.5)은 (i) 임의 임계 이분화
+ 유계 차분 천장효과, (ii) 라벨-행동 괴리(deadlock 에서 협력-라벨 유형이 실제 배신)로
협력을 과대·왜곡 측정한다. 특히 deadlock 의 ORE 유역이 인공적 고점(≈0.9)을 보이나 실제
상호협력은 거의 없음이 드러났다(§ORE 검증).

개정: 종착 조성의 **행동적 CC율** CC(x)=xᵀ·CCm·x 로 전면 대체.
  · CCm[i,j] = i·j 다이애드의 실제 CC 발생률(관점 무관 대칭). estimate_payoff_matrix·
    estimate_variable_payoff_matrix 가 함께 반환(CC, CC_raw 키).
  · evo.population_cc_rate / re_terminal_cc / ore_terminal_cc — 조성·RE종착·ORE종착 CC율.
  · 초기점 X0 공유(CRN)·Π·CCm 공통 시드 부트스트랩으로 짝지은 비교 유지.

적용 범위(사전등록 확증 재기반):
  VP  C-VP2  = RE CC-widening(변동레짐)>0
  ORE C-ORE1 = ORE 종착 CC율 > RE 종착 CC율(레짐 짝지음)
      C-ORE2 = ORE adaptive CC-widening>0
  H8E cc_widening_positive (basin_analysis → re_terminal_cc, adaptive 유−무)
  H11 후보별 CC-widening(_widening_all → re_terminal_cc); 주 확증(CC 기울기)은 원래
      행동 기반이라 불변
  H12 basin_slot → re_terminal_cc: C1(위협축 한계 keystone 기울기)·C2(중복축 비대체성)·
      C3(keystone 프런티어 교차)를 CC율 곡면 위에서 재계산. Shapley φ 도 CC율 기반.

불변: 상태공간 위상도(basin_map_3 끌개 유역)는 동역학 구조 시각화로 유지(협력 정량
지표 아님). 골든 회귀(controllability off)·H1–H7·GS 결과 불변.

================================================================================
v0.6.2 — 확증 검정 replicate 단위 개정 (레짐 → 시드)
================================================================================
문제: v0.6.1 검증(seeds=16)에서 VP C-VP2·ORE C-ORE1/2 가 전 레짐 부호 일관(4/4)인데도
p≈0.06~0.13 으로 미지지. 원인은 확증 검정이 레짐 수준(n=3~4)에서 짝지은 순열이라
p 하한(~1/2^n)에 막힌 것 — 시드를 늘려도 해결 불가능한 구조적 결손.

개정: evo.per_seed_terminal_cc(pi_raw, cc_raw, X0) 추가 — 시드 s 마다 Π_s·CCm_s(그
시드의 다이애드만)로 RE·ORE 종착 CC율을 계산. 확증은 (레짐×시드) pooled replicate 로
검정하고, 레짐 내 시드 인덱스로 짝지음(CRN X0 공유). 레짐 수준 부호 일관성은 탐색 병기.

적용: VP C-VP2(시드×레짐 단일표본), ORE C-ORE1(시드×레짐 짝지음)·C-ORE2(단일표본).
검증 결과(seeds=16): 6/6 확증 지지 — C-VP1 p=0.0002, C-VP2 p=0.006, C-ABA1 p=0.0002
(T=240), C-ABA2 p=0.0005, C-ORE1 p=0.0002, C-ORE2 p=0.0002.
불변: 효과크기·지표 정의(행동적 CC율)는 그대로. 검정 구성만 교정.


================================================================================
v0.6.3 — 확증 시각화 · 보수-매개 empathy_shift · 층화 끌개
================================================================================
(1) 확증 전용 그림: fig_VP/ABA/ORE 말미에 fig_*_confirmatory 훅. 공용 _conf_panel
    (지터+그룹/전체 평균±부트CI+임계선+통계 주석). exp_VP/exp_ORE 반환에 replicate
    배열 추가(cvp1/cvp2_replicates, per_reg.seed_re/seed_ore/seed_sw_*).
(2) constants.empathy_shift(λ,p) = (T−S)λ + (R−T+P−S)p + (S−P). 기본 보수에서
    레거시 5λ−p−1 과 비트 동일(10^5 표본 max diff 0.0; ref 골든 3다이애드 diff 0.0).
    배선: inversion._empathy_shift / validate_tom_recovery 매개상대 / ann_tom
    ParametricOpponent + 식별가능 합성 계수 (T−S). set_payoffs 가 R 만 변경하므로
    맥락 의존은 p 계수 (R−T+P−S) 로 유입(harmony +3, deadlock −4.5). 부수효과:
    payoff-blind 상대에 대한 ToM 모형 불일치로 VP 효과 감소 — seeds=24 재검증으로
    6/6 확증 지지 확인.
(3) evolution.stratified_simplex_points + attractor_analysis 개정(n=1500 기본,
    corner_frac=0.3, corner_weight=0.8): 발견=전층, basin_frac=균등층 전용,
    corner_convergence=유형별 코너 수렴 분포. replicator_ends_batch 벡터화 사용.
    H8E: n_attr=1500(quick 150), '코너 강건성' 탐색 로깅 추가.

================================================================================
v0.6.4 — ToM 복원: 양방향 + 회귀자 중심화 배터리
================================================================================
(1) core/constants.set_payoff_matrix(R,T,S,P) / reset_payoffs() 추가 — (R,T,S,P)
    직접 지정. 복원 과제 전용(유효 PD 조건 미강제; 본 실험 미사용 → H1–H12 골든
    diff 0.0 유지). set_payoffs/set_coop_index 는 불변(T=5,S=0 고정).
(2) validate_tom_recovery.py 전면 개정:
    · 4-파라미터(α,ρ,β,λ_j) — λ_j 고정 해제, THETA_RANGES 명시·보고(복원 지표는
      표집 구간에 의존하므로 필수).
    · DESIGNS = {legacy: 단일맥락, centered: u=(T−S)∈{+2,0,−2} 중심화 ×
      v=(R−T+P−S)p+(S−P)∈{∓0.8} 직교 요인설계}. design_diagnostics() 가 u/v 의
      평균·SD·corr 를 로깅 — 이 진단이 초기 블록표의 산술오류(corr=−0.23)를 검출.
    · 정방향: bias·rmse·r·R²·recovery_slope·coverage_90·shrinkage + **혼동행렬**
      corr(참_i,추정_j) 와 대각지배 판정.
    · 역방향: predictive_check() — θ̂ 로 재생성한 상대의 맥락별 협력확률 vs 참값,
      입자 사후예측 90% 구간 커버리지(가중분위수).
    · 그림 2×4: (a-d) 참vs추정 산점(legacy 회색 대비), (e,f) 혼동행렬 히트맵,
      (g) 커버리지, (h) 역방향 사후예측.
    · JSON: designs{legacy,centered} + summary(centered, 하위호환) + theta_ranges.
(3) 검증(n=60, 240라운드): centered 가 α r=0.19→0.95, λ_j 0.62→0.90, 혼동 최대
    비대각 0.78→0.35(대각 지배). 정직 보고: legacy 는 정방향 파탄에도 역방향
    r=0.99 — 역방향은 복원의 필요조건일 뿐. ρ·β 는 βρ 결합·포화로 중간 수준 잔존.

================================================================================
v0.6.5 — ToM 복원 완결 (계단식 v + 능동 설계)
================================================================================
(1) _ctx(u,v[,p]) : (u,v)→(R,T,S,P,p) 역산 (R=(v−(S−P))/p+T−P+S; p=.5 → 2v+u+1).
    DESIGNS 를 dict 로 개편: legacy/centered {"blocks":...}, adaptive {"menu": 15맥락
    (u∈{±2,0}×v∈{0,±0.8,±1.6}), "adaptive": True}. 검산: 메뉴 전점 u·v 재계산 일치.
(2) recover_once_adaptive: 후보 30개(메뉴×f±1) 획득함수 = (ρ,β) 기대 사후분산
    감소(가상 가중갱신, 재표집 없음, (N,C) 벡터화). ε=0.15, 번인 2×30 라운드로빈.
    선택된 맥락만 set_payoff_matrix 로 반영(상대·필터 동일 맥락 유지).
(3) predictive_check 를 공통 EVAL_CONTEXTS(centered 배터리)로 통일 — 설계 간 비교
    가능 + 맥락 외 전이 검사로 승격(legacy in-ctx r=.99 → out-ctx .57, 축퇴 노출).
(4) 검증(60상대·240라운드): adaptive 가 α .93 / ρ .68 / β .83 / λ .84 (R²),
    편향≈0, 기울기 .92–1.12, 커버리지 .87–.98, 혼동 비대각 .31(대각 지배).
    정직 잔여: ρ 최약(모형의 f±1 구조 기인 — 우도 변경 없이는 설계 범위 밖).
(5) 모형·패키지 불변(스크립트만 수정) — 골든/H1–H12 무영향.

================================================================================
v0.6.6 — 사영 연구 + ToM 갱신 오프바이원 수정 (브레이킹)
================================================================================
(1) scripts/validate_projection.py 신규 (492줄): Q1 복원성/Q2 분리성/Q3 충실도/
    Q4 동치류 분해. LABELS 8종(전략 내부 error=0, env 잡음 대칭). θ*=장지평
    probe 극한(--star-rounds/--star-seeds). run_probe_dyad 가 run_dyad 역학 미러
    + _force_focal_action 으로 **정책 수준** 강제 배신(자기모형 일관 정정 — 방출만
    바꾸면 ToM 이 보복을 f=+1 에 오귀속). perm_test_decode(라벨 순열),
    bhattacharyya, icc_oneway, model_signature(CC≡CD/DC≡DD 붕괴 구조 노출).
    그림 projection_tom.png 2×3 + projection_tom.json.
    quick 결과: probe 0.55(p=.0164) vs onpolicy 0.20, Q4 이득 +0.35;
    시그니처 RMSE wsls 0.444(구조적 잔차 예측 적중) vs TFT 0.053/allc 0.018.
(2) ipd/agent.py step: 갱신용 ctx_upd 분리 — f_upd = my_actions[-2](=a_{k−2}).
    근거(데이터 수준): step(k) 관측 opp_{k−1} 과 일치도 my_actions[-2]=1.000 vs
    my_actions[-1]=0.495. 결정용 ctx 는 불변(원래 정렬).
    실다이애드 TFT ρ̂ 0.17 → +0.88.
(3) 골든 스냅숏 삭제 후 **v0.6.6 재기준**(재생성 → PASS, 재실행 PASS).
    ref(v0.2.1) 대비 diff 0.0 은 의도적으로 성립하지 않음.
(4) 확증 재검증(축소 조건 스팟체크): ABA/ORE 지지 유지(효과크기 소폭 감소),
    **VP C-VP1/C-VP2 미지지로 반전** — 단 seeds=16 vs 기준선 24 로 검정력 교락.
    사용자 머신 seeds=24 재실행으로 확정 필요. 정직 보고 원칙에 따라 미지지 명시.

================================================================================
v0.7.0 — 반사실(counterfactual) 인과 귀인 [모형 개정] (명세서 §1·§2·§4)
================================================================================
(1) core/allostasis.py: counterfactual_disposition() 신규.
    기질 = P(D|f=+1) = 1 − σ(β̂(α̂+ρ̂·(+1)+ω̂g+η̂·fg+s(λ̂_j,p)))
    유발 = P(D|f=−1) − P(D|f=+1)
    ρ 는 valence 가 아니라 **반사실 시나리오를 구성하는 조건화 변수**로만 쓰인다.
    LambdaRegulator 에 disposition_mode ∈ {legacy, counterfactual} (기본 legacy),
    cf_g_handling ∈ {marginalize, plus} (기본 marginalize — §7.5 권장),
    cf_attr_gate_dedup (기본 True — §4).
(2) 사전 확인(실측, v0.6.6 legacy, seeds=4·240R): ρ 항의 무정보성 재현.
    ALLC ρ항 0.397 ≈ ALLD 0.490 (Δ=0.093) vs α항 0.507 vs 0.926 (Δ=0.419).
    **명세서에 없던 추가 발견**: TFT ρ항 0.256 < ALLC 0.397 — 호혜 상대가 무조건
    협력자보다 나쁜 기질로 평가되는 **역전**. ρ 는 무정보가 아니라 역정보였다.
(3) 반사실 판별력(seeds=8·240R, 폐루프): ALLD−TFT +0.269 → **+0.667**(목표 .506),
    ALLD−ALLC +0.275 → **+0.699**(목표 .607). 목표 초과.
    유발분: TFT +0.260, GTFT +0.263, ALLD **+0.004≈0**(순수 기질), WSLS −0.020.
    정직 보고: 유발분 크기가 명세서(TFT +0.585)보다 작다. 명세서 값은 legacy 조절
    하 θ̂ 에 반사실 공식을 사후 적용한 것이고, 본 값은 반사실이 **폐루프로 작동하는**
    동역학에서 얻은 것 — 보복을 오귀인하지 않아 λ 가 덜 떨어지고 → 내가 덜 배신하고
    → TFT 도 덜 보복하여 유발분 자체가 감소. **개정이 의도대로 작동한 결과**이며
    판별력은 오히려 목표를 초과했다.
    WSLS 반사실 disp=0.504·유발분≈0 — §7 이 예측한 f-기저 구조적 표현 불가의 직접
    증거(XOR 이 f 로 주변화되면 P(D|f=+1)≈P(D|f=−1) 로 반사실 대비 소멸).
(4) §2 계수 1.5/1.0/0.7 제거: counterfactual 은 항 자체 소멸; legacy 는 (1,1,1)
    균등화(자유 파라미터 3개 제거, Occam). ρ 항의 개념적 지위는 주석으로 보존.
(5) §4 세 귀인 기제 파이프라인 명문화: (1) ControllabilityAttribution → attr_gate
    (결과의 자기-기인성) → (2) 반사실 → 유발분 제거(배신의 자기-도발성) →
    (3) attr_gate 게이팅. counterfactual 모드에서 유발분이 이미 제거된 뒤 attr_gate
    를 그대로 곱하면 같은 성분을 이중 할인 → 반사실이 설명한 몫만큼 게이트 완화.
(6) §1.5 H5 재정의(모형 개정): 정교형=반사실 사용, 즉각형=관측 배신 직접 반응.
    dd_charges 는 이제 정의가 아니라 결과. exp_H5 가 두 조작화(legacy vs
    counterfactual)를 나란히 실행 — 확증은 신규 정의 기준, legacy 는 탐색 병기.
    hypotheses.yaml H5 rationale 전면 개정 + H5_OPS(조작화 절제) 추가.
    fig_H5 2×3 확장(조작화 절제 + 유발분 진단 패널, 평균±SD + seed jitter).
(7) 검증: AST → --quick smoke → 전체 경로 → 18실험 통합 → 골든 재기준(PASS, 재실행
    PASS). 골든은 §2 균등화로 legacy 수치가 변하므로 v0.7.0 기준 재설정.

================================================================================
v0.7.1 — rollout ρ 전파 (명세서 §3)
================================================================================
(1) ipd/tom/opponent_simulator.py: rollout_reciprocity 토글(기본 False).
    step>0 에서 static ToM 후퇴 대신 P(a_j=C|f_virt) = σ(β̂(α̂+ρ̂f_virt+ω̂g+η̂fg+s)).
    신뢰도 게이팅(r)로 step 0 GatedToM 과 동일 논리 유지.
(2) ipd/tom/sophisticated_planner.py: evaluate_policy 가 가상 궤적을 따라
    my_prev/opp_prev 를 갱신하며 rollout → "협력→상대 협력 유도→내 미래 보수↑"
    도구적 경로가 G_self 에 자연 발생.
(3) **경계 준수**: LambdaRegulator 는 전혀 건드리지 않는다. ρ 의 도구적 가치를 λ 로
    처리하면 λ 가 공감이 아닌 전략 파라미터가 되어 구성개념이 붕괴한다.
    exp_RR 이 λ 궤적 불변을 탐색 지표로 명시 검증.
(4) 실측(horizon=2, seeds=4·120R): TFT coop 0.667→0.902, pay 297.8→329.8;
    GTFT coop 0.817→0.906. **음성 대조 성립**: ALLD coop 0.244→0.263,
    pay 115.0→112.5 (quick 격자에서는 Δcoop=0.000) — ρ̂≈0 이라 전파할 것이 없음.
(5) exp_RR/fig_RR 신규 + hypotheses.yaml RR(C-RR1 호혜 협력↑, C-RR2 착취자 무영향).
    horizon=1 에서는 rollout 이 없어 무영향(구조적 보장).

================================================================================
v0.8.0 — 우도 기억-1 완전화: g·fg 항 [브레이킹] (명세서 §7)
================================================================================
(1) ipd/tom/inversion.py: likelihood_basis ∈ {f, fg} (기본 f).
    P(a_j=C|f,g,θ) = σ(β(α + ρ·f + ω·g + η·f·g + s(λ_j,p))).
    THETA_AXES_FG 6축, _PRIOR ω/η ~ N(0,1) clip ±3, jitter 0.10.
    f 기저에서 omega/eta 는 0 배열 → 항 소멸 → v0.6.6 우도와 동일(하위호환).
    predict_coop(f, g), expected_infogain(a, f_next, g_next), posterior_means/stds
    에 ω·η 추가. _feature_g() 신규 (their_last_action 은 v0.6.6 에도 있었으나
    우도가 미사용 — fg 에서 비로소 소비).
(2) **§7.2 시제 표 명문화 (오프바이원 재발 방지)**:
      갱신용: 관측 opp_{k−1} → f=my_{k−2}, g=opp_{k−2}  (둘 다 한 시점 더 과거)
      결정용: 예측 opp_k    → f=my_{k−1}, g=opp_{k−1}  (둘 다 최신)
    agent.py 가 opp_actions 를 _regulate **직후** append 하므로 갱신 시점의
    opp_actions[-1] == opp_{k−2} (순서 의존 — 변경 금지, 주석 명시).
    **데이터-수준 검증(§7.4-1)**: 무작위 focal·200R, TFT/WSLS 모두
    f==my[k−1] 1.000 / g==op[k−1] 1.000 vs 대안 시제 0.510/0.515(우연).
    ground truth: TFT opp[k]==my[k−1] 1.000. 시제 정합 확인.
(3) **§7.4-2 WSLS 표현 회복(probe, 6 seeds·600R·600 입자)**:
      WSLS Q3 RMSE 0.453 → **0.040** (목표 ≤0.10 달성)
      β*_wsls 0.14 → **1.55** (목표 ≥1.5 달성 — 결정론성 회복)
      η̂_wsls **+2.44**, TFT +0.06 / ALLC +0.42 / ALLD +0.67 (음성 대조 성립)
      경험 [0.963,0.048,0.035,0.939] vs fg 모형 [0.984,0.024,0.009,0.981]
    validate_projection --basis fg 에서도 WSLS RMSE 0.016 확인.
(4) §7.3 fg_centered 복원 배터리(필수 동반): g 는 직접 조작 불가 → 합성 상대에 한해
    **상대 행동열 조건부 강제**로 (f,g) 4셀 균형 순회. recover_once_fg 신규.
    design_diagnostics 다회귀자 확장(u,v,f-빈도,g-빈도,fg-균형) → 실측
    f_mean=g_mean=fg_mean=0.0, corr_f_g=0.0, design_rank=4 full-rank, centered=True.
    THETA_RANGES 에 ω∈[−1.2,1.2], η∈[−1.6,1.6] 추가. 입자 600(차원 보상),
    기본 rounds 480(셀당 ~30 관측, §7.3 권장). 6×6 혼동행렬 + fg 4셀 전부 포함하는
    역방향 평가맥락. run_design/predictive_check/그림 전부 축수 적응형으로 개정.
(5) scripts/validate_projection.py: --basis {f,fg}. model_implied_signature 가
    상태 k=(my,opp) 를 divmod 로 분해해 f·g 를 모두 조건에 넣는다(fg 기저에서
    4자유도 전 공간 표현). _ident_coords 는 축 이름 인덱싱 + fg 에서 (βω, βη) 추가.
(6) exp_FG/fig_FG 신규 + hypotheses.yaml FG(C-FG1 RMSE≤0.10, C-FG2 β̂≥1.5,
    C-FG3 η̂ 특이성). **정직 보고**: ALLC/ALLD 는 자기 행동이 상수라 (f,g) 4셀 중
    2.2셀만 점유 → η 가 약식별. 시그니처 RMSE 가 미정의인 경우 검정을 등록하지 않고
    점유 구조를 그대로 보고(널을 p 로 위장 금지). η̂ 음성 대조군은 **4셀 점유
    전략만**(TFT/GTFT) 사용.
(7) 전역 토글: run_ipd_experiment.py --disposition-mode / --likelihood-basis /
    --rollout-reciprocity. MODEL_REVISION 이 agent_spec 에 주입되어 전 실험 일관 전파
    (실험이 명시한 값은 덮어쓰지 않음 — H5 조작화 절제가 이에 의존).
(8) 검증 층위 전부 통과: AST(30파일) → --quick smoke → 전체 경로(H5/RR/FG) →
    18실험 통합(H1 H2H3 H4 H5 H6 H7 H7H H8 H8E H9H10 H11 H12 GS VP ABA ORE RR FG)
    → 골든 재기준 PASS(재실행 PASS).

================================================================================
v0.8.1 — VP/ORE 예산 명시화 · 본 실험 posterior SD 로깅
================================================================================
(1) run_ipd_experiment.py: 은닉 캡 폐지.
    종전: exp_VP/exp_ORE 내부에 T = min(max(rounds), 120), seeds = min(seeds, 24)
    가 하드코딩되어 --rounds/--seeds 인자와 무관하게 축소 실행되었고, 실효 조건이
    config 에 기록되지 않았다(은닉 불일치·재현성 결함).
    개정: --vp-rounds/--vp-seeds/--ore-rounds/--ore-seeds (기본 240/60) 신설.
    EXP_BUDGET 전역이 두 실험에 주입되며, 실효값은 (a) vars(args) 경유로
    summary.json 의 config, (b) 각 실험 반환값의 "T"/"seeds", (c) 시작 로그 배너
    세 곳에 기록된다. --quick 은 예산도 30/3 으로 축소(args 갱신 → config 정직).
    **주의**: 기본 예산 실행(VP 7레짐 × T=240 × seeds=60)은 v0.6.6 대비 ~5×
    비용 — 사용자 로컬(16코어)에서 실행 전제. v0.6.6 결과(T=120·seeds=24)와
    대조할 때는 지평·시드가 교락 축이 됨을 hypotheses.yaml 에 명시.
(2) ipd/agent.py: 매 라운드 inversion.posterior_stds() 를 로그에 기록 —
    sd_alpha/sd_rho/sd_beta/sd_lambda_j/sd_omega/sd_eta.
    목적: recovery 배터리는 이상 조건(T=480·강제 균형 점유)의 **상한 검증**이므로,
    본 실험(T=60/240·on-policy)에서 θ̂ 의존 기제(β-게이팅·반사실 귀인) 해석 시
    실험 내 식별 상태를 사후 진단할 수단이 필요했다(T=60 에서 β coverage 0.58
    과신 확인). f 기저에서 sd_omega=sd_eta=0(상수축). 스모크 실측: T=60 에서
    sd_beta 1.17→0.90 (사전 1.8 대비 약식별 상태가 그대로 드러남 — 도구 유효).
    recovery 자체의 rounds(480)는 변경하지 않는다 — §7.3 통계적 요구.
(3) 검증: AST → 로그 키 스모크(f/fg 양 기저) → --quick VP·ORE(예산 배선 확인)
    → 골든 PASS(로그 키 추가는 수치 무변).

================================================================================
부록 D — 시뮬레이션 실행 안내 (v0.8.1 기준, 로컬 16코어 전제)
================================================================================

D.0 설치·공통 규약
--------------------------------------------------------------------------------
  pip install numpy matplotlib scipy pyyaml
  pip install inferactively-pymdp            # 선택: --backend pymdp 등가성 검증
  pip install "jax[cpu]" equinox optax       # 선택: validate_ann_tom.py 전용

  · 실행 위치: 리포지토리 루트(Halloreg/). 모든 경로는 여기 기준.
  · --jobs 16 (물리 코어 수). -1(논리 32)은 SMT 이득 없음 — 쓰지 말 것.
  · 소스를 편집하며 A/B 비교할 때는 반드시:
      find . -name __pycache__ -type d -prune -exec rm -rf {} +
      python -B <script> ...
    (stale 바이트코드가 "차이 없음"이라는 그럴싸한 오답을 만든다. 두 조건이
    소수점 끝까지 같으면 발견이 아니라 캐시를 의심할 것.)
  · 모든 산출물은 AIF_IPD/results/ 에 저장 (그림 png+pdf+caption.json, 수치 json).
  · 골든 회귀: python -B AIF_IPD/tests/test_golden.py (스냅숏은 v0.7.0 재기준).

D.1 메인 실험 러너 — run_ipd_experiment.py
--------------------------------------------------------------------------------
  # 전체 실험, 기본 예산 (seeds=240, rounds=60·240 이중 지평, VP/ORE 240/60)
  python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16

  # 특정 실험만 / 단일 지평만
  python AIF_IPD/scripts/run_ipd_experiment.py --experiments H5 RR FG --jobs 16
  python AIF_IPD/scripts/run_ipd_experiment.py --rounds 60 --jobs 16

  # 스모크 (seeds=3, rounds=[30], VP/ORE 예산 30/3)
  python AIF_IPD/scripts/run_ipd_experiment.py --quick --jobs 1

  실험 목록(18): H1 H2H3 H4 H5 H6 H7 H7H H8 H8E H9H10 H11 H12 GS RR FG VP ABA ORE
  · per-T 실험(H1~H6, H7, H8, H9H10, H11, GS, RR): --rounds 의 각 지평에서 반복,
    결과 키 H5_T60 식으로 태깅.
  · GLOBAL 실험(H7H, H8E, H12, VP, ABA, ORE, FG): 자체 지평 규약 —
      H7H  : 지평 자체가 조작 변수 (dΔ/dT 기울기가 확증 지표)
      H8E/H12 : {60,240} ∪ 요청 지평
      ABA  : 요청 지평 각각에서 3국면 분할
      VP/ORE : --vp-rounds/--ore-rounds (아래 D.2)
      FG   : 내부 probe 지평 max(rounds, 240)

D.2 [v0.8.1] VP/ORE 예산 플래그
--------------------------------------------------------------------------------
  --vp-rounds 240   --vp-seeds 60     # VP 지평·시드 (기본값)
  --ore-rounds 240  --ore-seeds 60    # ORE 지평·시드 (기본값)

  의미: VP/ORE 는 CI 레짐 격자를 자체 추정하는 GLOBAL 실험이라 --rounds 와
  독립적인 예산을 갖는다. v0.8.0 까지는 은닉 캡(120/24)이 있었고, v0.8.1 이
  이를 명시적 플래그로 전환하며 기본값을 본 실험 지평(240)과 seeds=60 으로
  복원했다. 실효값은 summary.json 의 config 와 각 실험 결과의 "T"/"seeds" 에서
  확인한다.
  v0.6.6 결과와 같은 조건으로 재현하려면: --vp-rounds 120 --vp-seeds 24
  (ORE 동일). 예산이 다른 두 summary 를 대조하면 효과 차이가 개정 효과인지
  검정력/지평 차이인지 교락된다 — compare_versions.py 가 경고를 띄운다.

D.3 모형 개정 토글 (v0.8.2 — **기본값 = 개정 모형**; 독립·조합 가능)
--------------------------------------------------------------------------------
  [v0.8.2 기본값] counterfactual · fg(입자 자동 600) · rollout on · horizon 2.
  플래그 없이 실행하면 개정 모형으로 시뮬레이션된다.

  **v0.6.6 재현(레거시) 실행**:
  python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16 \
      --disposition-mode legacy --likelihood-basis f \
      --no-rollout-reciprocity --planning-horizon 1
  (+VP/ORE 까지 v0.6.6 조건으로: --vp-rounds 120 --vp-seeds 24 --ore-rounds 120
   --ore-seeds 24)

  아래는 각 토글의 의미 (기본/레거시 값 표기는 v0.8.2 기준으로 읽을 것):
--------------------------------------------------------------------------------
  ① --disposition-mode {legacy, counterfactual}      [§1, 기본 counterfactual]
     무엇인가: 상대 '기질' 판단 방식. legacy 는 σ(−α̂)·(1−λ̂)·σ(−ρ̂) 가중평균으로
     ρ 를 valence 처럼 쓴다(실측상 무정보 — ALLC 0.397 vs ALLD 0.490 — 이며
     TFT 0.256 < ALLC 로 역정보이기까지 하다). counterfactual 은
     기질 = P(D|f=+1) ("내가 협력했더라도 배신했을까"), 유발분 = P(D|f=−1)−P(D|f=+1)
     로 분해해 내 도발로 유발된 배신을 기질 증거에서 제외한다. ρ 는 반사실을
     구성하는 조건화 변수로만 쓰인다. 판별력 ALLD−TFT +0.269 → +0.667.
     로그의 "provoked" 가 유발분 궤적. 관련 에이전트 인자: disposition_mode,
     cf_g_handling(marginalize/plus), cf_attr_gate_dedup.
     켜기: python AIF_IPD/scripts/run_ipd_experiment.py --disposition-mode counterfactual

  ② --likelihood-basis {f, fg}                        [§7, 기본 fg]
     무엇인가: ToM 우도의 조건 변수. f 기저 σ(β(α+ρf+s)) 는 기억-1 시그니처의
     2자유도 부분공간만 표현 — WSLS(XOR, [1,0,0,1])는 구조적으로 표현 불가
     (v0.6.6 RMSE 0.408 ≈ 구조 하한 0.406; 실패는 추정이 아니라 표현).
     fg 기저 σ(β(α+ρf+ωg+η·fg+s)) 는 (1,f,g,fg) 로 전 공간(4자유도)을 스팬.
     g=상대 자신의 직전 행동(관성 ω), η=결과-조건성(WSLS성). probe 검증에서
     WSLS RMSE 0.453→0.040, β̂ 0.14→1.55, η̂=+2.44(타 전략 ≈0).
     주의 1: on-policy 에서는 점유 편중으로 η 식별이 제한된다(§7.5 부분 성립 —
     REVISION_v070_v080_NOTES.md). fg 로 돌린 θ̂ 의존 지표는 sd_* 로그로 식별
     상태를 함께 확인할 것(D.6).
     주의 2: [v0.8.2] 러너 경유(agent_spec) fg 실행은 입자 600 자동 상향.
     에이전트를 직접 생성하는 커스텀 코드에서는 n_particles=600 을 직접 지정.
     켜기: --likelihood-basis fg

  ③ --rollout-reciprocity / --no-rollout-reciprocity  [§3, 기본 on]
     무엇인가: planning horizon≥2 의 rollout 에서 상대 예측을 ρ̂ 로 내 가상
     직전행동에 조건화한다. 종전에는 step>0 이 static ToM 으로 후퇴해 "내 협력이
     되돌아온다"는 계산이 없었고, 협력은 λ·G_other 로만 발생했다. 켜면 도구적
     호혜가 G_self 에 자연 발생: TFT coop 0.667→0.902, pay 297.8→329.8.
     음성 대조: ALLD 는 ρ̂≈0 이라 무영향(Δ=0.000).
     경계: LambdaRegulator 는 불변 — ρ 의 도구적 가치를 λ 로 처리하면 공감
     구성개념이 붕괴한다(λ 궤적 불변을 exp_RR 이 탐색 지표로 검증).
     [v0.8.2] 기본 horizon=2 라 기본 상태에서 실효. --planning-horizon 1 로
     내리면 rollout 은 구조적으로 무영향이 된다(끈 것과 행동 동일).
     끄기: --no-rollout-reciprocity

  조합 예:
  python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16 \
      --disposition-mode counterfactual --likelihood-basis fg --rollout-reciprocity
  시작 배너("모형 개정 토글: ...")와 summary.json config 로 실효 조건을 확인.

D.4 검증 스크립트
--------------------------------------------------------------------------------
  # ToM 파라미터 복원 (기본: 4설계 legacy/centered/adaptive/fg_centered,
  #  agents=60, rounds=480, particles=400 — fg_centered 는 내부에서 600 상향)
  python AIF_IPD/scripts/validate_tom_recovery.py
  #  rounds=480 은 §7.3 통계적 요구((f,g) 4셀 × 6블록, 셀당 ~30 관측) — 낮추면
  #  ω·η 식별 실패. v0.6.6 조건 재현: --designs legacy centered adaptive --rounds 240

  # 사영(projection) 검증 (기본 basis=f; fg 는 --basis fg)
  python AIF_IPD/scripts/validate_projection.py
  python AIF_IPD/scripts/validate_projection.py --basis fg

  # pymdp 등가성 / ANN-ToM (선택)
  python AIF_IPD/scripts/run_ipd_experiment.py --check-equivalence
  python AIF_IPD/scripts/validate_ann_tom.py

D.5 버전 대조 — compare_versions.py
--------------------------------------------------------------------------------
  python AIF_IPD/scripts/compare_versions.py \
      --old /path/v066/summary.json --new AIF_IPD/results/summary.json \
      --old-label v0.6.6 --new-label v0.8.1

  · 확증 지표를 (hyp, T-태그)로 짝지어 판정 대조 매트릭스·효과크기 산점·Δdz·
    신규/제거 목록·요약 카운트를 그린다 (version_compare.png/json).
  · config 불일치(seeds/rounds/backend) 시 교락 경고. **대조는 동일 조건에서만
    해석 가능** — 실측 예: quick(n=9) vs full(n=72) 대조에서 VP C-VP2 는 효과가
    커졌는데도(0.049→0.103) 검정력 손실로 '지지→미지지' 반전.
  · v0.6.6 대조 시 VP/ORE 는 예산 자체가 다르다(120/24 → 240/60). 예산 효과와
    개정 효과를 분리하려면 v0.8.1 을 --vp-rounds 120 --vp-seeds 24 로 한 번 더
    돌려 3점 비교(v0.6.6 ↔ v0.8.1@구예산 ↔ v0.8.1@신예산)를 권장.

D.6 [v0.8.1] posterior SD 로그 활용
--------------------------------------------------------------------------------
  agent.log 의 sd_alpha/sd_rho/sd_beta/sd_lambda_j/sd_omega/sd_eta.
  · 용도: θ̂ 의존 지표(H2 λ̂_final, H3 E[β], β-게이팅, 반사실 disposition) 해석
    시 해당 시점 θ̂ 가 사전에서 실제로 좁혀졌는지 확인. sd 가 사전 폭 근처면
    약식별 — 그 지표는 대조(contrast) 주장으로만 읽고 보정(calibration) 주장으로
    읽지 말 것.
  · 참고 사전 SD: α 2.0, ρ 1.2, β 1.8, λ_j 0.3, ω/η 1.0.
  · f 기저에서 sd_omega=sd_eta=0 (상수축 — 정상).

D.7 권장 실행 순서 (로컬 전면 재실행)
--------------------------------------------------------------------------------
  1. python -B AIF_IPD/tests/test_golden.py            # 환경 정합 확인
  2. python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16
     # 기본값 = legacy/f/off + VP·ORE 240/60 → v0.8.1 기준선 확보
  3. compare_versions.py 로 v0.6.6 summary 와 대조
     # VP/ORE 반전 시: --vp-rounds 120 --vp-seeds 24 재실행으로 예산 교락 분리
  4. (선택) 개정 토글 켠 조건들: --disposition-mode counterfactual 등 —
     기준선과 같은 seeds/rounds 로 A/B (D.0 캐시 규약 준수)
  5. validate_tom_recovery.py / validate_projection.py (+--basis fg)

================================================================================
v0.8.2 — 기본값 반전 · fg 입자 자동 상향 · horizon 플래그 [브레이킹]
================================================================================
(1) **기본값 반전**: 러너 기본 = 개정 모형.
      disposition_mode=counterfactual | likelihood_basis=fg |
      rollout_reciprocity=True | planning_horizon=2
    v0.6.6 재현은 이제 명시적 opt-in:
      --disposition-mode legacy --likelihood-basis f \
      --no-rollout-reciprocity --planning-horizon 1
    planning_horizon 기본을 2 로 함께 반전한 이유: rollout 은 horizon≥2 에서만
    존재하므로 horizon=1 + rollout=True 는 구조적 무효(§3) — "기본 = rollout" 이
    실효하려면 horizon 이 따라와야 한다.
    **에이전트 클래스 기본값은 legacy/f/400/h1 유지** — 골든 회귀(직접 생성 경로)
    와 라이브러리 하위호환 보존. 반전은 러너의 agent_spec 주입 층에서만 일어나며,
    실험이 스펙에 명시한 값(H5 절제의 legacy 팔, RR 의 rr=False·H 팔, FG 의 f/fg
    내부 스윕)은 전역보다 우선한다 — 단위 검증 7항목 PASS.
    validate_projection.py 의 --basis 기본도 fg 로 정합(러너와 일치).
(2) **fg 입자 자동 상향**: agent_spec 에서 최종 likelihood_basis=="fg" 이고
    n_particles 미명시면 600 주입(setdefault — 명시값 보존). 종전에는 exp_FG/
    recovery 내부에만 있어 "--likelihood-basis fg 본 실험 = 400 입자" 라는
    의도치 않은 중간 조건이 존재했다(v0.8.1 문답에서 발견).
(3) **--planning-horizon N** (기본 2): 전 실험 기본 horizon 을 전역 주입.
    RR 처럼 자체 지정하는 실험은 그쪽 우선. --summary-out FILE (기본
    summary.json) 신설 — horizon 1/2 별도 실행 결과를 덮어쓰지 않고 보존해
    compare_versions.py 대조에 쓴다.
(4) **horizon 1 vs 2 총체 비교 절차** (부록 D.8): 전 실험을 두 지평에서 각각
    실행 후 compare_versions.py 로 짝지어 대조. 샌드박스 quick 검증 실측은
    본 문서 D.8 끝의 주의 사례 참조.
(5) 비용: 기본 실행이 v0.8.1 대비 대략 3~5×/다이애드 — fg 입자 600(≈1.5×) ×
    horizon 2 planning(2^H 정책 열거, ≈2~3×). VP/ORE 240/60 과 결합되므로
    전면 실행은 로컬 16코어 전제.
(6) 검증: AST → agent_spec 단위 7항목 → quick 전 실험 h1/h2 스윕 + 대조 →
    legacy 재현 플래그 결정론 → 골든 PASS.

D.8 [v0.8.2] horizon 1 vs 2 총체 비교 절차
--------------------------------------------------------------------------------
  목적: 전 실험을 기본 horizon 1 과 2 에서 각각 시뮬레이션하고, 계획 지평이
  18개 가설의 판정·효과크기에 미치는 영향을 총체적으로 본다. horizon=2 조건은
  rollout ρ 전파가 실효하는 조건이기도 하므로(§3), 이 비교는 "1단계 반응 모형
  vs 도구적 호혜 모형"의 전면 대조이기도 하다.

  # ① 두 지평 실행 (summary 파일명 분리 — 서로 덮어쓰지 않게)
  python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16 \
      --planning-horizon 1 --summary-out summary_h1.json
  python AIF_IPD/scripts/run_ipd_experiment.py --jobs 16 \
      --planning-horizon 2 --summary-out summary_h2.json
  # (그림은 파일명이 같아 덮어쓰인다 — 그림도 보존하려면 실행 사이에
  #  results/ 를 통째로 복사해 둘 것)

  # ② 총체 대조
  python AIF_IPD/scripts/compare_versions.py \
      --old AIF_IPD/results/summary_h1.json \
      --new AIF_IPD/results/summary_h2.json \
      --old-label horizon=1 --new-label horizon=2 --out horizon_compare

  해석 주의:
  · horizon=1 에서는 rollout_reciprocity=True 여도 무영향이므로, 이 대조의
    "horizon 효과"는 (계획 심화 + rollout ρ 전파)의 **결합 효과**다. 순수 계획
    심화만 보려면 h2 를 --no-rollout-reciprocity 로 한 번 더 돌려 3점 비교.
  · seeds/rounds 는 두 실행에서 동일해야 한다(교락 방지 — config 경고 확인).
  · RR 실험은 자체 horizon(2,4)을 쓰므로 이 스윕의 영향을 받지 않는다 — 두
    summary 에서 RR 이 (거의) 동일하면 주입 경계가 지켜졌다는 무결성 검사가 된다.

--------------------------------------------------------------------------------
v0.8.2 샌드박스 검증 기록 (1코어, --quick 규모)
--------------------------------------------------------------------------------
(a) agent_spec 주입 의미론 단위검증 7항목 PASS: 기본 반전 / fg→600 / f 명시 시
    비상향 / 입자 명시값 보존(800) / H5 legacy 팔 보호 / RR rr=False·H=4 팔 보호 /
    tom_empathic 주입.
(b) horizon 스윕 기전 실증(D.8 절차 그대로): 13실험 부분집합을 h1/h2 각각
    --summary-out 분리 실행 → compare_versions 대조. config 에 planning_horizon
    1/2 정직 기록, seeds/rounds 동일(교락 경고 無), 17개 확증 지표 완전 짝지음
    (신규/제거 0). 무결성 검사 성립: RR(자체 horizon)·FG(내부 probe)는 두 실행에서
    **비트 단위 동일** — 전역 주입이 자체 지정 실험을 침범하지 않음.
(c) quick 규모 예고 신호(과잉해석 금지, n=3): VP C-VP1 이 h2 에서 지지로 반전
    — Δpay 0.031 [−0.127, 0.207] → 0.307 [0.170, 0.477]. rollout ρ 전파가 가변
    레짐에서 adaptive 보수 우위를 실효화한다는 §3 서사와 정합. 전면 확인은 로컬
    full-run 의 D.8 3점 비교로.
(d) h2 커버리지: 17/18 실험 완주 확인 (부분집합 13 + H8E·H11·H7H·H12 별도).
    **H8 은 샌드박스 시간 제약으로 h2 미검증** — 단 H8 은 H11/H8E 와 동일한
    집단 시뮬 경로의 규모 확장(133 집단)이라 코드 경로는 커버됨. 로컬 h2 전면
    실행에서 H8 이 최장 블록이 될 것(h1 quick 191s → h2 는 수 배).
(e) 잠재 버그 2건 수정(v0.6.6 기원, 신규 기본 동역학이 노출):
    비대칭 yerr(점추정−백분위CI)가 적은 부트에서 음수가 되어 matplotlib
    ValueError — fig_H12 Shapley 패널에서 실제 크래시 재현·수정. 동일 패턴 감사
    결과 6개 지점 중 1곳(ORE)만 원저자가 클립했었음 → 나머지 5곳(공용 bar_ci
    포함) 전부 np.maximum(err, 0.0) 클립. 그림 표시용이며 JSON 원 CI 는 보존.
(f) legacy 재현 opt-out 경로: 4플래그 조합으로 2회 실행 → summary 완전 동일
    (결정론), 신규 기본 실행과는 상이(반전 실효). 골든 PASS.
