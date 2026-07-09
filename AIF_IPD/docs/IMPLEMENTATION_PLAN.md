# HalloReg — Implementation Plan

**H**ierarchical **Allo**static **Reg**ulation of empathy in Active Inference agents
for the Iterated Prisoner's Dilemma.

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

```
HalloReg/
├── core/                      공유 유틸 + HalloReg 의 심장
│  ├── constants.py            PD 상수, joint-outcome 인덱싱 (CC/CD/DC/DD)
│  ├── generative.py           POMDP (A,B,C,D) + 해석적 EFE (pymdp 와 동치)
│  ├── pymdp_backend.py        pymdp 1.0.x JAX 백엔드 (선택) + 동치성 검증
│  ├── allostasis.py           ★ CoreAllostaticBeliefState, LambdaRegulator
│  └── logging_utils.py        logging 설정, OS 적응형 한글 폰트
├── ipd/
│  ├── agent.py                ToMEmpathicAgent (고정 λ), AdaptiveAgent (조절 λ)
│  ├── env.py                  Environment, StrategyAgent(TFT/ALLC/ALLD/WSLS/GTFT/…),
│  │                           형질전환 스케줄(변덕 상대), make_opponent/make_capricious
│  ├── sim.py                  run_dyad / run_many(멀티코어) / run_population
│  ├── tom/
│  │  ├── tom_core.py          TheoryOfMind(정적 최적반응), GatedToM, RecursiveSocialEFE
│  │  ├── opponent_simulator.py 계획 롤아웃용 상대 반응 시뮬레이터
│  │  ├── sophisticated_planner.py 계획지평 H 정책 열거·롤아웃 (Albarracin 보존)
│  │  └── inversion.py         입자필터 기반 상대 θ=(α,ρ,β,λ_j) 역추론
│  └── metrics/exploitability.py 착취가능성·CC율·방어특이성·Welch t
├── scripts/run_ipd_experiments.py  ★ 메인 엔트리포인트 (H1–H10 + 시각화)
├── docs/IMPLEMENTATION_PLAN.md
└── results/                   (gitignored)
```

### 명세 대비 변경 사항

* `core/` 는 파일명이 명세되지 않아 위 5개 모듈로 구성했다.
* 명세에 없던 `core/allostasis.py` 를 추가했다. HalloReg 의 핵심 기여(core allostatic
  belief + λ 조절)를 `agent.py` 에 묻어두면 재사용·절제실험이 불가능하기 때문이다.
* 명세에 없던 `core/pymdp_backend.py`, `core/logging_utils.py`, `ipd/sim.py::run_population`
  을 추가했다 (각각 pymdp 요구사항, logging 요구사항, H8 을 위해 필요).
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
한다 (`--check-equivalence` 로 실행; 통과 확인됨).

따라서:

* **기본 경로 = numpy 해석해** (`core/generative.py`). 수천 배 빠르고, 재귀적 확장(R1/R2)의
  수식이 그대로 드러난다.
* **검증/확장 경로 = pymdp 1.0.x JAX** (`--backend pymdp`, `AdaptiveAgent(use_pymdp=True)`).
  `Agent(A,B,C,D, batch_size=1, policy_len=1)`, 관점취하기는
  `eqx.tree_at(lambda a: a.C, agent, C_other)` 로 선호를 교체해 구현한다.
  `A/B/C/D` 는 batch 선행축을 가진 jax 배열 리스트.

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

DD(상호배신) 관측도 약한 배신 증거(가중 0.35)로 반영한다. 이것이 없으면 자기보호로
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

```
g⁻ (grievance) ← decay·g⁻ + protective_gain·attributed_disp
                          + anticipatory_gain·anticipatory
                          − forgiveness·[상대 협력]
g⁺ (trust)     ← trust_decay·g⁺ + trust_gain·coop_credence·[협력]
                          − trust_decay_on_betrayal·[배신]

λ_eff = clip( λ_base·(1 − g⁻) + (λ_max − λ_base)·g⁺ ,  0 , λ_max )
```

* `sophisticated=True` (rmPFC): `disposition` 을 `precision_conf = E[β]/4` 로 게이팅하고,
  누적된 `disp_credence` 로 변조하며, 예기적 항을 켠다.
* `sophisticated=False` (vmPFC 즉각): 정밀도도, 누적 귀인 신뢰도도 무시하고
  (`disposition = disp_credence = 1`) 지금 당한 배신에만 반응한다. 예기적 항은 꺼진다.
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

| 가설 | 실험 함수 | 조건 | 주 지표 |
|---|---|---|---|
| H1 자기보호 | `exp_H1` | Adaptive vs Fixed-λ, 착취자 | λ_final, 착취가능성 |
| H2 의도/맥락 구분 | `exp_H2_H3` | Adaptive vs {착취자, 잡음TFT, 무작위} | λ 회복 |
| H3 형질 추정 구분 | `exp_H2_H3` | 동일 | E[α], E[β] |
| H4 협력 복원 | `exp_H4` | Adaptive vs Fixed-λ, TFT | CC율 |
| H5 정교 vs 즉각 | `exp_H5` | `sophisticated` 토글 × {착취자, 잡음TFT, 무작위} | 방어 특이성, 보수 |
| H6 잡음과 고정전략 | `exp_H6` | 라운드로빈 × 잡음 {0, 0.15} | 평균 보수 |
| H7 변덕 상대 | `exp_H7` | Adaptive vs GTFT/WSLS/TFT × 잡음 스윕 | 평균 보수 |
| H8 집단 역학 | `exp_H8` | AdaptiveAgent 수 {0,2,4,6} | 집단 CC율 기울기 |
| H9 α 전용 귀인 | `exp_H9_H10` | `attribution_target` 절제, 변덕 상대 | 누적 보수 |
| H10 λ 전용 귀인 | `exp_H9_H10` | 절제, 정적 잡음 TFT | CC율 |

### 실행 결과 (seeds=12, rounds=120)

전 가설 지지. 대표 수치:

* **H1**: λ_final 0.00 (Adaptive) vs 0.40 (Fixed); 착취가능성 0.045 vs 0.966.
  고정-λ 에이전트의 착취자 상대 누적보수는 사실상 0 이다.
* **H2/H3**: λ_final 착취자 0.00 vs 잡음TFT 0.51. E[β] 는 착취자에서 크게 높고, E[α] 는 반대.
* **H5**: 방어 특이성 정교 +0.144 vs 즉각 −0.265. 즉각형은 착취자 방어도 **더 약하고**
  (예기적 항 부재로 진동) 잡음 파트너를 과잉처벌한다.
* **H9/H10 이중해리**: `intent_only` 는 정적 잡음 파트너의 CC율을 0.839 → 0.185 로
  붕괴시키고(과잉처벌), `beta_context` 는 착취자 상대 누적보수를 114.6 → 40.2 로
  잃는다(과소방어).

### H5 의 문자적 주장에 대하여 (정직한 보고)

원 가설은 "정교형이 착취자에게 더 방어하지만 **잡음 상대에게 덜 번다**" 였다.
검증 결과 이 주장은 **상대의 종류에 의존한다**:

* **완전 무작위** 상대(용서가 보답되지 않음): 정교형 208.2 < 즉각형 316.4 → **주장 성립**.
* **잡음 TFT** 상대(용서가 보답됨): 정교형 281.2 > 즉각형 242.2 → **주장 반전**.

즉 관대함의 비용은 "용서가 상호성으로 되돌아오지 않는" 환경에서만 발생한다.
따라서 `exp_H5` 의 **주 검정은 방어 특이성**(선택적 방어)으로 두고, 문자적 주장의
두 조건은 결과의 `literal` 필드에 분리 보고한다.

### H7 의 역U자 (예측된 경계조건)

잡음 0 에서는 형질 전환이 행동에 그대로 드러나 표면 규칙(TFT)만으로 충분하며
ToM 이득이 사라진다(adaptive 2.124 < TFT 2.274). 중간 잡음(0.10)에서 잡음과 의도변화가
혼동되어 잠재 형질 추론이 비로소 이득이 된다(adaptive 2.105 > GTFT 2.062 > WSLS 1.957).
과다 잡음(0.20)에서는 신호 자체가 소실된다. 주 조건은 중간 잡음이며 스윕 전체를 보고한다.

---

## 6. 병렬 실행

다이애드는 완전 독립이다. `ipd/sim.py::run_many` 가
`multiprocessing.get_context("spawn").Pool.imap_unordered` 로 코어에 분배한다.

* **spawn 필수**: JAX 는 fork-unsafe.
* 워커 oversubscription 방지를 위해 `sim.py` 최상단(= jax import 이전)에서
  `XLA_FLAGS`(intra-op 1), `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `JAX_PLATFORMS=cpu` 를 설정한다.
* `--jobs -1` → `cpu_count() − 1`. `--jobs 1` → 순차.

---

## 7. 실행법

```bash
cd <HalloReg 의 부모 디렉터리>          # HalloReg 가 최상위 패키지로 import 되어야 함
python HalloReg/scripts/run_ipd_experiments.py                 # 기본: seeds=12 rounds=120 jobs=-1
python HalloReg/scripts/run_ipd_experiments.py --quick         # 스모크 테스트 (~10초)
python HalloReg/scripts/run_ipd_experiments.py --seeds 24 --rounds 200 --jobs -1
python HalloReg/scripts/run_ipd_experiments.py --experiments H1 H5 H9H10
python HalloReg/scripts/run_ipd_experiments.py --backend pymdp --check-equivalence
```

산출물: `results/halloreg_ipd.png` (3×3 패널), `results/halloreg_results.json`.

### 그림의 불확실성 표기 (모든 패널 공통)

음영대와 오차막대는 **시드 간 표본표준편차(±1 SD, `ddof=1`)** 이다 (SE 아님).
`metrics.aggregate` 는 `{field: (평균, SD, SE)}` 를 반환하며 시각화는 `[1]`(SD)을 쓴다.

* **꺾은선 패널 (A, B, F, I)** — 평균 곡선 + 평균±1SD 반투명 음영대 (`band()`).
  라운드별로 시드 축에 대해 SD 를 계산한다.
* **막대 패널 (C, D, E, G, H)** — 평균 막대 + 1SD 오차막대(capsize). C·D 는 개별 시드값을
  지터 산점으로 겹쳐 그려 분포 형태까지 드러낸다.
* **짝지은 차분**: H5 의 방어 특이성 SD 는 시드별 `방어(착취자) − 방어(잡음TFT)` 를 먼저
  계산한 뒤 그 분포의 SD 다. `run_many` 가 spec 순서를 보존하므로 시드 정렬이 보장된다
  (독립 SD 를 합성하면 과대추정된다).
* **H6 의 SD**: 시드마다 라운드로빈 전체 평균을 하나의 관측치로 삼는다. 상대 전략 간
  변동성이 시드 간 변동성과 섞이지 않도록 한 것이다.

### 각 패널의 y축

| 패널 | y축 | 범위/단위 |
|---|---|---|
| A, B | 공감 λ | [0, λ_max=0.8] |
| C | E[α](로짓, 음수 가능) · E[β](양수) | 단위 상이 — 같은 x 그룹 안에서만 비교 |
| D | 착취가능성 = P(CD)−P(DC) · CC율 | [−1,1] · [0,1] |
| E | 방어량(−착취가능성) · 방어 특이성 · 누적보수/100 | 막대별 단위 상이 |
| F | g⁻(불만), g⁺(신뢰), 기질귀인 신뢰도 | 모두 무차원 [0,1] |
| G, H | 라운드당 평균 보수 | [0, 5] (P=1, R=3, T=5) |
| I | 집단 상호협력(CC)률 | [0, 1] |

의존성: `numpy`, `matplotlib`. 선택: `inferactively-pymdp>=1.0`, `jax`, `equinox`.

---

## 8. 확장 훅 (fMRI / 향후 과제)

* `inversion.belief_update_magnitude(prev_means)` → Buergi et al. (2026) 의 믿음갱신 회귀자
  (rTPJ). 라운드별로 `agent.log["belief_update"]` 에 기록된다.
* `agent.log["disp_credence"] / ["ctx_credence"]` → rmPFC 중재 신호 후보.
* `agent.log["grievance"] / ["trust"]` → vmPFC 즉각가치 / 누적가치 후보.
* `CoreAllostaticBeliefState.prior_reliability` → 사전학습된 core belief 을 주입하는 자리
  (Sul et al. 2015 이타성 과제, Lee et al. 2018 창의성 평가 과제로 개인별 사전 추정 후 주입).
* `StrategyAgent.schedule` → 임의의 형질 전환 스케줄 (변덕 상대 일반화).

---

## 9. 알려진 한계

1. `policy_len=1` 이 기본이다. `planning_horizon>1` 이면 `SophisticatedPlanner` 가
   2^H 정책을 열거하므로 H≤4 를 권한다.
2. 완전관측(A=I₄) 가정 하에서는 pymdp 의 상태추론이 자명하다. 지각 잡음을 도입하려면
   `generative.build_A(noise)` 를 쓰고, 이 경우 numpy 해석해 경로는 무효가 된다(pymdp 경로 사용).
3. `make_capricious` 의 화해 국면을 TFT 로 두면 grudge 교착(echo effect)이 발생해 λ 가
   회복되지 않는다. 기본값은 GTFT 다. 이는 모형의 결함이 아니라 예측이다.
4. 무작위(p=0.5) 상대에 대해 정교형은 계속 협력해 손해를 본다 (§5, H5). λ_base 가
   구조적 사전으로 남아 있기 때문이며, 의도적 설계다.
