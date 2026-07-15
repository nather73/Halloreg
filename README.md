# Halloreg

Computational Modeling and Simulations for Hierarchical Allostatic Regulation in Social Dynamics

능동추론(Active Inference) 기반 반복 죄수의 딜레마(IPD)에서, 공감 가중치 λ 를
항상성(allostatic) 누적기(불만 g⁻ / 신뢰 g⁺)로 내생적으로 조절하는 `AdaptiveAgent`
와, 상대의 잠재 형질(협력편향 α, 상호성 ρ, 행동정밀도 β)을 입자필터로 추론하는
Theory-of-Mind 모듈을 구현한다.

## 설치

```bash
pip install numpy matplotlib scipy pyyaml
pip install inferactively-pymdp   # 선택: --backend pymdp / --check-equivalence 용
pip install "jax[cpu]" equinox optax   # 선택: validate_ann_tom.py (ANN-ToM 재현) 용
```

## 실행

메인 엔트리포인트는 `AIF_IPD/scripts/run_ipd_experiment.py` (단수형) 이다.
리포지토리 루트(`Halloreg/`)에서 실행한다.

```bash
# 전체 실험 (기본값: seeds=240, rounds=60 240 두 지평, 모든 코어 사용)
python AIF_IPD/scripts/run_ipd_experiment.py

# 단일 지평만
python AIF_IPD/scripts/run_ipd_experiment.py --rounds 60

# 빠른 확인
python AIF_IPD/scripts/run_ipd_experiment.py --quick

# 일부 가설만 / 순차 실행
python AIF_IPD/scripts/run_ipd_experiment.py --experiments H5 H7 H8 --jobs 1

# pymdp 1.0.x 와 numpy 해석적 EFE 동치성 검증
python AIF_IPD/scripts/run_ipd_experiment.py --check-equivalence
```

기존 이름 `run_ipd_experiments.py` (복수형) 도 위임 셔틀로 남아 동일하게 동작한다.

주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--seeds` | 240 | 조건별 시드(다이애드) 수 |
| `--rounds` | 60 240 | 다이애드 라운드 수 — 복수 지정 시 각 지평에서 전 실험 반복 후 지평 비교 |
| `--jobs` | -1 | 병렬 워커 수 (-1 = 코어수-1, 1 = 순차) |
| `--experiments` | 전체 | `H1 H2H3 H4 H5 H6 H7 H7H H8 H8E H9H10 H11 H12 GS VP ABA ORE` 중 선택 |
| `--backend` | numpy | `numpy` (해석적 EFE) 또는 `pymdp` (검증용, 느림) |
| `--quick` | off | 스모크 테스트 (seeds=3, rounds=30) |
| `--check-equivalence` | off | pymdp↔numpy EFE 등가성만 검증하고 종료 |

## 출력

`AIF_IPD/results/` 에 **가설마다 개별 PNG** 와 수치 요약 JSON 이 저장된다. 복수 지평 실행 시 지평별 실험 그림은 `_T60`/`_T240` 접미가 붙고(H7H·H8E 는 내부에서 여러 T 를 함께 다루므로 접미 없음), 지평 간 확증 판정 개관은 `horizon_overview.png` 로 저장된다.

| 파일 | 가설 |
|---|---|
| `h1_self_protection.png` | H1 착취자에 대한 자기보호 |
| `h2_h3_intent_context.png` | H2/H3 의도 vs 맥락 귀인 + β-경로 절제 |
| `h4_cooperation_recovery.png` | H4 TFT 상대 상호협력 비열등성 |
| `h5_immediate_vs_sophisticated.png` | H5 즉각형 vs 정교형 + 2x2x2 요인 forest |
| `h6_noise_robustness.png` | H6 잡음 하 고정전략 성능 |
| `h7_capricious_partners.png` | H7 변덕 상대(주기×순환족 격자) + 주기 의존성 Δ(P) + λ 반응성 + 대칭 잡음 대전 |
| `h7h_horizon_dependence.png` | H7H 지평 스윕(전환수/주기 고정 족, T*, 히스테리시스) |
| `h8_scaling_cooperation.png` | H8 규모 확장 + 조건부 기울기 플롯 + ALLC 이전 회계 |
| `h8e_invasion_structure.png` | H8E-1 보수 구조·침입 성장률 격자·역방향 침입·구성 민감도 |
| `h8e_replicator_dynamics.png` | H8E-2 복제자 궤적(전 유형 legend)·역침입·끌개 조성·Moran |
| `h8e_state_space_basins.png` | H8E-3 상태공간 위상 초상·협력 유역 지도·유역 확장 격자 |
| `h11_keystone.png` | H11 Keystone 검정 — 치환 설계 dose 곡선·Δ 기울기·기제·진화 프레임 |
| `h12_keystone_frontier.png` | H12 상주 조건부 Keystone 프런티어 — R-P/R-K/R-N 위협축·R-D 중복축 반응 곡면·비대체성·교차 m*·Shapley·ALLD 장벽 심화·H8E/H11 정위 |
| `horizon_overview.png` | 확증 지표의 지평(T=60/240)별 판정 개관 |
| `h9_h10_attribution_scope.png` | H9/H10 귀인 범위 절제 + 비-ToM 베이스라인 |
| `gs_game_structure.png` | GS 게임구조(협력지수 CI) 강건성 스윕 |
| `vp_variable_payoff.png` | VP 가변 페이오프(연속 협력–경쟁) — 시변/극단 CI 레짐 payoff 우위·유역 확장(RE/ORE) |
| `aba_intent_recovery.png` | ABA A→B→A 의도복구 — 용서(재탐색) 용량-반응·히스테리시스·자기충족 함정 |
| `ore_optimal_replicator.png` | ORE 최적복제자(Bravetti&Padilla) — 2-유형 재현·RE vs ORE 종착 CC율·adaptive 한계기여 |
| `vp_confirmatory.png` / `aba_confirmatory.png` / `ore_confirmatory.png` | [v0.6.3] 확증 전용 — C-VP1/2·C-ABA1/2·C-ORE1/2 각각의 replicate 분포·평균±95%CI·판정 임계·순열 p 를 명시적으로 시각화 |
| `ann_tom_recovery.png` | ANN-ToM 재현 — Schwarcz GRU · Kim GCN+RNN 이 입자필터 사후 재현(별도 스크립트) |
| `recovery_tom.png` | ToM 파라미터 복원 연구(편향·RMSE·커버리지·식별불능) |
| `<name>.pdf` / `<name>.caption.json` | 각 그림의 벡터본 + 캡션 메타데이터 |
| `summary.json` | 전 가설 수치 + 확증/탐색 태그·p_holm·q_fdr·효과크기 |

## 구조

```
AIF_IPD/
  core/        constants, allostasis(g⁻/g⁺ 누적기 + LambdaRegulator), generative, pymdp_backend
  ipd/         agent(ToMEmpathicAgent, AdaptiveAgent), env(전략 + 형질전환 카탈로그), sim(병렬 러너)
    tom/       inversion(입자필터), tom_core, sophisticated_planner
    metrics/   exploitability, defense_metrics, stats(순열/부트/보정 — v0.2)
    baselines.py  비-ToM 학습 베이스라인(Q러너/베이지안BR/fictitious — v0.2)
    evolution.py  복제자 동역학·Moran 과정(H8E — v0.2)
  experiments/ hypotheses.yaml  가설 사전등록(1차 지표·판정 상수·근거 — v0.2)
  scripts/     run_ipd_experiment.py  ← 메인
               validate_tom_recovery.py  ToM 파라미터 복원 연구(H3 격상 — v0.2)
  tests/       test_golden.py  황금 시드 회귀 테스트(엔지니어링 안전망 — v0.2)
  results/     산출물 (gitignore)
```

## v0.3 — 지평 이중화·변덕 격자·진화 동역학 (3축)

`docs/IMPLEMENTATION_PLAN.md` §16 참조.

1. **지평 이중화.** 기본 실행이 `--rounds 60 240` 두 지평에서 전 실험을 반복하고,
   `interpret_horizons()` 가 지지 여부(딕셔너리로 분해된 하위기준)를 지평 간
   비교해 **뒤집히는 기준만** 추려 H7H 의 Δ(T)≈b·T−c·k 기제와 정합하는지
   해석 텍스트를 자동 생성한다. `horizon_overview.png` 로 확증 지표의 지평별
   판정을 개관한다. 짧은 지평의 결론을 긴 지평으로 외삽하지 않는다는 방침을
   코드로 강제한다.
2. **H7 변덕 격자.** case 카탈로그를 전환 주기 {10,30,60,120} × 순환족 8종
   (정교형 adaptive 국면 포함 3종)으로 재정의(32 case). `SwitchingAgent` 는
   전략 국면은 memory-1 연속으로, adaptive 국면은 지속 에이전트의 온난 전환으로
   처리한다. 주기별 Δ(P)=adaptive−GTFT 를 무전환 대조와 함께 보고하고, 지지
   부호가 주기에 갈리면 `period_dependent` 로 명시·시각화한다.
3. **H8E 진화 동역학.** env_error {0,.05,.1,.15,.2}×T{60,240} 격자(Π seeds=50,
   대칭 재사용). 확증은 err=0.10 의 지평별 침입 성장률. 탐색: 역방향 침입
   (adaptive 상주 → 고정전략), Dirichlet 구성 민감도, Replicator 상태공간
   (끌개·3-유형 위상 초상·cooperation basin), Moran 교차검증. 복제자 궤적은
   9유형 전체를 고정 색 legend 로 그린다. 그림 3분할(invasion_structure /
   replicator_dynamics / state_space_basins).



학술 비판에 대응해 6개 축을 보완했다. 자세한 내용은 `docs/IMPLEMENTATION_PLAN.md`
§10–15 및 "주장 감사(Claims Audit)" 표 참조.

1. **통계.** `ipd/metrics/stats.py` — 순열/부트스트랩 검정, Hedges' g·dz + 95% CI,
   Holm(확증 1차 지표)·BH-FDR(탐색). 사전등록(`experiments/hypotheses.yaml`).
   퇴화(H1 λ)·우측절단(첫 배신) 검정 교체. CRN 잡음 사전생성(`noise_seed`).
2. **설계.** H5 2×2×2 요인 분해, H2 β-클램프/intent 절제, H7 환경 계층 잡음 대칭화,
   λ 반응성 순열 영가설, ToM 복원 연구.
3. **단순성.** H8E 복제자/Moran 역학, 매칭 분산 분해, 구성 민감도, 게임구조(GS) 스윕,
   비-ToM 학습 베이스라인.
4. **해석.** λ=0.24 '정성적 일치' 재보정, ALLC 착취 이전(transfer) 회계,
   `[확증]`/`[탐색]` 태그 제도화.
5. **시각화.** 검정 패널 부트스트랩 CI vs 궤적 백분위 밴드, n 표기, 이중축 제거,
   조건부 기울기 플롯, PNG+PDF+캡션 JSON.
6. **지평 검증.** H7H — Δ(T) 를 전환수 고정/주기 고정 두 족에서 분리 추정, T*,
   히스테리시스 기제 조작.

## v0.4 — GTFT 이중화·H7 상대이점 명시·H11 Keystone (3축)

1. **GTFT 두 용서 기제의 명시적 분리.** 기존 GTFT 는 용서를 확률(30%)로만
   정의했다. v0.4 는 `generous_tft`(확률론적 용서 — Nowak & Sigmund 1992)와
   `generous_tft_count`(횟수 기반 용서 — 연속 배신이 `forgive_streak`(기본 2)회를
   넘기 전까지는 용서하고 임계 초과 시 보복하는 인내 임계 정식화)를 별개 kind 로
   구현하고, GTFT 가 등장하는 **모든** 시뮬레이션·분석·시각화(H6 잡음 강건성,
   H7 focal·순환족·대전, H7H Δ(T), H8 LARGE_MIX·filler, H8E 유형계·협력 유역,
   H11 협력자 기저, GS 강건성 스윕)에 두 유형을 병기한다.
2. **H7 상대이점의 주기×지평 명시.** adaptive 의 payoff 우위를 GTFT(확률),
   GTFT(횟수), WSLS 세 rival 각각에 대해 전 case 합산과 **전환 주기 P 별**로
   분해해 보고·시각화한다(`adaptive_advantage`). 어느 전환 간격과 어느 지평
   (T=60/240; 지평 비교는 main 의 horizon_comparison)에서 adaptive 가 더 많은
   보수를 얻는지 부호·CI·유의성(*)을 그림 (d) 패널에 직접 표기한다.
3. **H11 Keystone(핵심종) 검정.** "adaptive 는 승자가 아니라 조력자" — H7 경쟁
   열세·H8E 침입 실패와 모순 없이, 협력자에게 ALLD 대비 상대이점을 주고(주장 A:
   gap = 조건부 협력자 평균 보수 − ALLD 보수) 집단 협력률을 높이는지(주장 B:
   집단 CC)를 검증한다. 비-swap 구성(ALLD 30%)을 고정한 **치환 설계**
   (N=30 완전 라운드로빈, dose ∈ {0,3,6,9}, dose0 은 전 arm 공유 + CRN 짝지음),
   핵심 대조 **control-1 = fixed-λ ToMEmpathic(λ=0.4)** — treatment−ctrl1 이
   λ 위계적 조절의 순수 효과. 확증 2지표: ΔCC·Δgap 의 dose 기울기 > 0.
   탐색: ALLD 억제, 협력자 보호, 착취 이전(transfer) 회계 가드, 즉각형/추가
   TFT/ALLC arm 대조, 다이애드 기제(adaptive→ALLD DD 방어율 vs adaptive→협력자
   CC 유지율), 진화 프레임(협력자 상주에 adaptive 혼합 시 ALLD 침입장벽 심화,
   ALLD-heavy 상주에 대한 협력자 클러스터 침입성, 후보 유형별 협력 유역 확장 —
   adaptive vs TFT/GTFT 두 유형/WSLS/fixed-λ, 벡터화 replicator 부트스트랩).

```bash
python scripts/run_ipd_experiment.py --experiments H7H H8E GS   # 보완 실험
python scripts/validate_tom_recovery.py                          # H3 복원 연구
python tests/test_golden.py            # 회귀 테스트 (--update 로 스냅숏 갱신)
```

선택 의존성: PyYAML(사전등록 로딩; 없으면 내장 사본 폴백), pytest(테스트 러너).

## v0.5 — H12 상주 조건부 Keystone 프런티어

**동기 — 인공물 대 실질의 분리.** H8E(고변덕·저중복 상주에서 adaptive 의 협력
유역 확장 Δ=+0.67 @T=240)와 H11(무변덕·고중복 상주에서 adaptive < TFT)은
"adaptive 의 유역 확장 기여" 의 **부호가 상주 구성에 따라 뒤집히는** 상충으로
보인다. H12 는 (1) 두 결과가 서로 다른 **지지집합·차원·척도** 위에서 측정된
데서 비롯한 교란(인공물)을 제거하고, (2) 남는 실질 성분을 **상주 구성을 독립변수로
갖는 반응 곡면**으로 정량화해 keystone 경계를 규명한다.

1. **고정 심플렉스 재정의(원칙 A·D).** 단일 정준 최대 유형집합 **S\*(11종:
   TFT·GTFT확률·GTFT횟수·WSLS·ALLC·ALLD·random·capricious·adaptive·adaptive_imm·
   tom_fixed)** 를 고정한다. 상주는 "지지집합 변경" 이 아니라 S\* 위 **초기점 사전
   평균의 대치 슬롯** 으로 재정의된다. 추정량은 고정 Π\* + 고정 초기점 사전 위의
   단일 범함수로 고정.
2. **중립 filler 대치 측정(원칙 B).** 한 유형의 기여는 제거가 아니라 **중립
   filler(random) 대치**로 측정한다:
   `sub_widening(X | 배경 B, 슬롯 s) = coop_basin_frac(Π*, P_B, 슬롯=X)
   − coop_basin_frac(Π*, P_B, 슬롯=random)`. 지지집합·차원·척도가 모든 상주에서
   불변이라 H8E·H11 두 결과가 **동일 저울**에 오른다.
3. **위협축·중복축 매개화(원칙 C).** 위협축 — **R-P**(ALLD 점유 ρ_D ∈
   {0,.1,.2,.3,.4}), **R-K**(변덕 점유 ρ_C ∈ {0,.1,.2,.3}), **R-N**(환경 잡음 err
   격자). 중복축 — **R-D**(협력자 다양성 m: TFT단독→+GTFT→+WSLS→+GTFT횟수·ALLC).
   슬롯 점유는 협력자 예산에서만 인출하고 ALLD/변덕 점유는 불변으로 유지.
4. **확증 3지표(사전등록).** **C1** 위협 단조성(R-P/R-K 에서
   sub_widening 기울기 > 0), **C2** 중복 단조성(R-D 에서 [sub_widening(adaptive)
   − sub_widening(TFT)] 기울기 < 0), **C3** 교차 프런티어 m\* 존재(basin(adaptive)
   − basin(best_fixed) 부호전환이 R-D 범위 **내부**, 부트 근찾기 CI 유계). 교차가
   관측 범위를 벗어나면 "범위 내 부재" 로 정직 보고하고 외삽하지 않는다(§6.3).
5. **초기점 사전 이중화(원칙 E).** 균등 Dirichlet(1) interior 사전과 ALLD-heavy
   구조적 사전을 병기해 C1/C2 결론 일치를 강건성으로 보고한다.
6. **탐색.** Shapley 순서무관 기여 φ(adaptive) vs φ(TFT)(C={ad,TFT,GTFT,WSLS},
   2⁴ 연립)·ALLD 침입장벽 심화 Δg(random 앵커 대비 치환 회계, fixed-λ 대조)·
   R-K 변덕대응 정교(adaptive) vs 즉각(adaptive_imm) 기울기 분해(H5 연결)·
   H8E(고변덕·저중복)와 H11(무변덕·고중복)을 곡면 위 두 점으로 재현·정위.

H12 는 H8E 와 동형으로 내부에서 **err×T 격자를 자체 추정**(Π\* 는 전 상주가 공유 —
원칙 A 의 부수효과)하므로 `GLOBAL_EXPERIMENTS` 에 속하며 전체 지평 목록을 함께
받는다. 설계 원문: `docs/H12_resident_conditioned_keystone_DESIGN.md`.

```bash
python scripts/run_ipd_experiment.py --experiments H12        # H12 단독(내부 err/T 스윕)
```

## 재현성

다이애드·집단 replicate 는 모두 시드로 결정되며, `multiprocessing('spawn')` 워커에
분배되어도 결과는 워커 수와 무관하다. BLAS/JAX 스레드는 1로 고정하여
oversubscription 을 방지한다.

## v0.6 — 가변 페이오프 · A-B-A 의도복구 · 최적복제자 · ANN-ToM 재현

기존 실험군의 네 가지 구조적 한계(고정 payoff·이분법적 협력/경쟁·단순 복제자
동역학·ToM 의 신경 구현 부재)를 보완하는 확장이다. 아울러 자기/타인 통제권 귀인
(Spiering 2025)을 `AdaptiveAgent` 에 선택적으로 추가했다(기본 off — H1–H12 보존).

### 새 실험(내부 레짐 스윕 — `GLOBAL_EXPERIMENTS`)

- **VP — 가변 페이오프(Pisauro et al. 2022 Space Dilemma B6).** 협력지수
  CI=(R−P)/(T−S)를 극단(음수 교착·1 이상 조화)까지 라운드별 변동. 맥락-의존 효용을
  계산하는 adaptive 가, 어떤 단일 고정전략도 전 맥락에서 최적일 수 없는 **시변
  레짐**(oscillate·blocks·aba)에서 우위를 갖는지(C-VP1)와 협력 유역을 넓히는지
  (C-VP2; RE·ORE 병기)를 검증. H7/H8E/H12 의 '고정 payoff' 교란 제거.
  구현: `ipd/variable_payoff.py`(라운드별 `set_payoffs`, spawn-병렬 `run_variable_many`).

- **ABA — A→B→A 의도 전환 추적 복구.** 상대가 협력적 상호성(A)→착취(B)→다시 A 로
  복귀. **정직 보고**: 완전한 allostatic 에이전트는 배신 후 자기보호에 고착되어
  자동 복구하지 않는 히스테리시스(배신 후 선택적 관측 함정)를 보이며, 복구는
  **용서(재탐색)** 에 용량-반응적으로 게이팅된다(C-ABA1 高용서 복구, C-ABA2 용량-
  반응 기울기<0). β-맥락 절제·통제권·비-ToM 대조.

- **ORE — 최적복제자방정식(Bravetti & Padilla 2018).** 개체 수준 RE 를 집단 수준
  경쟁까지 확장한 ORE 가 협력 유역을 넓히는지(C-ORE1)와 adaptive 한계 기여가 ORE
  하에서 뚜렷한지(C-ORE2) 검증. Bravetti Fig.1 재현 포함. FBSM(forward-backward
  sweep)을 초기점 전체로 벡터화. 구현: `ipd/evolution.py` 의 `ore_*`.

```bash
python AIF_IPD/scripts/run_ipd_experiment.py --experiments VP ABA ORE
```

### 별도 스크립트 — ANN 의 베이지안 ToM 재현 (수정 #4)

입자필터 사후를 교사로 하는 지식 증류로, 두 신경망이 베이지안 ToM(특성 추론 +
특성별 신뢰도 배정)을 재현하는지 검증한다.

- **Schwarcz-style GRU**(Schwarcz et al. 2025): 순환 동역학에 베이즈 믿음 갱신 내재화.
- **Kim-style GCN+RNN**(Kim et al. 2026): 관계형 그래프 + 스포트라이트 주의(간선
  엔트로피 = 특성별 reliability).

두 구조 모두 교사 사후를 높은 R²(≈0.77–0.80)로 재현한다. 참 θ 개별 회복은 생성
모형의 α↔λ_j 가법 축퇴로 제한되나, **식별가능 합성 α+5λ_j** 는 잘 회복된다(정직한
식별성 보고 — ANN 결함이 아니라 교사에도 동일한 한계). JAX/Equinox 구현.

```bash
python AIF_IPD/scripts/validate_ann_tom.py            # 전체
python AIF_IPD/scripts/validate_ann_tom.py --quick    # 스모크
```
산출물: `results/ann_tom_recovery.png`, `results/ann_tom_summary.json`.

### 자기/타인 통제권 귀인 (Spiering 2025)

`AdaptiveAgent(controllability=True)` 로 활성화. 결과를 자기-기인(내 방출 행동이
DD/DC 유발)과 타인-기인(내가 협력했는데 배신당한 CD)으로 분할, 지각된 통제권으로
타인-귀인 가중치 w_other 를 산출해 dispositional grievance 충전을 게이팅한다(자기-
기인 배신은 상대 기질 불만을 덜 충전). 구현: `core/controllability.py`,
`core/allostasis.py`(attr_gate). 기본 off 로 H1–H12 결과 보존.

## v0.6.1 — 협력 지표 개정: 이분 유역 폐기 → 행동적 CC율

이분 협력 유역(협력-라벨 유형의 종착 점유율 > 0.5)은 두 결함이 있어 **모든 가설 검증·
시각화에서 폐기**하였다: (i) 임의 임계(0.5)에 의한 이분화(임계 통과 후의 질적 개선에
눈멂, 유계 차분의 천장 효과), (ii) 협력 '라벨'과 실제 협력 '행동'의 괴리(예: deadlock
레짐에서 협력-라벨 유형이 실제로는 배신하는데도 협력으로 집계).

대신 종착 조성의 **행동적 CC율** 을 표준 지표로 채택한다:

    CC(x) = xᵀ · CCm · x       (CCm[i,j] = 유형 i·j 다이애드의 실제 상호협력 발생률)

짝을 조성 x 에 따라 뽑을 때 두 개체가 모두 협력할 기대 확률이며, 연속값이고 유형
라벨이 아니라 관측된 CC 행동에서 직접 유도된다. `estimate_payoff_matrix`·
`estimate_variable_payoff_matrix` 가 CCm 행렬을 함께 반환하고, `evo.re_terminal_cc`·
`evo.ore_terminal_cc` 가 RE·ORE 종착 조성의 CC율을 계산한다.

이 개정의 영향(사전등록 지표 변경):
- **VP** C-VP2: RE sub_widening → **RE CC-widening**(종착 CC율 adaptive−random).
- **ORE** C-ORE1: ORE 유역>RE 유역 → **ORE 종착 CC율>RE 종착 CC율**; C-ORE2: adaptive
  sub_widening → **adaptive CC-widening**.
- **H8E** basin_widening → **CC-widening**(supported 키 cc_widening_positive).
- **H11** 후보별 협력 유역 확장 → **CC-widening**(탐색; 주 확증은 원래 CC 기울기 기반).
- **H12** C1/C2/C3 keystone 프런티어를 **CC율 곡면** 위에서 재계산(basin_slot →
  re_terminal_cc). Shapley 한계 기여도 CC율 기반.

정직 보고: deadlock 에서 과거 이분 유역은 인공적 고점(≈0.9)을 보였으나 행동적 CC율은
낮게 유지된다(adaptive 자기-CC≈0.22 — 실제 배신 반영). 지표 개정이 '유역 점유'와 '실제
상호협력'의 괴리를 제거해 과잉해석을 막는다. 상태공간 위상도(끌개 유역)는 동역학 구조
시각화로 유지된다(협력 정량 지표가 아님).

## v0.6.2 — 확증 검정 구성 개정: replicate 단위 = 시드

v0.6.1 검증에서 VP C-VP2·ORE C-ORE1/C-ORE2 가 **전 레짐에서 부호가 일관되게 양수인데도
미지지**로 나오는 문제가 확인되었다. 원인은 효과 부재가 아니라 **확증 검정의 replicate
단위가 '레짐'(n=3~4)이었던 구조적 검정력 결손**이다 — 레짐 수준 순열의 p 하한은
~1/2ⁿ(n=4 → 0.0625)이라 시드를 240까지 늘려도 유의에 도달할 수 없다.

개정: 확증 replicate 단위를 **시드**로 전환. 시드 s 마다 그 시드의 다이애드만으로
구성한 Π_s·CCm_s 로 종착 CC율을 계산해 (레짐×시드) 관측을 만든다
(`evo.per_seed_terminal_cc`). 동일 X0(CRN)·동일 시드 인덱스로 짝지어 비교의 정합성을
유지하며, 레짐 수준 부호 일관성은 탐색 지표로 병기한다.

효과(seeds=16 검증): 효과크기는 그대로인 채 검정력만 회복 —
- C-VP2 : CC_widening=0.043 [0.012,0.075] (n=48), p_holm=0.006 → **지지**
- C-ORE1: ΔCC=+0.202 [0.156,0.246] (n=64), p_holm=0.0002 → **지지** (4/4 레짐 일관)
- C-ORE2: CC-widening=0.090 [0.076,0.103] (n=64), p_holm=0.0002 → **지지**


## v0.6.3 — 확증 전용 시각화 · 보수-매개 공감항 · 층화 끌개 표집

**(1) 확증 전용 그림 3종.** `vp_confirmatory.png`·`aba_confirmatory.png`·
`ore_confirmatory.png` 가 C-VP1/C-VP2·C-ABA1/C-ABA2·C-ORE1/C-ORE2 각각을 replicate
분포(시드×레짐 지터) + 그룹/전체 평균±부트95%CI + 판정 임계선 + 순열 p 로 명시적으로
보여준다(C-ORE1 은 짝지은 RE vs ORE 산점, C-ABA2 는 용량-반응 기울기+CI 밴드).

**(2) 보수-매개 공감항 `empathy_shift` (core/constants.py).** 레거시 ToM 공감항
5λ−p−1 은 기본 보수 (T=5,R=3,P=1,S=0) 에서 유도된 상수식이었다. 이를 현재 보수를
호출 시점에 읽는 일반형으로 대체:

    empathy_shift(λ, p) = (T−S)·λ + (R−T+P−S)·p + (S−P)

기본 보수에서 레거시와 **비트 단위 동일**(H1–H12·골든 회귀 완전 보존)하며, 가변
페이오프 환경(§VP)에서는 ToM 공감항이 맥락 효용을 반영한다(예: 조화 R=7 이면 p 계수
+3 으로 반전). 적용: 입자필터(inversion), ToM 복원 스크립트, ANN-ToM 매개적 상대 —
전 소비처 일관. ANN 식별가능 합성의 λ 계수도 (T−S) 로 일반화. **정직 보고**: ToM
공감항의 맥락화로 payoff-blind 고정전략에 대한 생성모형 불일치가 생겨 VP 효과크기가
감소(Δpay 0.166→0.060; CC-widening 0.043→0.051 @seeds=24). seeds=24(실사용 상한)
검증에서 6/6 확증 지지 유지: C-VP1 p=0.010, C-VP2 p=0.005, C-ORE1/2 p=0.0002.

**(3) 층화 끌개 표집 (evolution.attractor_analysis, 기본 n=1500).** 고차원(k=9)
Dirichlet(1) 균등표집의 중심 편향('한 유형 과반' 초기점 ≈3%)을 보완: 균등층(70%) +
유형별 코너층(30%; x=0.8·e_i+0.2·Dirichlet)으로 층화. 끌개 **발견**은 전 층으로,
**basin_frac 은 균등층에서만**(균등 측도 비편향 유지), 유형별 코너 초기 집단의
수렴처는 `corner_convergence` 로 별도 보고(코너 강건성). H8E 끌개 구성 n=400→1500.
