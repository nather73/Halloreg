# Halloreg

Computational Modeling and Simulations for Hierarchical Allostatic Regulation in Social Dynamics

능동추론(Active Inference) 기반 반복 죄수의 딜레마(IPD)에서, 공감 가중치 λ 를
항상성(allostatic) 누적기(불만 g⁻ / 신뢰 g⁺)로 내생적으로 조절하는 `AdaptiveAgent`
와, 상대의 잠재 형질(협력편향 α, 상호성 ρ, 행동정밀도 β)을 입자필터로 추론하는
Theory-of-Mind 모듈을 구현한다.

## 설치

```bash
pip install numpy matplotlib
pip install inferactively-pymdp   # 선택: --backend pymdp / --check-equivalence 용
```

## 실행

메인 엔트리포인트는 `AIF_IPD/scripts/run_ipd_experiment.py` (단수형) 이다.
리포지토리 루트(`Halloreg/`)에서 실행한다.

```bash
# 전체 실험 (기본값: seeds=120, rounds=60, 모든 코어 사용)
python AIF_IPD/scripts/run_ipd_experiment.py

# 빠른 확인
python AIF_IPD/scripts/run_ipd_experiment.py --quick

# 일부 가설만 / 순차 실행
python AIF_IPD/scripts/run_ipd_experiment.py --experiments H5 H7 H8 --jobs 1

# pymdp 1.0.x 와 numpy 해석적 EFE 동치성 검증
python AIF_IPD/scripts/run_ipd_experiment.py --check-equivalence --experiments H1
```

기존 이름 `run_ipd_experiments.py` (복수형) 도 위임 셔틀로 남아 동일하게 동작한다.

주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--seeds` | 120 | 조건별 시드(다이애드) 수 |
| `--rounds` | 60 | 다이애드 라운드 수 |
| `--jobs` | -1 | 병렬 워커 수 (-1 = 코어수-1, 1 = 순차) |
| `--experiments` | 전체 | `H1 H2H3 H4 H5 H6 H7 H7H H8 H8E H9H10 GS` 중 선택 |
| `--backend` | numpy | `numpy` (해석적 EFE) 또는 `pymdp` (검증용, 느림) |
| `--quick` | off | 스모크 테스트 (seeds=3, rounds=30) |

## 출력

`AIF_IPD/results/` 에 **가설마다 개별 PNG** 와 수치 요약 JSON 이 저장된다.

| 파일 | 가설 |
|---|---|
| `h1_self_protection.png` | H1 착취자에 대한 자기보호 |
| `h2_h3_intent_context.png` | H2/H3 의도 vs 맥락 귀인 + β-경로 절제 |
| `h4_cooperation_recovery.png` | H4 TFT 상대 상호협력 비열등성 |
| `h5_immediate_vs_sophisticated.png` | H5 즉각형 vs 정교형 + 2x2x2 요인 forest |
| `h6_noise_robustness.png` | H6 잡음 하 고정전략 성능 |
| `h7_capricious_partners.png` | H7 변덕 상대 + λ 반응성(순열 영가설) + 대칭 잡음 대전 |
| `h7h_horizon_dependence.png` | H7H 지평 스윕(전환수/주기 고정 족, T*, 히스테리시스) |
| `h8_scaling_cooperation.png` | H8 규모 확장 + 조건부 기울기 플롯 + ALLC 이전 회계 |
| `h8e_evolutionary_dynamics.png` | H8E 복제자·Moran 역학(침입 성장률·유역) |
| `h9_h10_attribution_scope.png` | H9/H10 귀인 범위 절제 + 비-ToM 베이스라인 |
| `gs_game_structure.png` | GS 게임구조(협력지수 CI) 강건성 스윕 |
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

## v0.2 — 연구 보완 (6축)

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

```bash
python scripts/run_ipd_experiment.py --experiments H7H H8E GS   # 보완 실험
python scripts/validate_tom_recovery.py                          # H3 복원 연구
python tests/test_golden.py            # 회귀 테스트 (--update 로 스냅숏 갱신)
```

선택 의존성: PyYAML(사전등록 로딩; 없으면 내장 사본 폴백), pytest(테스트 러너).

## 재현성

다이애드·집단 replicate 는 모두 시드로 결정되며, `multiprocessing('spawn')` 워커에
분배되어도 결과는 워커 수와 무관하다. BLAS/JAX 스레드는 1로 고정하여
oversubscription 을 방지한다.
