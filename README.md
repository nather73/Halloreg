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
| `--experiments` | 전체 | `H1 H2H3 H4 H5 H6 H7 H8 H9H10` 중 선택 |
| `--backend` | numpy | `numpy` (해석적 EFE) 또는 `pymdp` (검증용, 느림) |
| `--quick` | off | 스모크 테스트 (seeds=3, rounds=30) |

## 출력

`AIF_IPD/results/` 에 **가설마다 개별 PNG** 와 수치 요약 JSON 이 저장된다.

| 파일 | 가설 |
|---|---|
| `h1_self_protection.png` | H1 착취자에 대한 자기보호 |
| `h2_lambda_recovery.png` | H2 λ 하향과 회복(의도 vs 맥락) |
| `h3_trait_estimates.png` | H3 형질 추정(E[α], E[β]) 구분 |
| `h4_cooperation_restoration.png` | H4 TFT 상대 상호협력 복원 |
| `h5_immediate_vs_sophisticated.png` | H5 즉각형(vmPFC) vs 정교형(rmPFC) 자기보호 |
| `h6_noise_fixed_strategies.png` | H6 잡음 수준별 고정전략 성능 |
| `h7_capricious_partners.png` | H7 다양한 형질전환(변덕) 상대 |
| `h8_population_dynamics.png` | H8 집단 역학과 상호협력 |
| `h9_alpha_only_attribution.png` | H9 α 전용 귀인의 한계 |
| `h10_lambda_only_attribution.png` | H10 λ 전용 귀인의 한계 + 이중해리 |
| `halloreg_results.json` | 전 가설 수치 요약 |

## 구조

```
AIF_IPD/
  core/        constants, allostasis(g⁻/g⁺ 누적기 + LambdaRegulator), generative, pymdp_backend
  ipd/         agent(ToMEmpathicAgent, AdaptiveAgent), env(전략 + 형질전환 카탈로그), sim(병렬 러너)
    tom/       inversion(입자필터), tom_core, sophisticated_planner
    metrics/   exploitability, defense_metrics, welch_t
  scripts/     run_ipd_experiment.py  ← 메인
  results/     산출물 (gitignore)
```

## 재현성

다이애드·집단 replicate 는 모두 시드로 결정되며, `multiprocessing('spawn')` 워커에
분배되어도 결과는 워커 수와 무관하다. BLAS/JAX 스레드는 1로 고정하여
oversubscription 을 방지한다.
