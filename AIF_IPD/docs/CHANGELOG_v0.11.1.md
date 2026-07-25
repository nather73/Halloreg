# HalloReg v0.11.1 — 변경 기록 (CHANGELOG)

이 문서는 v0.10.0 이후 v0.11.1 까지 반영된 **모든** 변경을 빠짐없이 기록한다.
코드 주석의 `[v0.11.1]` 태그가 본 변경들에 대응한다.

---

## 1. CoreAffect — θ-설명 잔차 affect (affect_mode="theta_residual", 기본)

**동기.** 정서는 사회적 맥락·타인 의도에 의존해 생성되어야 하나, v0.10.0 의
valence/uncertainty 는 실현보수 분포에만 접지되어 맥락-의존성이 없었다.

- **RPE 잔차화**: 기대보상 분포 Z(SelfModel setpoint 초기화 유지)와 실현보수 간
  RPE 를, 입자필터 θ 사후가 예측하는 **θ-조건부 기대보수로 회귀 제거**한다.
  `OpponentInversion.theta_reward_moments(my_action, ctx, PAYOFF_SELF)` 신설:
  입자별 협력확률 pc_k → 예상보수 r̂_k = pc_k·U(a,C)+(1−pc_k)·U(a,D) →
  E_θ[r]=Σw_k r̂_k, epistemic std=√Σw_k(r̂_k−E_θ)², aleatoric std(참고용).
- **잔차 귀인**: 잔여 mean → valence `V=(r_obs−E_θ[r])/(R−P)`,
  **θ 입자간(epistemic) std** → uncertainty `U=epi_std/u_ref` (aleatoric 미반영,
  사양). θ 예측 없으면 E_θ→E0(SelfModel) fallback.
- **Δλ 게이팅 재설계**: `Δλ = k_affect·V·(1−U)` — valence 부호가 방향,
  uncertainty 는 확신도 **게이트**(확실한 협력/배신→|Δλ|↑, 불확실→|Δλ|↓).
  k_V/k_U 가중합 폐기(k_U 제거, K_AFFECT_DEFAULT=0.7, U_GATE_REF_DEFAULT=0.5).
- **효과**: 예측 가능한 상대(ALLD·ALLC)는 affect 침묵(V≈±0.02 — "예상된 배신은
  놀랍지 않다"), 예측 불가(random)만 큰 affect(V≈−0.85). 예측처리 정론 구현.
- A/B: `affect_mode="valence_uncertainty"`(v0.10.0), `"legacy_attrib"`(구 귀인)
  토글 보존. legacy_attrib+quantile 은 구 골든 비트동일 재현 확인.

## 2. SelfModel — λ_base 의 self/context 중재 (자기보호 역설 해소)

**동기.** 순수 θ-잔차는 "θ 가 착취자를 잘 예측할수록 affect 가 침묵해 방어가
약해지는" allostatic 역설을 재도입했다(v0.11.0 중간 상태에서 H1 방향 역전 관측,
dz=+4.0). 또한 구 λ_base 는 거리가중 집계 단일식이라 착취자 방어가 거리 소멸로
자기제한적이었다.

- **분해식**: `λ_base = (1−w_cd)·λ_baseSelf + w_cd·λ_baseContext(θ_j)`
  - `lambda_base_self()`: **발달 엔트리만**의 거리가중 협력성 집계(상대 무관
    자기 특질, tonic setpoint). 동적 identity 엔트리 제외.
  - `lambda_base_context(ident)`: 현재 상대 θ 맞춤 setpoint. **거리와 무관** —
    **상호적(reciprocal) 사상** 채택:
    `base_j = (1−m_recip)·ĉ_θ + m_recip·λ̂_j`,
    `λ_ctx = λ_floor + (λ_ceil−λ_floor)·base_j`
    (ĉ_θ=θ-예측 협력확률, λ̂_j=상대가 나를 향한 공감 추정,
    M_RECIPROCAL_DEFAULT=0.35 — "나에게 공감하는 이에게 공감한다").
  - `w_cd`: focal agent 고유 **context-dependency** 개인특질
    (W_CD_DEFAULT=0.5, 고정값 노출 — 추후 학습화 검토).
  - 상대 θ 미기록 시 λ_baseSelf 로 자연 축약.
- **효과 — tonic/anticipatory/acute 분해 복원**: λ_baseSelf=tonic,
  λ_baseContext(θ)=**anticipatory**(예측된 착취가 놀람 없이도 setpoint 하강),
  잔차 valence=acute. ALLD: λ_base→0.164, λ→0.13 (affect V=−0.02 침묵 유지) —
  **H1 방향 복원(dz=−2.31, 방향성립)**. ALLC: λ→0.58, cc 0.82.
- `CoreAffect.step (0)`이 `lambda_base(partner_identity)` 로 상대-조건부 setpoint
  수신.

## 3. SelfModel — social distance 순수 trial-기반화

**동기.** 구 거리 갱신이 상대 성향(p_noncoop)에 의존해 친숙도와 호오(好惡)가
오염되어 있었다.

- 거리는 **오직 trial 수 n 의 함수** — 지수 포화:
  `d(n) = D_min + (D_start−D_min)·exp(−n/τ_fam)`
  (D_INIT=1.6, D_MIN=0.5, TAU_FAMILIARITY=60). 상대 협력확률·성향은 거리에
  일절 관여하지 않는다. 검증: 협력자·착취자 거리궤적 **완전 동일**(1.582→0.555).
- '오래 본 착취자'는 가깝되(친숙) 신뢰하지 않는(λ_ctx 낮음) 상태로 표현 —
  성향 방어는 전적으로 λ_baseContext 가 담당(역할 분리).
- 구 상수 D_FINAL/D_FAR/ETA_DISTANCE_SLOW 제거.

## 4. SelfModel — p_noncoop EMA 제거 (이중 평활 해소)

**동기 질의에 대한 답**: EMA 의 제1원리적 근거는 없었다(비정상성 추적·안정화
관례). 입자필터가 이미 원리적 순차 베이지안 추론기이므로 그 출력(ĉ_θ)에 EMA 를
덧씌우는 것은 **이중 평활**.

- `update_identity`: `theta_coop`/`theta_lambda_j` 를 사후에서 **즉시 기록**
  (EMA 없음). `p_noncoop=1−ĉ_θ` 는 집계 호환 필드로 동기화. `contextual_share`
  만 진단·legacy prior 용 완만 EMA 유지.
- `SelfEntry` 신규 필드: `theta_coop`, `theta_lambda_j`.
- agent 는 `predict_coop`(현재 ctx)와 사후 `lambda_j` 를 매 라운드 주입.

## 5. 신 검증 배터리 V12 — w_cd 스윕

`--experiments V12`. w_cd∈{0, .25, .5, .75, 1} × 상대{ALLD, TFT, ALLC} × CRN 시드.

| 항목 | 가설 | 예비 결과(seeds=6, Holm) |
|------|------|------|
| V12.1 | 분화: w_cd↑→상대별 λ 분산↑ | 지지 dz=13.5, r=0.98 |
| V12.2 | anticipatory 방어: λ_ALLD(w=1)<(w=0) | 지지 dz=−3.16 |
| V12.3 | 공감확장: λ_ALLC(w=1)>(w=0) | 지지 dz=24.2 |
| V12.4 | 방어의 보수실익 vs ALLD | 지지 pay 0.84>0.69, dz=2.95 |
| V12.5 | w_cd=0 → λ_base 상대무관 축약 | 확인 SD=0.0000 |

그림 `v12_wcd_sweep.png`(3패널: 상대별 λ 표현형 / 분화 / 방어 실익) 생성 확인.
정식 보고는 로컬 적정 seeds(≥30) 재실행 권장.

## 6. 가설 재분류 (hypotheses.yaml `reclassification_v0_11_1`)

- **유효(재기준선)**: H1(방향 복원 확인)·H2·H3·H4.
- **재정의**: V10.1(단조성 기제 affect→λ_baseContext), V10.2(U 의미 하방반편차→
  epistemic + 역할 가산→게이팅). V10.4/5/6 유지.
- **폐기 유지**: H5(귀인 기제 부재, §9.7).

## 6.5 v0.11.1 검증 실행 결과 요약 (본 환경, 검정력 제한 seeds)

| 가설 | 결과 | 비고 |
|------|------|------|
| H1 착취 자기보호 | 방향성립, dz=−2.31 | v0.11.0 역전 → 복원 |
| H2 λ 서열 | 방향성립, dz=3.07 | |
| H3 β 판별 | 방향성립, dz=3.17 | |
| H4 CC 비열등 | 방향성립, dz=2.00 | |
| V10.1 valence 단조성 | 방향성립, dz=14.3 | 기제 재해석(λ_baseContext) |
| V10.2 방어(게이팅 재정의) | 방향성립, dz=−1.38 | 재정의 후 약화 — 예상대로 |
| V10.3 정적 expectile | 미지지(정직 null 유지) | |
| V10.4 E0 초기화 | 정확(1.30/1.60/2.00) | ⚠ 부속지표 '험한 환경 빠른 붕괴'
  **역전**(half .15=38.2 > .50=22.8): v0.11.1 서 붕괴가 초기 setpoint 가 아니라
  λ_baseContext 의 **θ 수렴 속도**에 지배되기 때문. 정직 기록 — 지표 재정의 필요 |
| V10.5 identity 기억 | 확인(gap=0) | |
| V10.6 가변-payoff expectile | 방향성립(스트림 dz=−3.3, 에이전트 0.56<1.25) | |
| V12.1–5 (신규) | 전 항목 지지(Holm) | §5 표 참조 |

p 값은 본 환경의 소 seeds(3–6) 검정력 제한으로 경계에 있는 항목이 있으며, 정식
확증은 로컬 적정 seeds(기본 240 또는 ≥30)에서 재실행할 것. 전 실험이 시뮬레이션→
통계→시각화까지 크래시 없이 완료됨을 확인(그림·caption 산출 포함).

## 7. 기타

- 버전 배너 v0.11.1. 코드 태그 `[v0.11.1]` 통일.
- 하위호환: 기본 SelfModel(캘리브레이션 미사용) λ_base=0.5459 불변. affect A/B
  3종 토글, identity_memory 토글, --allostasis-legacy/--legacy-efe 모두 보존.
- 미해결/알려진 한계: w_cd·m_recip·k_affect 는 노출된 고정 초기값(제1원리 유도
  아님 — w_cd 학습화·진화적 최적화는 추후 과제). 집단 실행의 identity 재조우
  기억은 agent 지속 구조 도입 시 발현(배선 완료). V10 배터리의 V10.1/V10.2 는
  재정의된 해석으로 읽어야 함(수치 자체는 실행 가능).
