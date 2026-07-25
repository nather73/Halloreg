# HalloReg v0.11.2 — 변경 기록 (CHANGELOG)

v0.11.1 이후 반영된 다섯 가지 구조 변경. 코드 태그 `[v0.11.2]` 가 이에 대응한다.

---

## 1. λ_ctx = ĉ_θ 단독 (상호성 혼합 폐기)

**변경 전**: `base_j = (1−m)·ĉ_θ + m·λ̂_j`, m=0.35.
**변경 후**: `λ_ctx = λ_floor + (λ_ceil−λ_floor)·ĉ_θ`.

**근거**: ĉ_θ 의 정책 로짓 `σ(β(α + ρf + ωg + η·fg + s_empathy))` 에 이미
empathy shift(λ 관련 항)가 포함되어 있어, λ̂_j 를 다시 더하면 λ 정보가 **이중
계상**된다. `theta_lambda_j` 는 기록·진단용으로만 유지(λ 경로 미진입).
`M_RECIPROCAL_DEFAULT` 는 0.0 으로 DEPRECATED(하위호환 인자만 존치).

**검증**: ĉ=0.05 고정 하에 λ̂_j 를 0.01↔0.95 로 바꿔도 λ_ctx=0.0400 불변.

## 2. SelfModel 이 Z_i(s,a) 까지 기록 — model-based/free 통합 명부

`SelfEntry` 에 `z_sa: Optional[np.ndarray]` 신설. 이제 boundary 내 타인에 대해
**(id, dist, p^nc, Z_i(s,a))** 를 함께 기록한다:

- `p^nc` ← `OpponentInversion`(model-based, θ 추론)
- `Z_i(s,a)` ← `CoreAffect`(model-free, 분포적 가치) — (4,2,M) 스냅샷 전체

`CoreAffect._store_reward_belief` 가 매 라운드 `value.snapshot()` 을 기록하고,
`reset_for_partner` 는 재조우 시 `value.restore(z_sa)` 로 **8분포를 그대로 복원**
한다(스냅샷 부재 시 (E,σ) 축약 복원으로 폴백).

**검증**(V10.5 재작성): 재조우 복원 격자 최대오차 0.00e+00, 신규 identity 무변.

## 3. 분포적 TD 오차 + Z(s,a) 8분포 + 좌표계 통일

### 3.1 §3.1 스칼라 계약 폐기 → 분포적 TD 오차 δ_i
`QuantileValue.td_errors/td_weights/td_moments` 신설:

```
δ_i = r_obs − z_i,   α_i = |η_i − 1{δ_i<0}|
mean^α = Σα_i δ_i / Σα_i,   Var^α_Z[δ] = Σα_i (δ_i − mean^α)² / Σα_i
```

α 가중이 expectile 하방 비대칭을 물려받아, 하방으로 벌어진 분포일수록 Var 가
커진다(구 `downside_semidev` 정신의 계승). 구 스칼라 `r_pred` 기반 rpe 는
로깅으로만 존치 — 표상 불일치(분포 Z ↔ 스칼라 RPE) 해소.

### 3.2 Z(s,a) — 4상태 × 2행동 = 8분포
`SAValueBank` 신설. **근거**: `OpponentInversion` 입력이 이미 (직전 공동상태 f,
상대 이력 g)를 포함하므로, model-free 측도 동일 좌표계로 조건부화해야 정합하다.
무조건부 Z 의 두 문제 — (i) θ-예측은 a-조건부인데 비교 기준은 a-무관인 비대칭,
(ii) CC/DC 등 이질 보수 혼재로 분산이 인위적으로 부풀려짐 — 이 동시 해소된다.

**시제 규약(중요)**: Z(s,a) 의 s 는 **결정 시점 상태** s_{t−2}=joint(a_{t−2},
opp_{t−2}) 이며, 결과 상태 s_{t−1} 이 아니다(보수 자신이 조건이 되면 순환).
에이전트가 `s_decision = 2·my_actions[-2] + opp_actions[-1]` 로 계산해 주입하고,
이력 부족(초기 1~2R)이면 None→CC(0) 초기상태 관례를 따른다.
갱신은 **정준적 on-policy** — 관측된 (s,a) 셀 하나만.

**검증**: 8셀이 실제로 분화(TFT 상대 예: (CC,C)=2.26, (CC,D)=2.78, (DC,C)=1.29),
미갱신 셀 독립 유지, 스냅샷 shape (4,2,21).

### 3.3 좌표계 이질성 해소
구 버전은 λ_baseSelf 가 affine[0.1,0.7], target 캘리브레이션은 항등[0,1],
λ_ctx 는 또 [0.1,0.7] 로 **세 좌표계가 혼재**했다. 전 성분을 단일
**[floor, ceil] = [0.0, 0.8]** (λ_max 와 동일)로 통일. `_build_anchored` 는
항등 강제를 폐기하고 통일 좌표계 위에서 폐형해를 푼다(frac* = (target−floor)/span).

**⚠ 하위호환 영향(정직 보고)**: 좌표계 통일은 모든 경로의 λ_base 를 이동시킨다.
기본 DEV prior λ_baseSelf 가 **0.5459 → 0.5945**, legacy_attrib A/B 의 구 골든
(ALLD/ALLC 0.097/0.746)도 **0.144/0.793 으로 이동**해 더 이상 비트동일이 아니다.
이는 사양(좌표계 통일)의 불가피한 귀결이며 버그가 아니다 — 구 골든은 무효화된다.
target 캘리브레이션 경로는 통일 좌표계에서도 정확(target=0.24 → 0.2400).

## 4. λ 합성 재편 — affect 를 self 축에 귀속

**변경 전**: `λ_base = (1−w_cd)·λ_baseSelf + w_cd·λ_ctx` 를 만든 뒤 affect 를 더함.
**변경 후**(`CoreAffect.step` 내부에서 합성):

```
λ_self = λ_baseSelf + λ_affect
λ      = (1 − w_cd)·λ_self + w_cd·λ_ctx(θ_j)        [θ 미상 시 λ_self 로 축약]
```

정서는 '나의 반응'이지 '상대의 맥락'이 아니므로 self 축에 더해지고, 그 뒤에
맥락 의존성 w_cd 로 상대맞춤 setpoint 와 중재된다. 이로써 SelfModel(λ_baseSelf)과
CoreAffect(λ_affect)가 λ_self 안에서 결합해 "따로 노는" 구조가 해소된다.
`SelfModel.lambda_base()` 는 affect=0 인 경우의 중재값(분포 초기화·진단용)으로 존치.

반환 dict 에 `lam_self`, `lam_self_base`, `lam_ctx`, `lambda_affect` 신설.

## 5. λ_affect 의 새 계산식 (학술적 정식화)

```
V = (r_obs − E_θ[r]) / (R−P)                                  ← 잔차 mean
U = √(σ²_epi + w_Z·Var^α_Z[δ]) / σ_ref     (clip 0..1)        ← 잔차 std
λ_affect = k_affect · V · (1 − U)
```

- **V(valence)**: θ 회귀 제거 후 잔차의 mean. δ_i 와 θ-예측 δ̂_i 가 −z_i 를
  공유해 상쇄되므로 잔차 mean 은 r_obs − E_θ[r] 로 귀결된다(격자 독립).
- **U(uncertainty)**: **전분산 법칙**으로 두 원천 재결합 — σ_epi(입자간,
  epistemic '상대를 모름') + Var^α_Z[δ](그 (s,a) 기대분포의 α-가중 폭,
  aleatoric '결과가 흔들림'). w_Z=1(자연값, 노출 파라미터).
- **게이팅**: V 부호가 방향, U 가 확신도 게이트(모르면 |λ_affect|↓).

**이론적 정당성**: (i) RPE 가 분포 RL 과 표상 일관(δ_i), (ii) appraisal(V)/
arousal(U) 이중 표상에 충실, (iii) v0.11.1 의 epistemic-only U 가 잃었던
aleatoric 감수성을 Var^α_Z 로 복원(하방 비대칭 자동 반영), (iv) k_U 없는 단일
스케일 유지.

---

## 검증 결과 (본 환경, 소 seeds — 검정력 제한)

| 항목 | 결과 |
|------|------|
| 좌표계 통일 | floor/ceil=[0.0,0.8], target=0.24 → 0.2400 정확 |
| λ_ctx ĉ_θ 단독 | λ̂_j 무영향 확인(0.0400 불변) |
| Z(s,a) 8분포 | 셀 분화 확인, on-policy 갱신, 스냅샷 (4,2,21) |
| Z(s,a) identity 기억 | 재조우 복원오차 0.00e+00, 신규 무변 (V10.5) |
| H1 착취 자기보호 | 방향성립 dz=−2.31 |
| V12.1 분화 | 지지 dz=11.0, r=0.992 |
| V12.2 anticipatory 방어 | 지지 dz=−2.99 |
| V12.3 공감확장 | 지지 dz=68.7 |
| V12.4 방어 보수실익 | pay 0.897 > 0.669 (dz=2.90) |
| V12.5 w_cd=0 축약 | SD=0.0000 |
| V10.1/2/6 | 방향성립(dz=8.41 / −1.66 / −3.31) |
| V10.3 정적 expectile | 미지지 유지(정직 null) |

전 실험이 시뮬레이션→통계→시각화(PNG+PDF+caption)까지 크래시 없이 완료.

## 알려진 한계·후속 과제

- **w_Z=1 의 부작용(관찰)**: U 에 Var^α_Z 가 더해지며 U 가 전반적으로 상승했고
  (예: TFT U 0.11→0.55), 그만큼 λ_affect 게이트가 닫혀 acute 성분이 약화됐다.
  λ 분화는 λ_ctx 가 담당하므로 행동 표현형(V12)은 유지되나, affect 기여를
  키우려면 w_Z 하향(0.3~0.5) 또는 u_gate_ref 상향이 필요하다 — 사양대로 우선
  w_Z=1 로 구현했으며, 조정은 후속 결정 사항.
- **구 골든 무효화**: §3.3 참조. legacy A/B 는 행동적으로 유사하나 비트동일 아님.
- **V10.4 부속지표 역전 유지**: '험한 환경 빠른 붕괴'는 v0.11.1 부터 역전 상태
  (붕괴가 초기 setpoint 가 아니라 θ 수렴 속도에 지배). 지표 재정의 대기.
- w_cd·k_affect·w_Z 는 여전히 노출된 고정 파라미터(제1원리 유도 아님).
