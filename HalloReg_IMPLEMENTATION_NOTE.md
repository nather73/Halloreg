# HalloReg v0.11.2 — 구현 노트 (implementation note)

> **v0.11.2: 분포적 TD 오차 · Z_i(s,a) 8분포 · λ 합성 재편.** 다섯 변경:
> (1) λ_ctx = ĉ_θ **단독**(m·λ̂_j 이중계상 제거), (2) SelfModel 이 (id, dist,
> p^nc, **Z_i(s,a)**)를 기록 — model-based/free 통합 명부, (3) §3.1 스칼라 계약
> 폐기 → **분포적 TD 오차 δ_i = r_obs − z_i** + Z(s,a) 4상태×2행동 8분포(결정
> 시점 상태 s_{t−2} 조건, on-policy 갱신) + **좌표계 [0.0, 0.8] 통일**,
> (4) **λ_self = λ_baseSelf + λ_affect → λ = (1−w_cd)λ_self + w_cd·λ_ctx**
> (affect 를 self 축에 귀속), (5) λ_affect = k·V·(1−U) 에서
> **U = √(σ²_epi + w_Z·Var^α_Z[δ])/σ_ref** (전분산 법칙으로 epistemic+aleatoric
> 재결합, α 가중이 하방 비대칭 계승).
>
> ⚠ **구 골든 무효화**: 좌표계 통일로 기본 λ_baseSelf 0.5459→0.5945, legacy A/B
> 도 비트동일 아님(0.144/0.793). 사양의 불가피한 귀결 — §9.8 재기준선 대상.
>
> 전문: `AIF_IPD/docs/CHANGELOG_v0.11.2.md`.

---

## [이하 v0.11.1 이력]

# HalloReg v0.11.1 — 구현 노트 (implementation note)

> **v0.11.1: λ_base 의 self/context 중재 + 순수 trial 거리 + EMA 제거.**
>
> **(1) setpoint 분해.** `λ_base = (1−w_cd)·λ_baseSelf + w_cd·λ_baseContext(θ_j)`.
> λ_baseSelf = **발달 엔트리만**의 거리가중 집계(상대 무관 자기 특질), λ_baseContext =
> 현재 상대 θ 맞춤 setpoint로 **거리와 무관**하게 θ 를 직접 반영. w_cd = focal agent
> 고유 context-dependency(기본 0.5, 노출). λ_baseContext 는 **상호적 사상**:
> `base_j = (1−m)·ĉ_θ + m·λ̂_j`, `λ_ctx = floor + (ceil−floor)·base_j` (m=0.35) —
> "나에게 공감하는 이에게 공감한다".
>
> **(2) social distance = 오직 trial 수.** `d(n) = D_min + (D_start−D_min)·e^{−n/τ_fam}`
> (D 1.6→0.5, τ=60). 상대 성향은 거리에 **일절 관여하지 않는다** — 친숙도와 호오의
> 분리. '오래 본 착취자'는 가깝되(친숙) 신뢰하지 않는(λ_ctx 낮음) 상태로 표현된다.
> 검증: 협력자·착취자 거리궤적 **완전 동일**(1.582→0.555).
>
> **(3) p_noncoop EMA 제거.** 입자필터가 이미 원리적 순차 베이지안 추론기이므로 그
> 출력(ĉ_θ)에 EMA 를 덧씌우는 것은 **이중 평활**이며 근거가 없다 → θ 즉시 반영.
>
> **역설 해소 확인.** v0.11.0 의 자기보호 역설(θ가 착취자를 잘 예측할수록 잔차 affect
> 가 작아져 방어가 약해짐)이 해소됐다. ALLD 상대: λ_base→0.164, λ→0.13(v0.11 은
> 0.19에서 정체)이며 이는 **affect 가 아니라 λ_baseContext(ĉ_θ=0.02)** 가 만든
> anticipatory 방어다(V=−0.02로 affect 는 여전히 침묵 — 이론적으로 정확). H1 방향
> 복원(dz=−2.31, 방향성립). ALLC: λ→0.58, cc 0.82.
> 하위호환: 기본 SelfModel λ_base=0.5459 불변, affect A/B 토글 보존.

---

## [이하 v0.11.x 이력]

# HalloReg v0.11.0 — 구현 노트 (implementation note)

> **v0.11.0: θ-설명 잔차 affect (맥락-의존 정서).** CoreAffect 의 valence/uncertainty
> 를 "실현보수 분포 그 자체"가 아니라 "**입자필터 θ 로 설명되고 남은 잔차**"로 재정의.
> RPE 를 θ-조건부 기대보수(입자 회귀)로 잔차화 → 잔여 mean→valence, θ 입자간
> (epistemic) std→uncertainty. Δλ = k·V·(1−U) 게이팅(k_U 제거). λ_base 는 상대
> identity 의 θ 에 의존(θ-예측 비협력으로 p_noncoop 갱신, social distance 극완만
> θ-목표 접근). affect_mode="theta_residual" 기본; valence_uncertainty(v0.10)·
> legacy_attrib 는 A/B 보존(비트동일 검증).
>
> **정직한 발견 — allostatic 자기보호 역설의 재출현.** θ 가 착취자를 정확히 예측하면
> RPE 가 대부분 설명돼 잔차 affect 가 작아지고(V≈−0.02, "예상된 배신은 놀랍지 않다"),
> 따라서 affect-구동 λ 보호가 약해진다. ALLD 상대 λ 가 0.19 까지만 하강(v0.10 은
> 0.00)하고 H1(착취 자기보호, T=30) 방향이 뒤집힌다(dz +4.0). 이는 예측처리 순수성
> (예측된 사건=무affect)의 이론적 귀결이자, 초기 HalloReg 가 tonic+acute+anticipatory
> 분해로 풀었던 역설의 재출현이다. 단기 보호를 회복하려면 λ_base(θ)의 **anticipatory
> 성분**(예측된 착취가 놀람 없이도 setpoint 를 낮추도록)이 필요하나, 이는 "social
> distance 극완만" 사양과 긴장한다 — 설계 결정으로 남긴다(아래 §미해결).
>
> 학술 novelty·함의·한계: `docs/NOVELTY_AND_LIMITATIONS.md`.

---

## [이하 v0.10.x 이력]

# HalloReg v0.10.0 — 구현 노트 (implementation note)

> v0.10.0: valence+uncertainty affect 정식 승격 + 가설 재분류 + V10 검증 배터리.
> k_V=0.58, k_U=0.24 를 여러 상대군 실측 (V,U)의 목표-λ 최소자승 적합으로 캘리브레이션.
> V10 결과: 단조성 지지(dz≈17, rank_r≈0.94) · 하방불확실 방어 지지(dz≈−3.7) ·
> SelfModel 초기화 확인 · identity 기억 확인 · **expectile 무편향은 정적 payoff서
> 미지지(정직 null)이나, 가변 payoff(V10.6)서 지지** — 스트림·전체 AdaptiveAgent
> 모두 expectile 추적오차가 quantile 의 ~절반. 경계조건(정상성 의존)을 나란히 보고.
> 가설 재분류·novelty 문서는 아래·docs/ 참조.
> 학술 novelty·함의·한계: `docs/NOVELTY_AND_LIMITATIONS.md`.
> 가설 재분류: `AIF_IPD/experiments/hypotheses.yaml`의 `reclassification_v10`.

---

## [이하 v0.9.x 구현 이력]

# HalloReg v0.9.0 — 구현 노트 (implementation note)

`HalloReg_v090_architecture_spec.md` 를 v0.3.0 코드베이스에 구현한 결과 요약.
검토는 `git diff v0.3.0` 로 전체 변경을 확인할 수 있다.

## 변경 파일

| 파일 | 상태 | 핵심 변경 |
|------|------|-----------|
| `AIF_IPD/core/allostasis_v9.py` | **신규** | 2계층 조절: `SelfModel`(느린 λ_base setpoint) + `CoreAffect`(빠른 RPE-귀인 λ 조절) + `QuantileValue`(분포적 RL) |
| `AIF_IPD/ipd/agent.py` | 수정 | `AdaptiveAgent` 기본을 2계층으로. `--allostasis-legacy` 로 구 적분기 A/B 복원. 신 로그키(lam_base, delta_lam, rpe, deficit, ctx_credence). self-projection 필터(θ̂_self) 생성 |
| `AIF_IPD/ipd/tom/inversion.py` | 수정 | 전 6축 histogram 정보이득(§5): `expected_infogain_allaxis`, `observed_infogain_allaxis`. H0 메모이제이션(순수 속도 개선, 수치 동일) |
| `AIF_IPD/ipd/tom/tom_core.py` | 수정 | `RecursiveSocialEFE` 전면 개정 — 대칭 pragmatic, 전축 IG_self/IG_other, depth-2 자연 임베딩, w_epi 제거, 단일 r_pred 계약(§4–7) |
| `AIF_IPD/ipd/tom/sophisticated_planner.py` | 수정 | rollout 단계마다 전축 IG 반영(§5.3) |
| `AIF_IPD/scripts/run_ipd_experiment.py` | 수정 | `MODEL_REVISION` 토글 확장, `--allostasis-legacy`/`--legacy-efe` CLI, 신 검증 실험 **V9**(§11.1–4) + 그림. H5 §9.7 조작화-무효 정직 보고 가드 |

## 명세 대응 (PART I · II)

- **§1–3 조절 재정초**: `CoreAllostaticBeliefState`+`LambdaRegulator`(leaky 적분기)를
  `SelfModel`+`CoreAffect`로 대체. 사건 = **부적 RPE**(CD 라벨 아님, §3.1). 분포적 RL
  분위수(M=21) + RPE 에 대한 베이지안 귀인 q(z)∈{contextual, dispositional}(§3.3).
  θ→원인 매핑은 **부호교정판**(§3.4). λ = λ_base(SelfModel) + Δλ_drive, 적분기 없음(§3.5).
  구 클래스는 `--allostasis-legacy` A/B 토글로만 보존(§9.1).
- **§4–7 G_social 엄밀화**: pragmatic 키만(상태 엔트로피 제거), self/other 대칭.
  전 6축(α,ρ,ω,η,β,λ_j) histogram 정보이득, 지평 2 기본. self-projection(θ̂_self)로
  IG_other 를 실제 정보이득으로 계산(수동조율 [0.5r,0.2r] 폐기, §6). w_epi 가중 삭제,
  최종형 G(π)=(1/H)Σ[(1−λ)(prag_self−IG_self)+λ(prag_other−IG_other)](§7).

## 신 검증 실험 V9 (§11) — 검증 결과

주 프로세스 직접 다이애드로 실행(계층 내부 상태 관측 필요). `--experiments V9`.

| 항목 | 가설 | 결과(seeds≈14) |
|------|------|------|
| V9.1 | RPE 스케일 민감성: T5→T10 시 \|RPE\| 증가 (legacy 는 rpe≡0 무반응 대조) | \|RPE\| 1.20→1.53, p=0.001 ✓ |
| V9.2 | ALLD 지속 시 λ_base 하향 drift + boundary 엔트리 발달고정 | drift 유의(p<0.001), 고정율 100% ✓ (drift 크기는 η_slow=0.02 로 소폭 — 설계상 느림) |
| V9.3 | 동일 부적 RPE 라도 contextual vs dispositional 귀인서 λ 분기 | λ ctx 0.68 > disp 0.21, p<0.001 ✓ |
| V9.4 | h=2 에서 IG_self 비상쇄(S2 해결, epistemic 부활) | 비상쇄율 0.80 ✓ |

## 정직 보고 (강요된 판정 없음)

- **H5 조작화 무효(§9.7)**: 2계층 allostasis 에서 λ 조절은 RPE 귀인이 전담하므로
  구 `dd_charges`·`disposition_mode` 로 가른 즉각/정교 유형이 동일 CoreAffect 궤적을
  갖는다. 착취자 상대 방어량이 양형 포화 → 짝지은 차 0 → dz 미정의. 이를 **'조작화
  무효 → 미지지'** 로 정직 보고(가드 추가). H5 재정의(§11.6)는 별도 과제로 남김.
- **골든 스냅샷 무효(§9.8)**: 조절 재정초로 `tests/test_golden.py` 는 구 스냅샷과
  불일치 예상 — 재기준선 필요.

## 실행

```bash
# 스모크 (빠름)
python3 -B AIF_IPD/scripts/run_ipd_experiment.py --quick --seeds 2 --experiments H1 --jobs 2
# 신 검증
python3 -B AIF_IPD/scripts/run_ipd_experiment.py --experiments V9 --seeds 30
# 전체 (기본 rounds=[60,240], seeds=240 — 16코어 권장: --jobs 16)
python3 -B AIF_IPD/scripts/run_ipd_experiment.py --jobs 16
# 구 경로 A/B 복원
python3 -B AIF_IPD/scripts/run_ipd_experiment.py --allostasis-legacy --legacy-efe --experiments H1
# [§10 개정 옵션] λ_base=0.24 (Albarracin 부호역전 경계) 중립 초기화
python3 -B AIF_IPD/scripts/run_ipd_experiment.py --target-lambda-base 0.24 --experiments H1
```

## λ_base 캘리브레이션 — Self-앵커 항등 재정식화 (§10 개정)

**동기.** 구 사상 `λ_base = floor + (ceil−floor)·frac` 의 floor/ceil 은 근거 없는
자유 상수였다. 임의 target 을 맞추려면 trait 를 비관적으로 밀거나 ceil 을 손으로
재중심화해야 했다(임의성). 재정식화는 이 자유도를 제거한다: **floor=0, ceil=1 로
항등 사상**하고, λ_base 의 경계·진폭을 **거리 구조 자체가 내생적으로 결정**하게 한다.

**Self-앵커 원리.** 항등 사상에서
```
λ_base = [coop_self + Σ_oth w·coop_oth] / [1 + W_oth],   W_oth = Σ_oth exp(−d/τ)
```
Self(d=0)의 가중치 1 이 Other 총합 W_oth 를 지배한다 — 이것이 **자기 EFE 가 타인
EFE 보다 기본적으로 우선**임을 거리 구조로 인코딩한 것이다. Self 의 협력성만
미지수로 두면 **폐형해**:
```
coop_self = target·(1 + W_oth) − Σ_oth w·coop_oth
```
Self 가 setpoint 를 앵커하고, **Other 거리가 흔들림 대역폭(W_oth)을 정한다**. 중간
앵커 기본(`MID_ANCHOR_OTHERS`: d=0.9/1.3/1.9)은 현행 DEV(0.5/0.8/1.5)보다 Other 를
조금 더 밀어 Self 지배(≈79%)를 강화하되, 최근접 Other(d=0.9)를 boundary 내에 유지해
§3.6 조기보정·§6.2 boundary-공감 경로를 보존한다. 도달 불가 target(coop_self∉[0,1])은
조용한 근사 대신 ValueError(Other 거리 조정 안내).

**사용:**
```python
SelfModel(target_lambda_base=0.24)                              # 중간앵커 기본
SelfModel(target_lambda_base=0.24, anchor_others=[...])         # Other 구조 지정
AdaptiveAgent(selfmodel_kwargs=dict(target_lambda_base=0.24))
```
```bash
python3 -B AIF_IPD/scripts/run_ipd_experiment.py --target-lambda-base 0.24 --experiments H1
```

**검증(target=0.24):** λ_base=0.240000 정확. Self coop=0.131(지배 79%), boundary Other
1개 유지. 행동: ALLD→0.00 붕괴 / TFT·ALLC→0.44 상승 / random→0.39(맥락 보존).

**하위호환.** `target_lambda_base` 미지정 시 **기본 경로는 완전 불변** — affine 사상·
DEV prior 그대로, default λ_base=0.5459. 기존 전 실험(H1–V9)·기준선 영향 없음(검증됨).

**정직한 한계 — drift 대역폭의 실현성.** → **[해소됨: §3.6 identity 재정의]** 아래 참조.

## §3.6 재정의 — identity 관계형성 (v0.9.2)

**교정된 의미론.** identity 는 유사성 추론이 아니라 **관측 가능한 표지**다. 종전
구현(trait-근접 매칭)은 검토 결과 판별력 0(전 상대가 동일 엔트리에 매칭)·기본
구성 실효 0(ablation |Δλ|=0)·방향 오류 가능으로 **§3.6 미구현에 가까웠다**.
재정의: (i) 모든 상호작용 상대는 identity 를 노출하고 환경(run_dyad)이
`begin_partner(ident)` 로 통지한다. (ii) 새 identity 는 **boundary 밖**(d=1.6)에
등록되며, 초기 trait 는 현재 λ_base 에 중립(p_noncoop=1−λ_base — 등록이 setpoint
를 흔들지 않음). (iii) 반복 trial 로 거리가 좁혀져 **정확히 W_admit(40) 라운드에
boundary 를 통과**(편입)하고 d=0.8 에서 정지한다. (iv) 그 상대에 대해 학습한
**잠재 z**(p_noncoop←배신 EMA, contextual_share←q_ctx EMA, η=0.05)가 엔트리에
저장된다. (v) 조기 보정 = **identity 동일성**: 편입된 identity 와의 상호작용에서
prior 가 저장된 z 로 편향되고, **재조우 시 q(z) 를 저장값으로 즉시 초기화**
(`reset_for_partner` — prior 지수 희석과 무관한 실효 보정). 발달 엔트리는 절대
불변(발달 고정, V9.2 정합 검증됨).

**setpoint drift 의 접지.** λ_base 이동이 이제 '특정 타자의 boundary 편입'으로
접지된다 — 검증(0.24 앵커, T=120): vs ALLC λ_base 0.240→0.343(+0.103 소폭 상향,
λ 말기 0.542 — 종전 0.44 천장이 setpoint 상승으로 자연 해제), vs ALLD
0.240→0.209(하향, λ→0). 종전의 '무력한 drift(Δ≈6e-5)' 한계가 해소되었다.

**토글·호환.** `identity_memory=True` 기본. `False` 면 종전 경로(trait-근접 +
far-drift) 완전 재현(비트 동일 검증: ALLD/TFT/ALLC λ_final 0.097/0.739/0.746).
기본 ON 은 §9.8 재기준선 대상. 집단 실행은 현행상 다이애드마다 fresh agent 라
세션 내 재조우 기억이 미발현 — agent 지속 구조 도입 시 자동 발현(설계 노트).

## §3.2·3.5 재정초 — expectile affect (valence+uncertainty) (v0.9.3)

**동기.** (i) 분포 코드를 quantile 로 둘 근거로 든 "support-free"는 expectile 도
공유해 변별력이 없었고, 프로젝트 지식(Dabney 2020 dopamine)은 DAN tuning 이
heaviside(quantile)보다 bilinear(expectile)에 부합함을 보여 **expectile** 을 가리킨다.
(ii) 구 λ 조절은 분포에서 mean 하나만 쓰고(deficit) uncertainty 를 β 사후편차(상대
성향 추정 불확실성 — 분포 자체가 아님)에서 빌려온 임의적 게이팅이었다.

**재정초 3축.**
1. **Expectile code(기본).** `z += α·|η−1{u<0}|·u` (L2 비대칭). η=0.5 expectile 이
   정확히 평균이라 valence 가 편향 없이 나온다 — mean 은 **중앙 격자점(η=0.5)** 에서
   읽는다(격자 산술평균은 치우친 분포서 편향; 검증: expectile mean 오차<0.04 vs 참값).
2. **SelfModel 함의 초기화(지시 2).** 분포 중심 E0 = P + λ_base·(R−P) (협력적 환경→
   높은 기대), 퍼짐 σ0 = 엔트리 협력성 거리가중 표준편차. R=3 임의 상수 폐기.
   deficit 기준점도 R 고정→setpoint 함의 기대로 이동(allostasis 정합).
3. **valence+uncertainty λ 사상(지시 3).** β-deficit·q_z 귀인 게이팅 **완전 폐기**:
   ```
   V = (E[r] − E0)/(R−P)              # valence — setpoint 대비 부호
   U = downside_semidev / σ_ref       # 하방 반편차 √E[(E0−r)_+²]
   Δλ = k_V·V − k_U·U ;  λ = clip(λ_base+Δλ, 0, λ_max)
   ```
   방어 비대칭(구 g_disp>g_ctx)이 임의 상수가 아니라 **분포 하방구조**에서 내생.
   affect = valence+arousal 의 저차원 표상(Theriault 2021·Hesp 2020 정합).

**identity 재배선(지시 3).** 재조우 저장·복원 대상이 q_z→**기대보상 분포(E,σ)**.
편입 identity 를 다시 만나면 value 를 학습된 (E,σ)로 재초기화('아는 상대는 학습된
기대분포로'). 지시 2 초기화와 동일 통로 — 검증: ALLC 편입 후 (E,σ)=(2.94,0.59)
저장, 재조우 시 분포중심 1.48→2.94 복원, 신규 identity 는 무보정.

**검증(0.24 앵커).** valence/uncertainty 서명이 해석가능·정합:
ALLD V=−0.22 U=0.51→λ→0(cc0.00) / TFT V=+0.58 U=0.27→λ0.41(cc0.60) /
ALLC V=+0.81 U=0.14→λ0.62(cc0.72) / random V=+0.37 U=0.36→λ0.20. 착취자=음
valence+고 하방불확실→붕괴, 협력자=양 valence+저 불확실→확장.

**토글·호환.** `affect_mode="valence_uncertainty"`(기본) / `"legacy_attrib"`(구 귀인).
`code="expectile"`(기본) / `"quantile"`. legacy_attrib+quantile 은 R-init 복원으로
**구 골든 비트동일 재현**(ALLD/TFT/ALLC 0.097/0.739/0.746). 기본 ON 은 §9.8 재기준선.

## 성능 주의

전축 histogram IG(6축) × 지평 2 × self-projection 필터로 **스텝당 비용이 증가**했다
(명세 완전반영, 런타임 단축용 단순화 없음). H8/GS/H11/H12/VP/ORE 등 대규모 실험은
16코어(`--jobs 16`) 야간 실행 권장. 본 환경(소코어)에서 전 실험이 **크래시 없이 실행**됨을
확인(대규모는 느리되 정상). H0 메모이제이션으로 IG 핫패스 일부를 무손실 가속.
