"""
HalloReg — Hierarchical Allostatic Regulation of empathy in Active Inference agents.

Albarracin et al. (2026) 의 고정 공감(λ) 능동추론 IPD 에이전트를,
core allostatic belief state 에 기반한 **내생적 λ 위계 조절** 에이전트로 확장한다.

[v0.7.0]  §1 반사실(counterfactual) 인과 귀인 — 기질 귀인에서 ρ 축출
          §2 disposition 손튜닝 계수(1.5/1.0/0.7) 제거
          §4 attr_gate·통제권 귀인과 반사실 귀인의 통합 정리
[v0.7.1]  §3 rollout ρ 전파 — ρ 의 도구적 가치를 λ 가 아닌 EFE/rollout 으로
[v0.8.0]  §7 우도 기억-1 완전화 (g·fg 항) — WSLS 표현 불가 해소
          §7.3 fg_centered 복원 배터리 (필수 동반)
[v0.8.1]  VP/ORE 실행 예산 명시화 (은닉 캡 T≤120·seeds≤24 폐지, 기본 240/60)
          본 실험 로그에 입자 사후 SD 추가 (실험 내 식별 상태 사후 진단)

[v0.8.2]  **기본값 반전** — 러너 기본 = 반사실·fg(입자 자동 600)·rollout(horizon 2).
          v0.6.6 재현은 opt-in: --disposition-mode legacy --likelihood-basis f
          --no-rollout-reciprocity --planning-horizon 1
          (에이전트 클래스 기본값은 legacy/f 유지 — 골든·라이브러리 호환)
"""

__version__ = "0.8.2"
