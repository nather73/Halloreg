"""
AIF_IPD.experiments
===================

가설별 실험 모듈.

  common            : 실행 설정, 결과 등록기, 시각화 유틸
  arch_validation   : ARCH — 아키텍처 구현 정합성 검증 (가설 이전 필수)
  h1_intent         : H1  — 전략적 의도 추론
  h1a_tracking      : H1A — 변동 의도 추적 및 λ 복원
  h2_protection     : H2/H2A — 착취자 방어와 잡음-의도 판별
  h3_h4_population  : H3/H3A/H4/H4A — 정상/비정상 혼합 집단
  h5_evolution      : H5  — RE/ORE 진화 동역학에서의 생존
  h6_recovery       : H6  — 파라미터 복원과 자기-사영
"""

from .common import Config, Registry

__all__ = ["Config", "Registry"]
