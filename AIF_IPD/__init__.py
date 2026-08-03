"""
AIF_IPD — HalloReg
==================

**Adaptive Prosociality Through Hierarchical Allostatic Regulation in Social
Dynamics: A Simulation Study** (Choi, Albarracin, Pae, & Kim)

반복 죄수의 딜레마(IPD)에서 친사회성이 **내생적으로 조절되는** 능동추론
에이전트의 구현.

  (i)   조망수용    — 상대 모형 파라미터에 대한 입자필터 베이지안 추론
  (ii)  핵심정서    — 기대보상 분포로부터 구성되는 2차원(valence × arousal) 정서
  (iii) 공감        — 핵심정서와 추론된 맥락을 결합해 친사회성 λ 를 조절
  (iv)  자기모형    — 사회적 환경에 대한 믿음으로부터 할로스타틱 설정점을 확립
"""

__version__ = "1.0.0"
