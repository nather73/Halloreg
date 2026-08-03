"""
core.logging_utils
==================

로거 설정과 OS별 한글 폰트 자동 선택 유틸.

시각화 결과의 축·범례에 한글이 들어가므로, matplotlib 이 기본 폰트로 두부(□)를
찍지 않도록 실행 환경에 맞는 한글 폰트를 rcParams 에 주입한다.
"""

from __future__ import annotations

import logging
import platform

_CONFIGURED = False


def get_logger(name: str = "HalloReg", level: int = logging.INFO) -> logging.Logger:
    """
    콘솔 핸들러가 붙은 로거를 반환한다.

    'HalloReg' 루트 로거에만 핸들러를 한 번 붙이고 propagate 를 끄므로,
    하위 모듈이 몇 번 호출해도 핸들러가 중복되지 않는다(로그 중복 출력 방지).
    """
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        root = logging.getLogger("HalloReg")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger


def set_korean_font(plt, font_manager) -> None:
    """
    OS별 한글 폰트 자동 선택.

    Windows : 맑은 고딕(Malgun Gothic)
    macOS   : AppleGothic
    Linux   : NanumGothic → Noto Sans CJK KR → (없으면) 파일 경로 직접 등록

    어느 것도 없으면 DejaVu Sans 로 남고 한글이 깨질 수 있다(치명적이지 않음).
    `axes.unicode_minus=False` 는 한글 폰트에 U+2212(−) 글리프가 없어 음수 부호가
    깨지는 문제를 막는다.
    """
    candidates = {
        "Windows": ["Malgun Gothic"],
        "Darwin": ["AppleGothic"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP",
                  "DejaVu Sans"],
    }.get(platform.system(), ["DejaVu Sans"])

    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break
    if chosen is None:
        # 설치 목록에 이름이 안 잡히는 경우(컨테이너 등) 파일 경로로 직접 등록 시도
        for fp in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            try:
                font_manager.fontManager.addfont(fp)
                chosen = font_manager.FontProperties(fname=fp).get_name()
                break
            except Exception:
                continue

    # 폰트 **폴백 목록**을 지정한다. 한글 폰트에는 결합 곡절부호(θ̂ 의 ̂)나
    # 일부 기호 글리프가 없는 경우가 많은데, 뒤에 DejaVu Sans 를 두면
    # matplotlib 이 없는 글리프만 대체 폰트에서 가져온다(두부 □ 방지).
    fallback = ["DejaVu Sans"]
    plt.rcParams["font.family"] = ([chosen] + fallback) if chosen else fallback
    # 등폭(monospace) 계열에도 같은 폴백을 건다. 아키텍처 모식도처럼 등폭으로
    # 정렬하는 텍스트에 한글이 섞이면, 등폭 폰트에 한글 글리프가 없어 깨진다.
    if chosen:
        plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", chosen]
    plt.rcParams["axes.unicode_minus"] = False
