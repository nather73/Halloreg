"""
core.logging_utils
==================

시뮬레이션 로깅 설정과 OS별 한글 폰트 자동 선택 유틸.
"""

from __future__ import annotations

import logging
import platform

_CONFIGURED = False


def get_logger(name: str = "HalloReg", level: int = logging.INFO) -> logging.Logger:
    """콘솔 핸들러가 붙은 로거를 반환(중복 핸들러 방지)."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root = logging.getLogger("HalloReg")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger


def set_korean_font(plt, font_manager) -> None:
    """OS별 한글 폰트 자동 선택 (Windows: 맑은고딕 / macOS: AppleGothic / Linux: 나눔)."""
    candidates = {
        "Windows": ["Malgun Gothic"],
        "Darwin": ["AppleGothic"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"],
    }.get(platform.system(), ["DejaVu Sans"])
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
            try:
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.family"] = \
                    font_manager.FontProperties(fname=fp).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
