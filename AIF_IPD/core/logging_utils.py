"""
core.logging_utils
==================

Logger setup and cross-platform font configuration for matplotlib.

Historically the figures carried Korean labels, so a CJK-capable font
was injected into rcParams per OS. All labels are English as of
v3.9.x; the helper is retained as a harmless glyph-fallback setup for
mixed-script environments.
"""

from __future__ import annotations

import logging
import platform

_CONFIGURED = False


def get_logger(name: str = "HalloReg", level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a console handler.

    The handler is attached once to the 'HalloReg' root logger with
    propagate off, so repeated calls from submodules never duplicate
    handlers (no doubled log lines).
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
    Cross-platform CJK-capable font selection (legacy helper).

    Windows: Malgun Gothic; macOS: AppleGothic; Linux: NanumGothic ->
    Noto Sans CJK -> direct file registration. If none is found,
    DejaVu Sans remains. `axes.unicode_minus=False` avoids broken
    minus signs in fonts lacking the U+2212 glyph.
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
        # Names may be missing from the installed list (containers):
        # try registering font files directly.
        for fp in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            try:
                font_manager.fontManager.addfont(fp)
                chosen = font_manager.FontProperties(fname=fp).get_name()
                break
            except Exception:
                continue

    # Set a **fallback list**: CJK fonts often lack combining marks
    # or some symbol glyphs; DejaVu Sans at the end lets matplotlib
    # substitute only the missing glyphs (no tofu boxes).
    fallback = ["DejaVu Sans"]
    plt.rcParams["font.family"] = ([chosen] + fallback) if chosen else fallback
    # Apply the same fallback to the monospace family used by
    # monospace-aligned annotations.
    if chosen:
        plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", chosen]
    plt.rcParams["axes.unicode_minus"] = False
