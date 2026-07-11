#!/usr/bin/env python
"""
run_ipd_experiments.py (호환 셔틀)
==================================

메인 엔트리포인트가 `run_ipd_experiment.py` (단수형)로 이동했다.
기존 명령을 그대로 사용할 수 있도록 동일 인자를 위임 실행한다.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_NEW = Path(__file__).resolve().with_name("run_ipd_experiment.py")

if __name__ == "__main__":
    print(f"[안내] run_ipd_experiments.py 는 {_NEW.name} 로 대체되었습니다. 위임 실행합니다.",
          file=sys.stderr)
    runpy.run_path(str(_NEW), run_name="__main__")
