# -*- coding: utf-8 -*-
"""점수 특징 41종 추출 실행기 (GPU, 약 200발화/초).

계산 본체는 `mdd.gop_feats_v2` 에 있다. 그 모듈은 평가 스크립트 6개가 import 하는
라이브러리이기도 해서, 실행 진입점을 여기로 분리했다.

    $PY tasks/pronunciation-assessment/scripts/score/extract_gop_feats.py --split test --n 200000
"""

# -- 저장소 부트스트랩 (설치 불필요) --
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

from mdd import gop_feats_v2

if __name__ == "__main__":
    gop_feats_v2.main()
