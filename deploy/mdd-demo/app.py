# -*- coding: utf-8 -*-
"""발음 진단(MDD) 웹 데모 실행기 — 서버 기동만 담당한다.

진단 로직은 `mdd.pipeline` 에 있다. 예전에는 이 파일(`mdd_demo.py`)에 서버와 로직이
섞여 있었고, 그래서 평가 스크립트 6개가 **데모 파일을 라이브러리로 import** 하는
기묘한 구조였다. 로직을 패키지로 올리고 여기는 얇은 실행기만 남겼다.

    $PY deploy/mdd-demo/app.py --port 8901
"""

# -- 저장소 부트스트랩 (설치 불필요) --
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import os

from mdd import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()

    # HTML 은 이 디렉터리에 있다 (pipeline 이 자기 위치를 기준으로 찾지 않도록 명시)
    pipeline.HERE = os.path.dirname(os.path.abspath(__file__))

    import uvicorn
    uvicorn.run(pipeline.make_app(), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
