# -*- coding: utf-8 -*-
"""설치 없이 저장소 패키지를 import 가능하게 만든다.

`pip install -e .` 를 했다면 이 파일은 필요 없다. 하지만 이 서버의 fairseq_xlsr
환경(python 3.8 · pip<24.1 고정)에서는 editable 설치가 까다로워서, 스크립트가
어디서 실행되든 동작하도록 5줄짜리 헤더를 각 스크립트 맨 위에 둔다.
"""

import sys
from pathlib import Path


def add_repo_paths(start: str) -> Path:
    """`start`(보통 __file__)에서 위로 올라가 pyproject.toml을 찾고 경로를 등록."""
    root = Path(start).resolve()
    while not (root / "pyproject.toml").exists():
        if root.parent == root:
            raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
        root = root.parent
    for p in [root, *sorted(root.glob("tasks/*/src"))]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root
