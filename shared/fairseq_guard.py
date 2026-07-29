# -*- coding: utf-8 -*-
"""`import fairseq` 함정 회피 — 41개 파일에 흩어져 있던 sys.path 조작을 여기로 통일.

문제: fairseq 저장소 루트(`__init__.py` 없는 디렉터리)가 상위 경로를 통해
namespace package 로 인식되면서 editable 설치된 **진짜** fairseq 패키지를 가린다.
증상이 조용한 ImportError 나 엉뚱한 모듈 로드라서 디버깅이 오래 걸린다.

해결: fairseq 소스 루트를 sys.path **맨 앞**에 두고 나서 import 한다.
"""
import sys

from .paths import FAIRSEQ_ROOT


def ensure_fairseq():
    """`import fairseq` 앞에서 반드시 먼저 호출할 것."""
    p = str(FAIRSEQ_ROOT)
    if not sys.path or sys.path[0] != p:
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
