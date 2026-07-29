"""발음 진단(MDD) — XLS-R 300m + CTC 기반 음소 오류 검출 및 점수 산출.

주요 모듈
    mdd_eval               CTC 디코더(`Decoder`) — 20개 스크립트가 import 하는 중심 모듈
    pipeline               문장→표준음소열, 정렬, 오류 유형 판정 (구 `mdd_demo.py`)
    gop_feats_v2           강제정렬 기반 점수 특징 41종
    jamo_to_kog2p          자모 46 ↔ KoG2P 71음소 무손실 변환
    mdd_reproduce_official 공식 3-way 정렬 이식본
    l1prior/               모국어별 오류 사전(s0~s9 파이프라인)
"""
