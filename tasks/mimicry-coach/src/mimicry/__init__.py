# -*- coding: utf-8 -*-
"""성대모사 교정 도구 Phase 0 — 원본 대사와 내 녹음을 음절 단위로 비교해 어긋난 곳을 짚는다.

흐름: 대사 → 음절 + 정렬용 자모(`text`) / 녹음 → CTC 강제 정렬로 음절 경계(`align`)
      → 음높이·세기 곡선(`prosody`) → 음절별 특징(`analyze`) → 원본과 비교·우선순위(`compare`)
      → 교정 문장(`feedback`) → 보고서(`report`). `synth` 는 검증용 변형(알려진 어긋남 주입)이다.

학습은 없다. 정렬 모델은 발음 task 의 원어민 Stage 1 w2v-BERT(`w2vbert_stage1/step_10000`)를 그대로 쓴다.
"""
