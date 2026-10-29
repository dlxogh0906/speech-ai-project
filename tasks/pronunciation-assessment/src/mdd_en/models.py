# -*- coding: utf-8 -*-
"""영어 음소 인식기 목록 — 비교 스크립트·점수기 학습·평가·서버가 **같은 이름**으로 같은 모델을 읽는다.

    model_dir, kind, vocab_dir = resolve("lv60-espeak")
    model_dir, kind, vocab_dir = resolve("en037-v1")      # 우리가 037 로 추가 학습한 것($SPEECH_AI_DATA/checkpoints)

kind 는 `mdd_en.phones.token_map` 의 출력 기호 체계(espeak · arpa39 · arpa · slplab).
"""

import os

MODELS = {
    "lv60-espeak": {"hf": "facebook/wav2vec2-lv-60-espeak-cv-ft", "kind": "espeak",
                    "license": "Apache-2.0 (CV CC0 · LibriLight) — 상업 가능"},
    "xlsr53-espeak": {"hf": "facebook/wav2vec2-xlsr-53-espeak-cv-ft", "kind": "espeak",
                      "license": "Apache-2.0 (사전학습에 BABEL 포함 — 검토)"},
    "huper": {"hf": "huper29/huper_recognizer", "kind": "arpa39",
              "license": "MIT 가중치 · TIMIT 기반 — 법무 검토 필요"},
    "slplab-l2": {"hf": "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition", "kind": "slplab",
                  "license": "라이선스 없음 · 한국인 학습자 데이터 — 상업 불가(협약 필요)"},
    "charsiu-ctc": {"hf": "charsiu/en_w2v2_ctc_libris_and_cv", "kind": "arpa", "vocab_hf": "charsiu/tokenizer_en_cmu",
                    "license": "가중치 라이선스 표기 없음 — 확인 필요"},
    # lv60-espeak 를 AIHub 037(한국인 영어, AIHub 자동 음소 라벨)로 추가 학습 — scripts/train/en_finetune_ctc.py
    "en037-v1": {"path": os.path.join("checkpoints", "en_ctc_037_v1", "best"), "kind": "arpa",
                 "license": "lv60(Apache-2.0) + AIHub 037(학습 이용 가능 · 출처 표기)"},
    # en037-v1 을 한국인 오류 합성(Kokoro TTS)으로 추가 학습 — 1단계 비교용(3,000걸음 마지막 체크포인트)
    "synth-a": {"path": os.path.join("checkpoints", "en_ctc_synth_v1a", "step_3000"), "kind": "arpa",
                "license": "en037-v1 + Kokoro-82M(Apache-2.0) 합성 음성"},
    "synth-b": {"path": os.path.join("checkpoints", "en_ctc_synth_v1b", "step_3000"), "kind": "arpa",
                "license": "en037-v1 + Kokoro-82M 합성 + AIHub 037"},
    # 2단계: 합성 + 037 자가 학습 의사라벨(synth-a 1,000걸음이 한국인 대표 오류 자리를 판정 · 037 고득점 화자 5% 문턱)
    "synth-d": {"path": os.path.join("checkpoints", "en_ctc_synth_v2d", "step_3000"), "kind": "arpa",
                "license": "en037-v1 + Kokoro-82M 합성 + AIHub 037(의사라벨)"},
}


def resolve(name, local_only=True):
    """이름 → (모델 폴더, kind, vocab 폴더 또는 None). 'path' 는 $SPEECH_AI_DATA 기준 상대 경로."""
    m = MODELS[name]
    if "path" in m:
        from shared.paths import DATA_ROOT
        d = os.path.join(str(DATA_ROOT), m["path"])
        if not os.path.isdir(d):
            raise FileNotFoundError("모델 폴더가 없다: %s" % d)
        return d, m["kind"], None
    from huggingface_hub import snapshot_download
    md = snapshot_download(m["hf"], local_files_only=local_only)
    vd = snapshot_download(m["vocab_hf"], local_files_only=local_only) if m.get("vocab_hf") else None
    return md, m["kind"], vd
