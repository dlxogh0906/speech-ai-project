# -*- coding: utf-8 -*-
"""발음 운영 기록의 녹음에 국적 모델을 돌려 captures.sqlite 의 pron_nat 테이블에 넣는다(NPU 안에서).

발음 요청에는 국적이 없고, 국적 요청과 이어 붙일 열쇠(IP·세션·토큰)도 없다(2026-11-02 확인: 0건).
그래서 운영 국적 API(8905)와 **같은 모델 파일 · 같은 설정**(run_npu.sh)으로 추정한다. 다른 점은 둘뿐이다:
  - NPU(rbln) 대신 CPU(ONNX)로 돈다 — 운영 서비스의 NPU 장치를 쓰지 않는다.
  - HTTP 를 거치지 않는다 — 운영 기록(capture)·업로드 저장에 새 요청이 남지 않는다.
전처리는 운영 /predict 와 같다(0.5초 미만 · 말소리 부족은 판정하지 않고 사유만 남긴다).
결과는 **모델 추정**이다(사람이 준 국적이 아니다). 이미 넣은 rid 는 건너뛴다(자동화용).

    cd /root/nat_serve/bundle && ../venv/bin/python /root/capture_db/pron_nationality.py [--db ...]
"""
import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import time

BUNDLE = os.environ.get("NAT_BUNDLE", os.getcwd())  # 국적 서빙 번들 폴더 (update_all.sh 가 cd 해서 부름)
# 운영 run_npu.sh 와 같은 값 — backend 만 onnx
PROD_ENV = {"NAT_BACKEND": "onnx", "KD_MODEL_DIR": BUNDLE + "/models/zeroth", "KD_GATE_DIR": BUNDLE + "/models/deployed_gate",
            "KD_GATE_TAU": "0.9", "KD_GATE_INPUT": "raw", "KD_GATE_MIN_SEC": "1.0", "KD_LOWCONF": "0.45", "KD_SILENCE_NORM": "0",
            "POOL": "1", "THREADS": "4", "FFMPEG_BIN": os.environ.get("FFMPEG_BIN", "ffmpeg")}
SCHEMA = """CREATE TABLE IF NOT EXISTS pron_nat (
  rid TEXT PRIMARY KEY, status TEXT, top1_label TEXT, top1_prob REAL, top5 TEXT, gate_korea REAL, native INTEGER,
  low_confidence INTEGER, voiced_sec REAL, model TEXT, created_at TEXT);"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--model-tag", default="nat-silnorm-off-20261031 · onnx-cpu")
    a = ap.parse_args()
    for k, v in PROD_ENV.items():
        os.environ[k] = v
    for k in ("KD_SAVE_DIR", "KD_CAPTURE_DIR", "SPEECH_CAPTURE_DIR"):
        os.environ.pop(k, None)
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    todo = con.execute("""SELECT r.rid, r.audio_path FROM requests r JOIN pron_utts u USING (rid)
                          WHERE r.audio_path IS NOT NULL AND r.rid NOT IN (SELECT rid FROM pron_nat)""").fetchall()
    if not todo:
        print("[국적] 새로 할 것 없음", flush=True)
        return
    sys.path[:0] = [BUNDLE]
    spec = importlib.util.spec_from_file_location("nat_app", os.path.join(BUNDLE, "app", "app.py"))
    nat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nat)
    n_ok, n_skip = 0, 0
    for rid, path in todo:
        top5, gate_p, voiced, status = None, None, None, "ok"
        try:
            wav = nat.decode(open(path, "rb").read())
            if wav.size < nat.SR // 2:
                status = "too_short"
            else:
                voiced = nat.voiced_sec(wav)
                if voiced < nat.GATE_MIN_SEC:
                    status = "no_speech"
                else:
                    wav_raw = wav
                    if nat.SILENCE_NORM:
                        wav = nat.silence_normalize(wav, nat.GATE_RMS)
                    if nat.RMS_NORM > 0:
                        wav, _ = nat.rms_normalize(wav, nat.RMS_NORM)
                    top5, _, gate_p = nat.classify(wav, wav_raw)
        except nat.AudioError as e:
            status = e.code
        except FileNotFoundError:
            status = "audio_missing"
        native = None if not top5 else int(top5[0]["label"] == "Korea")
        low = None if not top5 else int(bool(nat.LOWCONF > 0 and not native and top5[0]["prob"] < nat.LOWCONF))
        con.execute("INSERT OR REPLACE INTO pron_nat VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            rid, status, top5[0]["label"] if top5 else None, top5[0]["prob"] if top5 else None,
            json.dumps(top5, ensure_ascii=False) if top5 else None, gate_p, native, low, voiced, a.model_tag,
            time.strftime("%Y-%m-%dT%H:%M:%S")))
        n_ok += status == "ok"; n_skip += status != "ok"
    con.commit()
    print("[국적] 판정 %d · 판정 안 함(짧음·말소리 부족 등) %d" % (n_ok, n_skip), flush=True)


if __name__ == "__main__":
    main()
