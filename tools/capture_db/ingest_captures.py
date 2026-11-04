# -*- coding: utf-8 -*-
"""운영 서버 요청 기록(`shared/request_capture.py` 가 남긴 request.json)을 SQLite 한 파일로 모은다.

    python3 ingest_captures.py [--root /data/speech-logs/captures] [--db /data/speech-logs/db/captures.sqlite]

- 표준 라이브러리만 쓴다(NPU 서버의 시스템 python3 로 돈다). 녹음은 옮기지 않고 경로·sha256 만 남긴다.
- 여러 번 돌려도 된다: rid 별로 request.json 의 mtime 을 기억해, 새로 생기거나 바뀐 요청만 다시 넣는다(자동화용).
- 개인 식별 값은 원문을 넣지 않는다: 접속 IP · session_id · tokenId · userId 는 서버 안의 salt 로 해시한다.
- 라벨의 성격이 다르다:
    pron_phonemes.actual_jamo · error_type  = 그 시점 운영 모델·규칙의 추정(사람 정답 아님)
    nat_preds.actual_nationality            = 앱이 보낸 사용자 입력(사람이 준 값)
    nat_preds.top1_label · dialect_preds.label = 모델 추정
- 발음 운영 버전(pron_version)은 시각으로 붙인다(10-10 이후 capture 의 deployment_id 가 한 값이라).
"""
import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
import time

READ_TYPE = {0: "정상", 1: "빠짐", 2: "다르게 읽음", 3: "오독", 4: "끼워 읽음"}
# 발음 운영(NPU 8932) 변경 시각(KST) — docs/발음평가_개선_반영_2026-10-17.md · 2026-10-21.md · 현재_서비스_모델.md
PRON_VERSIONS = [
    ("2026-10-01T00:00", "P1", "원래 모델 step_38000 + KoG2P 표준발음"),
    ("2026-10-17T21:24", "P2", "섞은 모델(50%) + 표준발음 보정"),
    ("2026-10-18T18:29", "P3", "P2 + 짧은 문장 말소리 기준(음절당 0.08초)"),
    ("2026-10-18T20:16", "P4", "P3 + 된·거센소리 예외"),
    ("2026-10-21T01:58", "P5", "M3 100% 모델(step_3000) + 빠짐 점수 50 미만 + ㄴ↔ㄹ 예외"),
    ("2026-10-22T01:18", "P6", "P5 + 통째로 안 들린 글자 글자 단위 판정"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  rid TEXT PRIMARY KEY, service TEXT, time TEXT, date TEXT, method TEXT, path TEXT, status INTEGER,
  error_code TEXT, ms REAL, deployment_id TEXT, version TEXT,
  client_hash TEXT, session_hash TEXT, client_type TEXT, app_version TEXT, os TEXT, os_version TEXT, device_type TEXT,
  user_agent TEXT, audio_path TEXT, audio_sha256 TEXT, audio_bytes INTEGER, audio_type TEXT, audio_sec REAL, sample_rate INTEGER,
  request_dir TEXT, src_mtime REAL, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS pron_utts (
  rid TEXT PRIMARY KEY, ref_text TEXT, token_hash TEXT, overall INTEGER, pronunciation INTEGER, fluency INTEGER,
  comprehensibility INTEGER, raw_pron REAL, raw_comp REAL, raw_flu REAL,
  n_words INTEGER, n_phonemes INTEGER, n_flags INTEGER, n_uncertain INTEGER);
CREATE TABLE IF NOT EXISTS pron_words (
  rid TEXT, w_idx INTEGER, word TEXT, read_type INTEGER, read_type_name TEXT, score INTEGER, inserted TEXT,
  PRIMARY KEY (rid, w_idx));
CREATE TABLE IF NOT EXISTS pron_phonemes (
  rid TEXT, w_idx INTEGER, p_idx INTEGER, seq INTEGER, alpha TEXT, alpha_jamo TEXT, position TEXT, pronunciation INTEGER,
  actual TEXT, actual_jamo TEXT, error_type TEXT, uncertain INTEGER, detector REAL,
  PRIMARY KEY (rid, seq));
CREATE TABLE IF NOT EXISTS nat_preds (
  rid TEXT PRIMARY KEY, actual_nationality TEXT, top1_label TEXT, top1_prob REAL, top5 TEXT, gate_korea REAL, native INTEGER,
  low_confidence INTEGER, sec REAL, voiced_sec REAL, n_parts INTEGER, rms_dbfs REAL, peak REAL, flatness REAL,
  per_part TEXT, encoder_outputs TEXT);
CREATE TABLE IF NOT EXISTS dialect_preds (
  rid TEXT PRIMARY KEY, sentence_id TEXT, label TEXT, label_ko TEXT, prob REAL, standard INTEGER, low_confidence INTEGER,
  probs TEXT, per_part TEXT);
CREATE TABLE IF NOT EXISTS pron_versions (version TEXT PRIMARY KEY, start_time TEXT, description TEXT);
CREATE INDEX IF NOT EXISTS ix_req_service_time ON requests(service, time);
CREATE INDEX IF NOT EXISTS ix_ph_error ON pron_phonemes(error_type);
CREATE VIEW IF NOT EXISTS v_pron_flags AS
  SELECT r.time, r.version, u.ref_text, w.word, p.* FROM pron_phonemes p
  JOIN requests r USING (rid) JOIN pron_utts u USING (rid) JOIN pron_words w ON w.rid = p.rid AND w.w_idx = p.w_idx
  WHERE p.error_type IS NOT NULL OR p.uncertain = 1;
CREATE VIEW IF NOT EXISTS v_scored AS
  SELECT r.rid, r.time, r.version, r.client_hash, r.audio_path, r.audio_sec, u.ref_text, u.overall, u.pronunciation, u.fluency,
         u.raw_pron, u.raw_comp, u.raw_flu, u.n_words, u.n_phonemes, u.n_flags, u.n_uncertain
  FROM requests r JOIN pron_utts u USING (rid) WHERE r.status = 200;
"""
CHILD = {"pronunciation": ("pron_utts", "pron_words", "pron_phonemes"), "nationality": ("nat_preds",), "dialect": ("dialect_preds",)}


def salt_of(db_path):
    p = os.path.join(os.path.dirname(os.path.abspath(db_path)), ".salt")
    if not os.path.exists(p):
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, os.urandom(16).hex().encode())
        os.close(fd)
    return open(p).read().strip()


def first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def pron_version(t):
    v = None
    for start, name, _ in PRON_VERSIONS:
        if t[:16] >= start:
            v = name
    return v


def ingest_one(con, path, root, h):
    r = json.load(open(path, encoding="utf-8"))
    rid, svc = r.get("rid"), r.get("service")
    if not rid:
        return None
    form, headers, res = r.get("form") or {}, r.get("headers") or {}, r.get("result")
    files = r.get("files") or []
    f0 = files[0] if files else {}
    t = r.get("time") or ""
    err = None
    if isinstance(res, dict) and (r.get("status") or 0) >= 400:
        err = res.get("code") or res.get("error") or res.get("detail")
        err = json.dumps(err, ensure_ascii=False)[:200] if not isinstance(err, str) else err[:200]
    version = pron_version(t) if svc == "pronunciation" else r.get("deployment_id")
    for tb in CHILD.get(svc, ()):
        con.execute("DELETE FROM %s WHERE rid = ?" % tb, (rid,))
    con.execute("INSERT OR REPLACE INTO requests VALUES (%s)" % ",".join("?" * 28), (
        rid, svc, t, t[:10], r.get("method"), r.get("path"), r.get("status"), err, r.get("ms"), r.get("deployment_id"), version,
        h(r.get("client")), h(first(form.get("session_id")) or first(form.get("session")) or headers.get("x-session-id")),
        first(form.get("client_type")) or headers.get("x-client-type"), first(form.get("app_version")) or headers.get("x-app-version"),
        first(form.get("os")), first(form.get("os_version")), first(form.get("device_type")), headers.get("user-agent"),
        os.path.join(os.path.dirname(path), os.path.basename(f0["path"])) if f0.get("path") else None,
        f0.get("sha256"), f0.get("bytes"), f0.get("content_type"), (f0.get("wav") or {}).get("sec"), (f0.get("wav") or {}).get("sample_rate"),
        os.path.relpath(os.path.dirname(path), root), os.path.getmtime(path), time.strftime("%Y-%m-%dT%H:%M:%S")))
    if not isinstance(res, dict) or r.get("status") != 200:
        return svc
    if svc == "pronunciation" and isinstance(res.get("result"), dict):
        rr = res["result"]; words = rr.get("words") or []; meta = res.get("_meta") or {}; raw = meta.get("raw_human_1to5") or {}
        seq, n_flag, n_unc = 0, 0, 0
        for wi, w in enumerate(words):
            rt = w.get("readType")
            con.execute("INSERT INTO pron_words VALUES (?,?,?,?,?,?,?)", (rid, wi, w.get("word"), rt, READ_TYPE.get(rt), w.get("score"),
                        json.dumps(w["inserted"], ensure_ascii=False) if w.get("inserted") else None))
            for pi, p in enumerate(w.get("phonemes") or []):
                n_flag += "errorType" in p; n_unc += bool(p.get("uncertain"))
                con.execute("INSERT INTO pron_phonemes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    rid, wi, pi, seq, p.get("alpha"), p.get("alpha_jamo"), p.get("position"), p.get("pronunciation"),
                    p.get("actual"), p.get("actual_jamo"), p.get("errorType"), int(bool(p.get("uncertain"))), p.get("detector")))
                seq += 1
        con.execute("INSERT INTO pron_utts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            rid, res.get("refText") or first(form.get("refText")), h(res.get("recordId") or first(form.get("tokenId"))),
            rr.get("overall"), rr.get("pronunciation"), rr.get("fluency"), rr.get("comprehensibility"),
            raw.get("pron"), raw.get("comp"), raw.get("flu"), len(words), seq, n_flag, n_unc))
    elif svc == "nationality" and res.get("top5") is not None:
        top = res.get("top5") or [{}]
        dg = r.get("diagnostics") or {}; aq = dg.get("audio_quality") or {}
        con.execute("INSERT INTO nat_preds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            rid, first(form.get("actual_nationality")), top[0].get("label"), top[0].get("prob"), json.dumps(top, ensure_ascii=False),
            res.get("gate_korea"), None if res.get("native") is None else int(bool(res.get("native"))),
            None if res.get("low_confidence") is None else int(bool(res.get("low_confidence"))), res.get("sec"), res.get("voiced_sec"),
            res.get("n"), aq.get("rms_dbfs"), aq.get("peak"), aq.get("flatness"),
            json.dumps(res.get("per_part"), ensure_ascii=False), json.dumps(dg.get("encoder_outputs"), ensure_ascii=False)))
    elif svc == "dialect" and res.get("label") is not None:
        con.execute("INSERT INTO dialect_preds VALUES (?,?,?,?,?,?,?,?,?)", (
            rid, first(form.get("sentence_id")), res.get("label"), res.get("label_ko"), res.get("prob"),
            None if res.get("standard") is None else int(bool(res.get("standard"))),
            None if res.get("low_confidence") is None else int(bool(res.get("low_confidence"))),
            json.dumps(res.get("probs"), ensure_ascii=False), json.dumps(res.get("per_part"), ensure_ascii=False)))
    return svc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="운영 서버 녹음 기록 폴더 (예: /data/speech-logs/captures)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-age", type=float, default=120, help="이보다 최근(초)에 바뀐 request.json 은 다음 실행으로 미룬다(받는 중일 수 있다)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.db)), mode=0o700, exist_ok=True)
    salt = salt_of(a.db)
    h = lambda v: hashlib.sha256((salt + str(v)).encode()).hexdigest()[:16] if v else None
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    con.executemany("INSERT OR REPLACE INTO pron_versions VALUES (?,?,?)", [(n, s, d) for s, n, d in PRON_VERSIONS])
    seen = dict(con.execute("SELECT rid, src_mtime FROM requests"))
    now, done, skip, bad = time.time(), {}, 0, 0
    for path in sorted(glob.glob(os.path.join(a.root, "*", "*", "*", "request.json"))):
        m = os.path.getmtime(path)
        rid = os.path.basename(os.path.dirname(path))
        if seen.get(rid) == m or now - m < a.min_age:
            skip += 1
            continue
        try:
            svc = ingest_one(con, path, a.root, h)
            done[svc] = done.get(svc, 0) + 1
        except Exception as e:                       # 깨진 기록 하나 때문에 전체가 멈추지 않게
            bad += 1
            print("[건너뜀] %s: %r" % (path, e), file=sys.stderr)
        if sum(done.values()) % 500 == 0:
            con.commit()
    con.commit()
    tot = dict(con.execute("SELECT service, COUNT(*) FROM requests GROUP BY service"))
    print("[%s] 새로 넣음 %s · 그대로 %d · 실패 %d · DB 합계 %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), done, skip, bad, tot), flush=True)
    con.close()


if __name__ == "__main__":
    main()
