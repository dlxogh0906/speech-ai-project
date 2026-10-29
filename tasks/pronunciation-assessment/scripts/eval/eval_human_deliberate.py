# -*- coding: utf-8 -*-
"""사내 녹음(일부러 틀리게 읽은 발음) 채점 — 발음 API 와 같은 경로(`mdd.api_offline`)로 돌려
"일부러 틀린 소리를 그대로 들었나 / 표준으로 고쳐 적었나 / 화면에 틀림으로 나왔나" 와
"맞게 읽은 녹음에서 헛지적이 나왔나" 를 센다.

문장 목록: `tasks/pronunciation-assessment/evalsets/deliberate_human_v1.csv`
  id · ref_text(채점 기준 문장) · wrong_text(일부러 읽을 형태, 비면 맞게 읽기만) · pair · position · group
녹음 파일 이름: <화자>_<id>_<c|w>.<확장자>   (c = 맞게, w = 일부러 틀리게 · 예: s03_d01_w.m4a)
  wav 가 아니면 ffmpeg 로 16 kHz mono 로 바꾼다.

출력: <out>/results.jsonl(녹음마다) · <out>/summary.md(문장·쌍·화자별)

    $PY .../eval_human_deliberate.py --recordings <녹음 폴더> --out <결과 폴더>
    $PY .../eval_human_deliberate.py --recordings … --ckpt <다른 모델> --scorer <짝 점수 모델>
    $PY .../eval_human_deliberate.py --recordings … --canon choose     # 표준발음 보정(mdd.canonical_fix) 효과
--canon: api = 지금 API(KoG2P) · fixed = 결정적 보정 · choose = 보정 + 허용 변이(녹음에 맞춰 고름, 목표 자리와 겹치는 변이는 안 씀)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, FFMPEG, MDD_OUTPUTS

import argparse, collections, csv, json, os, re, shutil, subprocess, wave

NAME_RE = re.compile(r"^(?P<spk>[^_]+)_(?P<id>[A-Za-z]\d+)_(?P<v>[cw])\.[A-Za-z0-9]+$")


def load_audio(path):
    import numpy as np
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as w:
                if w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass
    exe = shutil.which(FFMPEG) or FFMPEG
    p = subprocess.run([exe, "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise ValueError(p.stderr.decode("utf-8", "replace")[-200:])
    return np.frombuffer(p.stdout, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--items", default=str(_r / "tasks" / "pronunciation-assessment" / "evalsets" / "deliberate_human_v1.csv"))
    ap.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    ap.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    ap.add_argument("--canon", choices=("api", "fixed", "choose"), default="api")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mdd.api_offline import ApiEmulator
    from mdd.pair_rescue import FAMILY_OF
    from mdd import canonical_fix as cf

    items = dict((r["id"], r) for r in csv.DictReader(open(a.items, encoding="utf-8")))
    emu = ApiEmulator(a.ckpt, scorer=a.scorer)
    B = cf.CanonicalBuilder(emu.canonical)
    canon_of = emu.canonical if a.canon == "api" else B.fixed
    os.makedirs(a.out, exist_ok=True)
    res = []
    for fn in sorted(os.listdir(a.recordings)):
        m = NAME_RE.match(fn)
        if not m or m.group("id") not in items:
            continue
        it = items[m.group("id")]
        try:
            x = load_audio(os.path.join(a.recordings, fn))
        except ValueError as e:
            res.append({"file": fn, "error": str(e)}); continue
        wc = canon_of(it["wrong_text"]) if it["wrong_text"] else None
        rc = canon_of(it["ref_text"])
        tj = None
        if wc is not None and len(wc) == len(rc):
            d = [j for j in range(len(rc)) if rc[j] != wc[j]]
            tj = d[0] if len(d) == 1 else None
        if a.canon == "choose":
            lp = emu.lprobs(x)
            base, alts = B.variants(it["ref_text"])
            if tj is not None:
                alts = [v for v in alts if not (v[0] <= tj + 1 and tj - 1 <= v[1])]
            cc, used = cf.choose(lp[0], base, alts, emu.dec.d.index, blank=emu.dec.blank)
            if tj is not None and used:                 # 변이로 길이가 바뀌면 목표 자리를 옮긴다
                shift = sum(len(r) - (e - s_) for s_, e, r, _k, _g in used if e <= tj)
                tj += shift
            body, raw = emu.assess(x, it["ref_text"], canon=cc, lp=lp)
        else:
            body, raw = emu.assess(x, it["ref_text"], canon=rc)
        rec = {"file": fn, "spk": m.group("spk"), "id": it["id"], "version": m.group("v"), "pair": it["pair"],
               "position": it["position"], "group": it["group"], "canon_method": a.canon}
        if body is None:
            rec["rejected"] = raw["rejected"]; res.append(rec); continue
        canon = raw["canon"]; ph = emu.phonemes(body)
        rec["n_shown_errors"] = sum("errorType" in p for p in ph)
        rec["heard"] = "".join(s_["pred"] or "-" for s_ in raw["sites"])
        if tj is not None and tj < len(ph):
            p = ph[tj]
            rec.update(target=tj, canon=canon[tj], intended=wc[tj] if wc else None, heard_at_target=p.get("actual_jamo"),
                       shown_error=("errorType" in p), uncertain=bool(p.get("uncertain")), score=p.get("pronunciation"))
        res.append(rec)
    with open(os.path.join(a.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 요약
    by = collections.defaultdict(collections.Counter)
    for r in res:
        if "target" not in r and r.get("group") != "G2P":
            continue
        k = (r["id"], r["version"])
        by[k]["n"] += 1
        if r.get("rejected"):
            by[k]["rejected"] += 1; continue
        if "target" in r:
            by[k]["heard_intended"] += r["heard_at_target"] == r["intended"]
            by[k]["fixed_to_standard"] += r["heard_at_target"] == r["canon"]
            by[k]["shown_error"] += r["shown_error"]
        by[k]["any_shown_error"] += r.get("n_shown_errors", 0) > 0
    lines = ["# 일부러 틀린 발음 — 사내 녹음 채점 (%s · 표준발음 %s)" % (a.ckpt, a.canon), "",
             "| 문장 | 쌍 · 위치 | 일부러 틀리게: 그 소리로 들음 / 표준으로 고침 / 화면 틀림 | 맞게: 그 자리 화면 틀림 / 발화 어디든 틀림 | 녹음 수(거절) |",
             "|---|---|---|---|---|"]
    for iid, it in items.items():
        w, c = by.get((iid, "w"), collections.Counter()), by.get((iid, "c"), collections.Counter())
        if not (w["n"] or c["n"]):
            continue
        lines.append("| %s %s | %s · %s | %s | %s | %d(%d) / %d(%d) |" % (
            iid, it["ref_text"], it["pair"], it["position"] or "-",
            "%d / %d / %d" % (w["heard_intended"], w["fixed_to_standard"], w["shown_error"]) if w["n"] else "-",
            "%d / %d" % (c["shown_error"], c["any_shown_error"]) if c["n"] else "-", w["n"], w["rejected"], c["n"], c["rejected"]))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
