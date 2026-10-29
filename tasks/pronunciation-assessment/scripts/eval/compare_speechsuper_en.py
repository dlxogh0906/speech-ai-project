# -*- coding: utf-8 -*-
"""영어 발음 평가: 우리 API(8952) ↔ SpeechSuper 영어(sent.eval.promax) 같은 녹음 비교.

⚠️ **보내도 되는 녹음만** — 외부(SpeechSuper, 중국 쑤저우 법인)로 오디오가 나간다.
   - 동의를 받은 우리 녹음, 또는 speechocean762(CC BY 4.0)만 쓴다.
   - AIHub·L2-ARCTIC 오디오는 거부한다(이용 조건상 제3자 제공 금지 / 비상업).
   - 체험 키는 동시 5건 제한이 있어 한 건씩 보낸다. 요금은 요청당 약 $0.006(PAYG)이다.
   - 2026-10-04 기준 우리 키는 영어 coreType 이 막혀 있다(errId 41030 invalid coreType) — 권한을 받은 뒤 돌릴 것.

입력: --items <jsonl> 한 줄에 {"wav": 경로, "text": 영어 문장[, "id": …]}
비교
    문장   overall · pronunciation · fluency · rhythm · integrity 상관(PCC)과 평균 차
    단어   점수 상관 · "틀린 단어" 일치(우리: errorType 있는 음소 / 그쪽: 음소 readType 3·6 또는 단어 readType 3)
    원본 응답은 out/<id>.{ours,ss}.json 으로 남긴다(인증 항목은 지운다)

    source ~/.speechsuper_keys
    $PY .../compare_speechsuper_en.py --items my_recordings.jsonl --out $MDD_OUTPUTS/english/ss_compare/v1 --max 50
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse, hashlib, json, os, re, time, urllib.error, urllib.request, uuid

import numpy as np

SS_URL = "https://api.speechsuper.com/sent.eval.promax"
# SpeechSuper CMU 표기에 있는 비-CMUdict 기호 → ARPAbet 39
SS_PHONE = {"ix": "ih", "ux": "uw", "ax": "ah", "axr": "er", "oh": "ao"}


def allowed(path):
    """외부로 보내도 되는 파일인가 — AIHub·L2-ARCTIC 은 막는다."""
    p = os.path.realpath(path)
    root = os.path.realpath(str(DATA_ROOT))
    if not p.startswith(root + os.sep):
        return True                                    # 코퍼스 밖(우리 녹음)
    rel = os.path.relpath(p, root).replace(os.sep, "/")
    return rel.startswith("english/speechocean762/")


def multipart(params, audio):
    b = uuid.uuid4().hex
    body = ("--" + b + "\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n" + json.dumps(params) + "\r\n--" + b
            + "\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"rec.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            ).encode() + audio + ("\r\n--" + b + "--\r\n").encode()
    return body, {"Content-Type": "multipart/form-data; boundary=" + b, "Request-Index": "0"}


def post(url, params, audio, token=None, timeout=60):
    body, h = multipart(params, audio)
    if token:
        h["Authorization"] = "Bearer " + token
    try:
        res = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=timeout)
    except urllib.error.HTTPError as e:
        res = e
    with res:
        d = json.loads(res.read())
    for k in ("params", "applicationId"):              # 인증 항목은 남기지 않는다
        d.pop(k, None)
    return d


def ss_params(text, rid):
    app, secret = os.environ["SPEECHSUPER_APP_KEY"], os.environ["SPEECHSUPER_SECRET_KEY"]
    ts, user = str(int(time.time())), "en-compare"
    sig = lambda v: hashlib.sha1(v.encode()).hexdigest()
    return {"connect": {"cmd": "connect", "param": {"sdk": {"version": 16777472, "source": 9, "protocol": 2},
                                                    "app": {"applicationId": app, "timestamp": ts, "sig": sig(app + ts + secret)}}},
            "start": {"cmd": "start", "param": {
                "app": {"applicationId": app, "timestamp": ts, "userId": user, "sig": sig(app + ts + user + secret)},
                "audio": {"audioType": "wav", "channel": 1, "sampleBytes": 2, "sampleRate": 16000},
                "request": {"coreType": "sent.eval.promax", "refText": text, "tokenId": rid, "dict_type": "CMU",
                            "dict_dialect": "en_us", "phoneme_output": 1, "readtype_diagnosis": 1, "getParam": 0}}}}


def words_ss(d):
    out = []
    for w in ((d.get("result") or {}).get("words") or []):
        if w.get("charType", 0) != 0 or w.get("readType") == 4:      # 문장부호 · 덧붙인 단어 제외
            continue
        ph = w.get("phonemes") or []
        bad = w.get("readType") == 3 or any(p.get("readType") in (3, 6) for p in ph)
        out.append((re.sub(r"[^a-z0-9']", "", w.get("word", "").lower()), (w.get("scores") or {}).get("overall"), bad))
    return out


def words_ours(d):
    return [(re.sub(r"[^a-z0-9']", "", w["word"].lower()), w["score"], any(p.get("errorType") for p in w["phonemes"]))
            for w in (d.get("result") or {}).get("words", [])]


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return round(float(np.corrcoef(a, b)[0, 1]), 4) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours", default="http://127.0.0.1:8952/sent.eval")
    ap.add_argument("--ours-token", dest="ours_token", default=os.environ.get("PRON_EN_TOKEN"))
    ap.add_argument("--max", type=int, default=50, help="SpeechSuper 호출 상한(요금·체험 한도)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    bad = [it["wav"] for it in items if not allowed(it["wav"])]
    if bad:
        raise SystemExit("외부로 보내면 안 되는 오디오가 있다(AIHub·L2-ARCTIC 등): %s …" % bad[:3])
    if "SPEECHSUPER_APP_KEY" not in os.environ:
        raise SystemExit("키가 없다 — source ~/.speechsuper_keys")
    os.makedirs(a.out, exist_ok=True)
    S = {k: ([], []) for k in ("overall", "pronunciation", "fluency", "rhythm", "integrity")}
    wsc, wflag, n_ok = ([], []), [], 0
    for k, it in enumerate(items[: a.max]):
        rid = it.get("id") or "en-%04d" % k
        audio = open(it["wav"], "rb").read()
        ours = post(a.ours, {"start": {"param": {"app": {"userId": "compare"},
                                                  "request": {"refText": it["text"], "tokenId": rid}}}},
                    audio, token=a.ours_token)
        ss = post(SS_URL, ss_params(it["text"], rid), audio)
        json.dump(ours, open(os.path.join(a.out, rid + ".ours.json"), "w"), ensure_ascii=False)
        json.dump(ss, open(os.path.join(a.out, rid + ".ss.json"), "w"), ensure_ascii=False)
        if ss.get("errId") == 41030:
            raise SystemExit("SpeechSuper: 이 키에 영어 coreType 권한이 없다(41030 invalid coreType) — 권한 요청 필요")
        if "result" not in ss or "result" not in ours:
            print("[skip] %s ours=%s ss=%s" % (rid, ours.get("code") or ours.get("error"), ss.get("error")), flush=True)
            continue
        n_ok += 1
        for f in S:
            o, s = ours["result"].get(f), ss["result"].get(f)
            if o is not None and s is not None:
                S[f][0].append(o); S[f][1].append(s)
        wo, ws = words_ours(ours), words_ss(ss)
        if [w for w, _, _ in wo] == [w for w, _, _ in ws]:
            for (_, so, fo), (_, s_, fs) in zip(wo, ws):
                if so is not None and s_ is not None:
                    wsc[0].append(so); wsc[1].append(s_)
                wflag.append((fo, fs))
        time.sleep(0.3)
    rep = {"n": n_ok, "sentence": {}, "word": {}}
    for f, (o, s) in S.items():
        rep["sentence"][f] = {"pcc": pcc(o, s), "mean_ours": round(float(np.mean(o)), 1) if o else None,
                              "mean_ss": round(float(np.mean(s)), 1) if s else None}
    rep["word"]["score_pcc"] = pcc(*wsc)
    if wflag:
        fo = np.array([x for x, _ in wflag]); fs = np.array([y for _, y in wflag])
        rep["word"]["flag"] = {"both": int((fo & fs).sum()), "ours_only": int((fo & ~fs).sum()),
                               "ss_only": int((~fo & fs).sum()), "neither": int((~fo & ~fs).sum()),
                               "agreement": round(float((fo == fs).mean()), 4)}
    json.dump(rep, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
