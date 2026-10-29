# -*- coding: utf-8 -*-
"""서비스 점검 ④ — 엉뚱한 녹음에 어떻게 반응하나 + 대사 확인 문 후보 + 처리 시간.

실서비스 녹음엔 다른 대사, 대사 일부, 앞에 붙은 군말, 꺼진 마이크, 잡음·배경음만 든 녹음이 섞인다. 지금 도구는 대사 텍스트로
억지 정렬을 하므로 무엇이 들어오든 음절 지적을 만들 수 있다 — 그게 얼마나 되는지 잰다. 원본 = 원작 대사. 녹음 종류:
  correct        같은 대사 TTS 3개(정상 — 지적은 나와도 된다)
  wrong          다른 대사 TTS(원본마다 나머지 대사 × 3)
  partial        마지막 어절을 뺀 것(한 어절 대사면 마지막 음절) · first_missing 첫 어절(음절)을 뺀 것
  extra          앞에 다른 대사의 첫 어절을 붙인 것(군말) · repeat 대사를 두 번
  silence 무음 · hiss −50 dBFS 백색 잡음 · noise −25 dBFS 백색 잡음 · babble 웅성거림만 · music 배경음만 (각 2초)
  quiet          정상 녹음을 −35 dB · clipped 24 dB 넘게 찌그러뜨림 · padded 앞뒤 3초 무음
  whisper        정상 녹음을 속삭임으로(Praat LPC 로 성대 울림을 잡음으로 바꿈)
  playback       따라 하지 않고 원본을 스피커로 틀어 녹음한 것(잔향 + opus)
결과(데모와 같은 경로 — Coach.compare, 교차 확인 켬):
  rejected(데모 입력 검사: 0.4초 미만 · RMS 1.5e-3 미만) · error(분석 실패) · retake(경계 일치율 미달 → 다시 녹음 안내) ·
  advice(음절 지적을 보여 줌) · pass("원본과 거의 같아요")
대사 확인 문 후보: 받아 적기 JER(녹음 전체 greedy CTC vs 대사 자모열), 강제 정렬 평균 토큰 확신. 정상 vs 다른 대사 AUC 와 문턱별 막는 비율.
`--latency N`: 정상 쌍 N개로 compare 벽시계 시간만 잰다(단독 실행할 것 — CPU 는 MIMIC_FORCE_CPU=1 로 따로).

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/audit_inputs.py --clips /data/speech-ai/out/mimic/original_clips_v1 \\
        --samples /data/speech-ai/out/mimic/sample_clips_v1 --out <출력 폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 audit_channel(배경음)·validate_synthetic

import argparse, collections, json, os, time

import numpy as np

SR = 16000
KINDS = ["correct", "wrong", "partial", "first_missing", "drop_middle", "extra", "extra_after", "filler", "run_on", "repeat",
         "silence", "hiss", "noise", "babble", "music",
         "quiet", "clipped", "padded", "whisper", "playback"]
OUTCOMES = ["rejected", "error", "line", "whisper", "missing", "retake", "advice", "pass"]
DROPS = ("partial", "first_missing", "drop_middle")              # 대사 일부를 뺀 것 — "'…'가 안 들려요"가 맞다
EXTRAS = ("extra", "extra_after", "filler", "run_on", "repeat", "padded")   # 대사는 다 있고 군더더기가 붙은 것 — 잘라서 봐야 한다


def demo_check(y):
    """deploy/mimicry-demo 의 check_audio 와 같은 문턱(그쪽을 바꾸면 여기도)."""
    return len(y) >= SR * 0.4 and float(np.sqrt(np.mean(y[:SR * 20] ** 2))) >= 1.5e-3


def whisper(x, rng):
    """LPC(Burg 16차) 로 성도 필터를 잡고 잡음을 통과시킨다 — 세기 곡선은 원래 녹음에 맞춘다."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=SR)
    lpc = call(snd, "To LPC (burg)", 16, 0.025, 0.005, 50.0)
    noise = parselmouth.Sound(rng.standard_normal(len(x)) * 0.1, sampling_frequency=SR)
    y = np.asarray(call([lpc, noise], "Filter", "no").values[0], dtype=np.float64)[:len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    w = int(0.02 * SR)
    env = lambda v: np.sqrt(np.convolve(v ** 2, np.ones(w) / w, mode="same") + 1e-10)
    return (y * env(np.asarray(x, dtype=np.float64)) / env(y)).astype(np.float32)


def cut(x, a, b=None):
    return x[int(round(a * SR)):(None if b is None else int(round(b * SR)))]


def words_bounds(res):
    """→ [(어절 첫 음절 시작, 어절 끝 음절 끝)]."""
    out = {}
    for s in res["syllables"]:
        a, b = out.get(s["word"], (s["start"], s["end"]))
        out[s["word"]] = (min(a, s["start"]), max(b, s["end"]))
    return [out[k] for k in sorted(out)]


def target_ids(aligner, line):
    from mimicry.text import align_tokens, parse_line
    words, syls, _ = parse_line(line)
    toks, _ = align_tokens(words, syls)
    return [aligner.vocab[t] for ts in toks for t in ts if t in aligner.vocab]


def auc(pos, neg):
    """pos 가 neg 보다 클 확률(동점 0.5)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--latency", type=int, default=0, help="정상 쌍 N개로 처리 시간만 잰다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mimicry import synth
    from mimicry.audio import load_audio
    from mimicry.locate import greedy, jer
    from mimicry.pipeline import Coach

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    coach = Coach()
    orig = [c for c in json.load(open(os.path.join(a.clips, "clips.json"), encoding="utf-8"))["clips"]
            if (c.get("qc") or {}).get("ok", True)]
    samp = {c["id"]: c for c in json.load(open(os.path.join(a.samples, "clips.json"), encoding="utf-8"))["clips"]}
    orig = [c for c in orig if c["id"] in samp]
    tts = {c["id"]: {k: load_audio(os.path.join(a.samples, f)) for k, f in samp[c["id"]]["files"].items()
                     if k in ("ref", "take_a", "take_b")} for c in orig}
    refs = {}
    for c in orig:
        xr = load_audio(os.path.join(a.clips, c["files"]["ref"]))
        refs[c["id"]] = (xr, c["line"], coach.analyze(xr, c["line"]))

    if a.latency:
        import torch
        pairs = [(cid, y) for cid in refs for y in tts[cid].values()][:a.latency]
        for cid, y in pairs[:2]:                                   # 데우기
            coach.compare(refs[cid][0], y, refs[cid][1], ref=refs[cid][2])
        ts, dur = [], []
        for cid, y in pairs:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            coach.compare(refs[cid][0], y, refs[cid][1], ref=refs[cid][2])
            ts.append(time.time() - t0)
            dur.append(len(y) / SR)
        r = dict(device=coach.al.device, n=len(ts), sec_median=float(np.median(ts)), sec_p90=float(np.percentile(ts, 90)),
                 audio_sec_median=float(np.median(dur)), threads=torch.get_num_threads())
        print(json.dumps(r))
        json.dump(r, open(os.path.join(a.out, "latency_%s.json" % coach.al.device), "w"), indent=1)
        return

    from audit_channel import Ctx, load_bgm
    from validate_synthetic import zeroth_sources
    pool = zeroth_sources("train_data_01", 12, a.seed + 1, 8, 32)
    ctx = Ctx([load_audio(p) for p, _ in pool], load_bgm([a.clips]))
    ids = list(refs)
    takes = []                                              # (원본 id, 종류, 설명, 파형)
    for j, cid in enumerate(ids):
        xr, line, (A, FA) = refs[cid]
        for k, y in sorted(tts[cid].items()):
            takes.append((cid, "correct", k, y))
        for other in ids:
            if other != cid:
                for k, y in sorted(tts[other].items()):
                    takes.append((cid, "wrong", "%s/%s" % (other, k), y))
        other = ids[(j + 1) % len(ids)]
        for k in sorted(tts[cid]):
            y = tts[cid][k]
            B, _ = coach.analyze(y, line)
            wb, S = words_bounds(B), B["syllables"]
            syl = [s_["text"] for s_ in S]
            word_txt = [w for w in B["words"]]
            if len(wb) > 1:
                takes.append((cid, "partial", "%s:%s" % (k, word_txt[-1]), cut(y, 0, wb[-1][0] - 0.02)))
                takes.append((cid, "first_missing", "%s:%s" % (k, word_txt[0]), cut(y, wb[1][0] - 0.02)))
                if len(wb) > 2:
                    m = len(wb) // 2
                    takes.append((cid, "drop_middle", "%s:%s" % (k, word_txt[m]),
                                  np.concatenate([cut(y, 0, wb[m][0] - 0.01), np.zeros(int(0.05 * SR), dtype=np.float32),
                                                  cut(y, wb[m][1] + 0.01)])))
            else:
                takes.append((cid, "partial", "%s:%s" % (k, syl[-1]), cut(y, 0, S[-1]["start"] - 0.02)))
                takes.append((cid, "first_missing", "%s:%s" % (k, syl[0]), cut(y, S[1]["start"] - 0.02)))
            yo = tts[other][k]
            Bo, _ = coach.analyze(yo, refs[other][1])
            wo = words_bounds(Bo)
            gap = np.zeros(int(0.15 * SR), dtype=np.float32)
            first_o, last_o = cut(yo, max(0.0, wo[0][0] - 0.02), wo[0][1] + 0.02), cut(yo, max(0.0, wo[-1][0] - 0.02), wo[-1][1] + 0.05)
            takes.append((cid, "extra", "%s:앞에 '%s'" % (k, Bo["words"][0]), np.concatenate([first_o, gap, y])))
            takes.append((cid, "extra_after", "%s:뒤에 '%s'" % (k, Bo["words"][-1]), np.concatenate([y, gap, last_o])))
            takes.append((cid, "run_on", "%s:앞에 '%s' 쉬지 않고" % (k, Bo["words"][0]), np.concatenate([first_o, y])))
            Be, _ = coach.analyze(tts["eoi"][k], refs["eoi"][1])
            eo = Be["syllables"][0]
            filler = cut(tts["eoi"][k], max(0.0, eo["start"] - 0.02), eo["end"] + 0.03)   # '어'
            takes.append((cid, "filler", "%s:앞에 '어'" % k, np.concatenate([filler, np.zeros(int(0.25 * SR), dtype=np.float32), y])))
            takes.append((cid, "repeat", "%s:두 번" % k, np.concatenate([y, np.zeros(int(0.3 * SR), dtype=np.float32), y])))
        y = tts[cid]["take_a"]
        n2 = 2 * SR
        takes.append((cid, "silence", "", np.zeros(n2, dtype=np.float32)))
        takes.append((cid, "hiss", "", (rng.standard_normal(n2) * 10 ** (-50 / 20.0)).astype(np.float32)))
        takes.append((cid, "noise", "", (rng.standard_normal(n2) * 10 ** (-25 / 20.0)).astype(np.float32)))
        b = ctx.babble(n2, rng)
        takes.append((cid, "babble", "", (b / np.max(np.abs(b)) * 0.3).astype(np.float32)))
        mu = ctx.bgm(n2, rng)
        takes.append((cid, "music", "", (mu / (np.max(np.abs(mu)) + 1e-9) * 0.3).astype(np.float32)))
        takes.append((cid, "quiet", "−35 dB", (y * 10 ** (-35 / 20.0)).astype(np.float32)))
        takes.append((cid, "clipped", "+24 dB", synth.clip(y, 24.0)))
        z3 = np.zeros(3 * SR, dtype=np.float32)
        takes.append((cid, "padded", "take_a:앞뒤 3초", np.concatenate([z3, y, z3])))
        takes.append((cid, "whisper", "", whisper(y, rng)))
        takes.append((cid, "playback", "", synth.codec(synth.reverb(xr, 0.4, rng, drr_db=4.0), "opus", 32)))

    fo = open(os.path.join(a.out, "inputs.jsonl"), "w", encoding="utf-8")
    rows = []
    tids = {cid: target_ids(coach.al, refs[cid][1]) for cid in ids}
    t_start = time.time()
    clean_top1 = {}
    for cid, kind, desc, y in takes:
        xr, line, pre = refs[cid]
        row = dict(ref=cid, kind=kind, desc=desc, sec=round(len(y) / SR, 2))
        hyp = [k for k, _ in greedy(coach.al.log_probs(y), coach.al.blank)] if len(y) >= SR * 0.1 else []
        row["jer"] = round(jer(tids[cid], hyp), 3)
        row["demo_ok"] = demo_check(y)
        t0 = time.time()
        try:
            A, B, C = coach.compare(xr, y, line, ref=pre)
            q = C["quality"]
            reason = q.get("reason")
            row.update(agree=B.get("bound_agree"), prob=round(float(np.mean([s["prob"] for s in B["syllables"]])), 3),
                       match=C["match"], n_sel=len(C["selected"]), sel=[s["say"]["what"] for s in C["selected"]],
                       reason=reason, say=(q.get("say") or {}).get("what"), trimmed=B.get("trimmed"), crop=B.get("crop"),
                       outcome=("line" if reason in ("line", "silent") else reason) if reason in ("line", "silent", "whisper", "missing")
                       else ("retake" if not q["ok"] else ("advice" if C["selected"] else "pass")))
            base = desc.split(":")[0]
            if kind == "correct":
                clean_top1[(cid, desc)] = row["sel"][0] if row["sel"] else None
            elif (cid, base) in clean_top1:
                row["clean_top1"] = clean_top1[(cid, base)]
        except (ValueError, RuntimeError) as e:            # 데모는 이 둘을 400/500 안내 문구로 돌려준다
            row.update(outcome="error", err=str(e)[:160])
        except Exception as e:                             # 그 밖은 데모에서 처리 안 된 500 — 따로 센다
            row.update(outcome="error", crash=type(e).__name__, err=str(e)[:160])
        row["ms"] = int((time.time() - t0) * 1000)
        if not row["demo_ok"]:
            row["outcome_demo"] = "rejected"
        else:
            row["outcome_demo"] = row["outcome"]
        rows.append(row)
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    fo.close()
    print("%d개 · %.0fs" % (len(rows), time.time() - t_start), flush=True)
    md = summarize(rows)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(md)
    print(md)


def summarize(rows):
    L = ["# 서비스 점검 ④ 엉뚱한 녹음 — 도구가 어떻게 반응하나", "",
         "결과 = 데모 경로 그대로(입력 검사 → compare). rejected = 데모 입력 검사에서 거절, error = 분석 실패(오류 응답), "
         "line = '대사가 다르게 들려요'·'말소리가 잘 안 들려요', whisper = '속삭이듯 말해서 … 목소리를 내서', "
         "missing = \"'…'가 안 들려요\"(빠진 어절), "
         "retake = '녹음을 원본과 확실히 맞추지 못했어요', advice = 음절 지적을 보여 줌, pass = '원본과 거의 같아요'. "
         "JER = 녹음 전체를 받아 적은 자모열과 대사의 자모 오류율(군말 포함).", "",
         "| 녹음 | n | rejected | error | line | whisper | missing | retake | **advice** | pass | JER 중앙값 | 정렬 확신 중앙값 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for k in KINDS:
        R = by.get(k)
        if not R:
            continue
        c = collections.Counter(r["outcome_demo"] for r in R)
        pr = [r["prob"] for r in R if "prob" in r]
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | **%s** | %s | %.2f | %s |" % (
            k, len(R), *["%.0f%%" % (100.0 * c[o] / len(R)) for o in OUTCOMES], np.median([r["jer"] for r in R]),
            "%.2f" % np.median(pr) if pr else "—"))
    L += ["", "## 대사 일부를 뺀 녹음 · 군더더기가 붙은 녹음", "",
          "뺀 녹음은 \"'…'가 안 들려요\"가 맞다(이름 맞음 = 뺀 어절·음절이 안내 문장에 들어 있음). 군더더기 녹음은 잘라 내고 "
          "분석해야 맞다(1순위 같음 = 군더더기 없는 같은 녹음의 첫 지적과 같음).", "",
          "| 녹음 | n | 빠짐 안내 | 그중 이름 맞음 | 다른 대사로 막음 | 음절 지적(틀림) | 잘라 냄 | 분석함 | 1순위가 깨끗한 녹음과 같음 |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in DROPS + EXTRAS:
        R = by.get(k)
        if not R:
            continue
        n = len(R)
        miss = [r for r in R if r["outcome_demo"] == "missing"]
        named = [r for r in miss if r["desc"].split(":")[-1] in (r.get("say") or "")]
        ana = [r for r in R if r["outcome_demo"] in ("advice", "pass")]
        same = [r for r in ana if "clean_top1" in r and (r["sel"][0] if r["sel"] else None) == r["clean_top1"]]
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            k, n, "%.0f%%" % (100.0 * len(miss) / n), ("%d/%d" % (len(named), len(miss))) if miss else "—",
            "%.0f%%" % (100.0 * sum(r["outcome_demo"] == "line" for r in R) / n),
            "%.0f%%" % (100.0 * sum(r["outcome_demo"] == "advice" for r in R) / n) if k in DROPS else "—",
            "%.0f%%" % (100.0 * sum(bool(r.get("trimmed")) for r in R) / n),
            "%.0f%%" % (100.0 * len(ana) / n),
            ("%d/%d" % (len(same), len([r for r in ana if "clean_top1" in r]))) if k in EXTRAS else "—"))
    pos = [r for r in by.get("correct", [])]
    neg = [r for r in by.get("wrong", [])]
    L += ["", "## 대사 확인 문 후보 — 정상(같은 대사) vs 다른 대사", "",
          "AUC: 받아 적기 JER %.3f · 정렬 확신 %.3f (1 이면 완전 분리)" % (
              auc([-r["jer"] for r in pos], [-r["jer"] for r in neg]),
              auc([r["prob"] for r in pos if "prob" in r], [r["prob"] for r in neg if "prob" in r])), "",
          "| JER 문턱(넘으면 '대사가 달라요') | 정상을 막음 | 다른 대사를 막음 | 일부 뺌을 막음 | 군말·두 번을 막음 | 잡음·무음류를 막음 |",
          "|--:|--:|--:|--:|--:|--:|"]
    part = by.get("partial", []) + by.get("first_missing", [])
    ext = by.get("extra", []) + by.get("repeat", [])
    junk = [r for k in ("silence", "hiss", "noise", "babble", "music") for r in by.get(k, [])]
    for th in (0.3, 0.4, 0.5, 0.6):
        f = lambda R: "%.0f%%" % (100.0 * np.mean([r["jer"] > th for r in R])) if R else "—"
        L.append("| %.1f | %s | %s | %s | %s | %s |" % (th, f(pos), f(neg), f(part), f(ext), f(junk)))
    ms = [r["ms"] for r in rows if r["outcome"] != "error"]
    crash = collections.Counter((r["kind"], r["crash"]) for r in rows if r.get("crash"))
    L += ["", "처리 안 된 예외(데모에선 500): %s" % (", ".join("%s %s×%d" % (k, e, n) for (k, e), n in crash.items()) or "없음"),
          "", "compare 처리 시간(이 실행, GPU 공유 상태): 중앙값 %d ms · p90 %d ms" % (np.median(ms), np.percentile(ms, 90)), "",
          "## 지적이 나온 엉뚱한 녹음 예", ""]
    for k in ("wrong", "partial", "first_missing", "drop_middle", "extra", "run_on", "filler", "repeat", "noise", "whisper"):
        ex = [r for r in by.get(k, []) if r["outcome_demo"] == "advice"][:2]
        for r in ex:
            L.append("- %s · 원본 %s · %s → %s (JER %.2f, 일치 %s)" % (k, r["ref"], r["desc"] or "-", " / ".join(r["sel"]),
                                                                  r["jer"], r["match"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
