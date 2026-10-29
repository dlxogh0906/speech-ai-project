# -*- coding: utf-8 -*-
"""Phase 0 검증 — 정답을 아는 어긋남을 넣어 "그 글자를 그 종류로 짚는가", 방해 변형에서 "안 짚는가"를 잰다.

원천: Zeroth Korean test 화자(공개 원어민 낭독). 화자 메타데이터(AUDIO_INFO)는 읽지 않는다.
  - 방해 변형 9종: 목소리(성별 바꿈)·전체 음높이 ±3·전체 빠르기 ×0.8/×1.25·음량 −15 dB·잡음 20 dB·조합
    → 지적이 없어야 한다(같은 사람이 같은 억양으로 말한 것).
  - 주입 변형 7종 × 바탕 2종(같은 목소리 / 목소리 바꿈): 음절 하나에 음높이 ±4 반음 · 억양 방향 뒤집기 ·
    길이 ×1.8 / ×0.5 · 앞에 쉼 0.4초 · 세기 −10 dB → 그 음절을 그 종류로 짚어야 한다.
  - 경계 오차: 변형의 시간 사상으로 원본 경계를 옮긴 값이 정답. 녹음 강제 정렬(우리) vs 정렬 모델 특징 DTW(교차 확인용)
    vs MFCC-DTW 투영(계획서 6.4).
  ⚠ 같은 사람 신호를 비튼 것이라 DTW 에 유리하다 — 실제 다른 화자 쌍은 scripts/eval/real_pairs.py.
변형마다 음절 특징을 rows.jsonl 에 남긴다 — 허용치를 바꿔 다시 채점할 땐 `--rescore` (디코드 없이 수 초).

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/eval/validate_synthetic.py --n 40 --out <출력 폴더>
    PYTHONPATH=... $PY tasks/mimicry-coach/scripts/eval/validate_synthetic.py --rescore <출력 폴더> --tol energy=8
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, json, os, random, re, time

import numpy as np

NUISANCE = ["resynth", "pitch+3", "pitch-3", "tempo0.8", "tempo1.25", "gender", "noise20", "gain-15", "combo"]
INJECT = ["pitch_up", "pitch_down", "contour", "longer", "shorter", "pause_add", "weaker"]
BASES = ["same", "gender"]
TYPES = ["pitch_level", "pitch_dir", "duration", "pause", "energy"]
FEAT = ["start", "end", "dur", "pause_before", "level", "delta", "energy", "n_voiced", "prob", "bound_ok"]


def zeroth_sources(split, n, seed, min_syl, max_syl):
    from shared.paths import DATA_ROOT
    root = os.path.join(os.fspath(DATA_ROOT), "zeroth", "wav16k", split, "003")
    by_spk = {}
    for spk in sorted(os.listdir(root)):
        tf = os.path.join(root, spk, "%s_003.trans.txt" % spk)
        if not os.path.exists(tf):
            continue
        for line in open(tf, encoding="utf-8"):
            stem, _, text = line.strip().partition(" ")
            if not text or re.search(r"[0-9A-Za-z]", text):
                continue
            if min_syl <= len(re.sub(r"[^가-힣]", "", text)) <= max_syl:
                by_spk.setdefault(spk, []).append((os.path.join(root, spk, stem + ".wav"), text))
    rng = random.Random(seed)
    for v in by_spk.values():
        v.sort()
        rng.shuffle(v)
    out, spks = [], sorted(by_spk)
    while len(out) < n and any(by_spk.values()):
        for s in spks:
            if by_spk[s] and len(out) < n:
                out.append(by_spk[s].pop())
    return out


def pack(S):
    return [[s.get(k) for k in FEAT] for s in S]


def unpack(rows, text=None):
    out = []
    for i, r in enumerate(rows):
        d = dict(zip(FEAT, r), i=i, text=(text[i] if text else ""))
        if d.get("bound_ok") is None:
            d.pop("bound_ok", None)                      # 원본(교차 확인 안 함) · 옛 rows
        out.append(d)
    return out


def pick_target(A, kind, rng):
    ok = eligible(A, kind)
    return rng.choice(ok) if ok else None


def eligible(A, kind):
    """그 종류를 넣어 볼 만한 음절 번호들."""
    ok = []
    for s in A["syllables"][1:]:                      # 첫 음절은 말 시작 경계라 뺀다
        if kind in ("pitch_up", "pitch_down"):
            good = s["level"] is not None and s["n_voiced"] >= 8
        elif kind == "contour":
            good = s["delta"] is not None and s["n_voiced"] >= 10
        elif kind in ("longer", "shorter"):
            good = s["dur"] >= 0.1
        elif kind == "pause_add":
            good = s["pause_before"] < 0.1
        else:
            good = s["n_voiced"] >= 5
        if good:
            ok.append(s["i"])
    return ok


def bounds_of(S):
    return [s["start"] for s in S] + [S[-1]["end"]]


def run(a):
    from mimicry import dtw, synth
    from mimicry.align import FRAME, OFFSET
    from mimicry.analyze import crosscheck
    from mimicry.audio import load_audio
    from mimicry.pipeline import Coach

    os.makedirs(a.out, exist_ok=True)
    srcs = zeroth_sources(a.split, a.n, a.seed, a.min_syl, a.max_syl)
    coach = Coach()
    rng = random.Random(a.seed)
    fo = open(os.path.join(a.out, "rows.jsonl"), "w", encoding="utf-8")
    t_start = time.time()
    lat = []

    def evaluate(sid, x, A, FA, y, tmap, text, meta):
        t0 = time.time()
        B, FB = coach.analyze(y, text)
        if a.crosscheck:
            crosscheck(x, A, y, B, feats=(FA, FB), hop=FRAME, off=OFFSET)
        lat.append((time.time() - t0, len(y) / 16000.0))
        rb = np.array(bounds_of(A["syllables"]))
        truth = synth.map_time(tmap, rb + 1e-3) - 1e-3            # 음절 시작은 오른쪽 극한(쉼을 넣었으면 쉼 끝)
        row = dict(meta, src=sid, syl=pack(B["syllables"]), speech=B["speech"], bound_agree=B.get("bound_agree"),
                   line_jer=B.get("line_jer"), line_heard=B.get("line_heard"),
                   bounds=dict(truth=[round(float(v), 4) for v in truth], ctc=bounds_of(B["syllables"])))
        if a.crosscheck:
            S = B["syllables"]
            row["bounds"]["featdtw"] = [s["dtw"][0] for s in S] + [S[-1]["dtw"][1]]
        if a.dtw:
            row["bounds"]["mfccdtw"] = dtw.project(x, y, list(rb), A["speech"], B["speech"])
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    for sid, (path, text) in enumerate(srcs):
        x = load_audio(path)
        if len(x) / 16000.0 > a.max_sec:
            continue
        try:
            A, FA = coach.analyze(x, text)
        except Exception as e:
            print("[skip] src %d: %s" % (sid, e), flush=True)
            continue
        high = A["baseline_hz"] >= 165.0
        pr = A["pitch_range"]
        wide = (max(50.0, pr[0] * 0.8), min(1100.0, pr[1] * 1.25))   # 합성 쪽 분석 범위는 넉넉히(주입한 내림·올림이 잘리지 않게)
        fo.write(json.dumps(dict(variant="ref", src=sid, text=text, n=len(A["syllables"]), syl=pack(A["syllables"]),
                                 speech=A["speech"], baseline_hz=A["baseline_hz"], token_source=A["token_source"],
                                 line_jer=A.get("line_jer")),
                            ensure_ascii=False) + "\n")
        for kind in NUISANCE:
            y, tm = synth.nuisance(x, kind, wide, np.random.default_rng(rng.randrange(1 << 30)), high)
            evaluate(sid, x, A, FA, y, tm, text, dict(variant="nuisance", kind=kind))
        y0 = synth.psola(x, prange=wide)                 # 같은 목소리 바탕(PSOLA 재합성만)
        for base in BASES:
            for kind in INJECT:
                k = pick_target(A, kind, rng)
                if k is None:
                    continue
                src_sig = y0 if kind in ("pause_add", "weaker") else x
                y, tm, truth = synth.inject(src_sig, kind, A["syllables"][k], wide, ref_delta=A["syllables"][k]["delta"])
                if base == "gender":
                    y = synth.change_gender(y, wide, high)
                evaluate(sid, x, A, FA, y, tm, text, dict(variant="inject", kind=kind, base=base, k=k, truth=truth))
        fo.flush()
        el = time.time() - t_start
        print("[%d/%d] %d음절 · %.0fs 경과" % (sid + 1, len(srcs), len(A["syllables"]), el), flush=True)
    fo.close()
    if lat:
        L = np.array(lat)
        json.dump(dict(per_variant_sec_median=float(np.median(L[:, 0])), audio_sec_median=float(np.median(L[:, 1])),
                       device=coach.al.device, note="GPU 를 다른 학습과 같이 쓰는 상태에서 잼 — 단독 실행 수치가 아니다"),
                  open(os.path.join(a.out, "latency.json"), "w"), indent=1)


def dir_ok(chk, ttype, tdir):
    if ttype == "pitch_dir":
        return chk["take_dir"] == tdir
    return chk["dir"] == tdir


def score(rows, tol=None, use_cc=True):
    from mimicry.compare import compare
    refs = {r["src"]: r for r in rows if r["variant"] == "ref"}
    nu = collections.defaultdict(collections.Counter)
    inj = collections.defaultdict(collections.Counter)
    bnd = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["variant"] == "ref" or r["src"] not in refs:
            continue
        ref = refs[r["src"]]
        txt = re.sub(r"[^가-힣]", "", ref["text"])
        take = dict(syllables=unpack(r["syl"], txt), line_jer=r.get("line_jer"), line_heard=r.get("line_heard"))
        if use_cc and r.get("bound_agree") is not None:
            take["bound_agree"] = r["bound_agree"]
        else:
            for sd in take["syllables"]:
                sd.pop("bound_ok", None)
        C = compare(dict(syllables=unpack(ref["syl"], txt), line_jer=ref.get("line_jer")), take, tol=tol)
        n = ref["n"]
        flagged = {row["i"] for row in C["rows"] if any(v["flag"] for v in row["checks"].values())}
        if r["variant"] == "nuisance":
            st = nu[r["kind"]]
            st["utt"] += 1
            st["syl"] += n
            st["flag_syl"] += len(flagged)
            st["clean_utt"] += int(len(C["selected"]) == 0 and C["quality"]["ok"])
            st["shown"] += len(C["selected"])                    # 보여 준(고칠 곳) 잘못된 지적
            st["retake"] += int(not C["quality"]["ok"])
            st["uncertain_syl"] += C["quality"]["uncertain"]
            st["tempo_flag"] += int(any(x["type"] == "tempo" for x in C["issues"]))
            for t in TYPES:
                st["flag_" + t] += sum(1 for row in C["rows"] if row["checks"].get(t, {}).get("flag"))
        else:
            st = inj[(r["kind"], r["base"])]
            k, tt, td = r["k"], r["truth"]["type"], r["truth"]["dir"]
            chk = C["rows"][k]["checks"].get(tt)
            sel = C["selected"]
            st["n"] += 1
            st["retake"] += int(not C["quality"]["ok"])
            st["k_uncertain"] += int(C["rows"][k]["uncertain"])
            st["measurable"] += int(chk is not None)
            st["type_hit"] += int(bool(chk and chk["flag"] and dir_ok(chk, tt, td) and C["quality"]["ok"]))
            st["loc_hit"] += int(k in flagged)
            covers = lambda s: s.get("i") is not None and s["i"] <= k <= s.get("i_end", s["i"]) and s["type"] == tt
            st["top1"] += int(bool(sel) and covers(sel[0]))
            st["in_top3"] += int(any(covers(s) for s in sel))
            st["other_flag_syl"] += len(flagged - {k})
            st["other_syl"] += n - 1
        b = r.get("bounds")
        if b:
            key = r["kind"] if r["variant"] == "nuisance" else "inject:" + r["kind"]
            tr = np.array(b["truth"])
            for m in ("ctc", "featdtw", "mfccdtw"):
                if m in b:
                    bnd[key][m] += list(np.abs(np.array(b[m]) - tr) * 1000.0)
    return nu, inj, bnd


def pct(a, b):
    return "—" if not b else "%.1f%%" % (100.0 * a / b)


def write_summary(out, rows, tol=None, tag="", use_cc=True):
    nu, inj, bnd = score(rows, tol, use_cc)
    n_src = sum(1 for r in rows if r["variant"] == "ref")
    L = ["# Phase 0 합성 검증%s" % tag, "",
         "원천 발화 %d개(Zeroth test 원어민 낭독) · 허용치 %s · 경계 교차 확인 %s" % (
             n_src, json.dumps(tol or "초안", ensure_ascii=False), "켬" if use_cc else "끔"), "",
         "## 1. 방해 변형 — 지적이 없어야 한다", "",
         "지적된 음절 비율 = 허용치를 넘은 음절 / 전체 음절 · 보여 준 지적/발화 = 고칠 곳(음높이·세기 허용치 1.5배 · 길이 1.25배 · 나머지 1배 넘음)으로 보여 준 수 · "
         "지적 0 발화 = 보여줄 지적이 하나도 없음 · "
         "재녹음 = 경계 일치율 미달로 다시 녹음 안내 · 확신 없음 = 경계 불일치로 안 잰 음절", "",
         "| 변형 | 발화 | 지적된 음절 비율 | 보여 준 지적/발화 | 지적 0 발화 | 재녹음 | 확신 없음 | 음높이 | 방향 | 길이 | 쉼 | 세기 | 빠르기 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    tot = collections.Counter()
    for k in NUISANCE:
        st = nu.get(k)
        if not st:
            continue
        tot.update(st)
        L.append("| %s | %d | %s | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k, st["utt"], pct(st["flag_syl"], st["syl"]), st["shown"] / max(1, st["utt"]), pct(st["clean_utt"], st["utt"]),
            pct(st["retake"], st["utt"]),
            pct(st["uncertain_syl"], st["syl"]),
            *[pct(st["flag_" + t], st["syl"]) for t in TYPES], pct(st["tempo_flag"], st["utt"])))
    if tot:
        L.append("| **합계** | %d | **%s** | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            tot["utt"], pct(tot["flag_syl"], tot["syl"]), tot["shown"] / max(1, tot["utt"]), pct(tot["clean_utt"], tot["utt"]),
            pct(tot["retake"], tot["utt"]),
            pct(tot["uncertain_syl"], tot["syl"]),
            *[pct(tot["flag_" + t], tot["syl"]) for t in TYPES], pct(tot["tempo_flag"], tot["utt"])))
    L += ["", "## 2. 주입 변형 — 그 음절을 그 종류로 짚어야 한다", "",
          "type_hit = 주입한 음절의 그 항목이 허용치를 넘고 방향도 맞음 · 1순위 = 첫 지적이 바로 그것 · "
          "3개 안 = 보여주는 3개 안에 있음 · 다른 음절 지적 = 주입 안 한 음절이 걸린 비율", "",
          "| 주입 | 바탕 | n | 재녹음 | 그 음절 확신 없음 | 잴 수 있음 | type_hit | 1순위 | 3개 안 | 다른 음절 지적 |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in INJECT:
        for base in BASES:
            st = inj.get((k, base))
            if not st:
                continue
            L.append("| %s | %s | %d | %s | %s | %s | **%s** | %s | %s | %s |" % (
                k, base, st["n"], pct(st["retake"], st["n"]), pct(st["k_uncertain"], st["n"]), pct(st["measurable"], st["n"]),
                pct(st["type_hit"], st["n"]), pct(st["top1"], st["n"]),
                pct(st["in_top3"], st["n"]), pct(st["other_flag_syl"], st["other_syl"])))
    L += ["", "## 3. 음절 경계 오차(ms) — 중앙값 / p90 / 50ms 넘는 비율", "",
          "녹음 강제 정렬(경계로 쓰는 것) · 정렬 모델 특징 DTW(교차 확인) · MFCC-DTW(계획서 6.4). "
          "같은 사람 신호를 비튼 합성 쌍이라 DTW 에 유리하다.", "",
          "| 변형 | 강제 정렬 | 특징 DTW | MFCC-DTW |", "|---|--:|--:|--:|"]
    for k in NUISANCE + ["inject:" + x for x in INJECT]:
        if k not in bnd:
            continue
        fmt = lambda v: ("%.0f / %.0f / %.1f%%" % (np.median(v), np.percentile(v, 90), 100.0 * np.mean(v > 50))) if len(v) else "—"
        L.append("| %s | %s | %s | %s |" % (k, *[fmt(np.array(bnd[k].get(m, []))) for m in ("ctc", "featdtw", "mfccdtw")]))
    md = "\n".join(L) + "\n"
    name = "summary%s.md" % (("_" + tag.strip().replace(" ", "_")) if tag else "")
    open(os.path.join(out, name), "w", encoding="utf-8").write(md)
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="출력 폴더(새로 돌릴 때)")
    ap.add_argument("--rescore", help="이미 돈 출력 폴더 — rows.jsonl 만 다시 채점")
    ap.add_argument("--tol", default="", help="허용치 덮어쓰기, 예: pitch_level=2.5,energy=8")
    ap.add_argument("--tag", default="", help="요약 파일 이름 꼬리표(--rescore 때)")
    ap.add_argument("--split", default="test_data_01")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-syl", dest="min_syl", type=int, default=8)
    ap.add_argument("--max-syl", dest="max_syl", type=int, default=32)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=9.0)
    ap.add_argument("--no-dtw", dest="dtw", action="store_false")
    ap.add_argument("--no-crosscheck", dest="crosscheck", action="store_false",
                    help="경계 교차 확인(정렬 vs DTW) 없이 — 채점 때도 끈다")
    a = ap.parse_args()
    tol = {}
    for kv in filter(None, a.tol.split(",")):
        k, v = kv.split("=")
        tol[k.strip()] = float(v)
    if "duration" in tol:
        tol["duration"] = float(np.log(tol["duration"]))     # 배수로 받는다(예: duration=1.5)
    if "tempo" in tol:
        tol["tempo"] = float(np.log(tol["tempo"]))
    out = a.rescore or a.out
    if not out:
        ap.error("--out 또는 --rescore 가 필요합니다")
    if not a.rescore:
        run(a)
    rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"), encoding="utf-8")]
    print(write_summary(out, rows, tol or None, (" " + a.tag) if a.tag else "", use_cc=a.crosscheck))


if __name__ == "__main__":
    main()
