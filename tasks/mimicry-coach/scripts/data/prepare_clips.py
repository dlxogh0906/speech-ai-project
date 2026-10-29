# -*- coding: utf-8 -*-
"""원작 장면 오디오 → 대사 한 줄짜리 원본 클립(목소리만) + clips.json. 계획서 6.2 사전처리의 자동 부분.

장면마다(대사당 후보 여러 개):
  1. 목소리 분리(separate.Separator — Hybrid Demucs) → 16 kHz 모노 목소리 / 원래 섞인 소리
  2. 정렬 모델로 받아 적기(locate.transcribe) → 대사 표기 후보마다 근사 부분 매칭(locate.find) → 자모 오류율(JER)
     표기 후보 = 배우가 실제로 말한 형태가 철자와 다를 수 있어서(계획서 6.9 line_variants: "마이 무따 아이가" 등)
  3. 가장 잘 맞는 (장면, 표기, 분리본/섞인 소리)를 골라, 받아 적은 토큰 시각으로 앞뒤 다른 단어 사이에서 자르고
     그 안을 분석해 음절 경계 밖 앞 0.15초 · 뒤 0.3초만 남긴다(넓은 구간을 강제 정렬하면 남의 단어에 끌려간다)
  4. 장면 전체 받아 적기에서 자른 구간 안의 토큰만으로 JER 확인 — 0.35 넘으면 ok=false(데모가 안 쓴다)
출처(영상 id·제목·채널·자른 시각)를 clips.json 에 남긴다. 음절 경계 사람 검수(계획서 7.1)는 이 다음 단계다.

⚠️ 원작 음성은 권리자 허락 없이는 내부 시험용으로만 둔다 — 저장소에 넣지 말 것(wav 는 .gitignore),
   데모를 외부(funnel 공개 주소)로 열 때 이 클립을 같이 내보내지 말 것(계획서 7.3).

    TORCH_HOME=/data/speech-ai/hf_cache/torch PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 \\
    HF_HUB_OFFLINE=1 $PY tasks/mimicry-coach/scripts/data/prepare_clips.py \\
        --src-dir /data/speech-ai/out/mimic/original_src --out /data/speech-ai/out/mimic/original_clips_v1
장면 오디오는 yt-dlp 등으로 미리 받아 둔다(<영상 id>.<확장자> + <영상 id>.info.json). --spec 으로 목록을 바꿀 수 있다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, glob, json, os, time

import numpy as np

SPEC = [
    dict(id="hawaii", title="친구 (2001)", variants=["니가 가라 하와이"], sources=["dkPehjszHFE"]),
    dict(id="mai", title="친구 (2001)", variants=["마이 묵었다 아이가", "마이 무따 아이가", "마이 묵따 아이가", "마이 뭇다 아이가"],
         sources=["6ClOTzbXrj4", "Cvn-C7AQyLU"]),
    dict(id="sidabari", title="친구 (2001)", variants=["내가 니 시다바리가", "내는 니 시다바리가", "내가 니 시다바리냐"],
         sources=["lqAStMtrwd0", "WVMILGfcM8E", "JDjwDprAot8", "G-q8vDLdwKI", "J3pIOO4nx0A", "ACBvMPCFHKs"]),
    dict(id="idae", title="타짜 (2006)", variants=["나 이대 나온 여자야"], sources=["7jPNgor1csE", "Pso-sz3V0-o", "FadjAoKc6ls"],
         accept_jer=0.4, note="배경음악 속 끝 '자야'를 정렬 모델이 못 받아 적음 — 스펙트로그램으로 대사 끝까지 말소리 확인(2026-10-12). "
                              "X4kmOtaTU8M 은 코미디 리믹스라 뺌"),
    dict(id="bab", title="살인의 추억 (2003)", variants=["밥은 먹고 다니냐", "밥은 먹고 다니냐고"],
         sources=["i9jR6qmgaf4", "Hi-ilfr7cbw"]),
    dict(id="neona", title="친절한 금자씨 (2005)", variants=["너나 잘하세요"], sources=["-FJ6SmTuEsg", "AriNg372lDg"]),
    dict(id="eoi", title="베테랑 (2015)", variants=["어이가 없네"], sources=["zASuuYQFht4"]),
    dict(id="mwosi", title="곡성 (2016)", variants=["뭣이 중헌디", "뭣이 중한디", "뭣이 중허냐고"], sources=["QGgK1gyf46k", "AyHrh7PrE0Y"]),
    dict(id="seojang", title="범죄와의 전쟁 (2012)", variants=["느그 서장 남천동 살제", "느그 서장 남천동 살지"],
         sources=["TJJWAe0xqVM", "dwgmHBZtDMw"]),
    dict(id="salaitne", title="범죄와의 전쟁 (2012)", variants=["살아있네", "솰아있네"], sources=["IPmazTWInbI"]),
]
JER_OK = 0.35
PAD_FINAL = (0.15, 0.3)      # 음절 경계 밖으로 남길 폭


def tokens_of(text, vocab):
    from mimicry.text import align_tokens, parse_line
    w, s, _ = parse_line(text)
    return [vocab[t] for syl in align_tokens(w, s)[0] for t in syl if t in vocab]


def info_of(src_dir, vid):
    p = os.path.join(src_dir, vid + ".info.json")
    if not os.path.exists(p):
        return dict(video_id=vid)
    j = json.load(open(p, encoding="utf-8"))
    return dict(video_id=vid, url=j.get("webpage_url"), title=j.get("title"), channel=j.get("channel"),
                duration=j.get("duration"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", dest="src_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", help="SPEC 대신 쓸 JSON 목록 [{id,title,variants,sources}]")
    a = ap.parse_args()
    from mimicry import locate
    from mimicry.align import Aligner
    from mimicry.analyze import analyze
    from mimicry.audio import load_audio, save_wav
    from mimicry.separate import Separator, load_stereo

    spec = json.load(open(a.spec, encoding="utf-8")) if a.spec else SPEC
    cache = os.path.join(a.out, "_cache")
    os.makedirs(cache, exist_ok=True)
    al = Aligner()
    sep = None
    clips, t_start = [], time.time()
    for c in spec:
        tried = []
        for vid in c["sources"]:
            files = [f for f in glob.glob(os.path.join(a.src_dir, vid + ".*")) if not f.endswith(".json")]
            if not files:
                print("  [%s] %s 파일 없음" % (c["id"], vid))
                continue
            vp, mp = os.path.join(cache, vid + "_vocals.wav"), os.path.join(cache, vid + "_mix.wav")
            if not os.path.exists(vp):
                sep = sep or Separator()
                voc, rest = sep.split(load_stereo(files[0]))
                save_wav(vp, voc)
                save_wav(mp, np.clip(voc + rest, -1, 1))
            sig = dict(vocals=load_audio(vp), mix=load_audio(mp))
            for kind, x in sig.items():                    # 속삭임·절규는 분리에서 목소리가 깎일 수 있어 둘 다 본다
                hyp = locate.transcribe(al, x)
                for v in c["variants"]:
                    r = locate.find(hyp, tokens_of(v, al.vocab))
                    if r:
                        tried.append(dict(vid=vid, kind=kind, variant=v, t0=r[0], t1=r[1], jer=r[2], hyp=hyp, idx=r[3],
                                          sig=sig))
        if not tried:
            print("[%s] 후보 없음" % c["id"])
            continue
        # 1) 어느 장면·어느 표기인지는 분리본·섞인 소리 통틀어 가장 잘 맞는 것으로(절규처럼 분리본이 흐린 대사가 있다)
        top = min(tried, key=lambda x: (x["jer"], len(x["sig"]["vocals"])))
        # 2) 같은 표기면 목소리 분리본을 우선한다(배경음이 없어 분석이 깨끗하다 — 자리는 달라도 된다: 같은 대사를 여러 번
        #    하는 장면이 있다). 섞인 소리가 0.1 넘게 더 맞을 때만 그쪽
        voc = sorted([x for x in tried if x["kind"] == "vocals" and x["variant"] == top["variant"]],
                     key=lambda x: (x["jer"], len(x["sig"]["vocals"])))
        best = voc[0] if voc and voc[0]["jer"] <= top["jer"] + 0.1 else top
        sr = 16000
        x = best["sig"][best["kind"]]
        hyp, (j0, j1) = best["hyp"], best["idx"]
        # 받아 적은 토큰 시각으로 자른다 — 앞뒤 다른 단어 토큰과의 사이에서
        start = best["t0"] - 0.25
        if j0 > 0:
            start = max(start, (hyp[j0 - 1][1] + best["t0"]) / 2.0)
        end = best["t1"] + 0.7
        for k in range(j1 + 1, len(hyp)):                  # 다음 단어 = 0.12초 넘게 쉰 뒤 나온 토큰(같은 낱말 토큰은 건너뜀)
            if hyp[k][1] - hyp[k - 1][1] >= 0.12:
                end = min(end, hyp[k][1] - 0.12)          # 다음 단어 소리는 그 첫 토큰보다 먼저 시작한다
                break
        start, end = max(0.0, start), min(len(x) / sr, end)
        seg = x[int(start * sr): int(end * sr)]
        A = analyze(seg, best["variant"], al, keep_tracks=False)
        f0 = start + max(0.0, A["syllables"][0]["start"] - PAD_FINAL[0])
        f1 = start + min(len(seg) / sr, A["syllables"][-1]["end"] + PAD_FINAL[1])
        cut = {k: v[int(f0 * sr): int(f1 * sr)] for k, v in best["sig"].items()}
        save_wav(os.path.join(a.out, c["id"] + "_ref.wav"), cut[best["kind"]])
        save_wav(os.path.join(a.out, c["id"] + "_mix.wav"), cut["mix"])
        want = tokens_of(best["variant"], al.vocab)
        # 검사: 장면 전체를 받아 적은 토큰 중 자른 구간 안에 든 것만으로 JER — "대사가 자른 구간에 다 들어갔나".
        # (자른 1~2초만 따로 받아 적으면 정렬 모델이 문맥을 못 봐서 같은 소리도 JER 이 0.07 → 0.57 로 나빠진다)
        got = [k for k, t in hyp if f0 - 0.02 <= t <= f1 + 0.02]
        jer2 = locate.jer(want, got)
        entry = dict(id=c["id"], line=best["variant"], title=c["title"] + " — 원작 발췌 · 내부 시험용",
                     files=dict(ref=c["id"] + "_ref.wav", mix=c["id"] + "_mix.wav"), ref_kind=best["kind"],
                     source=dict(info_of(a.src_dir, best["vid"]), t0=round(f0, 3), t1=round(f1, 3)),
                     qc=dict(jer_search=round(best["jer"], 3), jer=round(jer2, 3), ok=jer2 <= c.get("accept_jer", JER_OK),
                             accept_jer=c.get("accept_jer", JER_OK), note=c.get("note"), sec=round(f1 - f0, 2),
                             tried=[dict(vid=t["vid"], kind=t["kind"], variant=t["variant"], jer=round(t["jer"], 3),
                                         t0=round(t["t0"], 2)) for t in sorted(tried, key=lambda t: t["jer"])[:8]]))
        clips.append(entry)
        print("[%s] %s ← %s(%s) %.1f~%.1fs · 찾기 JER %.2f · 자른 뒤 JER %.2f %s · %.1fs · %.0fs 경과" % (
            c["id"], best["variant"], best["vid"], best["kind"], f0, f1, best["jer"], jer2,
            "ok" if jer2 <= c.get("accept_jer", JER_OK) else "의심", f1 - f0, time.time() - t_start), flush=True)
    json.dump(dict(clips=clips, kind="original-excerpt", note="원작 음성 발췌 — 내부 시험용. 권리 확인 전 외부 공개 금지."),
              open(os.path.join(a.out, "clips.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
