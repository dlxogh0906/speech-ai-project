# -*- coding: utf-8 -*-
"""웹마이크 재학습용 오디오 인덱스 — AIHub 040~043 · 131(505) · 한국어 원어민.

왜
    배포 KD 모델은 505·041·043·한국어로만 학습돼, 040(영어권)·042(중·일) 화자는 학습에 없었다.
    040~043 웹마이크 녹음을 학습에 넣기 위해 음성 파일 전체를 한 번 훑어 인덱스로 만든다.

무엇을 읽나
    040~043   Training/Validation 의 음성(TS_*/VS_*) 중 라벨 JSON(TL/VL_pronunciation_lab ·
              TL/VL_Speech_lab)이 있는 것. 국적은 SpeakerMetadata.nationality 이며 화자당 최대 3개만
              읽는다(파일명 첫 칸 = UserID, 같은 화자면 메타데이터가 같다 — 불일치는 세어서 보고).
    131       Training 음성(TS_3 일본어 · TS_4 중국어 · TS_5 태국어 · TS_6 기타). 이 서버엔 라벨 JSON 이
              없어서 **라벨 없이** 적는다(그룹 = 화자ID 앞 2글자).
    한국어    KsponSpeech 16 kHz 변환본. 화자 정보가 없어 하위 폴더(1,000발화 묶음)를 화자 대용으로 쓴다.

출력 (--out_dir)
    index_040_043.csv  audio_path,stem,dataset,aihub_split,task,speaker,nationality,language,label_en,duration_sec
    index_131.csv      audio_path,stem,group,speaker,duration_sec
    index_kspon.csv    audio_path,stem,speaker,duration_sec
    index_meta.json    개수·불일치·헤드 밖 국적 통계

길이는 파일 크기로 계산한다(16 kHz · mono · 16bit 확인됨, 헤더 44바이트 가정).

    $PY tasks/nationality-classification/scripts/data/build_webmic_index.py \
        --out_dir /data/speech-ai/out/nationality/webmic_kd
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS, NAT_TASK

import argparse, collections, csv, glob, json, os, time
from multiprocessing import Pool

BYTES_PER_SEC = 16000 * 2
# SpeakerMetadata.nationality → 헤드 라벨 (eval_kd_onnx_variants.py 와 같다)
ALIAS = {"Russian Federation": "Russia", "Iran, Islamic Republic Of": "Iran", "Viet Nam": "Vietnam",
         "United States of America": "United States", "USA": "United States", "UK": "United Kingdom",
         "Korea, Republic Of": "Korea", "Hongkong": "Hong Kong"}


def walk(root, ext):
    """root 아래 ext 파일을 (stem, path, size) 로. os.walk 보다 stat 을 아끼려고 scandir 재귀."""
    out, stack = [], [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.name.endswith(ext):
                    out.append((e.name[: -len(ext)], e.path, e.stat().st_size if ext == ".wav" else 0))
    return out


def _walk_job(a):
    return a, walk(a[1], a[2])


def speaker_meta(paths):
    """JSON 최대 3개에서 (nationality, language, 불일치 여부)."""
    nats, langs = collections.Counter(), collections.Counter()
    for p in paths[:3]:
        try:
            j = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m = j.get("SpeakerMetadata") or {}
        nats[str(m.get("nationality") or "").strip()] += 1
        langs[str(m.get("language") or "").strip()] += 1
    if not nats:
        return None, None, False
    return nats.most_common(1)[0][0], langs.most_common(1)[0][0], len(nats) > 1


def _meta_job(a):
    return a[0], speaker_meta(a[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--kspon_root", default=str(DATA_ROOT / "kspon_wav16k"))
    ap.add_argument("--out_dir", default=str(NAT_OUTPUTS / "webmic_kd"))
    ap.add_argument("--workers", type=int, default=8)
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    t0 = time.time()
    ko2en = json.load(open(NAT_TASK / "configs" / "country41_ko2en.json", encoding="utf-8"))
    head_labels = set(ko2en.values())

    # ── ① 040~043: 라벨·음성 디렉터리 목록 ──
    jobs = []
    for ds in ["040", "041", "042", "043"]:
        base = glob.glob(os.path.join(A.aihub_root, f"{ds}.*", "01-1.정식개방데이터"))[0]
        for sp, lp, ap_ in [("Training", "TL", "TS"), ("Validation", "VL", "VS")]:
            for d in sorted(glob.glob(os.path.join(base, sp, "02.라벨링데이터", f"{lp}_*_lab"))):
                jobs.append((ds, d, ".json", sp))
            for d in sorted(glob.glob(os.path.join(base, sp, "01.원천데이터", f"{ap_}_*"))):
                jobs.append((ds, d, ".wav", sp))
    lab, wavs = {}, []
    with Pool(A.workers) as pool:
        for (ds, d, ext, sp), items in pool.imap_unordered(_walk_job, jobs):
            if ext == ".json":
                for st, p, _ in items:
                    lab[(ds, st)] = p
            else:
                wavs += [(ds, sp, st, p, sz) for st, p, sz in items]
            print(f"  [scan] {ds} {os.path.basename(d):28} {len(items):>9,} ({time.time()-t0:.0f}s)", flush=True)

    # ── ② 화자별 국적 (화자당 JSON 최대 3개, 과제가 섞이도록 고른다) ──
    by_spk = collections.defaultdict(list)
    no_label = collections.Counter()
    for ds, sp, st, p, sz in wavs:
        if (ds, st) in lab:
            by_spk[f"{ds}:{st.split('-')[0]}"].append(lab[(ds, st)])
        else:
            no_label[ds] += 1
    meta_jobs = []
    for k, ps in by_spk.items():
        ps = sorted(ps)
        meta_jobs.append((k, [ps[0], ps[len(ps) // 2], ps[-1]]))
    spk_meta, incons = {}, 0
    with Pool(A.workers) as pool:
        for k, (nat, lang, bad) in pool.imap_unordered(_meta_job, meta_jobs, chunksize=64):
            spk_meta[k] = (nat, lang)
            incons += bad
    print(f"  [meta] 화자 {len(spk_meta):,} · 국적 불일치 화자 {incons} ({time.time()-t0:.0f}s)", flush=True)

    # ── ③ 040~043 인덱스 ──
    outside = collections.Counter()
    n_rows = collections.Counter()
    with open(os.path.join(A.out_dir, "index_040_043.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "stem", "dataset", "aihub_split", "task", "speaker",
                    "nationality", "language", "label_en", "duration_sec"])
        for ds, sp, st, p, sz in sorted(wavs, key=lambda r: (r[0], r[2])):
            if (ds, st) not in lab:
                continue
            parts = st.split("-")
            spk = f"{ds}:{parts[0]}"
            nat, lang = spk_meta.get(spk, (None, None))
            if not nat:
                continue
            lb = ALIAS.get(nat, nat)
            if lb not in head_labels:
                outside[nat] += 1
                lb = ""
            task = "".join(ch for ch in (parts[5] if len(parts) > 5 else "") if ch.isalpha())
            w.writerow([p, st, ds, sp, task, spk, nat, lang, lb,
                        round(max(sz - 44, 0) / BYTES_PER_SEC, 2)])
            n_rows[(ds, bool(lb))] += 1
    print(f"  [040~043] " + " · ".join(f"{d} {n_rows[(d, True)]:,}(헤드 밖 {n_rows[(d, False)]:,})"
                                      for d in ["040", "041", "042", "043"]), flush=True)

    # ── ④ 131 (라벨 없음) ──
    b131 = glob.glob(os.path.join(A.aihub_root, "131.*", "*", "1.Training", "원천데이터*"))
    n131 = collections.Counter()
    with open(os.path.join(A.out_dir, "index_131.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "stem", "group", "speaker", "duration_sec"])
        with Pool(A.workers) as pool:
            dirs = [("131", d, ".wav", "") for b in b131 for d in sorted(glob.glob(os.path.join(b, "*")))]
            for _, items in pool.imap_unordered(_walk_job, dirs):
                for st, p, sz in sorted(items):
                    parts = st.split("_")
                    spk = parts[1] if len(parts) > 1 else parts[0]
                    w.writerow([p, st, spk[:2], f"131:{spk}", round(max(sz - 44, 0) / BYTES_PER_SEC, 2)])
                    n131[spk[:2]] += 1
    print(f"  [131] {dict(n131)} ({time.time()-t0:.0f}s)", flush=True)

    # ── ⑤ KsponSpeech ──
    nks = 0
    with open(os.path.join(A.out_dir, "index_kspon.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "stem", "speaker", "duration_sec"])
        dirs = [("kspon", d, ".wav", "") for d in sorted(glob.glob(os.path.join(A.kspon_root, "KsponSpeech_0*")))]
        with Pool(A.workers) as pool:
            for _, items in pool.imap_unordered(_walk_job, dirs):
                for st, p, sz in sorted(items):
                    w.writerow([p, st, f"KSPON:{os.path.basename(os.path.dirname(p))}",
                                round(max(sz - 44, 0) / BYTES_PER_SEC, 2)])
                    nks += 1
    print(f"  [kspon] {nks:,} ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"rows_040_043": {f"{d}{'' if k else '_헤드밖'}": v for (d, k), v in n_rows.items()},
               "wav_without_label": dict(no_label), "speakers": len(spk_meta),
               "inconsistent_speakers": incons, "outside_head": dict(outside.most_common()),
               "rows_131": dict(n131), "rows_kspon": nks, "elapsed_sec": round(time.time() - t0)},
              open(os.path.join(A.out_dir, "index_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[saved] {A.out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
