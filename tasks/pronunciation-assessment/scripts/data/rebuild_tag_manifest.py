# -*- coding: utf-8 -*-
"""TAG 매니페스트(manifest_tag_all) 재구성 — 지금 AIHub 배치(040~043 발음평가 zip 을 푼 그대로)에서.

원본 빌더 `build_tag_manifest_combined.py` 는 학습 서버의 옛 폴더 배치와 미리 리샘플해 둔
`mdd_gold_wav16k` 를 전제로 한다. 둘 다 새 머신에는 없어서 같은 규칙으로 다시 만든다.

원본과 같게 맞춘 것 (build_tag_manifest_combined.py · mdd_gold_spec.py)
  - split   md5(화자ID)[:8] % 10 → 0=test · 1=valid · 나머지=train   (화자ID = 파일명 첫 토큰)
  - .ltr    TextGrid "phones" 티어 → 호환 자모(U+3131~3163)만. "" . - spn <unk> 제외.
            인코딩은 utf-16 을 먼저 시도하고 실패하면 utf-8 (원본 순서 그대로)
  - .canon  LAB json RecordingMetadata.phonemic.phones[].phone → 자모만
  - 필터    tag·lab·wav 가 다 있음 · 16 kHz 샘플 수 8,000~240,000 · ltr·canon 이 비어 있지 않음
  - 리샘플  scipy resample_poly(→16 kHz), 채널 평균, clip×32767 → int16 wav
  - 같은 stem 이 여러 소스에 있으면 --sources 순서상 먼저 나온 것

원본과 다른 것
  - 행 순서: 원본은 정렬 안 된 glob 순서라 재현할 수 없다 → stem 정렬.
    **옛 *_preds.jsonl 의 i 와는 호환되지 않는다.** 모델끼리 비교할 땐 이 매니페스트로 새로 디코드한다.
  - 소스 구성: 원본은 041 을 TL 만 썼고 040·042 는 TL/VL 여부가 기록돼 있지 않았다.
    **기본값(041 TL · 043/040/042 TL+VL)으로 문서값과 정확히 일치함을 확인했다**(2026-09-14):
    test 178,049발화 · 프레임 8,192~238,934 / valid 188,570발화 (docs/04_데이터셋_사전.md).
    041 VL 을 넣으면 test 184,538 · valid 195,771 로 어긋난다. 이상 행은 길이 3,463 · 빈 라벨 73(두 split 합).

    $PY tasks/pronunciation-assessment/scripts/data/rebuild_tag_manifest.py --stats --splits test,valid
    $PY tasks/pronunciation-assessment/scripts/data/rebuild_tag_manifest.py --build --splits test \\
        --out $MDD_OUTPUTS/manifest_tag_rebuilt --workers 12
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse, collections, glob, hashlib, json, math, os, re, time, wave
from multiprocessing import Pool

import numpy as np

SKIP_LBL = {"", ".", "-", "spn", "<unk>"}
PARTS = {"TL": ("Training", "TS"), "VL": ("Validation", "VS")}
DEFAULT_SOURCES = "041:TL;043:TL,VL;040:TL,VL;042:TL,VL"   # 원본 SRC 순서 (041 은 TL 만)
MIN_FR, MAX_FR = 8000, 240000


def is_jamo(c):
    return 0x3131 <= ord(c) <= 0x3163


def split_of(spk):
    h = int(hashlib.md5(spk.encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "valid" if h == 1 else "train"


def parse_tag(path):
    """원본 parse_tag 와 같은 규칙. (자모 목록 또는 None, utf-16 으로 읽혔는가)"""
    txt, used16 = None, False
    try:
        with open(path, encoding="utf-16") as f:
            txt = f.read()
        used16 = True
    except Exception:
        try:
            with open(path, encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            return None, False
    m = re.search(r'name = "phones".*?(?=name = "|\Z)', txt, re.S)
    if not m:
        return None, used16
    toks = []
    for lbl in re.findall(r'text = "([^"]*)"', m.group(0)):
        lbl = lbl.strip()
        if lbl in SKIP_LBL:
            continue
        toks += [ch for ch in lbl if is_jamo(ch)]
    return toks, used16


def parse_lab(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    ph = d.get("RecordingMetadata", {}).get("phonemic", {})
    if not ph or "phones" not in ph:
        return None
    toks = []
    for x in ph["phones"]:
        toks += [ch for ch in str(x.get("phone", "")).strip() if is_jamo(ch)]
    return toks


def frames16k(path):
    """리샘플 후 샘플 수 — resample_poly 출력 길이는 ceil(n·up/down)."""
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
    if sr == 16000:
        return n
    g = math.gcd(int(sr), 16000)
    return int(math.ceil(n * (16000 // g) / (sr // g)))


def resample_to(src, dst):
    """mdd_gold_spec.load_resample_16k + resample_worker 와 같은 식. 이미 있으면 그대로 쓴다."""
    if os.path.exists(dst) and os.path.getsize(dst) > 44:
        try:
            with wave.open(dst, "rb") as w:
                return w.getnframes()
        except Exception:
            pass
    from scipy.signal import resample_poly
    with wave.open(src, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a = (a - 128) / 128 if sw == 1 else a / {2: 32768.0, 4: 2147483648.0}[sw]
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != 16000:
        g = math.gcd(int(sr), 16000)
        a = resample_poly(a, 16000 // g, sr // g)
    if a.size < 400:
        return None
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
    os.replace(tmp, dst)
    return len(pcm)


def scan_names(d, suffix):
    out = {}
    if not os.path.isdir(d):
        return out
    for e in os.scandir(d):
        if e.name.endswith(suffix):
            out[e.name[: -len(suffix)]] = e.path
    return out


def collect(root, sources, splits):
    """소스별 tag·lab·wav 색인 → 요청 split 에 드는 후보 (stem, 데이터셋, 파트, tag, lab, wav)."""
    cands, seen, per_src = [], set(), []
    for spec in sources.split(";"):
        code, parts = spec.split(":")
        bases = glob.glob(os.path.join(root, f"{code}.*", "01-1.정식개방데이터"))
        if len(bases) != 1:
            raise SystemExit(f"{code} 데이터셋 폴더를 하나로 못 찾았다: {bases}")
        base = bases[0]
        for part in parts.split(","):
            top, snd = PARTS[part]
            lbl = os.path.join(base, top, "02.라벨링데이터")
            tags = scan_names(os.path.join(lbl, f"{part}_pronunciation_tag"), ".TextGrid")
            labs = scan_names(os.path.join(lbl, f"{part}_pronunciation_lab"), ".json")
            wavs = {}
            for sd in sorted(glob.glob(os.path.join(base, top, "01.원천데이터", f"{snd}_pronunciation_sound*"))):
                if os.path.isdir(sd):
                    wavs.update(scan_names(sd, ".wav"))
            c = collections.Counter(tags=len(tags), labs=len(labs), wavs=len(wavs))
            for st in sorted(tags):
                if st in seen:
                    c["dup"] += 1; continue
                if st not in labs:
                    c["no_lab"] += 1; continue
                if st not in wavs:
                    c["no_wav"] += 1; continue
                sp = split_of(st.split("-")[0])
                seen.add(st)                      # 원본도 split 과 무관하게 먼저 나온 stem 을 잡는다
                if sp not in splits:
                    continue
                cands.append((st, code, part, tags[st], labs[st], wavs[st], sp))
            per_src.append({"source": f"{code}:{part}", **c})
            print(f"  [{code}:{part}] tag {c['tags']:,} · lab {c['labs']:,} · wav {c['wavs']:,} · "
                  f"lab없음 {c['no_lab']:,} · wav없음 {c['no_wav']:,} · 중복 {c['dup']:,}", flush=True)
    return cands, per_src


def check_one(item):
    st, code, part, tag, lab, wav, sp = item
    try:
        fr = frames16k(wav)
    except Exception:
        return item, None, "wav_unreadable", False
    if fr < MIN_FR or fr > MAX_FR:
        return item, None, "length", False
    ltr, used16 = parse_tag(tag)
    canon = parse_lab(lab)
    if not ltr or not canon:
        return item, None, "empty_label", used16
    return item, (fr, ltr, canon), None, used16


def build_one(args):
    item, wav_root = args
    st = item[0]
    dst = os.path.join(wav_root, st[:2], st + ".wav")
    try:
        n = resample_to(item[5], dst)
    except Exception as e:
        return st, None, f"{type(e).__name__}: {e}"
    return st, (dst, n), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stats", action="store_true", help="행 수·화자·시간만 센다 (리샘플 안 함)")
    mode.add_argument("--build", action="store_true", help="16 kHz 리샘플 + 매니페스트 작성")
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--sources", default=DEFAULT_SOURCES, help=f"기본 {DEFAULT_SOURCES}")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--wav_root", default=str(DATA_ROOT / "mdd_gold_wav16k"))
    ap.add_argument("--out", default=None, help="--build 출력 폴더 / --stats 결과 json 경로(선택)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.build and not a.out:
        ap.error("--build 는 --out 이 필요하다")
    splits = set(a.splits.split(","))

    t0 = time.time()
    print(f"[collect] {a.aihub_root} · sources={a.sources} · splits={sorted(splits)}", flush=True)
    cands, per_src = collect(a.aihub_root, a.sources, splits)
    print(f"[collect] 후보 {len(cands):,} ({time.time()-t0:.0f}s)", flush=True)

    rows, reasons, n16 = [], collections.Counter(), 0
    with Pool(a.workers) as pool:
        for i, (item, res, why, used16) in enumerate(pool.imap(check_one, cands, chunksize=256)):
            n16 += used16
            if res is None:
                reasons[why] += 1
            else:
                rows.append((item, res))
            if (i + 1) % 50000 == 0:
                print(f"  ...{i+1:,}/{len(cands):,} 유효 {len(rows):,}", flush=True)

    stats = {"sources": a.sources, "per_source": per_src, "rejected": dict(reasons),
             "tag_read_as_utf16": n16, "splits": {}}
    for sp in sorted(splits):
        rs = [r for r in rows if r[0][6] == sp]
        task = collections.Counter(r[0][0].split("-")[5][:2] for r in rs)
        stats["splits"][sp] = {
            "utts": len(rs), "speakers": len({r[0][0].split("-")[0] for r in rs}),
            "hours": round(sum(r[1][0] for r in rs) / 16000 / 3600, 1),
            "frames_min": min((r[1][0] for r in rs), default=None),
            "frames_max": max((r[1][0] for r in rs), default=None),
            "by_source": dict(collections.Counter(f"{r[0][1]}:{r[0][2]}" for r in rs)),
            "by_task": dict(task)}
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)

    if a.stats:
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=1)
        return 0

    os.makedirs(a.out, exist_ok=True)
    built, fails = {}, []
    with Pool(a.workers) as pool:
        jobs = [(item, a.wav_root) for item, _ in rows]
        for i, (st, res, err) in enumerate(pool.imap_unordered(build_one, jobs, chunksize=64)):
            if res is None or res[1] is None:
                fails.append((st, err or "too_short"))
            else:
                built[st] = res
            if (i + 1) % 20000 == 0:
                print(f"  [resample] {i+1:,}/{len(jobs):,} ({time.time()-t0:.0f}s)", flush=True)

    for sp in sorted(splits):
        n = mismatch = 0
        with open(os.path.join(a.out, f"{sp}.tsv"), "w") as ft, \
             open(os.path.join(a.out, f"{sp}.ltr"), "w", encoding="utf-8") as fl, \
             open(os.path.join(a.out, f"{sp}.canon"), "w", encoding="utf-8") as fc:
            ft.write("/\n")
            for item, (fr_est, ltr, canon) in rows:
                if item[6] != sp or item[0] not in built:
                    continue
                dst, _ = built[item[0]]
                fr = max(0, (os.path.getsize(dst) - 44) // 2)      # 원본 frames_of 와 같은 정의
                if fr != fr_est:
                    mismatch += 1
                if fr < MIN_FR or fr > MAX_FR:
                    continue
                ft.write(f"{dst.lstrip('/')}\t{fr}\n")
                fl.write(" ".join(ltr) + "\n")
                fc.write(" ".join(canon) + "\n")
                n += 1
        stats["splits"][sp]["written"] = n
        stats["splits"][sp]["frames_estimate_mismatch"] = mismatch
        print(f"[write] {sp}: {n:,}행 (길이 추정 불일치 {mismatch})", flush=True)
    stats["resample_failures"] = fails[:50]
    stats["n_resample_failures"] = len(fails)
    with open(os.path.join(a.out, "rebuild_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[완료] {a.out} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
