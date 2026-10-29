# -*- coding: utf-8 -*-
"""경량화 동등성 점검용 **도메인 표본 매니페스트** — TAG 밖의 두 도메인(LAR 문장 낭독 · 한국인 원어민).

경량화 변형(ONNX fp16 · int8)은 TAG test 에서만 원본과 같다고 확인했다. 서빙 모델 `tag_all_native` 는
LAR·원어민 도메인까지 학습했으므로, 그 도메인에서도 인식이 흔들리지 않는지 따로 본다.
두 도메인 모두 "제대로 읽은 발화"만 고르므로 정답은 문장의 표준발음(KoG2P canonical)이고,
PER 이 낮을수록 좋다(오탐이 적다). 학습 데이터와 겹치지 않게 **Validation 쪽**에서만 뽑는다.

  lar     041·043 VL LAR — `build_lar_mix.py` 와 같은 무오류 규칙(quality ok · 한글만 · 전사=프롬프트)
          records: `build_speech_records.py` 산출물. 오디오: VS_Speech_sound (16 kHz 변환본)
  native  001 자유대화(일반남여) Validation 3.스튜디오 — 전사(stt)가 한글·문장부호뿐인 발화

화자당 최대 --cap 발화 → 전체에서 --n 발화 무작위(seed 고정). 0.5~20초만.
출력: <out>/<domain>/{test.tsv, test.ltr, test.canon, dict.ltr.txt, items.jsonl, stats.json}
      (`decode_manifest_backend.py --split test` 와 `domain_parity_mdd.py` 가 읽는 형식)

    $PY tasks/pronunciation-assessment/scripts/data/build_domain_check_manifest.py --domain lar \
        --out /data/speech-ai/out/compress/mdd/domain_check
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
from shared.paths import DATA_ROOT, MDD_ANALYSIS, MDD_OUTPUTS

import argparse, collections, glob, json, os, random, shutil, wave

H = lambda c: 0xAC00 <= ord(c) <= 0xD7A3


def clean(s):
    return "".join(c for c in s if H(c) or c == " ").strip()


def hangul_only(s):
    return all(H(c) or c in " .?!,~" for c in s.strip())


def wav_frames(path):
    """16 kHz mono int16 이면 프레임 수, 아니면 None."""
    try:
        with wave.open(path, "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                return None
            return w.getnframes()
    except Exception:
        return None


def lar_candidates(aihub):
    """(stem, wav, spk, text) — 041·043 Validation 의 무오류 LAR."""
    vl, wavs = set(), {}
    for g in ("041", "043"):
        base = glob.glob(os.path.join(aihub, f"{g}.*", "01-1.정식개방데이터", "Validation"))
        if len(base) != 1:
            raise SystemExit(f"{g} Validation 폴더를 하나로 못 찾았다: {base}")
        lab = os.path.join(base[0], "02.라벨링데이터", "VL_Speech_lab")
        vl.update(f for f in os.listdir(lab) if "LAR" in f and f.endswith(".json"))
        for r, _, fs in os.walk(os.path.join(base[0], "01.원천데이터", "VS_Speech_sound")):
            for f in fs:
                if "LAR" in f and f.endswith(".wav"):
                    wavs[f[:-4]] = os.path.join(r, f)
    rec = os.path.join(str(MDD_ANALYSIS), "040_041_042_043_both", "records.jsonl")
    out = []
    for line in open(rec, encoding="utf-8"):
        r = json.loads(line)
        if r["file"] not in vl or r["task"] != "LAR" or r.get("quality_flag") != "ok":
            continue
        stem = r["file"][:-5]
        if stem not in wavs:
            continue
        p, t = r.get("prompt", ""), r.get("orthographic_clean", "")
        pr, tr = clean(p), clean(t)
        if not (pr and hangul_only(p) and hangul_only(t)) or pr.replace(" ", "") != tr.replace(" ", ""):
            continue
        out.append((stem, wavs[stem], stem.split("-")[0], pr))
    return out, {"vl_lar_labels": len(vl), "vs_lar_wav": len(wavs)}


def native_candidates(aihub):
    """(stem, wav, spk, text) — 001 Validation 3.스튜디오."""
    root = os.path.join(aihub, "001.자유대화(일반남여)", "01.데이터", "2.Validation")
    labels, wavs = {}, {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json"):
                labels[f[:-5]] = os.path.join(r, f)
            elif f.endswith(".wav"):
                wavs[f[:-4]] = os.path.join(r, f)
    if not wavs:
        raise SystemExit(f"{root} 에 wav 가 없다 — 109 VS 3.스튜디오 음성을 먼저 받는다")
    out, n_bad = [], 0
    for stem, jp in sorted(labels.items()):
        if stem not in wavs:
            continue
        try:
            j = json.load(open(jp, encoding="utf-8"))
            stt, spk = j["발화정보"]["stt"], j["녹음자정보"]["recorderId"]
        except Exception:
            n_bad += 1
            continue
        if not (clean(stt) and hangul_only(stt)):
            continue
        out.append((stem, wavs[stem], spk, clean(stt)))
    return out, {"labels": len(labels), "wav": len(wavs), "label_read_fail": n_bad}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=["lar", "native"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--dict_src", default=str(MDD_OUTPUTS / "manifest_tag_rebuilt"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=5, help="화자당 최대 발화")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ensure_fairseq()
    from mdd import pipeline as MD                         # canonical = 서빙과 같은 KoG2P 규칙
    from g2p import readRules, ver_info
    MD.S["ri"], MD.S["ro"] = readRules(ver_info[0], MD.RULEBOOK)   # build_lar_mix.py 와 같은 초기화(모델은 안 올린다)

    cands, src_stats = (lar_candidates if a.domain == "lar" else native_candidates)(a.aihub_root)
    dict_jamo = set(l.split()[0] for l in open(os.path.join(a.dict_src, "dict.ltr.txt"), encoding="utf-8"))
    rng = random.Random(a.seed)
    by_spk = collections.defaultdict(list)
    for c in cands:
        by_spk[c[2]].append(c)
    pool = []
    for spk in sorted(by_spk):
        lst = sorted(by_spk[spk]); rng.shuffle(lst)
        pool.extend(lst[:a.cap])
    rng.shuffle(pool)

    rows, skip = [], collections.Counter()
    for stem, wav, spk, text in pool:
        if len(rows) >= a.n:
            break
        n = wav_frames(wav)
        if n is None:
            skip["not_16k_mono"] += 1; continue
        if not 8000 <= n <= 320000:
            skip["duration"] += 1; continue
        try:
            canon = list(MD.canonical(text)[0])
        except Exception:
            skip["g2p_fail"] += 1; continue
        if not canon or not set(canon) <= dict_jamo:
            skip["dict"] += 1; continue
        rows.append((stem, wav, spk, text, n, canon))

    od = os.path.join(a.out, a.domain)
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/test.tsv", "w") as ft, open(f"{od}/test.ltr", "w", encoding="utf-8") as fl, \
         open(f"{od}/test.canon", "w", encoding="utf-8") as fc, open(f"{od}/items.jsonl", "w", encoding="utf-8") as fi:
        ft.write("/\n")
        for stem, wav, spk, text, n, canon in rows:
            ft.write(f"{os.path.abspath(wav).lstrip('/')}\t{n}\n")
            fl.write(" ".join(canon) + "\n"); fc.write(" ".join(canon) + "\n")
            fi.write(json.dumps({"stem": stem, "spk": spk, "text": text}, ensure_ascii=False) + "\n")
    shutil.copy(os.path.join(a.dict_src, "dict.ltr.txt"), f"{od}/dict.ltr.txt")
    stats = {"domain": a.domain, "source": src_stats, "candidates": len(cands), "speakers": len(by_spk),
             "cap": a.cap, "pool": len(pool), "n": len(rows), "n_spk": len({r[2] for r in rows}),
             "hours": round(sum(r[4] for r in rows) / 16000 / 3600, 2), "skipped": dict(skip), "seed": a.seed}
    json.dump(stats, open(f"{od}/stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if len(rows) < a.n:
        print(f"⚠️ 목표 {a.n} 발화보다 적다({len(rows)})", flush=True)


if __name__ == "__main__":
    main()
