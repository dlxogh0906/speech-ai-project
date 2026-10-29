# -*- coding: utf-8 -*-
"""외국인 TAG + 원어민 KsponSpeech 혼합 train 매니페스트 (E2 · E2-F).

비율은 **오디오 시간 기준**이다 — fairseq 가 max_tokens(샘플 수)로 배치를 채우므로 시간 비율이 곧 gradient 몫이다.
    외국인 train 전부 + 원어민 = 외국인 시간 × native_ratio/(1-native_ratio)  (기본 20% → 외국인 200h 면 원어민 50h)
원어민 표본은 --seed 로 결정적이며, 같은 정규화 문장은 한 번만 뽑는다(문장 다양성 우선).
valid · test 계열은 외국인 매니페스트의 것을 그대로 복사한다(평가 split 이 E1 과 같아야 비교된다).
dict.ltr.txt 가 두 매니페스트에서 같은지 assert 한다.

추가 산출: native_train_texts.txt (학습에 들어간 원어민 정규화 문장) → eval_actual_ctc.py --exclude-texts 로
Strict-text-unseen(외국인 train·원어민 train 어디에도 없는 문장) 을 따로 채점한다.

    $PY .../build_mix_manifest.py --foreign $MDD_OUTPUTS/manifest_foreign_clean_v2 \\
        --native $MDD_OUTPUTS/manifest_kspon_native_v1 --native-ratio 0.2 --seed 42 --out $MDD_OUTPUTS/manifest_mix_f80_n20
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, hashlib, json, os, random, shutil, time

SR = 16000


def read(mdir, split):
    tsv = open(os.path.join(mdir, f"{split}.tsv"), encoding="utf-8").read().splitlines()
    ltr = open(os.path.join(mdir, f"{split}.ltr"), encoding="utf-8").read().splitlines()
    meta = [json.loads(l) for l in open(os.path.join(mdir, f"{split}.meta.jsonl"), encoding="utf-8")]
    root, rows = tsv[0], tsv[1:]
    assert len(rows) == len(ltr) == len(meta)
    return root, rows, ltr, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foreign", required=True); ap.add_argument("--native", required=True)
    ap.add_argument("--native-ratio", dest="ratio", type=float, default=0.2, help="혼합 train 에서 원어민 오디오 시간 비율")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"출력 폴더가 비어 있지 않다(덮어쓰기 금지): {a.out}")
    df, dn = (open(os.path.join(m, "dict.ltr.txt"), encoding="utf-8").read() for m in (a.foreign, a.native))
    assert df == dn, "외국인·원어민 dict.ltr.txt 가 다르다"

    froot, frows, fltr, fmeta = read(a.foreign, "train")
    nroot, nrows, nltr, nmeta = read(a.native, "train")
    f_sec = sum(int(r.split("\t")[1]) for r in frows) / SR
    target = f_sec * a.ratio / (1 - a.ratio)
    order = list(range(len(nrows)))
    random.Random(a.seed).shuffle(order)
    picked, seen, n_sec = [], set(), 0.0
    for i in order:
        if n_sec >= target:
            break
        t = nmeta[i]["norm_text"]
        if t in seen:
            continue
        seen.add(t); picked.append(i); n_sec += int(nrows[i].split("\t")[1]) / SR
    picked.sort()

    os.makedirs(a.out, exist_ok=True)
    # 루트가 달라 절대경로로 쓴다 (tsv 첫 줄 "/")
    with open(os.path.join(a.out, "train.tsv"), "w", encoding="utf-8") as ft, \
         open(os.path.join(a.out, "train.ltr"), "w", encoding="utf-8") as fl, \
         open(os.path.join(a.out, "train.source"), "w", encoding="utf-8") as fs:
        ft.write("/\n")
        for r, l in zip(frows, fltr):
            p, fr = r.split("\t")
            ft.write(f"{os.path.join(froot, p).lstrip('/')}\t{fr}\n"); fl.write(l + "\n"); fs.write("foreign\n")
        for i in picked:
            p, fr = nrows[i].split("\t")
            ft.write(f"{os.path.join(nroot, p).lstrip('/')}\t{fr}\n"); fl.write(nltr[i] + "\n"); fs.write("native\n")
    for fn in os.listdir(a.foreign):
        if fn.startswith("train.") or fn in ("stats.json", "stats.md"):
            continue
        shutil.copy2(os.path.join(a.foreign, fn), os.path.join(a.out, fn))
    with open(os.path.join(a.out, "native_train_texts.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(seen)) + "\n")

    ftxt = {m["norm_prompt"] for m in fmeta}
    ut = {json.loads(l)["norm_prompt"] for l in open(os.path.join(a.foreign, "test_unseen.meta.jsonl"), encoding="utf-8")}
    h = hashlib.sha256()
    for ext in ("tsv", "ltr"):
        h.update(open(os.path.join(a.out, f"train.{ext}"), "rb").read())
    stats = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a),
             "foreign": {"utterances": len(frows), "hours": round(f_sec / 3600, 2), "unique_texts": len(ftxt)},
             "native": {"utterances": len(picked), "hours": round(n_sec / 3600, 2), "unique_texts": len(seen),
                        "pool_utterances": len(nrows)},
             "native_hour_ratio": round(n_sec / (n_sec + f_sec), 4),
             "native_utt_ratio": round(len(picked) / (len(picked) + len(frows)), 4),
             "test_unseen_prompts_in_native_train": sorted(ut & seen),
             "train_sha256": h.hexdigest()}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
