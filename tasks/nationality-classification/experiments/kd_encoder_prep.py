# -*- coding: utf-8 -*-
"""인코더 증류 준비 — 화자 split 고정 + 오디오 경로 기반 학습/평가 매니페스트 생성.

왜 필요한가:
- 지금까지의 KD 는 특징 캐시 행 기준이었다. 인코더 학습에는 **오디오 원본**이 필요한데
  캐시에는 경로가 없고, 소스 CSV 는 추출 후 재생성돼 순서가 다르다(505 는 1,868조각).
  → 조각 매칭 대신 **교사 신호를 서브셋에 대해 재추출**하기로 했다. 그 준비 단계다.
- 화자 분리 배정(assign_speakers seed42)은 인덱스 '순서' 의존이라, CSV 에서 다시 만들면
  배정이 달라진다. **캐시 인덱스 기준 배정을 그대로 계산해 speaker→split 을 JSON 으로
  고정**하고, 이후 모든 단계(추출·학습·평가)가 이 파일만 참조한다 — 헤드 KD 실험과
  같은 시험지를 보장한다.

출력 ($NAT_TASK/outputs/kd_encoder/):
  speaker_split.json   화자 → train/valid/test (+ 생성 근거 메타)
  train_subset.csv     학습 서브셋 (per-class cap, 꼬리는 전량) — audio_path,label_en,speaker,domain
  valid.csv / test.csv 평가용 전량 — 같은 스키마

    $PY tasks/nationality-classification/experiments/kd_encoder_prep.py --cap 6000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), str(_r / "tasks/nationality-classification/experiments"),
                 *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_DATA, NAT_TASK

import argparse, csv, json, os
from collections import Counter, defaultdict
import numpy as np

import kd_2b_to_300m as K
from country_classifier.make_speaker_split import assign_speakers
from country_classifier.make_country40_probe import SPLIT, LANG2C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=6000, help="클래스당 학습 발화 상한 (꼬리는 전량 유지)")
    ap.add_argument("--out_dir", default=str(NAT_TASK / "outputs" / "kd_encoder"))
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)

    # ── ① 캐시 인덱스에서 화자 배정을 '한 번' 계산해 고정 ──
    print("[1/3] 캐시 인덱스 기준 화자 split 고정", flush=True)
    _, _, _, _, sps, lab, dom = K.build_index()
    ko2en = json.load(open(NAT_TASK / "configs" / "country41_ko2en.json", encoding="utf-8"))
    lab_en = np.array([ko2en[l] for l in lab], dtype=object)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab_en)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    json.dump({"_설명": "캐시 인덱스(kd_2b_to_300m.build_index) 순서 기준 assign_speakers seed42."
                        " 헤드 KD 실험과 동일한 시험지. 이후 단계는 전부 이 파일만 참조할 것.",
               "assign": assign},
              open(f"{A.out_dir}/speaker_split.json", "w", encoding="utf-8"), ensure_ascii=False)
    cnt = Counter(assign.values())
    print(f"    화자 {len(assign):,}명 → {dict(cnt)}", flush=True)

    # ── ② CSV 4종 → 오디오 경로 기반 행 목록 (라벨은 헤드 KD 와 동일 규칙) ──
    print("[2/3] CSV 병합 (오디오 경로 기반)", flush=True)
    rows_f = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows_f:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
    def c40(s):
        mt = spk_mt.get(s)
        return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)
    e2l = {r["speaker_id"]: r["label_name"]
           for r in csv.DictReader(open(f"{NAT_DATA}/euro_country40_manifest.csv", encoding="utf-8"))}
    a2l = {}
    for r in csv.DictReader(open(f"{NAT_DATA}/asian_country40_manifest.csv", encoding="utf-8")):
        a2l.setdefault(r["speaker_id"], r["label_name"])

    merged = []           # (audio_path, label_en, speaker_key, domain, duration)
    def add_csv(path, labeler, domain, spk_prefix=""):
        n0 = len(merged)
        for r in csv.DictReader(open(path, encoding="utf-8")):
            s = r["speaker_id"]; l = labeler(s)
            if l is None: continue
            merged.append((r["audio_path"], ko2en[l], spk_prefix + s, domain,
                           float(r.get("duration_sec") or 0)))
        print(f"    {domain:8} {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: {len(merged)-n0:,}행", flush=True)

    for sp in ["train", "valid", "test"]:
        add_csv(f"{NAT_DATA}/aihub_nationality_fine_811/{sp}.csv", c40, "asia")
    add_csv(f"{NAT_DATA}/korean_native_for16s/train.csv", lambda s: "한국어", "korean", spk_prefix="KOR:")
    add_csv(f"{NAT_DATA}/euro_country40_sent/train.csv", lambda s: e2l.get(s), "euro")
    add_csv(f"{NAT_DATA}/asian_country40/train.csv", lambda s: a2l.get(s), "asia043")

    # ── ③ split 적용 + 학습 서브셋 (cap) ──
    print("[3/3] split 적용 + 서브셋", flush=True)
    by_split = {"train": [], "valid": [], "test": [], "": []}
    miss = 0
    for row in merged:
        a = assign.get(row[2])
        if a is None: miss += 1; continue
        by_split[a].append(row)
    print(f"    행: train {len(by_split['train']):,} · valid {len(by_split['valid']):,} · "
          f"test {len(by_split['test']):,} · 배정없음 {miss:,}", flush=True)

    rng = np.random.RandomState(42)
    per_class = defaultdict(list)
    for row in by_split["train"]:
        per_class[row[1]].append(row)
    subset = []
    for c, rr in sorted(per_class.items()):
        if len(rr) > A.cap:
            idx = rng.choice(len(rr), A.cap, replace=False)
            rr = [rr[i] for i in idx]
        subset += rr
    rng.shuffle(subset)
    print(f"    학습 서브셋: {len(subset):,}행 (cap {A.cap}/클래스) · "
          f"총시간 ≈ {sum(r[4] for r in subset)/3600:.0f}h", flush=True)

    hdr = ["audio_path", "label_en", "speaker", "domain", "duration_sec"]
    for name, rows in [("train_subset", subset), ("valid", by_split["valid"]), ("test", by_split["test"])]:
        with open(f"{A.out_dir}/{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in rows: w.writerow(r)
    dist = Counter(r[1] for r in subset)
    json.dump({"cap": A.cap, "train_subset": len(subset), "valid": len(by_split["valid"]),
               "test": len(by_split["test"]), "class_dist": dict(dist)},
              open(f"{A.out_dir}/prep_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {A.out_dir}/ (speaker_split.json · train_subset.csv · valid.csv · test.csv)")


if __name__ == "__main__":
    main()
