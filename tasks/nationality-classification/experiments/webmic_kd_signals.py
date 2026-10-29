# -*- coding: utf-8 -*-
"""웹마이크 재학습용 교사 신호 — 현재 배포 KD 모델(ONNX 에서 되돌린 체크포인트)의 출력.

왜
    2B 교사는 이 서버에 없다. 대신 **지금 배포된 모델 자신**을 교사로 쓴다 — 목적은 '더 잘하기'가 아니라
    옛 모델이 알던 것(505 도메인·한국어·041/043)을 새 데이터로 학습하는 동안 **잃지 않기**다.
    - train: 학생과 같은 입력(앞 --train_sec 초)으로 logits·pooled·margin.
    - valid/test: 서빙과 같은 앞 12초로 logits — 새 split 에서의 **현재 모델 기준선**이자 131(라벨 없음)
      일치율 평가의 기준.

출력 (--out_dir)
    teacher_train.npz   row · logits(n,41) · pooled(n,1024, fp16) · margin(n)
    teacher_valid.npz / teacher_test.npz   row · logits
    행 번호는 매니페스트 CSV 의 행 순서(0부터). 읽기 실패로 빠진 행은 row 에 없다.

    $PY tasks/nationality-classification/experiments/webmic_kd_signals.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS, XLSR_300M

import argparse, csv, os, time
import numpy as np

WD = NAT_OUTPUTS / "webmic_kd"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(WD))
    ap.add_argument("--ckpt", default=str(WD / "init_kd_from_onnx.pt"))
    ap.add_argument("--xlsr300", default=str(XLSR_300M))
    ap.add_argument("--out_dir", default=str(WD))
    ap.add_argument("--train_sec", type=float, default=8.0)
    ap.add_argument("--eval_sec", type=float, default=12.0)
    ap.add_argument("--max_batch_sec", type=float, default=600.0)
    ap.add_argument("--splits", default="valid,test,train")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 행 제한")
    ap.add_argument("--reuse_dir", default="", help="train.csv + teacher_train.npz 가 있는 폴더 — 같은 audio_path 는 "
                                                   "재추출하지 않고 가져온다(같은 --train_sec·같은 교사일 때만)")
    ap.add_argument("--reuse_only", action="store_true",
                    help="재사용분만 저장하고 새 행은 추출하지 않는다(GPU 불필요). 학습 스크립트는 신호 없는 라벨 행을 "
                         "CE 만으로 배운다")
    A = ap.parse_args()
    if A.reuse_only:
        assert A.reuse_dir and A.splits == "train", "--reuse_only 는 --reuse_dir 와 --splits train 이 필요하다"

    import torch
    from torch.utils.data import DataLoader
    from country_classifier.dataset import AudioCountryDataset, collate_audio_batch
    from country_classifier.kd_student import (LengthBucketBatchSampler, build_student, forward_pooled,
                                               write_loader_csv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if A.reuse_only:
        labels = [str(x) for x in np.load(os.path.join(A.reuse_dir, "teacher_train.npz"), allow_pickle=True)["labels"]]
    else:
        enc, head, mu, sd, labels, layer = build_student(A.xlsr300, A.ckpt, dev)
        enc.eval(); head.eval()
    c2i = {c: i for i, c in enumerate(labels)}

    for split in A.splits.split(","):
        rows = list(csv.DictReader(open(os.path.join(A.data_dir, f"{split}.csv"), encoding="utf-8")))
        if A.limit:
            rows = rows[:A.limit]
        sec = A.train_sec if split == "train" else A.eval_sec
        reuse = {}
        if split == "train" and A.reuse_dir:
            old_rows = list(csv.DictReader(open(os.path.join(A.reuse_dir, "train.csv"), encoding="utf-8")))
            zr = np.load(os.path.join(A.reuse_dir, "teacher_train.npz"), allow_pickle=True)
            assert float(zr["sec"]) == sec and [str(x) for x in zr["labels"]] == labels, "재사용 신호의 조건이 다르다"
            r_log, r_pool = zr["logits"], zr["pooled"]
            for j, i in enumerate(zr["row"]):
                reuse[old_rows[i]["audio_path"]] = (r_log[j], r_pool[j])
            print(f"  [{split}] 재사용 후보 {len(reuse):,}행", flush=True)
        todo = [i for i, r in enumerate(rows) if r["audio_path"] not in reuse]
        R, L, P = [], [], []
        hit = [i for i, r in enumerate(rows) if r["audio_path"] in reuse]
        if hit:
            R.append(np.array(hit))
            L.append(np.stack([reuse[rows[i]["audio_path"]][0] for i in hit]))
            P.append(np.stack([reuse[rows[i]["audio_path"]][1] for i in hit]))
            print(f"  [{split}] 재사용 {len(hit):,}행 · 새로 추출 {len(todo):,}행", flush=True)
        if A.reuse_only:
            print(f"  [{split}] --reuse_only: 새 행 {len(todo):,}개는 신호 없이 둔다", flush=True)
            todo = []
        sub = [rows[i] for i in todo]
        tmp = write_loader_csv(sub, os.path.join(A.out_dir, f"_sig_{split}.csv"))
        ds = AudioCountryDataset(tmp, 16000, min_duration_sec=0.25, max_duration_sec=sec, skip_errors=True)
        bs = LengthBucketBatchSampler([float(r["duration_sec"] or 0) for r in sub], sec, A.max_batch_sec,
                                      256, shuffle=False)
        ld = DataLoader(ds, batch_sampler=bs, num_workers=8, collate_fn=collate_audio_batch, pin_memory=True)
        t0 = time.time(); done = 0
        with torch.no_grad():
            for bi, bt in enumerate(ld):
                if bt is None:
                    continue
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    pooled = forward_pooled(enc, bt["waveforms"].to(dev), bt["padding_mask"].to(dev), layer)
                z = head((pooled - mu) / sd).float()
                R.append(np.array([todo[int(s)] for s in bt["speaker_ids"]]))       # 하위 목록 → 원래 행번호
                L.append(z.cpu().numpy())
                if split == "train":
                    P.append(pooled.cpu().numpy().astype(np.float16))
                done += len(bt["speaker_ids"])
                if bi % 100 == 0:
                    sp = done / max(time.time() - t0, 1e-9)
                    print(f"  [{split}] {done:,}/{len(sub):,} ({sp:.0f} utt/s · 잔여 ~{(len(sub)-done)/max(sp,1e-9)/60:.1f}분)",
                          flush=True)
        row = np.concatenate(R); logits = np.concatenate(L).astype(np.float32)
        order = np.argsort(row, kind="stable")
        row, logits = row[order], logits[order]
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        srt = np.sort(p, 1)
        margin = (srt[:, -1] - srt[:, -2]).astype(np.float16)
        out = {"row": row, "logits": logits, "margin": margin, "labels": np.array(labels, dtype=object),
               "sec": sec}
        if split == "train":
            out["pooled"] = np.concatenate(P).astype(np.float16)[order]
        np.savez(os.path.join(A.out_dir, f"teacher_{split}.npz"), **out)

        # 요약: 라벨 있는 행의 교사 정확도 (도메인별)
        dom = np.array([rows[i]["domain"] for i in row], dtype=object)
        lab = np.array([c2i.get(rows[i]["label_en"], -1) for i in row])
        top5 = np.argsort(-logits, 1)[:, :5]
        print(f"[{split}] 생존 {len(row):,}/{len(rows):,} · {time.time()-t0:.0f}s")
        for d in sorted(set(dom)):
            m = dom == d
            if (lab[m] >= 0).all():
                h1 = (top5[m, 0] == lab[m]).mean(); h5 = (top5[m] == lab[m][:, None]).any(1).mean()
                print(f"    {d:9} n={m.sum():>7,} top1 {h1:.4f} top5 {h5:.4f}")
            else:
                pred = np.array(labels)[top5[m, 0]]
                vals, cnts = np.unique(pred, return_counts=True)
                top = sorted(zip(cnts, vals), reverse=True)[:4]
                print(f"    {d:9} n={m.sum():>7,} (라벨 없음) 교사 top1 분포 " +
                      ", ".join(f"{v} {c / m.sum():.0%}" for c, v in top))


if __name__ == "__main__":
    main()
