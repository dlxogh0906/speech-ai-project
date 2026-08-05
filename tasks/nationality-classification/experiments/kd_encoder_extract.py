# -*- coding: utf-8 -*-
"""인코더 증류용 교사 신호 재추출 — 2B(L31) 를 서브셋 오디오에 직접 forward.

캐시 행↔CSV 순서가 어긋나 있어(재생성) 조각 매칭 대신 재추출을 택했다.
모든 산출물이 **서브셋 행 번호**로 키가 잡히므로 정렬 문제가 원천적으로 없다.

트릭: build_dataloader 는 읽기 실패 클립을 조용히 건너뛰므로, CSV 의 speaker_id 칸에
행 번호를 넣어 배치가 어느 행에서 왔는지 복원한다 (화자 정보는 서브셋 CSV 에 이미 있음).

출력 ($NAT_TASK/outputs/kd_encoder/teacher_signals.npz):
  row      서브셋 행 번호 (생존 행만)
  pooled31 (n,1920) fp16 — 2B layer31 마스크 평균풀링 (clip/nan 은 교사 헤드 규약대로)
  logits   (n,41) fp32 — 0702 교사 헤드, **학생 클래스(영문 정렬) 순서로 재배열 완료**
  margin   (n,) fp16 — 교사 확신도 (게이팅 KD 용)

    $PY tasks/nationality-classification/experiments/kd_encoder_extract.py          # ~70분 GPU
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), str(_r / "tasks/nationality-classification/experiments"),
                 *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_TASK, NAT_OUTPUTS, FAIRSEQ_ROOT

import argparse, csv, json, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(NAT_TASK / "outputs/kd_encoder/train_subset.csv"))
    ap.add_argument("--xlsr_ckpt", default=f"{DATA_ROOT}/models/xlsr2_2B_1000k_inference.pt",
                    help="옵티마이저 제거된 추론용 2B (8.2GB)")
    ap.add_argument("--teacher_head", default=f"{NAT_OUTPUTS}/0702_country41_sent_best/head.pt")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 행 제한")
    ap.add_argument("--out", default=str(NAT_TASK / "outputs/kd_encoder/teacher_signals.npz"))
    A = ap.parse_args()

    import torch
    from country_classifier.dataset import build_dataloader
    from country_classifier.model import build_model

    # ── 서브셋 → 로더 호환 CSV (speaker_id 칸 = 행번호) ──
    rows = list(csv.DictReader(open(A.subset, encoding="utf-8")))
    if A.limit: rows = rows[:A.limit]
    classes = sorted({r["label_en"] for r in rows} | {"Korea"})
    # 클래스 집합은 교사 헤드와 대조해 최종 확정 (아래에서 assert)
    tmp = str(NAT_TASK / "outputs/kd_encoder/_loader_tmp.csv")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"])
        for i, r in enumerate(rows):
            w.writerow([r["audio_path"], "", str(i), r["label_en"], 0, r["duration_sec"]])
    print(f"[subset] {len(rows):,}행 → {tmp}", flush=True)

    # ── 교사 헤드 (mu/sd/W/b) + 학생 클래스 순서로 재배열 준비 ──
    th = torch.load(A.teacher_head, map_location="cpu")
    ko2en = json.load(open(NAT_TASK / "configs/country41_ko2en.json", encoding="utf-8"))
    t_en = [ko2en[l] for l in th["labels"]]
    classes = sorted(t_en)                                   # 41 클래스 확정 (영문 정렬)
    assert {r["label_en"] for r in rows} <= set(classes), "서브셋에 교사 밖 클래스"
    reorder = [t_en.index(c) for c in classes]
    W = th["head"]["weight"].numpy().astype(np.float32)[reorder]
    b = th["head"]["bias"].numpy().astype(np.float32)[reorder]
    mu = np.asarray(th["mu"], np.float32); sd = np.asarray(th["sd"], np.float32)

    print(f"[model] 2B 로드: {A.xlsr_ckpt}", flush=True)
    model = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=A.xlsr_ckpt,
                        fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).cuda().eval()

    def pool(x_tbc, pad):
        x = x_tbc.transpose(0, 1)
        if pad is None: return x.mean(1)
        v = (~pad).to(x.dtype).unsqueeze(-1)
        return (x * v).sum(1) / v.sum(1).clamp_min(1.0)

    loader = build_dataloader(tmp, 16000, A.batch_size, shuffle=False, seed=0,
                              max_duration_sec=A.max_sec, num_workers=8)
    R, P = [], []
    done = 0; t0 = time.time()
    with torch.no_grad():
        for bi, bt in enumerate(loader):
            if bt is None: continue
            wav = bt["waveforms"].cuda(); pm = bt["padding_mask"].cuda()
            with torch.autocast("cuda"):
                res = model.encoder.extract_features(source=wav, padding_mask=pm, mask=False)
            x = pool(res["layer_results"][31][0], res.get("padding_mask"))
            x = torch.nan_to_num(x.float(), posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
            P.append(x.cpu().numpy())
            R.extend(int(s) for s in bt["speaker_ids"])          # 행번호 복원
            done += wav.size(0)
            if bi % 100 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{len(rows):,} ({done/max(el,1e-9):.0f} utt/s · "
                      f"잔여 ~{(len(rows)-done)/max(done/max(el,1e-9),1e-9)/60:.0f}분)", flush=True)
    X = np.concatenate(P, 0)
    row = np.array(R, np.int64)
    Z = (X - mu) / sd
    logits = Z @ W.T + b
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    srt = np.sort(p, 1); margin = (srt[:, -1] - srt[:, -2]).astype(np.float16)

    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    np.savez_compressed(A.out, row=row, pooled31=X.astype(np.float16),
                        logits=logits.astype(np.float32), margin=margin,
                        classes=np.array(classes, dtype=object))
    # 스모크 sanity: 서브셋 라벨로 교사 정확도 (도메인별)
    lab = np.array([rows[i]["label_en"] for i in row], dtype=object)
    dom = np.array([rows[i]["domain"] for i in row], dtype=object)
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in lab])
    acc = (logits.argmax(1) == y)
    print(f"\n[sanity] 교사@서브셋 top1 {acc.mean():.4f} (참고: 학습화자 포함이라 낙관적)")
    for d in ["asia", "euro", "korean", "asia043"]:
        m = dom == d
        if m.sum(): print(f"  {d:8} n={int(m.sum()):>7,} top1={acc[m].mean():.4f}")
    print(f"[saved] {A.out} · 생존 {len(row):,}/{len(rows):,}행 · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
