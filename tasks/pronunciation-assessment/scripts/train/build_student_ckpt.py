# -*- coding: utf-8 -*-
"""발음 진단 학생 모델 초기화 — 교사(24층) 체크포인트에서 층 일부를 골라 K층 체크포인트를 만든다.

경량화 4단계의 출발점. 만들어진 파일은 **기존 코드가 그대로 읽는 형식**이다
(`checkpoint_utils.load_model_ensemble_and_task` — `mdd_eval.Decoder`, 평가 스크립트, `deploy/mdd-api`).

어떻게
    - 슬림 체크포인트의 `cfg.model.w2v_args` 는 완전한 설정이라 `w2v_path`(사전학습 백본)를 열지 않는다.
      거기서 `encoder_layers` 만 K 로 바꾼다. (cfg 를 안 바꾸고 모델만 자르면 strict 로드가 실패한다)
    - 가중치: 층이 아닌 키(conv 특징추출기 · pos_conv · 최종 layer_norm · CTC proj)는 그대로,
      `...encoder.layers.{keep[j]}.*` → `...encoder.layers.{j}.*` 로 옮긴다.
    - 옵티마이저 상태는 넣지 않는다(교사 슬림본에도 없다).
    - ⚠️ 모델을 `copy.deepcopy` 하지 말 것 — torch 2.1 에서 pos_conv 의 옛 weight_norm 때문에 죽는다.
      그래서 state dict 수준에서만 다룬다.

층 고르기 (--preset 또는 --layers)
    odd        1,3,…,23        — 최상층(23)을 남긴다. CTC 헤드·최종 LN 과 붙어 있던 층
    even23     0,2,…,20,23
    bottom_top 0~5 + 18~23
    top        12~23

학습 없이 자르기만 한 모델은 정확도가 크게 떨어진다 — 여기서 재는 건 **어느 조합이 출발점으로 나은가**다.
저장 후 로드 검증을 하고, --probe_wav_dir 을 주면 교사 대비 greedy 편집률도 잰다.

    $PY tasks/pronunciation-assessment/scripts/train/build_student_ckpt.py --preset odd --out student12_odd.pt
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()

import argparse, copy, glob, json, os, re, time

PRESETS = {
    "odd": list(range(1, 24, 2)),
    "even23": list(range(0, 22, 2)) + [23],
    "bottom_top": list(range(0, 6)) + list(range(18, 24)),
    "top": list(range(12, 24)),
}
PRE = "w2v_encoder.w2v_model.encoder.layers."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default=os.environ.get("MDD_SERVE_CKPT"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=sorted(PRESETS))
    g.add_argument("--layers", help="쉼표로 구분한 교사 층 번호 (0-based, 오름차순)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=os.environ.get("MDD_DATA_DIR"), help="dict.ltr.txt 폴더")
    ap.add_argument("--probe_wav_dir", default=None, help="교사 대비 greedy 편집률을 잴 16 kHz wav 폴더")
    ap.add_argument("--probe_n", type=int, default=100)
    a = ap.parse_args()
    if not a.teacher:
        ap.error("--teacher 가 없고 .env 에 MDD_SERVE_CKPT 도 없다")
    keep = PRESETS[a.preset] if a.preset else [int(x) for x in a.layers.split(",")]

    import torch
    raw = torch.load(a.teacher, map_location="cpu")
    w2v_model_cfg = raw["cfg"]["model"]["w2v_args"]["model"]
    n_teacher = int(w2v_model_cfg["encoder_layers"])
    if sorted(set(keep)) != keep or keep[0] < 0 or keep[-1] >= n_teacher:
        raise SystemExit(f"--layers 는 0..{n_teacher-1} 범위의 오름차순 고유 번호여야 한다: {keep}")

    sd = raw["model"]
    new_sd, moved = {}, 0
    pat = re.compile(re.escape(PRE) + r"(\d+)\.(.*)")
    remap = {t: j for j, t in enumerate(keep)}
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            new_sd[k] = v
            continue
        t = int(m.group(1))
        if t in remap:
            new_sd[f"{PRE}{remap[t]}.{m.group(2)}"] = v
            moved += 1

    st = {k: v for k, v in raw.items() if k != "last_optimizer_state"}
    st["cfg"] = copy.deepcopy(raw["cfg"])
    st["cfg"]["model"]["w2v_args"]["model"]["encoder_layers"] = len(keep)
    st["model"] = new_sd
    extra = dict(st.get("extra_state") or {})
    extra["student_of"] = {"teacher": os.path.abspath(a.teacher), "teacher_layers": keep,
                           "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    st["extra_state"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(st, a.out)
    n_params = sum(v.numel() for v in new_sd.values())
    n_teacher_params = sum(v.numel() for v in sd.values())
    print(f"[build] 층 {keep} → {len(keep)}층 · 텐서 {len(new_sd)} (층 텐서 {moved}) · "
          f"파라미터 {n_params/1e6:.1f}M / 교사 {n_teacher_params/1e6:.1f}M · "
          f"{os.path.getsize(a.out)/2**30:.2f} GiB → {a.out}", flush=True)

    # ── 기존 로더로 읽히는지 (Decoder 가 쓰는 것과 같은 호출) ──
    os.environ["MDD_FORCE_CPU"] = os.environ.get("MDD_FORCE_CPU", "0")
    from mdd import mdd_eval
    if a.data_dir:
        os.environ["MDD_DATA_DIR"] = a.data_dir
    mdd_eval.CKPT = a.out
    dec = mdd_eval.Decoder()
    got = len(dec.model.w2v_encoder.w2v_model.encoder.layers)
    assert got == len(keep), (got, len(keep))
    report = {"out": os.path.abspath(a.out), "teacher": os.path.abspath(a.teacher), "teacher_layers": keep,
              "params": n_params, "teacher_params": n_teacher_params, "bytes": os.path.getsize(a.out),
              "load_ok": True, "device": dec.device}

    if a.probe_wav_dir:
        import editdistance
        from mdd.gop_feats_v2 import forward_lprobs
        wavs = sorted(glob.glob(os.path.join(a.probe_wav_dir, "*.wav")))[:a.probe_n]
        mdd_eval.CKPT = a.teacher
        tdec = mdd_eval.Decoder()
        edits = toks = eq = 0
        for p in wavs:
            w = mdd_eval.load_wav(p)
            outs = []
            for d in (tdec, dec):
                lp, Ls = forward_lprobs(d, [w])
                outs.append(lp[0, :Ls[0]].argmax(-1).unique_consecutive().tolist())
            t_seq = [x for x in outs[0] if x != 0]
            s_seq = [x for x in outs[1] if x != 0]
            edits += editdistance.eval(t_seq, s_seq); toks += len(t_seq); eq += t_seq == s_seq
        report["probe"] = {"n": len(wavs), "token_edit_rate_vs_teacher": edits / max(1, toks),
                           "seq_equal_rate": eq / max(1, len(wavs))}
        print(f"[probe] 교사 대비 토큰 편집률 {edits/max(1,toks):.4f} · 완전 일치 {eq}/{len(wavs)} (학습 전)")
    with open(os.path.splitext(a.out)[0] + ".build.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[load] 기존 Decoder 로 로드 OK · {got}층 · {dec.device}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
