# -*- coding: utf-8 -*-
"""발음 진단 학생 모델 학습 — CTC + 지식증류(KD). 경량화 4단계.

학생은 `build_student_ckpt.py` 로 교사 층 일부를 복사해 만든 K층 체크포인트에서 출발한다.
교사(배포본 tag_all_native)는 fp16 · eval · 기울기 없이 **같은 배치에서 온라인으로** 돌린다
(프레임 단위 KD 라 캐시로 저장하기엔 크다).

손실 (문장 수로 정규화 — 원 학습의 sentence_avg 와 같다)
    L = (1-α)·CTC  +  α·T²·Σ_frame w_f·KL(p_teacher^T ‖ p_student^T)  +  β·Σ_frame mean_j(1 - cos(h_s[j], h_t[keep[j]]))
    - CTC 는 fairseq `ctc` criterion 과 같은 식(reduction=sum, zero_infinity).
    - KL 은 **패딩 프레임을 뺀 유효 프레임에서만** 잰다. fairseq get_logits 가 패딩 프레임 로짓에 -inf 를 넣어서
      그대로 쓰면 NaN 이 된다.
    - w_f: 교사 argmax 가 blank 인 프레임의 가중(--kd_blank_weight). CTC 출력은 blank 가 ~88% 라
      그대로 두면 KD 가 "blank 흉내"에 쏠린다.
    - β 항은 학생 j번째 층 출력과 그 층을 복사해 온 교사 층 출력의 코사인 거리(둘 다 1024차원).
    국적 KD(α=0.3 · T=2 · β=0.5)를 출발점으로 쓴다 — tasks/nationality-classification/docs/지식증류_인수인계.md

데이터: 교사·학생 체크포인트와 같은 fairseq `audio_finetuning` 태스크로 `--manifest` 의 {train,valid}.tsv/.ltr 을
       읽는다(길이 버킷 배치 · 발화별 정규화 · 자르기 없음). 폴더에 dict.ltr.txt 가 있어야 한다.
최적화: Adam(0.9,0.98) · tri-stage LR(0.1/0.4/0.5, 끝 ×0.05) · fp16(autocast+GradScaler) · update_freq.
검증: --valid_every 마다 valid 부분집합에서 CTC 손실 · greedy PER(정답 .ltr 대비) · 교사 대비 토큰 편집률.
저장: out/checkpoint_best.pt · checkpoint_last.pt — **학생 체크포인트와 같은 형식**이라 Decoder·평가·배포 코드가
     그대로 읽는다. 이어하기용 상태는 out/trainer_last.pt 에 따로.

⚠️ 채택 게이트는 여기서 판단하지 않는다. 학습이 끝나면 test 매니페스트를 decode_manifest_backend.py 로 디코드해
   mdd_eval_tag71.py(F1 ≥0.95, 오류 F1 유지) · mdd_eval_fixedref.py 로 재고, 점수 회귀기는 **다시 학습**해야 한다
   (체크포인트와 회귀기는 짝이다 — docs/WEIGHTS.md).

    $PY tasks/pronunciation-assessment/scripts/train/train_student_kd.py \\
        --student student12_odd.pt --manifest $MDD_OUTPUTS/manifest_tag_rebuilt --out <폴더> --max_update 40000
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

import argparse, json, math, os, time


def tri_stage_lr(step, max_update, peak, phases=(0.1, 0.4, 0.5), init_scale=0.01, final_scale=0.05):
    """fairseq tri_stage 와 같은 모양: 선형 warmup → 유지 → 지수 감쇠."""
    w = int(phases[0] * max_update)
    h = int(phases[1] * max_update)
    d = max(1, max_update - w - h)
    if step < w:
        return peak * (init_scale + (1 - init_scale) * step / max(1, w))
    if step < w + h:
        return peak
    t = min(1.0, (step - w - h) / d)
    return peak * math.exp(math.log(final_scale) * t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="build_student_ckpt.py 출력")
    ap.add_argument("--teacher", default=os.environ.get("MDD_SERVE_CKPT"))
    ap.add_argument("--manifest", required=True, help="{train,valid}.tsv/.ltr + dict.ltr.txt 폴더")
    ap.add_argument("--train_subset", default="train")
    ap.add_argument("--valid_subset", default="valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--kd_blank_weight", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_update", type=int, default=40000)
    ap.add_argument("--max_tokens", type=int, default=1600000, help="배치당 오디오 샘플 수 상한")
    ap.add_argument("--update_freq", type=int, default=4)
    ap.add_argument("--layerdrop", type=float, default=0.0, help="학생 LayerDrop (교사 설정은 0.1)")
    ap.add_argument("--clip_norm", type=float, default=0.0)
    ap.add_argument("--valid_every", type=int, default=2000)
    ap.add_argument("--valid_max_batches", type=int, default=0, help="검증 배치 상한 (0 = valid 전체)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", default=os.environ.get("MDD_DATA_DIR"),
                    help="체크포인트 로드용 dict.ltr.txt 폴더 (--manifest 에 dict.ltr.txt 가 있으면 그쪽을 쓴다)")
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    import editdistance
    from fairseq import checkpoint_utils, utils

    if not torch.cuda.is_available():
        raise SystemExit("학습은 GPU 가 필요하다")

    # fairseq 원 학습은 모델 전체를 half 로 올리는 fp16 옵티마이저를 쓴다. 여기서는 autocast 를 쓰는데,
    # 그러면 마스킹(apply_mask)의 `index_put(x_half, idx, mask_emb_float32)` 가 dtype 불일치로 죽는다.
    # fairseq 소스는 건드리지 않고 wav2vec2 모듈이 import 한 이름만 바꿔 값을 대상 dtype 으로 맞춘다.
    from fairseq.models.wav2vec import wav2vec2 as _w2v
    _orig_index_put = _w2v.index_put

    def _index_put_cast(tensor, indices, value):
        if torch.is_tensor(value) and value.dtype != tensor.dtype:
            value = value.to(tensor.dtype)
        return _orig_index_put(tensor, indices, value)

    _w2v.index_put = _index_put_cast
    if not os.path.exists(os.path.join(a.manifest, "dict.ltr.txt")):
        raise SystemExit(f"{a.manifest}/dict.ltr.txt 가 없다 — 가중치 번들의 final/pronunciation/dict.ltr.txt 를 복사할 것")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")

    ov = {"data": a.manifest}
    s_models, s_cfg, task = checkpoint_utils.load_model_ensemble_and_task([a.student], arg_overrides=ov)
    student = s_models[0].to(dev).float()
    t_models, _, _ = checkpoint_utils.load_model_ensemble_and_task([a.teacher], arg_overrides=ov)
    teacher = t_models[0].to(dev).half().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    s_raw = torch.load(a.student, map_location="cpu")
    keep = (s_raw.get("extra_state") or {}).get("student_of", {}).get("teacher_layers")
    if a.beta > 0 and not keep:
        raise SystemExit("--beta > 0 인데 학생 체크포인트에 teacher_layers 기록이 없다 (build_student_ckpt.py 로 만들 것)")
    s_enc = student.w2v_encoder.w2v_model.encoder
    s_enc.layerdrop = a.layerdrop
    d = task.target_dictionary
    blank, pad, eos = d.bos(), d.pad(), d.eos()

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler()
    state = {"num_updates": 0, "epoch": 1, "best_per": None, "teacher_valid": None}
    tr_path = os.path.join(a.out, "trainer_last.pt")
    if os.path.exists(tr_path):
        ck = torch.load(tr_path, map_location="cpu")
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        print(f"[resume] update {state['num_updates']} · epoch {state['epoch']}", flush=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), ensure_ascii=False, indent=1)

    task.load_dataset(a.train_subset)
    task.load_dataset(a.valid_subset)
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[data] train {len(task.dataset(a.train_subset)):,} · valid {len(task.dataset(a.valid_subset)):,} · "
          f"학생 {n_s/1e6:.1f}M · 층 {len(s_enc.layers)} (교사 층 {keep})", flush=True)

    def batches(subset, epoch, shuffle):
        it = task.get_batch_iterator(dataset=task.dataset(subset), max_tokens=a.max_tokens,
                                     max_positions=task.max_positions(), ignore_invalid_inputs=True,
                                     required_batch_size_multiple=1, seed=a.seed, num_workers=a.num_workers,
                                     epoch=epoch)
        return it.next_epoch_itr(shuffle=shuffle)

    def ctc_loss(model, net, sample):
        lprobs = model.get_normalized_probs(net, log_probs=True).contiguous()      # (T,B,C) — 패딩 프레임 마스킹 포함
        pm = net.get("padding_mask")
        if pm is not None:
            in_len = (~pm).long().sum(-1)
        else:
            in_len = lprobs.new_full((lprobs.size(1),), lprobs.size(0), dtype=torch.long)
        tmask = (sample["target"] != pad) & (sample["target"] != eos)
        tflat = sample["target"].masked_select(tmask)
        tlen = sample.get("target_lengths", tmask.sum(-1))
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(lprobs.float(), tflat, in_len, tlen, blank=blank, reduction="sum", zero_infinity=True)
        return loss, lprobs, in_len, pm

    def valid_frames(pm, T, B):
        return (~pm).t() if pm is not None else torch.ones(T, B, dtype=torch.bool, device=dev)

    def save_student(name, extra):
        out = {k: v for k, v in s_raw.items() if k != "last_optimizer_state"}
        out["model"] = {k: v.detach().float().cpu() for k, v in student.state_dict().items()}
        es = dict(out.get("extra_state") or {})
        es["student_kd"] = {**extra, "args": vars(a)}
        out["extra_state"] = es
        tmp = os.path.join(a.out, name + ".tmp")
        torch.save(out, tmp)
        os.replace(tmp, os.path.join(a.out, name))

    @torch.no_grad()
    def validate():
        student.eval()
        tot = {"n": 0, "ctc": 0.0, "err": 0, "len": 0, "t_err": 0, "t_len": 0, "tt_err": 0}
        need_teacher = state["teacher_valid"] is None
        for bi, sample in enumerate(batches(a.valid_subset, 1, False)):
            if a.valid_max_batches and bi >= a.valid_max_batches:
                break
            sample = utils.move_to_cuda(sample)
            if not sample:
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                net_s = student(**sample["net_input"])
                net_t = teacher(**sample["net_input"])
            loss, lp_s, in_len, _ = ctc_loss(student, net_s, sample)
            lp_t = teacher.get_normalized_probs(net_t, log_probs=True).float()
            tot["ctc"] += float(loss); tot["n"] += sample["target"].size(0)
            for b in range(sample["target"].size(0)):
                L = int(in_len[b])
                tgt = [int(x) for x in sample["target"][b] if x != pad and x != eos]
                ps = [int(x) for x in lp_s[:L, b].argmax(-1).unique_consecutive() if x != blank]
                pt = [int(x) for x in lp_t[:L, b].argmax(-1).unique_consecutive() if x != blank]
                tot["err"] += editdistance.eval(ps, tgt); tot["len"] += len(tgt)
                tot["tt_err"] += editdistance.eval(ps, pt); tot["t_len"] += len(pt)
                if need_teacher:
                    tot["t_err"] += editdistance.eval(pt, tgt)
        student.train()
        res = {"valid_utts": tot["n"], "ctc_per_sent": tot["ctc"] / max(1, tot["n"]),
               "per": tot["err"] / max(1, tot["len"]), "edit_vs_teacher": tot["tt_err"] / max(1, tot["t_len"])}
        if need_teacher:
            state["teacher_valid"] = {"per": tot["t_err"] / max(1, tot["len"]), "utts": tot["n"]}
        res["teacher_per"] = state["teacher_valid"]["per"]
        return res

    log_f = open(os.path.join(a.out, "train_log.jsonl"), "a", encoding="utf-8")
    student.train()
    acc = {"loss": 0.0, "ctc": 0.0, "kd": 0.0, "hid": 0.0, "sent": 0, "frames": 0}
    t0 = time.time(); last_log = state["num_updates"]; micro = 0
    opt.zero_grad(set_to_none=True)
    done = state["num_updates"] >= a.max_update
    while not done:
        for sample in batches(a.train_subset, state["epoch"], True):
            sample = utils.move_to_cuda(sample)
            if not sample:
                continue
            if hasattr(student, "set_num_updates"):
                student.set_num_updates(state["num_updates"])
            nsent = sample["target"].size(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                net_t = teacher(**sample["net_input"])
            with torch.autocast("cuda", dtype=torch.float16):
                net_s = student(**sample["net_input"])
                ctc, _, _, pm = ctc_loss(student, net_s, sample)
                T_, B_ = net_s["encoder_out"].shape[:2]
                vf = valid_frames(pm, T_, B_)
                zs = net_s["encoder_out"].float()[vf]
                zt = net_t["encoder_out"].float()[vf]
                lps = F.log_softmax(zs / a.temp, -1)
                lpt = F.log_softmax(zt / a.temp, -1)
                kl = (lpt.exp() * (lpt - lps)).sum(-1)
                w = torch.where(zt.argmax(-1) == blank, torch.full_like(kl, a.kd_blank_weight), torch.ones_like(kl))
                kd = (w * kl).sum() * a.temp ** 2
                hid = zs.new_zeros(())
                if a.beta > 0:
                    lr_s, lr_t = net_s.get("layer_results"), net_t.get("layer_results")
                    if not lr_s or not lr_t:
                        raise SystemExit("encoder 출력에 layer_results 가 없다 — --beta 0 으로 돌릴 것")
                    for j, tj in enumerate(keep):
                        hs = lr_s[j][0].float()[vf]
                        ht = lr_t[tj][0].float()[vf]
                        hid = hid + (1 - F.cosine_similarity(hs, ht, dim=-1)).sum()
                    hid = hid / len(keep)
                loss = ((1 - a.alpha) * ctc + a.alpha * kd + a.beta * hid) / nsent
            if not torch.isfinite(loss):
                print(f"[warn] 비유한 손실 — 배치 건너뜀 (ctc {float(ctc):.1f} kd {float(kd):.1f})", flush=True)
                opt.zero_grad(set_to_none=True); micro = 0
                continue
            scaler.scale(loss / a.update_freq).backward()
            acc["loss"] += float(loss) * nsent; acc["ctc"] += float(ctc); acc["kd"] += float(kd)
            acc["hid"] += float(hid); acc["sent"] += nsent; acc["frames"] += int(vf.sum())
            micro += 1
            if micro < a.update_freq:
                continue
            micro = 0
            lr = tri_stage_lr(state["num_updates"], a.max_update, a.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            if a.clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, a.clip_norm)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            state["num_updates"] += 1
            u = state["num_updates"]

            if u % a.log_every == 0:
                el = time.time() - t0
                rec = {"update": u, "epoch": state["epoch"], "lr": lr,
                       "loss": acc["loss"] / max(1, acc["sent"]), "ctc": acc["ctc"] / max(1, acc["sent"]),
                       "kd": acc["kd"] / max(1, acc["sent"]), "hid": acc["hid"] / max(1, acc["sent"]),
                       "ups": (u - last_log) / max(1e-9, el), "scale": scaler.get_scale(),
                       "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                acc = {k: 0 for k in acc}; t0 = time.time(); last_log = u

            if u % a.valid_every == 0 or u >= a.max_update:
                v = validate()
                v["update"] = u
                better = state["best_per"] is None or v["per"] < state["best_per"]
                if better:
                    state["best_per"] = v["per"]
                    save_student("checkpoint_best.pt", {"update": u, "valid": v})
                save_student("checkpoint_last.pt", {"update": u, "valid": v})
                torch.save({"student": student.state_dict(), "optimizer": opt.state_dict(),
                            "scaler": scaler.state_dict(), "state": state}, tr_path + ".tmp")
                os.replace(tr_path + ".tmp", tr_path)
                v["best"] = better
                print("[valid] " + json.dumps(v, ensure_ascii=False), flush=True)
                log_f.write(json.dumps({"valid": v}) + "\n"); log_f.flush()
            if u >= a.max_update:
                done = True
                break
        else:
            state["epoch"] += 1
    log_f.close()
    print(f"[완료] update {state['num_updates']} · best valid PER {state['best_per']} → {a.out}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
