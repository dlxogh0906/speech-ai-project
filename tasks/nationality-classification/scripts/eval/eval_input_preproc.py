# -*- coding: utf-8 -*-
"""추론 전처리로 입력 분포를 학습 데이터 쪽으로 맞추기 — 학습 없이 전처리만 바꿔 test 와 실제 한국인 녹음을 함께 잰다 (2026-10-31).

왜: 학습 목록의 한국 6.3만 행은 Kspon · Zeroth 와 그 웹마이크 흉내 사본뿐이고, 진짜 웹마이크 녹음은 전부 외국인이다.
그래서 모델이 녹음 조건을 국적 단서로 쓰고 실제 한국인 웹마이크 녹음을 외국인으로 부른다(results/46 · 최종 점검 §2.4).
입력을 다듬어(무음 · 음량 · 마이크 주파수 특성 · 정지 잡음) 그 단서를 줄이면 한국인이 돌아오는지, 외국인은 얼마나 잃는지 본다.

변형 (클래스와 무관하게 모든 입력에 같은 규칙):
  raw        원본
  sil        무음 정규화 — 앞뒤 무음을 고정 룸톤으로 (서비스 설정, wav_augment.silence_normalize)
  sil_rms    + 말소리 음량을 학습 한국어와 외국 웹마이크 중앙값의 가운데로
  sil_rmsko  진단: 말소리 음량을 학습 한국어 중앙값으로
  sil_eqmid  + 장기 평균 스펙트럼(LTAS) 모양을 학습 한국어와 외국 웹마이크 평균의 가운데로 — 마이크 주파수 특성 지우기
  sil_eqko   진단: LTAS 모양을 학습 한국어 평균으로 (모델이 스펙트럼 모양을 얼마나 단서로 쓰는지)
  sil_eqweb  진단: LTAS 모양을 외국 웹마이크 평균으로
  sil_dn     + 정지 잡음 빼기 (스펙트럼 게이팅, 잡음 = 주파수별 하위 10% 파워)
  sil_rms_eqmid  + LTAS 가운데 → 음량 (잡음 빼기 없이)
  sil_all    + 잡음 빼기 → LTAS 가운데 → 음량
무음 정규화를 거는 변형은 말소리 구간(앞뒤 여유 포함)만 바꾸고 앞뒤 룸톤은 서비스와 똑같이 둔다. 그 뒤 앞 12초.
기준 LTAS · 음량은 학습 목록에서 무작위로 뽑은 파일로 잰다(--n_ref, 고정 seed).

재는 것:
  test(화자 8:1:1) — 웹 문장 · 웹 단어 · 131 · 한국어 코퍼스 정답률, 외국인을 한국으로 부른 비율.
    sil(서비스 설정) 대비 차이는 화자 복원추출 95% 구간.
  실제 한국인 녹음(--korean_list, 묶음별) — 한국 1위 · 한국 top-5 수, 대신 1위로 나온 나라. 집계만 쓴다(경로 · 파일 이름 없음).
  분포 — 학습 한국어 · 외국 웹마이크 · 131 · 실제 한국인 녹음의 말소리 음량 · 잡음 바닥 · LTAS 모양 차이.

    $PY scripts/eval/eval_input_preproc.py --ckpt <best.pt> --train <train.csv> --test <test.csv> \
        --korean_list <csv: audio_path,label_en,group> --out <폴더> [--variants raw,sil,...] [--check_logits <eval logits.npz>]
"""
# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, csv, json, os, time
import numpy as np

SR = 16000
NFFT, HOP = 512, 128
BAND = (100.0, 7000.0)                 # LTAS 모양을 비교 · 정규화하는 대역
VOICED_DB = 35.0                       # 가장 큰 프레임보다 이만큼 안쪽이면 말소리 프레임
EQ_MAX_DB, EQ_SMOOTH = 12.0, 9         # 보정 한도 · 주파수 방향 이동 평균(칸)
DN_OVER, DN_FLOOR = 1.5, 0.1           # 잡음 빼기 과감도 · 최소 이득(−20 dB)
VARIANTS = ["raw", "sil", "sil_rms", "sil_rmsko", "sil_eqmid", "sil_eqko", "sil_eqweb", "sil_dn", "sil_rms_eqmid", "sil_all"]
KO_TRAIN = ("kspon", "zeroth", "kor_aug_train")
KO_TEST = ("kspon", "kor_aug_test", "zeroth_test")


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_head(path, dev):
    """finetune_full_encoder · kd_student 형식 체크포인트 → (인코더, 헤드, mu, sd, 라벨, 층)."""
    import torch
    from shared.paths import MODEL_ROOT
    from country_classifier.model_zoo import _ff
    ff = _ff()
    ck = torch.load(path, map_location="cpu")
    if ck.get("xh_cfg") or ck.get("mf_cfg"):
        raise SystemExit("운율 · mel 융합 체크포인트는 이 스크립트가 다루지 않는다")
    xlsr = ck.get("xlsr") or str(MODEL_ROOT / ff.BACKBONES[ck.get("backbone", "300m")][0])
    enc = ff.load_backbone(xlsr, ck["layer"], dev)
    enc.load_state_dict(ck["encoder"])
    enc.eval()
    labels = [str(x) for x in ck["labels"]]
    head = torch.nn.Linear(ck["head"]["weight"].shape[1], len(labels)).to(dev)
    head.load_state_dict(ck["head"])
    head.eval()
    mu = torch.as_tensor(np.asarray(ck["mu"], np.float32).reshape(-1)).to(dev)
    sd = torch.as_tensor(np.asarray(ck["sd"], np.float32).reshape(-1)).to(dev)
    return enc, head, mu, sd, labels, int(ck["layer"])


class Pre:
    """GPU 에서 STFT 로 다듬는다. 이득은 시간에 따라 바뀌지 않는 필터(EQ)이거나 프레임별 게이트(잡음 빼기)다."""

    def __init__(self, dev):
        import torch
        self.torch, self.dev = torch, dev
        self.win = torch.hann_window(NFFT, device=dev)
        f = torch.arange(NFFT // 2 + 1, device=dev) * SR / NFFT
        self.band = (f >= BAND[0]) & (f <= BAND[1])
        self.target = {}                       # 이름 → (F,) dB 모양
        self.rms_db = {}                       # 이름 → 목표 말소리 dBFS

    def spec(self, x):
        return self.torch.stft(x, NFFT, HOP, window=self.win, center=True, return_complex=True)

    def inv(self, X, n):
        return self.torch.istft(X, NFFT, HOP, window=self.win, center=True, length=n)

    def _voiced(self, P):
        e = 10 * self.torch.log10(P.sum(0) + 1e-12)
        return e >= e.max() - VOICED_DB

    def shape_db(self, P):
        """말소리 프레임의 평균 파워 → dB, 대역 평균을 0 으로 (모양만)."""
        m = P[:, self._voiced(P)].mean(1)
        d = 10 * self.torch.log10(m + 1e-12)
        return d - d[self.band].mean()

    def stats(self, x):
        """→ (말소리 RMS dB, 잡음 바닥 dB, LTAS 모양 dB (F,)) — 분포 비교용."""
        P = self.spec(x).abs() ** 2
        e = 10 * self.torch.log10(P.sum(0) / (NFFT / 2 * self.win.pow(2).sum()) + 1e-12)   # 프레임 평균 제곱 → dBFS
        v = self._voiced(P)
        voiced = float(e[v].mean())
        floor = float(self.torch.quantile(e, 0.1))
        return voiced, floor, self.shape_db(P)

    def eq(self, x, name):
        X = self.spec(x)
        g = self.target[name] - self.shape_db(X.abs() ** 2)
        k = EQ_SMOOTH
        g = self.torch.nn.functional.avg_pool1d(
            self.torch.nn.functional.pad(g[None, None], (k // 2, k // 2), mode="replicate"), k, 1)[0, 0]
        g = g.clamp(-EQ_MAX_DB, EQ_MAX_DB)
        return self.inv(X * (10 ** (g / 20))[:, None], x.numel())

    def dn(self, x):
        X = self.spec(x)
        P = X.abs() ** 2
        N = self.torch.quantile(P, 0.1, dim=1, keepdim=True)
        G = (1 - DN_OVER * N / (P + 1e-12)).clamp(min=DN_FLOOR)
        G = self.torch.nn.functional.avg_pool1d(G[None], 3, 1, padding=1, count_include_pad=False)[0]
        return self.inv(X * G, x.numel())

    def rms(self, x, name):
        voiced, _, _ = self.stats(x)
        return x * (10 ** ((self.rms_db[name] - voiced) / 20))


def split_sil(w):
    """silence_normalize 와 같은 자르기 — (말소리 구간 + 여유) 만 돌려준다. 말소리가 없으면 None."""
    from country_classifier.wav_augment import _speech_bounds
    b = _speech_bounds(w, 1.5e-3)
    if b is None:
        return None
    return w[max(0, b[0] - int(0.15 * SR)): min(w.size, b[1] + int(0.15 * SR))]


def room_tone():
    """silence_normalize 의 앞 · 뒤 룸톤(고정 seed 라 모든 입력에 같다)."""
    from country_classifier.wav_augment import silence_normalize
    probe = np.zeros(SR, np.float32)
    probe[SR // 4: SR // 2] = 0.1                       # 말소리로 잡힐 구간 하나
    s = silence_normalize(probe)
    core = split_sil(probe)
    k = int(0.9 * SR)
    lead, trail = s[:k], s[k + core.size:]
    assert np.allclose(np.concatenate([lead, core, trail]), s)
    return lead, trail


def input_len(w, core, var):
    n12 = 12 * SR
    if var == "raw" or core is None:
        return min(w.size, n12)
    return min(int(0.9 * SR) + core.size + int(0.7 * SR), n12)


def make_input(pre, w, core, var, tone):
    """파형(np) + 미리 자른 말소리 구간(split_sil) + 변형 이름 → 모델 입력 1차원 텐서 (앞 12초)."""
    torch = pre.torch
    n12 = 12 * SR
    if var == "raw" or core is None:                    # 말소리를 못 찾으면 서비스처럼 원본 그대로
        return torch.from_numpy(np.ascontiguousarray(w[:n12])).to(pre.dev)
    x = torch.from_numpy(np.ascontiguousarray(core)).to(pre.dev)
    if x.numel() >= NFFT:
        if var in ("sil_dn", "sil_all"):
            x = pre.dn(x)
        if var in ("sil_eqmid", "sil_rms_eqmid", "sil_all"):
            x = pre.eq(x, "mid")
        elif var == "sil_eqko":
            x = pre.eq(x, "ko")
        elif var == "sil_eqweb":
            x = pre.eq(x, "web")
        if var in ("sil_rms", "sil_rms_eqmid", "sil_all"):
            x = pre.rms(x, "mid")
        elif var == "sil_rmsko":
            x = pre.rms(x, "ko")
    return torch.cat([tone[0], x.float(), tone[1]])[:n12]


def boot_delta(spk, masks, a, b, n_boot, seed):
    """화자 복원추출로 (b − a) 평균 차이의 95% 구간. masks: 이름 → 행 마스크, a/b: 이름 → 행별 값(0/1)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(spk, return_inverse=True)
    S = uniq.size
    W = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(np.float64)    # (B, S) 화자별 뽑힌 횟수
    out = {}
    for k, m in masks.items():
        cnt = np.bincount(inv[m], minlength=S).astype(np.float64)
        da = np.bincount(inv[m], weights=a[k][m], minlength=S)
        db = np.bincount(inv[m], weights=b[k][m], minlength=S)
        den = W @ cnt
        ok = den > 0
        diff = (W @ (db - da))[ok] / den[ok]
        out[k] = {"delta": round(float((b[k][m].mean() - a[k][m].mean())), 4),
                  "ci": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--train", required=True, help="기준 LTAS · 음량을 잴 학습 목록")
    ap.add_argument("--test", required=True)
    ap.add_argument("--korean_list", default="", help="실제 한국인 녹음 csv (audio_path,label_en,group)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n_ref", type=int, default=600, help="기준을 잴 묶음별 학습 파일 수")
    ap.add_argument("--limit", type=int, default=0, help="test 에서 무작위로 이만큼만 (시험용)")
    ap.add_argument("--skip_test", action="store_true", help="test 는 건너뛰고 실제 한국인 녹음만 (로짓 저장용)")
    ap.add_argument("--keep_sec", type=float, default=20.0, help="test 파일을 앞에서 이만큼만 메모리에 둔다")
    ap.add_argument("--check_logits", default="", help="같은 체크포인트의 저장된 test 로짓 — raw 의 1위 일치율 확인")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--batch_sec", type=float, default=160.0, help="한 배치의 입력 길이 합(초) — 길이순으로 묶고 남는 칸은 패딩 마스크")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_mem_frac", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    import torch
    from country_classifier.audio import load_wav_mono
    from country_classifier.kd_student import forward_pooled
    torch.set_num_threads(A.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if A.gpu_mem_frac > 0 and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(A.gpu_mem_frac)
    os.makedirs(A.out, exist_ok=True)
    variants = [v for v in A.variants.split(",") if v]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit("모르는 변형: %s" % bad)

    def load(p, sec):
        w, sr = load_wav_mono(p, SR)
        return np.asarray(w, np.float32).reshape(-1)[: int(sec * SR)]

    pre = Pre(dev)
    rng = np.random.default_rng(A.seed)

    # ── 기준 LTAS · 음량, 분포 ──
    t0 = time.time()
    tr = read_csv(A.train)
    pools = {"ko": [r for r in tr if r["label_en"] == "Korea" and r["domain"] in KO_TRAIN],
             "web": [r for r in tr if r["label_en"] != "Korea" and r["domain"].startswith("web")],
             "p505": [r for r in tr if r["domain"].startswith("p505")]}
    kor_rows = read_csv(A.korean_list) if A.korean_list else []
    dist, shapes, levels = {}, {}, {}
    with torch.no_grad():
        for g, rows in list(pools.items()) + ([("real_korean", kor_rows)] if kor_rows else []):
            pick = rows if g == "real_korean" else [rows[i] for i in rng.choice(len(rows), min(A.n_ref, len(rows)), replace=False)]
            vs, fs, ss = [], [], []
            for r in pick:
                core = split_sil(load(r["audio_path"], A.keep_sec))
                if core is None or core.size < SR // 2:
                    continue
                v, f, s = pre.stats(torch.from_numpy(core).to(dev))
                vs.append(v); fs.append(f); ss.append(s)
            S = torch.stack(ss)
            shapes[g] = S.mean(0)
            levels[g] = float(np.median(vs))
            dist[g] = {"files": len(vs), "voiced_db_median": round(float(np.median(vs)), 1),
                       "floor_db_median": round(float(np.median(fs)), 1),
                       "snr_db_median": round(float(np.median(np.array(vs) - np.array(fs))), 1)}
    pre.target = {"ko": shapes["ko"], "web": shapes["web"], "mid": (shapes["ko"] + shapes["web"]) / 2}
    pre.rms_db = {"mid": (levels["ko"] + levels["web"]) / 2, "ko": levels["ko"]}
    for g in dist:                                      # LTAS 모양 차이 (대역 평균 |dB|)
        for t in ("ko", "web"):
            dist[g]["ltas_vs_" + t + "_db"] = round(float((shapes[g] - shapes[t])[pre.band].abs().mean()), 2)
    print("[dist] %s · %.0fs" % (json.dumps(dist, ensure_ascii=False), time.time() - t0), flush=True)
    tone = tuple(torch.from_numpy(np.ascontiguousarray(t)).to(dev) for t in room_tone())

    # ── 모델 · 입력 ──
    enc, head, mu, sd, labels, layer = load_head(A.ckpt, dev)
    c2i = {c: i for i, c in enumerate(labels)}
    ki = c2i["Korea"]
    rows = read_csv(A.test)
    if A.skip_test:
        rows = rows[:1]
    sel = np.arange(len(rows))
    if A.limit and A.limit < len(rows):
        sel = np.sort(np.random.default_rng(A.seed).choice(len(rows), A.limit, replace=False))
        rows = [rows[i] for i in sel]
    t0 = time.time()
    waves = [load(r["audio_path"], A.keep_sec) for r in rows]
    kwaves = [load(r["audio_path"], 60.0) for r in kor_rows]
    cores = [split_sil(w) for w in waves]
    kcores = [split_sil(w) for w in kwaves]
    print("[load] test %d · 한국인 녹음 %d · %.0fs" % (len(waves), len(kwaves), time.time() - t0), flush=True)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def run(ws, cores, var):
        """길이순으로 묶어 패딩 마스크와 함께 돌린다(평가와 같은 방식)."""
        L = np.zeros((len(ws), len(labels)), np.float64)
        lens = np.array([input_len(w, c, var) for w, c in zip(ws, cores)])
        order = np.argsort(lens)
        cap = int(A.batch_sec * SR)
        batches, cur = [], []
        for i in order:
            if cur and (len(cur) + 1) * lens[i] > cap:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        with torch.no_grad():
            for b in batches:
                xs = [make_input(pre, ws[i], cores[i], var, tone) for i in b]
                n = max(x.numel() for x in xs)
                x = torch.zeros(len(xs), n, device=dev)
                pm = torch.ones(len(xs), n, dtype=torch.bool, device=dev)
                for j, xi in enumerate(xs):
                    x[j, :xi.numel()] = xi
                    pm[j, :xi.numel()] = False
                with torch.autocast(**amp):
                    pooled = forward_pooled(enc, x, pm, layer)
                L[b] = head((pooled - mu) / sd).double().cpu().numpy()
        return L

    from country_classifier.model_zoo import _ff
    _sys.path.insert(0, str(_r / "tasks/nationality-classification/experiments"))
    score_all = _ff().score_all
    dom = np.array([r["domain"] for r in rows], dtype=object)
    task = np.array([r["task"] for r in rows], dtype=object)
    spk = np.array([r["speaker"] for r in rows], dtype=object)
    y = np.array([c2i.get(r["label_en"], -1) for r in rows])
    web = np.array([d.startswith("web") for d in dom])
    masks = {"web_sent": web & (task != "RW") & (y >= 0), "web_word": web & (task == "RW") & (y >= 0),
             "p505": np.array([d.startswith("p505") for d in dom]) & (y >= 0),
             "ko_corpus": np.isin(dom, KO_TEST) & (y == ki)}
    foreign = (y >= 0) & (y != ki)

    rep = {"ckpt": A.ckpt, "name": A.name, "test": A.test, "n_test": len(rows), "korean_list_rows": len(kor_rows),
           "dist": dist, "rms_target_db": {k: round(v, 1) for k, v in pre.rms_db.items()}, "variants": {}}
    hits = {}
    for var in variants:
        t0 = time.time()
        L = run(waves, cores, var)
        np.savez_compressed(os.path.join(A.out, "logits_%s.npz" % var), L=L.astype(np.float32), labels=np.array(labels))
        top5 = np.argsort(-L, 1)[:, :5]
        h1 = (top5[:, 0] == y).astype(np.float64)
        h5 = (top5 == y[:, None]).any(1).astype(np.float64)
        to_ko = (top5[:, 0] == ki).astype(np.float64)
        hits[var] = {"t1": h1, "t5": h5, "ko": to_ko}
        r = {"score": score_all(rows, L, np.ones(len(rows), bool), c2i)}
        for k, m in masks.items():
            r[k] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4)}
        r["foreign_as_korea"] = {k: round(float(to_ko[m & foreign].mean()), 4) for k, m in
                                 (("web", web), ("web_sent", masks["web_sent"]), ("web_word", masks["web_word"]),
                                  ("p505", masks["p505"]))}
        if kor_rows:
            Lk = run(kwaves, kcores, var)
            np.savez_compressed(os.path.join(A.out, "logits_korean_%s.npz" % var), L=Lk.astype(np.float32),
                                labels=np.array(labels), group=np.array([r_["group"] for r_ in kor_rows]))   # 목록 행 순서 · 경로 없음
            P = np.exp(Lk - Lk.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            tk = np.argsort(-Lk, 1)[:, :5]
            grp = np.array([r_["group"] for r_ in kor_rows], dtype=object)
            r["real_korean"] = {}
            for g in sorted(set(grp)) + ["전체"]:
                m = np.ones(len(grp), bool) if g == "전체" else grp == g
                instead = collections.Counter(labels[j] for j in tk[m, 0] if j != ki)
                r["real_korean"][g] = {"n": int(m.sum()), "korea_top1": int((tk[m, 0] == ki).sum()),
                                       "korea_top5": int((tk[m] == ki).any(1).sum()),
                                       "korea_prob_median": round(float(np.median(P[m, ki])), 4),
                                       "instead": instead.most_common(3)}
        if A.check_logits and var == "raw":
            z = np.load(A.check_logits, allow_pickle=True)
            assert [str(c) for c in z["labels"]] == labels
            ok = z["ok"].astype(bool)[sel]
            r["parity_raw_top1_agree"] = round(float((z["L"][sel][ok].argmax(1) == L[ok].argmax(1)).mean()), 4)
        r["sec"] = round(time.time() - t0)
        rep["variants"][var] = r
        print("[%s] 웹문장 %.4f/%.4f · 외국인→한국(웹) %.4f · 한국 코퍼스 %.4f · 실제 한국인 %s · %ds" % (
            var, r["web_sent"]["top1"], r["web_sent"]["top5"], r["foreign_as_korea"]["web"], r["ko_corpus"]["top1"],
            ("%d/%d" % (r["real_korean"]["전체"]["korea_top1"], r["real_korean"]["전체"]["n"])) if kor_rows else "-",
            r["sec"]), flush=True)
        json.dump(rep, open(os.path.join(A.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ── sil(서비스 설정) 대비 차이 — 화자 복원추출 ──
    if "sil" in hits:
        bm = {"web_sent_top1": masks["web_sent"], "web_sent_top5": masks["web_sent"], "web_word_top1": masks["web_word"],
              "foreign_as_korea_web": web & foreign, "ko_corpus_top1": masks["ko_corpus"]}
        key = {"web_sent_top1": "t1", "web_sent_top5": "t5", "web_word_top1": "t1", "foreign_as_korea_web": "ko",
               "ko_corpus_top1": "t1"}
        base = {k: hits["sil"][key[k]] for k in bm}
        for var in variants:
            if var != "sil":
                rep["variants"][var]["vs_sil"] = boot_delta(spk, bm, base, {k: hits[var][key[k]] for k in bm},
                                                            A.n_boot, A.seed)
    json.dump(rep, open(os.path.join(A.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done] %s" % os.path.join(A.out, "report.json"), flush=True)


if __name__ == "__main__":
    main()
