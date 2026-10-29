# -*- coding: utf-8 -*-
"""영어 음소 CTC 추가 학습 — 한국인 영어를 "들린 그대로" 받아 적게 만드는 것이 목적.

출발점
  · facebook/wav2vec2-lv-60-espeak-cv-ft(Apache-2.0): 몸통은 그대로 두고 **출력층을 ARPAbet 39 + blank 로 바꿔**
    학습한다(espeak IPA 392종 → 41종). 새 출력층은 같은 소리로 가던 espeak 토큰 가중치의 평균에서 시작한다.
  · 이미 ARPAbet 출력층이 있는 체크포인트(예: en037-v1)를 --init 으로 주면 출력층을 그대로 이어 학습한다.

학습 데이터(--train 경로[:비율], 여러 번 줄 수 있다 — 비율 1 = 한 바퀴에 전부, 0.5 = 무작위 절반, 2 = 두 번)
  · AIHub 037 PRN — 정답 = phonemeText(**AIHub 자동 라벨**, 사람 전사 아님)  `build_037_ctc_manifest.py`
  · 한국인 오류 합성(Kokoro TTS) — 정답 = 실제로 읽힌 소리  `synth_ko_errors_kokoro.py`
    --aug-synth 면 합성 녹음에 무작위 크기·잡음(SNR 10~40 dB)을 넣는다(깨끗한 TTS 와 실제 녹음의 차이 줄이기)

평가(사람 라벨로 판정한다)
  dev037    037 에서 뗀 화자 40명 — 기계 라벨과의 음소 오류율(PER)
  devsynth  합성 검증 문장(학습에 없는 문장) — PER · 넣은 오류를 그대로 적은 비율
  l2arctic  한국인 4명 600문장 **사람 주석**(CC BY-NC · 평가 전용) — 받아쓰기로 오류 판별(P/R/F1·오지적),
            한국인 대표 오류 짝별(ð→d · z→s …) 그대로 적은 비율(재현)과 맞게 읽은 자리에 그 소리를 적은 비율(헛지적)
            기준선: lv60 엄격 P 0.232 / R 0.231 · 대표 오류 재현 0.119 / 헛지적 0.029 ·
                    en037-v1 엄격 P 0.232 / R 0.239 · 재현 0.110 / 헛지적 0.022

산출: <out>/step_<N>/ (HF 형식 + vocab.json — `mdd_en.recognizer.PhoneRecognizer(kind="arpa")` 로 바로 읽힌다)
      <out>/metrics.jsonl

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/en_py38 HF_HOME=... \\
    $PY tasks/pronunciation-assessment/scripts/train/en_finetune_ctc.py --steps 8000     # tmux 에서
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, json, os, random, shutil, time

import numpy as np

SR = 16000
MAJOR = ("DH>D", "Z>S", "TH>S", "F>P", "V>B", "IH>IY", "R>L", "L>R", "EH>AE", "OW>AO")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import torch, torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def norm(x):
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def augment(x, rng):
    """합성 녹음용: 크기 ±6 dB, 70% 확률로 백색·분홍 잡음(SNR 10~40 dB)."""
    x = x * (10 ** (rng.uniform(-6, 6) / 20))
    if rng.random() < 0.7:
        p = float(np.mean(x ** 2)) + 1e-9
        snr = rng.uniform(10, 40)
        n = np.random.RandomState(rng.randint(0, 2 ** 31 - 1)).randn(len(x)).astype(np.float32)
        if rng.random() < 0.5:                                  # 분홍 잡음(대략): 누적합으로 저역을 키운다
            n = np.cumsum(n) * 0.02
            n = n - np.convolve(n, np.ones(400) / 400, mode="same")
        n *= np.sqrt(p / (np.mean(n ** 2) + 1e-9) / (10 ** (snr / 10)))
        x = x + n
    return x


def make_batches(rows, batch_sec, rng):
    """길이순으로 묶어(패딩 낭비를 줄임) 배치마다 '가장 긴 길이 × 개수 ≤ batch_sec', 배치 순서는 섞는다."""
    idx = sorted(range(len(rows)), key=lambda i: rows[i]["sec"])
    out, cur, mx = [], [], 0.0
    for i in idx:
        s = rows[i]["sec"]
        if cur and max(mx, s) * (len(cur) + 1) > batch_sec:
            out.append(cur); cur, mx = [], 0.0
        cur.append(i); mx = max(mx, s)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def epoch_rows(sources, rng):
    """[(행들, 비율)] → 이번 바퀴에 쓸 행. 비율 ≤ 1 은 무작위 일부, > 1 은 반복 + 나머지 무작위."""
    out = []
    for rows, w in sources:
        k = int(round(len(rows) * w))
        while k >= len(rows):
            out += rows; k -= len(rows)
        if k > 0:
            out += rng.sample(rows, k)
    return out


class Data:
    """torch Dataset — 배치 하나를 통째로 돌려준다(DataLoader batch_size=None)."""

    def __init__(self, rows, batches, tok, aug_synth=False, seed=0):
        self.rows, self.batches, self.tok, self.aug, self.seed = rows, batches, tok, aug_synth, seed

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        import torch
        rng = random.Random("%d-%d" % (self.seed, k))
        xs, ys = [], []
        for i in self.batches[k]:
            r = self.rows[i]
            x = load_wav(r["wav"])
            if self.aug and r.get("synthetic"):
                x = augment(x, rng)
            xs.append(norm(x))
            ys.append([self.tok[p] for p in r["phones"].split()])
        T = max(len(x) for x in xs)
        L = max(len(y) for y in ys)
        X = np.zeros((len(xs), T), np.float32)
        M = np.zeros((len(xs), T), np.int64)
        Y = np.full((len(xs), L), -100, np.int64)
        for j, (x, y) in enumerate(zip(xs, ys)):
            X[j, : len(x)] = x; M[j, : len(x)] = 1; Y[j, : len(y)] = y
        return torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(Y)


def build_model(init, names):
    """--init 의 출력층이 이미 ARPAbet(names 와 같은 어휘)이면 그대로, 아니면(espeak) 새 출력층으로 바꾼다."""
    import torch
    from transformers import Wav2Vec2ForCTC
    from mdd_en.phones import ESPEAK_EN
    m = Wav2Vec2ForCTC.from_pretrained(init)
    vocab = json.load(open(os.path.join(init, "vocab.json"), encoding="utf-8"))
    if vocab == dict((p, i) for i, p in enumerate(names)):
        m.config.ctc_loss_reduction = "mean"
        m.config.ctc_zero_infinity = True
        m.freeze_feature_encoder()
        return m
    old_w, old_b = m.lm_head.weight.data.clone(), m.lm_head.bias.data.clone()
    H = old_w.shape[1]
    new = torch.nn.Linear(H, len(names))
    torch.nn.init.normal_(new.weight, std=0.02); torch.nn.init.zeros_(new.bias)
    pad_old = m.config.pad_token_id
    with torch.no_grad():
        new.weight[0] = old_w[pad_old]; new.bias[0] = old_b[pad_old]          # <pad> = blank
        for k, p in enumerate(names):
            if k < 2:
                continue
            rows = [i for t, i in vocab.items() if ESPEAK_EN.get(t) == p and i < old_w.shape[0]]
            if rows:
                new.weight[k] = old_w[rows].mean(0); new.bias[k] = old_b[rows].mean()
    m.lm_head = new
    m.config.vocab_size = len(names)
    m.config.pad_token_id = 0
    m.config.ctc_loss_reduction = "mean"
    m.config.ctc_zero_infinity = True
    m.freeze_feature_encoder()
    return m


def greedy(logits, names):
    ids = logits.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i >= 2:
            out.append(names[i])
        prev = i
    return out


def per(ref, hyp):
    from mdd_en.align import align
    if not ref:
        return 0, 0
    _, d = align(ref, hyp)
    return d, len(ref)


def actual_per_canon(canon, actual):
    """합성 발화: 표준 자리마다 실제로 읽힌 소리(삽입은 버리고, 탈락은 None)."""
    from mdd_en.align import align
    sites, _ = align(canon, actual)
    return [s["actual"] for s in sites]


def evaluate(model, devs, l2, names, device):
    import torch
    from mdd_en.metrics import pair_report, pair_sites, prf, site_eval
    model.eval()
    res = {}

    def run(x):
        return greedy(model(torch.from_numpy(x)[None].to(device)).logits[0].float(), names)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for name, items in devs.items():
            num = den = 0
            prs = []
            for r, x in items:
                hyp = run(x)
                d, n = per(r["phones"].split(), hyp)
                num += d; den += n
                if r.get("synthetic"):
                    c = r["canon"].split()
                    prs += pair_sites(c, actual_per_canon(c, r["phones"].split()), hyp, "arpa")
            res["%s_per" % name] = round(num / max(den, 1), 4)
            if prs:
                pr = pair_report(prs)["_all"]
                res["%s_err_recall" % name] = pr["recall"]
                res["%s_false_alarm" % name] = pr["false_alarm"]
        sites, prs, num, den = [], [], 0, 0
        for r, x in l2:
            hyp = run(x)
            ph = [p for p in r["phones"] if p["canon"] is not None]
            sites += site_eval([p["canon"] for p in ph], [p["actual"] for p in ph],
                               [p["type"] in ("s", "d") and p["actual"] != p["canon"] for p in ph],
                               [p["stressed"] for p in ph], hyp, "arpa")
            prs += pair_sites([p["canon"] for p in ph], [p["actual"] for p in ph], hyp, "arpa")
            d, n = per([p["actual"] for p in r["phones"] if p["actual"] is not None], hyp)
            num += d; den += n
        res["l2_strict"] = prf(sites, 1)
        res["l2_per_vs_human"] = round(num / max(den, 1), 4)
        rep = pair_report(prs)
        res["l2_pairs_all"] = rep["_all"]
        res["l2_pairs"] = dict((k, [rep[k]["recall"], rep[k]["false_alarm"], rep[k]["n_err"]]) for k in MAJOR if k in rep)
    model.train()
    return res


def show(step, r):
    s, p = r["l2_strict"], r["l2_pairs_all"]
    extra = " · ".join("%s %s" % (k, r[k]) for k in sorted(r) if k.endswith("_per") or k.endswith("_recall")
                       or k.endswith("_false_alarm"))
    pairs = " ".join("%s %.2f/%.3f" % (k, v[0], v[1]) for k, v in r["l2_pairs"].items())
    return ("[eval] step %d | L2 사람: P %.3f R %.3f F1 %.3f 오지적 %.3f PER %.3f | 대표오류 재현 %.3f 헛 %.4f | %s\n"
            "        짝별(재현/헛): %s" % (step, s["precision"], s["recall"], s["f1"], s["frr"], r["l2_per_vs_human"],
                                        p["recall"], p["false_alarm"], extra, pairs))


def save(model, out, step, init, names):
    d = os.path.join(out, "step_%d" % step)
    model.save_pretrained(d)
    json.dump(dict((p, i) for i, p in enumerate(names)), open(os.path.join(d, "vocab.json"), "w"), ensure_ascii=False)
    shutil.copy(os.path.join(init, "preprocessor_config.json"), os.path.join(d, "preprocessor_config.json"))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m037 = os.path.join(str(MDD_OUTPUTS), "english", "aihub037_ctc")
    ap.add_argument("--init", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    ap.add_argument("--train", action="append", default=None, help="학습 목록[:비율] — 여러 번(기본: 037 train)")
    ap.add_argument("--dev037", default=os.path.join(m037, "dev.jsonl"))
    ap.add_argument("--dev-synth", dest="dev_synth", default=None, help="합성 검증 목록(split=dev 인 행만 쓴다)")
    ap.add_argument("--aug-synth", dest="aug_synth", action="store_true")
    ap.add_argument("--out", default=os.path.join(str(DATA_ROOT), "checkpoints", "en_ctc_037_v1"))
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", dest="head_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch-sec", dest="batch_sec", type=float, default=160.0)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=1000)
    ap.add_argument("--dev-n", dest="dev_n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="시험용 — 학습·검증·L2 를 앞에서 N개만")
    a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from mdd_en.audio import trim_edges
    from mdd_en.l2arctic import load_speakers
    from mdd_en.phones import ARPA39

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = a.init if os.path.isdir(a.init) else snapshot_download(a.init)
    names = ["<pad>", "<unk>"] + ARPA39
    tok = dict((p, i) for i, p in enumerate(names))
    sources = []
    for spec in (a.train or [os.path.join(m037, "train.jsonl")]):
        path, _, w = spec.partition(":")
        rows = [r for r in read_jsonl(path) if r.get("split", "train") == "train"]
        if a.limit:
            rows = rows[: a.limit * 20]
        sources.append((rows, float(w) if w else 1.0))
        print("[train] %s · %d발화 %.1f h · 비율 %s" % (path, len(rows), sum(r["sec"] for r in rows) / 3600, w or 1),
              flush=True)
    devs = {}
    d037 = read_jsonl(a.dev037)
    d037 = random.Random(1).sample(d037, min(a.limit or a.dev_n, len(d037)))
    devs["dev037"] = [(r, norm(load_wav(r["wav"]))) for r in d037]
    if a.dev_synth:
        ds = [r for r in read_jsonl(a.dev_synth) if r.get("split") == "dev"]
        ds = random.Random(1).sample(ds, min(a.limit or 600, len(ds)))
        devs["devsynth"] = [(r, norm(load_wav(r["wav"]))) for r in ds]
    l2_rows = load_speakers(a.l2arctic)
    l2 = [(r, norm(trim_edges(load_wav(r["wav"])))) for r in (l2_rows[: a.limit] if a.limit else l2_rows)]
    print("[data] 검증 %s · l2arctic %d" % (dict((k, len(v)) for k, v in devs.items()), len(l2)), flush=True)

    model = build_model(init, names).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("lm_head") and p.requires_grad]
    body = [p for n, p in model.named_parameters() if not n.startswith("lm_head") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=0.01)
    base = [a.lr, a.head_lr]

    def lr_at(s):
        return s / max(a.warmup, 1) if s < a.warmup else max(0.0, (a.steps - s) / max(a.steps - a.warmup, 1))

    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), ensure_ascii=False, indent=1)
    mf = open(os.path.join(a.out, "metrics.jsonl"), "a", encoding="utf-8")
    r0 = evaluate(model, devs, l2, names, device)
    r0["step"] = 0
    mf.write(json.dumps(r0, ensure_ascii=False) + "\n"); mf.flush()
    print(show(0, r0), flush=True)

    step, t0, run, ep = 0, time.time(), [], 0
    model.train()
    while step < a.steps:
        train = epoch_rows(sources, rng)
        batches = make_batches(train, a.batch_sec, rng)
        loader = torch.utils.data.DataLoader(Data(train, batches, tok, a.aug_synth, a.seed * 1000 + ep),
                                             batch_size=None, shuffle=False, num_workers=a.workers,
                                             persistent_workers=False, prefetch_factor=4)
        ep += 1
        for X, M, Y in loader:
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(X.to(device), attention_mask=M.to(device), labels=Y.to(device))
            loss = out.loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run.append(float(loss))
            if step % 100 == 0:
                print("[train] step %d loss %.4f lr %.2e  %.0f s/100 · mem %.1f GB" % (
                    step, np.mean(run[-100:]), opt.param_groups[0]["lr"], (time.time() - t0),
                    torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0), flush=True)
                t0 = time.time()
            if step % a.eval_every == 0 or step == a.steps:
                r = evaluate(model, devs, l2, names, device)
                r["step"] = step
                r["ckpt"] = save(model, a.out, step, init, names)
                mf.write(json.dumps(r, ensure_ascii=False) + "\n"); mf.flush()
                print(show(step, r), flush=True)
            if step >= a.steps:
                break
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
