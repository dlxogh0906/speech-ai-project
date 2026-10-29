# -*- coding: utf-8 -*-
"""국적 증강 실험 — Qwen3-TTS VoiceDesign 으로 '새 목소리 + 지시한 억양'의 한국어 발화를 만든다 (2026-10-18).

결과 18 · 20 은 실제 화자를 복제(Base · ICL)했다. 억양이 지워졌고, 4개국에만 넣어서 "합성 음색 → 그 4개국"
지름길이 생겼다. 여기서는
  - 새 목소리를 글 설명으로 만든다(VoiceDesign) — 병목인 화자 수를 늘리는 쪽이다
  - 41개 클래스 모두에 같은 양을 넣는다 — 합성 음색이 어느 국가의 단서도 되지 못하게
  - 문장은 AIHub 131 낭독 문장 풀(15~50자)에서 고르고, valid · test 화자가 읽은 문장은 뺀다

  plan    목소리 설명(성별 · 나이 · 음색 · 속도 · 말투 + 국가별 억양)과 문장 배정 → <out>/plan.csv  (gate · aug 두 묶음)
  gen     생성 (qwentts 환경) → 16 kHz PCM16 wav. 길이가 비정상이면 seed 를 바꿔 한 번 더, 그래도 이상하면 버린다
  rows    생성된 wav → 국적 학습 형식 CSV (<out>/synth_<묶음>.csv)
  asr     whisper-large-v3-turbo 로 받아 적어 문자 오류율 (qwentts 환경) → <out>/asr_<묶음>.csv
  gate    판정 logits(finetune_full_encoder.py --eval_only) → 억양 신호: 의도한 국가 top-1/top-5 · 한국 판정 · lift
  build   대조군 매니페스트 + 합성 행(train 에만) → 새 매니페스트 폴더

    Q=/data/speech-ai/envs/qwentts/bin/python   (HF_HOME=/data/speech-ai/hf_cache)
    $PY scripts/data/tts_voicedesign.py plan
    $Q  scripts/data/tts_voicedesign.py gen --split gate
    $PY scripts/data/tts_voicedesign.py rows --split gate
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS, NAT_OUTPUTS

import argparse, collections, csv, json, os, random, re, shutil, time, wave
from pathlib import Path

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
KOREA = "Korea"
FIELDS = ["audio_path", "label_en", "speaker", "domain", "task", "duration_sec"]
COPY = ("valid.csv", "test.csv", "test_koraug_zeroth.csv", "speaker_split.json", "prep_meta.json", "restore_meta.json")
SPLITS = {"gate": (6, 4, 1), "aug": (40, 15, 2)}          # 묶음 → (클래스당 목소리, 목소리당 문장, seed)

# 클래스 → (국적 형용사, 억양을 낼 모국어)
ACCENT = {
    "Bangladesh": ("Bangladeshi", "Bengali"), "Brazil": ("Brazilian", "Brazilian Portuguese"),
    "Bulgaria": ("Bulgarian", "Bulgarian"), "Cambodia": ("Cambodian", "Khmer"), "Chile": ("Chilean", "Chilean Spanish"),
    "China": ("Chinese", "Mandarin Chinese"), "Colombia": ("Colombian", "Colombian Spanish"),
    "Ethiopia": ("Ethiopian", "Amharic"), "France": ("French", "French"), "Germany": ("German", "German"),
    "Hong Kong": ("Hong Kong", "Cantonese"), "India": ("Indian", "Hindi"), "Indonesia": ("Indonesian", "Indonesian"),
    "Iran": ("Iranian", "Persian"), "Italy": ("Italian", "Italian"), "Japan": ("Japanese", "Japanese"),
    "Kazakhstan": ("Kazakh", "Kazakh"), "Kyrgyzstan": ("Kyrgyz", "Kyrgyz"), "Malaysia": ("Malaysian", "Malay"),
    "Mongolia": ("Mongolian", "Mongolian"), "Myanmar": ("Burmese", "Burmese"), "Nepal": ("Nepali", "Nepali"),
    "Pakistan": ("Pakistani", "Urdu"), "Peru": ("Peruvian", "Peruvian Spanish"), "Philippines": ("Filipino", "Tagalog"),
    "Romania": ("Romanian", "Romanian"), "Russia": ("Russian", "Russian"), "Rwanda": ("Rwandan", "Kinyarwanda"),
    "Singapore": ("Singaporean", "Singaporean English"), "Spain": ("Spanish", "Castilian Spanish"),
    "Tajikistan": ("Tajik", "Tajik"), "Thailand": ("Thai", "Thai"), "Turkey": ("Turkish", "Turkish"),
    "Turkmenistan": ("Turkmen", "Turkmen"), "Ukraine": ("Ukrainian", "Ukrainian"),
    "United Kingdom": ("British", "British English"), "United States": ("American", "American English"),
    "Uzbekistan": ("Uzbek", "Uzbek"), "Vietnam": ("Vietnamese", "Vietnamese"), "Zambia": ("Zambian", "Zambian English"),
}
TIMBRE = ["Low-pitched", "Medium-pitched", "High-pitched", "Slightly husky", "Bright, clear", "Deep, resonant", "Soft",
          "Slightly nasal"]
PACE = ["slow", "moderate", "slightly fast"]
TONE = ["calm", "neutral", "cheerful", "careful", "slightly nervous", "confident"]
STRENGTH = ["mild", "noticeable", "strong"]
REGION = ["Seoul", "Gyeonggi Province", "Incheon", "Busan", "Daegu", "Gwangju", "Daejeon"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv(path, fields, rows):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def slug(c):
    return c.replace(" ", "_")


def instruct(c, rng):
    gender = rng.choice(["man", "woman"])
    age = rng.randint(20, 55)
    style = f"{rng.choice(TIMBRE)} voice, {rng.choice(PACE)} pace, {rng.choice(TONE)} tone, reading a sentence aloud."
    if c == KOREA:
        return gender, age, (f"A {age}-year-old native Korean {gender} from {rng.choice(REGION)}, "
                             f"speaking standard Korean naturally. {style}")
    adj, l1 = ACCENT[c]
    return gender, age, (f"A {age}-year-old {adj} {gender} who learned Korean as a second language, speaking Korean "
                         f"with a {rng.choice(STRENGTH)} {l1} accent. {style}")


# ── plan ────────────────────────────────────────────────────────────────

def cmd_plan(a):
    out = Path(a.out)
    _, train = read_csv(Path(a.manifest) / "train.csv")
    classes = sorted({r["label_en"] for r in train})
    missing = [c for c in classes if c != KOREA and c not in ACCENT]
    if missing:
        raise SystemExit(f"억양 표가 없는 클래스: {missing}")
    held = set()                                         # valid · test 화자가 읽은 131 문장은 뺀다
    for name in ("valid.csv", "test.csv", "test_koraug_zeroth.csv"):
        p = Path(a.manifest) / name
        if p.is_file():
            held |= {os.path.splitext(os.path.basename(r["audio_path"]))[0] for r in read_csv(p)[1]}
    pool, drop = set(), set()
    with open(a.pool, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get("prompt") or "").strip()
            (drop if j.get("stem") in held else pool).add(t)
    pool = sorted(t for t in pool - drop if 15 <= len(t) <= 50 and re.fullmatch(r"[가-힣 .,?!~]+", t))
    n_gate = len(classes) * SPLITS["gate"][0] * SPLITS["gate"][1]
    n_aug = len(classes) * SPLITS["aug"][0] * SPLITS["aug"][1]
    if len(pool) <= n_gate:
        raise SystemExit(f"문장 풀 {len(pool):,}종이 gate {n_gate:,}개보다 작다")
    order = pool[:]
    random.Random(0).shuffle(order)
    gate_t, aug_t = order[:n_gate], order[n_gate:]      # gate 문장은 aug 에 다시 쓰지 않는다 — 학습 뒤 gate 점검이 깨끗하게
    print(f"[plan] 문장 풀 {len(pool):,}종 (valid·test 문장 {len(drop):,}종 뺌) · gate {n_gate:,}종(재사용 없음) · "
          f"aug {n_aug:,}발화에 {len(aug_t):,}종(최대 {-(-n_aug // len(aug_t))}번)", flush=True)
    pools = {"gate": iter(gate_t), "aug": iter(aug_t * (n_aug // len(aug_t) + 1))}
    rows = []
    for split, (n_voice, n_text, seed) in SPLITS.items():
        rng, texts = random.Random(seed), pools[split]
        for ci, c in enumerate(classes):
            for v in range(n_voice):
                gender, age, ins = instruct(c, rng)
                voice = f"{split[0]}{ci:02d}{v:02d}"
                for k in range(n_text):
                    rows.append({"split": split, "voice": voice, "label_en": c, "gender": gender, "age": age,
                                 "instruct": ins, "text": next(texts),
                                 "out": str(out / "wav" / split / slug(c) / f"{voice}_{k:02d}.wav")})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "plan.csv", list(rows[0]), rows)
    for split in SPLITS:
        n = sum(r["split"] == split for r in rows)
        print(f"[plan] {split}: {n:,}발화 = 클래스 {len(classes)} × 목소리 {SPLITS[split][0]} × 문장 {SPLITS[split][1]}", flush=True)


# ── gen (qwentts 환경) ───────────────────────────────────────────────────

def bad(sec, text):
    """길이가 문장에 비해 비정상 — 음절당 0.2초 기준으로 3배+2초 넘거나 0.3배 못 미침 (결과 20 과 같은 기준)."""
    e = max(1, len(re.findall(r"[가-힣]", text))) * 0.2
    return sec > 3 * e + 2 or sec < 0.3 * e


def cmd_gen(a):
    import numpy as np, soundfile as sf, librosa, torch
    from qwen_tts import Qwen3TTSModel
    _, plan = read_csv(Path(a.out) / "plan.csv")
    rows = [x for x in plan if x["split"] == a.split]
    # 남은 것만 길이순으로 줄 세워 번갈아 나눈다 — 다시 돌려도 조각마다 양과 길이가 고르다 (같은 순서라야 조각이 안 겹친다)
    left = sorted((r for r in rows if not os.path.exists(r["out"])), key=lambda r: (len(r["text"]), r["out"]))
    todo = left[a.shard::a.nshards]
    print(f"[gen {a.split} {a.shard}/{a.nshards}] 이 조각 {len(todo):,}개 · 전체 남음 {len(left):,}/{len(rows):,}", flush=True)
    if not todo:
        return
    m = Qwen3TTSModel.from_pretrained(a.model, device_map="cuda:0", dtype=torch.bfloat16)

    def run(batch, seed):
        torch.manual_seed(seed)
        ws, sr = m.generate_voice_design(text=[r["text"] for r in batch], instruct=[r["instruct"] for r in batch],
                                         language=["Korean"] * len(batch), max_new_tokens=300)
        torch.cuda.empty_cache()
        return ws, sr

    def save(r, w, sr):
        os.makedirs(os.path.dirname(r["out"]), exist_ok=True)
        w16 = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=16000)
        tmp = r["out"] + ".tmp.wav"
        sf.write(tmp, np.clip(w16, -1, 1), 16000, subtype="PCM_16")
        os.replace(tmp, r["out"])

    t0, done, retry, dropped = time.time(), 0, [], 0
    for bi in range(0, len(todo), a.batch):
        batch = todo[bi:bi + a.batch]
        ws, sr = run(batch, a.shard * 100003 + bi)
        for r, w in zip(batch, ws):
            if bad(len(w) / sr, r["text"]):
                retry.append(r)
            else:
                save(r, w, sr); done += 1
        if (bi // a.batch) % 20 == 0:
            print(f"  {bi + len(batch):,}/{len(todo):,} · 씀 {done:,} · 다시 {len(retry)} · {time.time() - t0:.0f}s", flush=True)
    for bi in range(0, len(retry), a.batch):                 # 한 번만 다시 — 그래도 이상하면 버린다
        batch = retry[bi:bi + a.batch]
        ws, sr = run(batch, 7 + a.shard * 100003 + bi)
        for r, w in zip(batch, ws):
            if bad(len(w) / sr, r["text"]):
                dropped += 1
            else:
                save(r, w, sr); done += 1
    print(f"[gen {a.split} {a.shard}/{a.nshards}] 씀 {done:,} · 버림 {dropped} · {time.time() - t0:.0f}s", flush=True)


# ── rows · build ────────────────────────────────────────────────────────

def cmd_rows(a):
    _, plan = read_csv(Path(a.out) / "plan.csv")
    out_rows, per = [], collections.Counter()
    for r in plan:
        if r["split"] != a.split or not os.path.exists(r["out"]):
            continue
        with wave.open(r["out"]) as w:
            sec = w.getnframes() / w.getframerate()
        out_rows.append({"audio_path": r["out"], "label_en": r["label_en"], "speaker": f"ttsvd_{r['voice']}",
                         "domain": "tts_vd", "task": "TTS", "duration_sec": round(sec, 2)})
        per[r["label_en"]] += 1
    planned = sum(r["split"] == a.split for r in plan)
    hours = sum(float(r["duration_sec"]) for r in out_rows) / 3600
    write_csv(Path(a.out) / f"synth_{a.split}.csv", FIELDS, out_rows)
    print(f"[rows {a.split}] {len(out_rows):,}/{planned:,}발화 · {hours:.1f}시간 · 클래스당 {min(per.values()):,}~"
          f"{max(per.values()):,} → synth_{a.split}.csv", flush=True)


def cmd_build(a):
    src, dst = Path(a.src), Path(a.dst)
    fields, train = read_csv(src / "train.csv")
    _, synth = read_csv(a.synth)
    dst.mkdir(parents=True, exist_ok=True)
    write_csv(dst / "train.csv", fields, train + [{k: r.get(k, "") for k in fields} for r in synth])
    for fn in COPY:
        if (src / fn).is_file():
            shutil.copy2(src / fn, dst / fn)
    per = collections.Counter(r["label_en"] for r in synth)
    meta = {"src": str(src), "synth": str(a.synth), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": {"src": len(train), "synth": len(synth), "dst": len(train) + len(synth)}, "synth_per_class": dict(per),
            "note": "train 에만 합성 행을 더했다. valid · test 는 대조군과 같다."}
    (dst / "tts_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[build] train {len(train):,} + 합성 {len(synth):,} = {len(train) + len(synth):,}행 → {dst}", flush=True)


# ── asr (qwentts 환경) · gate ─────────────────────────────────────────────

def norm(s):
    return re.sub(r"[^가-힣0-9a-zA-Z]", "", s)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def cmd_asr(a):
    import soundfile as sf, torch
    from transformers import pipeline
    _, plan = read_csv(Path(a.out) / "plan.csv")
    rows = [r for r in plan if r["split"] == a.split and os.path.exists(r["out"])]
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16,
                   device="cuda:0")
    out = []
    for s in range(0, len(rows), 16):
        batch = rows[s:s + 16]
        res = asr([sf.read(r["out"], dtype="float32")[0] for r in batch], batch_size=16,
                  generate_kwargs={"language": "korean", "task": "transcribe"})
        for r, o in zip(batch, res):
            hyp = o["text"].strip()
            out.append({"audio_path": r["out"], "label_en": r["label_en"], "text": r["text"], "asr": hyp,
                        "cer": f"{cer(r['text'], hyp):.3f}"})
    write_csv(Path(a.out) / f"asr_{a.split}.csv", list(out[0]), out)
    print(f"[asr {a.split}] {len(out):,}발화 → asr_{a.split}.csv", flush=True)


def cmd_gate(a):
    import numpy as np
    _, rows = read_csv(Path(a.out) / f"synth_{a.split}.csv")
    asr = {}
    p = Path(a.out) / f"asr_{a.split}.csv"
    if p.is_file():
        asr = {r["audio_path"]: float(r["cer"]) for r in read_csv(p)[1]}
    lines = [f"# VoiceDesign 억양 점검 — {a.split} {len(rows):,}발화", "",
             "| 판정 모델 | 의도한 국가 top-1 / top-5 (외국 40개국 평균) | 한국으로 판정 | 한국 빼고 40개국 중 top-1 | lift | 한국 목소리 → 한국 |",
             "|---|---|--:|--:|--:|--:|"]
    res = {}
    for spec in a.models:
        name, npz = spec.split("=", 1)
        d = np.load(npz, allow_pickle=True)
        labels = [str(x) for x in d["labels"]]
        if len(d["ok"]) != len(rows):
            raise SystemExit(f"{npz}: 행 수가 synth_{a.split}.csv 와 다르다")
        c2i = {c: i for i, c in enumerate(labels)}
        ok = d["ok"].astype(bool)
        L = d["L"]
        y = np.array([c2i[r["label_en"]] for r in rows])
        ko = c2i[KOREA]
        top5 = np.argsort(-L, 1)[:, :5]
        hit1, hit5 = top5[:, 0] == y, (top5 == y[:, None]).any(1)
        Lnk = L.copy(); Lnk[:, ko] = -np.inf
        hit1nk = Lnk.argmax(1) == y
        foreign = sorted({int(v) for v in y[ok] if v != ko})
        per = {}
        own, other = [], []
        for c in foreign:
            m = ok & (y == c)
            o = ok & (y != c) & (y != ko)
            if not m.any() or not o.any():
                continue
            per[labels[c]] = {"n": int(m.sum()), "top1": float(hit1[m].mean()), "top5": float(hit5[m].mean()),
                              "korea": float((top5[m, 0] == ko).mean())}
            own.append((top5[m] == c).any(1).mean()); other.append((top5[o] == c).any(1).mean())
        fm = ok & (y != ko)
        km = ok & (y == ko)
        r = {"top1": float(np.mean([v["top1"] for v in per.values()])), "top5": float(np.mean([v["top5"] for v in per.values()])),
             "korea": float((top5[fm, 0] == ko).mean()), "top1_no_korea": float(hit1nk[fm].mean()),
             "lift": float(np.mean(own) / max(1e-9, np.mean(other))), "korea_ok": float(hit1[km].mean()) if km.any() else None,
             "per_class": per}
        res[name] = r
        lines.append(f"| {name} | {r['top1']:.3f} / {r['top5']:.3f} | {r['korea']:.3f} | {r['top1_no_korea']:.3f} | "
                     f"{r['lift']:.2f} | {r['korea_ok'] if r['korea_ok'] is None else round(r['korea_ok'], 3)} |")
    if asr:
        f_cer = sorted(asr[r["audio_path"]] for r in rows if r["label_en"] != KOREA and r["audio_path"] in asr)
        k_cer = sorted(asr[r["audio_path"]] for r in rows if r["label_en"] == KOREA and r["audio_path"] in asr)
        med = lambda v: v[len(v) // 2] if v else float("nan")
        lines += ["", f"whisper 문자 오류율 중앙값 — 외국 억양 지시 {med(f_cer):.3f} · 한국 원어민 지시 {med(k_cer):.3f}"]
        res["cer_median"] = {"foreign": med(f_cer), "korea": med(k_cer)}
    lines += ["", "비교 (결과 18, run_zeroth 판정): 실제 외국인 녹음 lift 4.6 · 문자 오류율 0.071 / "
                  "Qwen3-TTS Base 모국어 참조(조건 B) top-1 0.14 · top-5 0.45 · 한국 판정 0.62 · lift 2.1 · 문자 오류율 0.000"]
    text = "\n".join(lines)
    print(text)
    report = Path(a.report or Path(a.out) / f"gate_{a.split}.md")
    report.write_text(text + "\n", encoding="utf-8")
    report.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "gen", "rows", "asr", "gate", "build"])
    ap.add_argument("--out", default=f"{NAT_OUTPUTS}/tts_voicedesign_20261018")
    ap.add_argument("--manifest", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth", help="plan: 클래스 · valid/test 문장 기준")
    ap.add_argument("--pool", default=f"{MDD_OUTPUTS}/manifest_505_weak_v1/pool.meta.jsonl", help="plan: AIHub 131 낭독 문장")
    ap.add_argument("--split", default="gate", choices=list(SPLITS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--models", nargs="+", default=[], help="gate: 이름=logits.npz (synth_<split>.csv 를 잰 것)")
    ap.add_argument("--report", default="", help="gate: 보고서 .md")
    ap.add_argument("--src", default="", help="build: 대조군 매니페스트 폴더")
    ap.add_argument("--synth", default="", help="build: synth_aug.csv")
    ap.add_argument("--dst", default="", help="build: 새 매니페스트 폴더")
    a = ap.parse_args()
    {"plan": cmd_plan, "gen": cmd_gen, "rows": cmd_rows, "asr": cmd_asr, "gate": cmd_gate, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
