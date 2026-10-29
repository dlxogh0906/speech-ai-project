# -*- coding: utf-8 -*-
"""웹마이크 재학습 준비 — 화자 분리 split 고정 + 학습/평가 매니페스트.

왜
    배포 KD 모델은 040(영어권)·042(중·일) 화자를 학습에 안 써서 웹마이크 녹음에서 top-5 가 0.56 이다.
    040~043 을 학습에 넣되, 옛 모델이 알던 505 도메인(131)과 한국어 원어민은 **증류로 보존**한다.

split (결정적 — 입력 순서와 무관)
    화자 키의 md5 순서로 정렬한 뒤 앞에서부터 test → valid → train.
    040~043  클래스 안에서 배정: test = n≥3 이면 max(1, round(10%)) · valid = n≥8 이면 max(1, round(5%)).
             꼬리 클래스도 test 화자가 최소 1명 생기게 하려는 것이다(n<3 이면 전부 train).
    131·한국어  전체에서 md5 백분위로 test 10% · valid 5%.
    ⚠️ AIHub 의 Training/Validation 폴더는 화자가 겹친다(11,469명 중 10,846명) — 폴더 구분은 쓰지 않는다.
    ⚠️ 040~043 의 nationality=Korea(10명)는 해외 학습자라 한국어 원어민 클래스에서 뺀다.

학습 서브셋
    웹마이크(라벨)  클래스당 --cap. 화자를 돌아가며 한 발화씩(단어 과제 RW 는 각 화자의 뒤로 미룸) —
                   큰 클래스는 화자 다양성을, 꼬리 클래스는 전량을 쓰게 된다.
    131(라벨 없음)  그룹(JP·CN·TH·EX)당 --cap_131. 옛 모델 출력만 따라 배운다(보존용).
    한국어(라벨)    001 자유대화 Validation 과 KsponSpeech 반반, 합계 --cap.
평가 셋 (화자당 상한, 시드 고정)
    webmic  문장·자유발화 --eval_sent 개 + 단어 --eval_word 개 / 화자
    131·korean  --eval_other 개 / 화자

출력 (--out_dir): speaker_split.json · train.csv · valid.csv · test.csv · prep_meta.json
    스키마: audio_path,label_en,speaker,domain,task,duration_sec  (label_en 빈칸 = 라벨 없음)

    $PY tasks/nationality-classification/experiments/webmic_kd_prep.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS

import argparse, collections, csv, glob, hashlib, json, os, random

HDR = ["audio_path", "label_en", "speaker", "domain", "task", "duration_sec"]
SENT_TASKS = {"RS", "LAR", "ATQ"}
FREE_TASKS = {"EIG", "PDT", "RP", "SPT", "SSO"}


def h(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def task_group(task):
    return "word" if task == "RW" else ("free" if task in FREE_TASKS else "sent")


def round_robin(by_spk, cap, rng, word_last=True):
    """화자를 돌아가며 한 발화씩. 각 화자 안에서는 무작위 순서(단어 과제는 뒤로)."""
    queues = []
    for s in sorted(by_spk, key=h):
        rows = list(by_spk[s])
        rng.shuffle(rows)
        if word_last:
            rows.sort(key=lambda r: r["task"] == "RW")        # 안정 정렬 — 무작위 순서 유지
        queues.append(collections.deque(rows))
    out = []
    while queues and len(out) < cap:
        nxt = []
        for q in queues:
            if len(out) >= cap:
                break
            out.append(q.popleft())
            if q:
                nxt.append(q)
        queues = nxt
    return out


def per_speaker(by_spk, n_sent, n_word, rng):
    out = []
    for s in sorted(by_spk):
        rows = sorted(by_spk[s], key=lambda r: r["audio_path"])
        rng.shuffle(rows)
        words = [r for r in rows if r["task"] == "RW"]
        other = [r for r in rows if r["task"] != "RW"]
        out += other[:n_sent] + words[:n_word]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index_dir", default=str(NAT_OUTPUTS / "webmic_kd"))
    ap.add_argument("--out_dir", default=str(NAT_OUTPUTS / "webmic_kd"))
    ap.add_argument("--k001_root", default=str(DATA_ROOT / "aihub" / "001.자유대화(일반남여)" / "01.데이터" / "2.Validation"))
    ap.add_argument("--cap", type=int, default=8000, help="웹마이크 클래스당·한국어 학습 발화 상한")
    ap.add_argument("--cap_131", type=int, default=10000, help="131 그룹당 학습 발화 상한 (라벨 없음)")
    ap.add_argument("--min_sec", type=float, default=1.0)
    ap.add_argument("--eval_sent", type=int, default=12)
    ap.add_argument("--eval_word", type=int, default=4)
    ap.add_argument("--eval_other", type=int, default=10)
    ap.add_argument("--valid_sent", type=int, default=6)
    ap.add_argument("--valid_word", type=int, default=2)
    ap.add_argument("--p505_labeled", default="",
                    help="build_131_index.py 의 index_131_labeled.csv — 주면 131 을 **라벨과 함께** 쓴다(음성 있는 행만). "
                         "클래스당 --cap_131 · 화자 split 은 라벨 없을 때와 같은 md5 백분위")
    ap.add_argument("--seed", type=int, default=42)
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    rng = random.Random(A.seed)

    # ── 040~043 (라벨) ──
    web = collections.defaultdict(lambda: collections.defaultdict(list))   # label → speaker → rows
    for r in csv.DictReader(open(os.path.join(A.index_dir, "index_040_043.csv"), encoding="utf-8")):
        lb = r["label_en"]
        if not lb or lb == "Korea" or float(r["duration_sec"]) < A.min_sec:
            continue
        web[lb][r["speaker"]].append({"audio_path": r["audio_path"], "label_en": lb, "speaker": r["speaker"],
                                      "domain": "web" + r["dataset"], "task": r["task"],
                                      "duration_sec": r["duration_sec"]})
    assign = {}
    for lb, spk in web.items():
        order = sorted(spk, key=h)
        n = len(order)
        n_te = 0 if n < 3 else max(1, round(0.10 * n))
        n_va = 0 if n < 8 else max(1, round(0.05 * n))
        for i, s in enumerate(order):
            assign[s] = "test" if i < n_te else ("valid" if i < n_te + n_va else "train")

    def pct_split(key):
        v = int(h(key)[:8], 16) % 100
        return "test" if v < 10 else ("valid" if v < 15 else "train")

    # ── 131 ──
    g131 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ([] if A.p505_labeled else csv.DictReader(open(os.path.join(A.index_dir, "index_131.csv"), encoding="utf-8"))):
        if float(r["duration_sec"]) < A.min_sec:
            continue
        g131[r["group"]][r["speaker"]].append({"audio_path": r["audio_path"], "label_en": "", "speaker": r["speaker"],
                                               "domain": "p505_" + r["group"], "task": "RB",
                                               "duration_sec": r["duration_sec"]})
        assign.setdefault(r["speaker"], pct_split(r["speaker"]))
    lab131 = collections.defaultdict(lambda: collections.defaultdict(list))     # label → speaker → rows
    if A.p505_labeled:
        for r in csv.DictReader(open(A.p505_labeled, encoding="utf-8")):
            if not r["label_en"] or not r["audio_path"] or float(r["duration_sec"] or 0) < A.min_sec:
                continue
            lab131[r["label_en"]][r["speaker"]].append({"audio_path": r["audio_path"], "label_en": r["label_en"],
                                                        "speaker": r["speaker"], "domain": "p505",
                                                        "task": "RB", "duration_sec": r["duration_sec"]})
            # 라벨 없이 쓰던 때와 같은 배정(md5 백분위) — 이미 본 화자가 test 로 새지 않는다
            assign.setdefault(r["speaker"], pct_split(r["speaker"]))

    # ── 한국어: 001 자유대화 Validation + KsponSpeech ──
    kor = {"k001": collections.defaultdict(list), "kspon": collections.defaultdict(list)}
    for p in glob.glob(os.path.join(A.k001_root, "2.원천데이터*", "*", "*", "*", "*.wav")):
        spk = "K001:" + os.path.basename(os.path.dirname(p))
        sec = max(os.path.getsize(p) - 44, 0) / 32000
        if sec < A.min_sec:
            continue
        kor["k001"][spk].append({"audio_path": p, "label_en": "Korea", "speaker": spk, "domain": "kor001",
                                 "task": "KOR", "duration_sec": round(sec, 2)})
    for r in csv.DictReader(open(os.path.join(A.index_dir, "index_kspon.csv"), encoding="utf-8")):
        if float(r["duration_sec"]) < A.min_sec:
            continue
        kor["kspon"][r["speaker"]].append({"audio_path": r["audio_path"], "label_en": "Korea", "speaker": r["speaker"],
                                           "domain": "kspon", "task": "KOR", "duration_sec": r["duration_sec"]})
    for src in kor.values():
        for s in src:
            assign.setdefault(s, pct_split(s))

    # ── 서브셋 ──
    def part(groups, split):
        return {s: rows for s, rows in groups.items() if assign[s] == split}

    train, valid, test = [], [], []
    stats = {}
    for lb in sorted(web):
        tr = round_robin(part(web[lb], "train"), A.cap, rng)
        va = per_speaker(part(web[lb], "valid"), A.valid_sent, A.valid_word, rng)
        te = per_speaker(part(web[lb], "test"), A.eval_sent, A.eval_word, rng)
        train += tr; valid += va; test += te
        stats[lb] = {"spk": {k: sum(assign[s] == k for s in web[lb]) for k in ["train", "valid", "test"]},
                     "train_utt": len(tr), "valid_utt": len(va), "test_utt": len(te)}
    for g in sorted(g131):
        train += round_robin(part(g131[g], "train"), A.cap_131, rng, word_last=False)
        valid += per_speaker(part(g131[g], "valid"), A.eval_other // 2, 0, rng)
        test += per_speaker(part(g131[g], "test"), A.eval_other, 0, rng)
    stats131 = {}
    for lb in sorted(lab131):
        tr = round_robin(part(lab131[lb], "train"), A.cap_131, rng, word_last=False)
        train += tr
        valid += per_speaker(part(lab131[lb], "valid"), A.eval_other // 2, 0, rng)
        te = per_speaker(part(lab131[lb], "test"), A.eval_other, 0, rng)
        test += te
        stats131[lb] = {"spk": {k: sum(assign[s] == k for s in lab131[lb]) for k in ["train", "valid", "test"]},
                        "train_utt": len(tr), "test_utt": len(te)}
    for name, src in kor.items():
        train += round_robin(part(src, "train"), A.cap // 2, rng, word_last=False)
        valid += per_speaker(part(src, "valid"), A.eval_other // 2, 0, rng)
        test += per_speaker(part(src, "test"), A.eval_other, 0, rng)
    rng.shuffle(train)

    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        with open(os.path.join(A.out_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HDR)
            w.writeheader()
            w.writerows(rows)
    json.dump({"_설명": "화자 키 → split. md5 순서 결정적 배정 (webmic_kd_prep.py 문서 참고).", "assign": assign},
              open(os.path.join(A.out_dir, "speaker_split.json"), "w", encoding="utf-8"), ensure_ascii=False)

    def summ(rows):
        c = collections.Counter(r["domain"] for r in rows)
        t = collections.Counter(task_group(r["task"]) for r in rows if r["domain"].startswith("web"))
        return {"rows": len(rows), "by_domain": dict(sorted(c.items())), "web_task_group": dict(t),
                "hours": round(sum(min(float(r["duration_sec"]), 12.0) for r in rows) / 3600, 1)}
    meta = {"args": vars(A), "train": summ(train), "valid": summ(valid), "test": summ(test), "per_class": stats,
            "per_class_131": stats131}
    json.dump(meta, open(os.path.join(A.out_dir, "prep_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in ["train", "valid", "test"]:
        print(f"[{k}] {meta[k]}")
    print("[class] 화자(train/valid/test) · 학습 발화")
    for lb in sorted(stats, key=lambda x: stats[x]["train_utt"]):
        s = stats[lb]
        print(f"  {lb:15} {s['spk']['train']:>5}/{s['spk']['valid']:>3}/{s['spk']['test']:>3} · train {s['train_utt']:>5} · test {s['test_utt']:>5}")
    if stats131:
        print("[131] 화자(train/valid/test) · 학습 발화")
        for lb in sorted(stats131, key=lambda x: stats131[x]["train_utt"]):
            t = stats131[lb]
            print(f"  {lb:15} {t['spk']['train']:>4}/{t['spk']['valid']:>3}/{t['spk']['test']:>3} · train {t['train_utt']:>6} · test {t['test_utt']:>4}")
    print(f"[saved] {A.out_dir}")


if __name__ == "__main__":
    main()
