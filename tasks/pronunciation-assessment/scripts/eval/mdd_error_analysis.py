# -*- coding: utf-8 -*-
"""오발음 검출(MDD) 상세 오류 분석 — 모델은 건드리지 않고 **저장된 예측**만 다시 센다.

무엇이 다른가: `eval_actual_ctc.py score` 는 전체 수치 한 벌을 낸다. 이 스크립트는 그 수치를
**오류 유형·음절 위치·자모·혼동쌍·문장·L1** 으로 쪼개고, 오발음 보존/되돌림을 같은 축으로 분해하며,
margin GOP 로 confidence 분리 가능성까지 본다. 지표 정의는 `eval_actual_ctc` 의 함수를 그대로
import 해서 쓴다(같은 이름 다른 정의를 만들지 않는다).

세 가지 라벨을 절대 섞지 말 것:
  canonical  = 문장 →KoG2P→ 표준발음 (`.canon`)          ← "내야 할 소리"
  actual/TAG = 전사자가 들은 실제 발음 (`.ltr`)           ← **MDD 정답**
  pred       = 모델 CTC 그리디 출력(캐시 `*.preds.jsonl`)  ← 평가 대상
MDD P/R/F1 은 `eval_actual_ctc.aggregate` 와 같은 정의다(TR/(TR+FR), TR/(TR+FA)).
**legacy weighted F1 과는 다른 지표다** — 이 스크립트는 legacy 를 내지 않는다.

    $PY .../mdd_error_analysis.py --manifest $MDD_OUTPUTS/manifest_full_tag_replay10 --split test \\
        --preds $MDD_OUTPUTS/experiments/w2vbert/w2vbert_test.preds.jsonl \\
        --gop $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --name w2vbert_step38000_seen_sent --out $MDD_OUTPUTS/analysis/mdd_error_2026-09-27
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))      # 같은 폴더의 평가 지표 재사용
from shared.paths import MDD_OUTPUTS

import argparse, collections, csv, json, os, random, time
import numpy as np
import edlib
from eval_actual_ctc import read_split, score_utt, aggregate, file_sha

VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
LOW_SUPPORT = 200            # 이보다 적은 자모/조건은 "과해석 금지" 표시
COR, SUB, DEL, INS = "correct", "substitution", "deletion", "insertion"


def align_gaps(canon, seq):
    """canonical 자리별 대응 기호(없으면 None) + **자리 앞에 끼어든 기호 목록**.

    `eval_actual_ctc.project_on_canon` 과 같은 정렬이지만 삽입을 버리지 않고 gap 으로 돌려준다.
    """
    c, s = u"".join(canon), u"".join(seq)
    gaps = [[] for _ in range(len(canon) + 1)]
    if not s:
        return [None] * len(canon), gaps
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    proj, j = [], 0
    for qc, ts in zip(a["query_aligned"], a["target_aligned"]):
        if qc == "-":
            gaps[j].append(ts)
            continue
        proj.append(None if ts == "-" else ts)
        j += 1
    assert len(proj) == len(canon)
    return proj, gaps


def position_of(canon, i):
    """초성 / 중성 / 종성. ㅇ 은 종성으로 본다(`mdd.pipeline` 과 같은 규칙)."""
    c = canon[i]
    if c in VOWELS:
        return u"중성"
    if c == u"ㅇ":
        return u"종성"
    return u"초성" if (i + 1 < len(canon) and canon[i + 1] in VOWELS) else u"종성"


class PRF(object):
    """자리 단위 오발음 검출 카운터 — 정의는 aggregate() 와 같다."""
    __slots__ = ("tp", "fp", "fn", "tn", "support")

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = self.support = 0

    def add(self, truth_err, pred_err):
        self.support += 1
        if truth_err and pred_err: self.tp += 1
        elif truth_err and not pred_err: self.fn += 1
        elif (not truth_err) and pred_err: self.fp += 1
        else: self.tn += 1

    def row(self):
        p = self.tp / float(self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / float(self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"support": self.support, "truth_errors": self.tp + self.fn, "TP": self.tp, "FP": self.fp,
                "FN": self.fn, "TN": self.tn, "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(f, 4), "low_support": int(self.tp + self.fn < LOW_SUPPORT)}


class Pres(object):
    """오발음 자리의 보존 / 표준발음 되돌림 / 제3오류."""
    __slots__ = ("preserve", "revert", "other", "other_del")

    def __init__(self):
        self.preserve = self.revert = self.other = self.other_del = 0

    def add(self, kind, pred_is_none):
        setattr(self, kind, getattr(self, kind) + 1)
        if kind == "other" and pred_is_none:
            self.other_del += 1

    def row(self):
        n = self.preserve + self.revert + self.other
        r = lambda v: round(v / float(n), 4) if n else None
        return {"mispron_sites": n, "preserved": self.preserve, "reverted": self.revert, "other": self.other,
                "preservation_rate": r(self.preserve), "reversion_rate": r(self.revert),
                "other_rate": r(self.other), "other_is_deletion": self.other_del,
                "low_support": int(n < LOW_SUPPORT)}


def write_csv(path, rows, key_names):
    if not rows:
        open(path, "w").close()
        return
    cols = list(key_names) + [k for k in rows[0] if k not in key_names]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--preds", default=None, help="캐시된 예측 jsonl ({i, pred}). --gop 만 있으면 그쪽 pred 를 쓴다")
    ap.add_argument("--gop", default=None, help="stem 별 margin GOP jsonl (confidence 분석용)")
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "analysis", "mdd_error_2026-09-27"))
    ap.add_argument("--name", required=True, help="산출 하위 폴더 이름")
    ap.add_argument("--top-pairs", dest="top_pairs", type=int, default=60)
    ap.add_argument("--hard-per-bucket", dest="hard_per", type=int, default=120)
    ap.add_argument("--min-group", dest="min_group", type=int, default=100, help="문장·L1 표에 남길 최소 발화 수")
    ap.add_argument("--no-plots", dest="no_plots", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    outdir = os.path.join(a.out, a.name)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    wavs, tags, canons, metas = read_split(a.manifest, a.split)
    stems = [os.path.basename(p)[:-4] for p, _ in wavs]
    print("[data] %s/%s · 발화 %d" % (a.manifest, a.split, len(wavs)), flush=True)

    # 예측 — 캐시 우선, 없으면 gop 파일의 pred
    preds = [None] * len(wavs)
    if a.preds:
        for line in open(a.preds, encoding="utf-8"):
            d = json.loads(line)
            preds[d["i"]] = [t for t in d["pred"] if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
    margins = {}
    gop_pred = {}
    if a.gop:
        for line in open(a.gop, encoding="utf-8"):
            d = json.loads(line)
            margins[d["stem"]] = d.get("gops_m")
            if d.get("pred"):
                gop_pred[d["stem"]] = [t for t in d["pred"] if 0x3131 <= ord(t) <= 0x3163]
    if not a.preds:
        for i, s in enumerate(stems):
            preds[i] = gop_pred.get(s)
    n_have = sum(1 for p in preds if p is not None)
    print("[pred] 예측 있는 발화 %d · margin 있는 발화 %d" % (n_have, len(margins)), flush=True)

    by_type = collections.defaultdict(PRF)         # 진짜 오류 유형별 검출
    by_pos = collections.defaultdict(PRF)
    by_phone = collections.defaultdict(PRF)
    pres_type, pres_pos, pres_phone, pres_pair = (collections.defaultdict(Pres) for _ in range(4))
    pres_l1, pres_sent = collections.defaultdict(Pres), collections.defaultdict(Pres)
    prf_l1, prf_sent = collections.defaultdict(PRF), collections.defaultdict(PRF)
    conf_mat = collections.Counter()               # (truth 유형, pred 유형)
    pair_truth = collections.Counter()             # canonical → 사람이 들은 소리
    pair_pred = collections.Counter()              # canonical → 모델
    pair_diag = collections.Counter()              # 오발음 자리에서 사람 → 모델
    utt_n = collections.Counter()
    ins_prf = PRF()
    conf = {"margin": [], "group": []}             # 0 correct · 1 true mispron · 2 model error
    hard = collections.defaultdict(list)
    evs = []

    for i in range(len(wavs)):
        pred = preds[i]
        if pred is None:
            continue
        canon, tag = canons[i], tags[i]
        if not canon:
            continue
        evs.append(score_utt(tag, canon, pred))
        ptag, gtag = align_gaps(canon, tag)
        ppred, gpred = align_gaps(canon, pred)
        m = metas[i] if i < len(metas) else {}
        parts = stems[i].split("-")
        l1 = m.get("language") or m.get("nationality") or (parts[3] if len(parts) > 3 else "?")
        sent = m.get("norm_prompt") or m.get("prompt") or (parts[5] if len(parts) > 5 else "?")
        mg = margins.get(stems[i])
        mg = mg if (mg and len(mg) == len(canon)) else None
        utt_n[l1] += 1

        for j, c in enumerate(canon):
            tj, pj = ptag[j], ppred[j]
            t_type = COR if tj == c else (DEL if tj is None else SUB)
            p_type = COR if pj == c else (DEL if pj is None else SUB)
            truth_err = t_type != COR
            pred_err = p_type != COR
            conf_mat[(t_type, p_type)] += 1
            pos = position_of(canon, j)
            for d in (by_type[t_type], by_pos[pos], by_phone[c], prf_l1[l1], prf_sent[sent]):
                d.add(truth_err, pred_err)
            pair_truth[(c, tj or u"∅")] += 1
            pair_pred[(c, pj or u"∅")] += 1
            if truth_err:
                kind = "preserve" if pj == tj else "revert" if pj == c else "other"
                for d in (pres_type[t_type], pres_pos[pos], pres_phone[c],
                          pres_pair[(c, tj or u"∅")], pres_l1[l1], pres_sent[sent]):
                    d.add(kind, pj is None)
                pair_diag[(c, tj or u"∅", pj or u"∅")] += 1
            if mg is not None:
                conf["margin"].append(float(mg[j]))
                conf["group"].append(1 if truth_err else (2 if pred_err else 0))
            # hard example 후보 — 조건별로 무작위 표본만 남긴다
            bucket = None
            if truth_err and pj == c:
                bucket = "canonical_reversion"
            elif (not truth_err) and pred_err:
                bucket = "normal_site_error"
            elif truth_err and pj != tj and pj is not None:
                bucket = "wrong_diagnosis"
            elif t_type == DEL and p_type != DEL:
                bucket = "missed_deletion"
            if bucket:
                rec = {"bucket": bucket, "audio_path": wavs[i][0], "stem": stems[i], "speaker_id": parts[0],
                       "l1": l1, "sentence": m.get("prompt") or sent, "pos_index": j, "phone_position": pos,
                       "canonical_jamo": c, "actual_jamo": tj, "predicted_jamo": pj,
                       "truth_error_type": t_type, "pred_error_type": p_type,
                       "margin_gop": (round(float(mg[j]), 4) if mg is not None else None),
                       "canonical_seq": u"".join(canon), "actual_seq": u"".join(tag),
                       "predicted_seq": u"".join(pred)}
                lst = hard[bucket]
                if len(lst) < a.hard_per:
                    lst.append(rec)
                elif random.random() < a.hard_per / float(a.hard_per + len(lst)):
                    lst[random.randrange(a.hard_per)] = rec
        # 삽입 — canonical 자리 사이 gap 단위로 센다
        for j in range(len(canon) + 1):
            t_ins, p_ins = bool(gtag[j]), bool(gpred[j])
            if t_ins or p_ins:
                ins_prf.add(t_ins, p_ins)
                conf_mat[(INS if t_ins else COR, INS if p_ins else COR)] += 1
                if t_ins:
                    kind = ("preserve" if (p_ins and gpred[j] == gtag[j])
                            else "revert" if not p_ins else "other")
                    pres_type[INS].add(kind, not p_ins)
        if len(evs) % 40000 == 0:
            print("   ... %d 발화 (%.0fs)" % (len(evs), time.time() - t0), flush=True)

    base = aggregate(evs)
    print("[baseline 재현] MDD P %s R %s F1 %s · 보존 %s 되돌림 %s · 정상자리오류 %s"
          % (base["mdd"]["precision"], base["mdd"]["recall"], base["mdd"]["f1"],
             base["sites"]["preservation"], base["sites"]["reversion"], base["sites"]["normal_site_error"]), flush=True)

    # ── 표 ──────────────────────────────────────────────────────────
    # 유형별 표 — **precision 은 "그 유형으로 예측한 자리" 기준**이어야 한다.
    # truth 유형만으로 묶으면 그 안에 FP 가 있을 수 없어 precision 이 항상 1.0 으로 나온다(오해 유발).
    type_rows = []
    for t in (SUB, DEL, INS):
        truth_n = sum(conf_mat[(t, p)] for p in (COR, SUB, DEL, INS))
        pred_n = sum(conf_mat[(tt, t)] for tt in (COR, SUB, DEL, INS))
        tp_same = conf_mat[(t, t)]
        det = sum(conf_mat[(t, p)] for p in (SUB, DEL, INS))      # 유형은 틀려도 오류라고 잡은 것
        tp_any = sum(conf_mat[(tt, t)] for tt in (SUB, DEL, INS))  # 그 유형으로 예측한 것 중 진짜 오류
        pr = tp_same / float(pred_n) if pred_n else 0.0
        rc = tp_same / float(truth_n) if truth_n else 0.0
        type_rows.append({"error_type": t, "truth_sites": truth_n, "predicted_sites": pred_n,
                          "TP_same_type": tp_same, "precision_same_type": round(pr, 4),
                          "recall_same_type": round(rc, 4),
                          "f1_same_type": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                          "detection_recall_any_type": round(det / float(truth_n), 4) if truth_n else None,
                          "precision_any_error": round(tp_any / float(pred_n), 4) if pred_n else None,
                          "low_support": int(truth_n < LOW_SUPPORT)})
    type_rows.append({"error_type": COR, "truth_sites": sum(conf_mat[(COR, p)] for p in (COR, SUB, DEL, INS)),
                      "predicted_sites": sum(conf_mat[(t, COR)] for t in (COR, SUB, DEL, INS)),
                      "TP_same_type": conf_mat[(COR, COR)], "precision_same_type": None,
                      "recall_same_type": None, "f1_same_type": None,
                      "detection_recall_any_type": None, "precision_any_error": None, "low_support": 0})
    write_csv(os.path.join(outdir, "mdd_by_error_type.csv"), type_rows, ["error_type"])
    # 참고: 진짜 오류 유형별 "검출됐나"만 보는 표(precision 은 위 표에서 볼 것)
    write_csv(os.path.join(outdir, "mdd_by_truth_error_type_recall.csv"),
              [dict({"truth_error_type": k}, **v.row()) for k, v in sorted(by_type.items())]
              + [dict({"truth_error_type": INS}, **ins_prf.row())], ["truth_error_type"])

    write_csv(os.path.join(outdir, "mdd_by_position.csv"),
              [dict({"position": k}, **v.row()) for k, v in sorted(by_pos.items())], ["position"])
    write_csv(os.path.join(outdir, "mdd_by_phone.csv"),
              sorted([dict({"phone": k}, **v.row()) for k, v in by_phone.items()],
                     key=lambda r: -r["support"]), ["phone"])

    pairs = []
    for (c, t), n in pair_truth.most_common(a.top_pairs):
        p = pres_pair.get((c, t))
        pairs.append({"kind": "truth(canonical→actual)", "canonical": c, "other": t, "count": n,
                      "share_of_sites": round(n / float(max(1, sum(pair_truth.values()))), 5),
                      "preservation_rate": (p.row()["preservation_rate"] if p else None),
                      "reversion_rate": (p.row()["reversion_rate"] if p else None)})
    for (c, p_) in [k for k, _ in pair_pred.most_common(a.top_pairs)]:
        n = pair_pred[(c, p_)]
        pairs.append({"kind": "model(canonical→pred)", "canonical": c, "other": p_, "count": n,
                      "share_of_sites": round(n / float(max(1, sum(pair_pred.values()))), 5),
                      "preservation_rate": None, "reversion_rate": None})
    write_csv(os.path.join(outdir, "confusion_pairs.csv"), pairs, ["kind", "canonical", "other", "count"])

    diag = [{"canonical": c, "actual": t, "predicted": p, "count": n}
            for (c, t, p), n in pair_diag.most_common(a.top_pairs)]
    write_csv(os.path.join(outdir, "confusion_triples.csv"), diag, ["canonical", "actual", "predicted", "count"])

    pres_rows = []
    for label, table, keyname in (("error_type", pres_type, "value"), ("position", pres_pos, "value"),
                                  ("phone", pres_phone, "value"), ("L1", pres_l1, "value")):
        for k, v in sorted(table.items(), key=lambda kv: -(kv[1].preserve + kv[1].revert + kv[1].other)):
            pres_rows.append(dict({"breakdown": label, "value": k}, **v.row()))
    for (c, t), v in sorted(pres_pair.items(), key=lambda kv: -(kv[1].preserve + kv[1].revert + kv[1].other))[:a.top_pairs]:
        pres_rows.append(dict({"breakdown": "confusion_pair", "value": u"%s→%s" % (c, t)}, **v.row()))
    write_csv(os.path.join(outdir, "preservation_analysis.csv"), pres_rows, ["breakdown", "value"])

    srows = []
    for k, v in sorted(prf_sent.items(), key=lambda kv: -kv[1].support):
        if v.support < a.min_group:
            continue
        p = pres_sent.get(k)
        srows.append(dict({"sentence": k}, **dict(v.row(), **{"reversion_rate": (p.row()["reversion_rate"] if p else None),
                                                              "preservation_rate": (p.row()["preservation_rate"] if p else None)})))
    write_csv(os.path.join(outdir, "sentence_analysis.csv"), srows, ["sentence"])

    lrows = []
    for k, v in sorted(prf_l1.items(), key=lambda kv: -kv[1].support):
        p = pres_l1.get(k)
        lrows.append(dict({"l1": k, "utterances": utt_n[k]}, **dict(v.row(), **{
            "reversion_rate": (p.row()["reversion_rate"] if p else None),
            "preservation_rate": (p.row()["preservation_rate"] if p else None)})))
    write_csv(os.path.join(outdir, "mdd_by_l1.csv"), lrows, ["l1", "utterances"])

    cm_rows = []
    for t in (COR, SUB, DEL, INS):
        row = {"truth": t}
        for p in (COR, SUB, DEL, INS):
            row["pred_" + p] = conf_mat.get((t, p), 0)
        cm_rows.append(row)
    write_csv(os.path.join(outdir, "confusion_matrix.csv"), cm_rows, ["truth"])

    # ── confidence (margin GOP) ────────────────────────────────────
    cinfo = {"available": False}
    if conf["margin"]:
        mar = np.array(conf["margin"], dtype=np.float32)
        grp = np.array(conf["group"], dtype=np.int8)
        q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 25, 50, 75, 95])]
        names = {0: "correct", 1: "true_mispronunciation", 2: "model_recognition_error"}
        cinfo = {"available": True, "n_sites": int(len(mar)), "groups": {}}
        for g, nm in names.items():
            v = mar[grp == g]
            if len(v):
                cinfo["groups"][nm] = {"n": int(len(v)), "mean": round(float(v.mean()), 3),
                                       "p5_p25_p50_p75_p95": q(v)}
        from sklearn.metrics import roc_auc_score
        mask = grp != 2
        y = (grp[mask] == 1).astype(np.int8)
        cinfo["auc_true_mispron_vs_correct"] = round(float(roc_auc_score(y, -mar[mask])), 4)
        sweep = []
        for thr in (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3):
            flag = mar < thr
            tp = int((flag & (grp == 1)).sum()); fp = int((flag & (grp != 1)).sum())
            fn = int(((~flag) & (grp == 1)).sum())
            p = tp / float(tp + fp) if tp + fp else 0.0
            rc = tp / float(tp + fn) if tp + fn else 0.0
            sweep.append({"margin_threshold": thr, "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                          "precision": round(p, 4), "recall": round(rc, 4),
                          "f1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
                          "flag_rate": round((tp + fp) / float(len(mar)), 4)})
        write_csv(os.path.join(outdir, "confidence_threshold_sweep.csv"), sweep, ["margin_threshold"])
        write_csv(os.path.join(outdir, "confidence_groups.csv"),
                  [dict({"group": k}, **v) for k, v in cinfo["groups"].items()], ["group"])

    with open(os.path.join(outdir, "hard_examples.jsonl"), "w", encoding="utf-8") as f:
        for b in sorted(hard):
            for rec in hard[b]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "name": a.name, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": os.path.abspath(a.manifest), "split": a.split,
        "tsv_sha16": file_sha(os.path.join(a.manifest, "%s.tsv" % a.split)),
        "ltr_sha16": file_sha(os.path.join(a.manifest, "%s.ltr" % a.split)),
        "preds": os.path.abspath(a.preds) if a.preds else None,
        "gop": os.path.abspath(a.gop) if a.gop else None,
        "utterances_scored": len(evs), "utterances_in_split": len(wavs),
        "labels": {"mdd_truth": "TAG (.ltr) — 전사자가 들은 실제 발음",
                   "canonical": ".canon — KoG2P 표준발음", "pred": "그리디 CTC, LM·렉시콘·프롬프트 없음"},
        "metric_definition": "eval_actual_ctc.aggregate 와 동일 (precision=TR/(TR+FR) · recall=TR/(TR+FA))",
        "baseline": base, "insertion_gap_level": ins_prf.row(), "confidence": cinfo,
        "notes": "legacy weighted F1 은 내지 않는다 — MDD F1 과 다른 지표다",
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), ensure_ascii=False, indent=1)

    if not a.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ph = sorted([r for r in ([dict({"phone": k}, **v.row()) for k, v in by_phone.items()])],
                        key=lambda r: -r["support"])
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter([r["truth_errors"] for r in ph], [r["f1"] for r in ph], s=14)
            ax.set_xscale("log"); ax.set_xlabel("true error sites (support)"); ax.set_ylabel("MDD F1")
            ax.set_title("phone support vs F1"); fig.tight_layout()
            fig.savefig(os.path.join(outdir, "support_vs_f1.png"), dpi=110); plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4))
            M = np.array([[conf_mat.get((t, p), 0) for p in (COR, SUB, DEL, INS)] for t in (COR, SUB, DEL, INS)],
                         dtype=float)
            Mn = M / np.maximum(1.0, M.sum(axis=1, keepdims=True))
            im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(4)); ax.set_xticklabels(["cor", "sub", "del", "ins"])
            ax.set_yticks(range(4)); ax.set_yticklabels(["cor", "sub", "del", "ins"])
            for r in range(4):
                for c2 in range(4):
                    ax.text(c2, r, "%.2f" % Mn[r, c2], ha="center", va="center", fontsize=9,
                            color="white" if Mn[r, c2] > 0.5 else "black")
            ax.set_xlabel("predicted"); ax.set_ylabel("truth (TAG)"); ax.set_title("error-type confusion (row-normalized)")
            fig.colorbar(im); fig.tight_layout()
            fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=110); plt.close(fig)

            if conf["margin"]:
                fig, ax = plt.subplots(figsize=(6, 4))
                mar = np.array(conf["margin"], dtype=np.float32); grp = np.array(conf["group"], dtype=np.int8)
                for g, nm, cl in ((0, "correct", "#2b8a3e"), (1, "true mispron.", "#c92a2a"),
                                  (2, "model error", "#f59f00")):
                    v = mar[grp == g]
                    if len(v):
                        ax.hist(np.clip(v, -12, 12), bins=60, density=True, histtype="step", label=nm, color=cl)
                ax.set_xlabel("margin GOP"); ax.set_ylabel("density"); ax.legend()
                ax.set_title("margin distribution by site group"); fig.tight_layout()
                fig.savefig(os.path.join(outdir, "confidence_distribution.png"), dpi=110); plt.close(fig)
            print("[plot] png 3개", flush=True)
        except Exception as e:                       # 그림은 실패해도 분석은 남긴다
            print("[plot] 건너뜀 — %s: %s" % (type(e).__name__, e), flush=True)

    print("[done] %s · %.0fs" % (outdir, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
