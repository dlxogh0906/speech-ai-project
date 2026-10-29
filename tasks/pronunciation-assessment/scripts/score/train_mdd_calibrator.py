# -*- coding: utf-8 -*-
"""자리 단위 MDD 보정기 — **학습 · 선택 · 최종 평가를 분리한** 절차.

    select : train split 의 화자를 80:20 으로 나눠 calibrator_train 으로 학습하고,
             **임계값·특징 묶음·재녹음 기준은 calibrator_valid 에서만** 고른다. test 는 열지 않는다.
    final  : select 에서 잠근 번들 하나를 test 에 **딱 한 번** 적용한다. 결과 파일이 이미 있으면 거부한다.

왜 이렇게 하나 — 첫 버전은 임계값을 test 에서 골랐다(정밀도 0.65 를 test 에 맞춘 셈).
이 절차에서는 test 가 선택에 전혀 쓰이지 않으므로 final 수치가 곧 일반화 추정치다.

특징 묶음(`mdd.mdd_calibrator.FEATURE_SETS`) — 축소 실험:
    full(29) · full_mism(30) · tiny(margin·mismatch·자모·위치) · minimal(margin·mismatch)
선택 규칙: valid 의 incorrect F1 이 최고값에서 `--simplicity-tol` 안이면 **가장 작은 묶음**.

판정 규칙은 `MddCalibrator.states` 와 같다:
    incorrect = p ≥ t_inc AND mismatch
    uncertain = (incorrect 아님) AND (p ≥ t_unc OR (mismatch AND p ≥ t_unc/2))
    correct   = 나머지
삽입 오류는 표준발음에 자리가 없어 이 보정기의 대상이 **아니다**(자리 단위 판정이므로).

    A=$MDD_OUTPUTS/analysis/mdd_calibrator
    $PY .../train_mdd_calibrator.py select --gop $MDD_OUTPUTS/score_phoneme/gop_margin_train200k.jsonl \\
        --manifest $MDD_OUTPUTS/manifest_tag_rebuilt --split train --out-dir $A
    $PY .../train_mdd_calibrator.py final  --gop $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --manifest $MDD_OUTPUTS/manifest_tag_rebuilt --split test --out-dir $A
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os, time
import numpy as np
import edlib
import joblib
from mdd.mdd_calibrator import (FEATURE_SETS, ALL_COLUMNS, site_features, column_index, position_code,
                                RERECORD_RATIO)

HARD_PAIRS = [(u"ㅅ", u"ㅆ"), (u"ㅈ", u"ㅊ"), (u"ㄱ", u"ㅋ"), (u"ㄸ", u"ㄷ"), (u"ㅉ", u"ㅈ"), (u"ㄲ", u"ㄱ"),
              (u"ㄴ", u"ㅇ"), (u"ㄴ", u"ㄹ"), (u"ㄹ", u"ㄴ"), (u"ㅇ", u"ㄱ")]
POS_NAME = {1: u"중성", 2: u"초성", 3: u"종성"}
SERVICE_FLAG_MAX = 60      # 현재 서비스 규칙(speechsuper.FLAG_MAX): 전사 불일치 AND 음소점수 < 60


def proj(canon, seq):
    """canonical 자리별 대응 기호(삭제면 None). eval_actual_ctc.project_on_canon 과 같은 정렬."""
    c, s = u"".join(canon), u"".join(seq)
    if not s:
        return [None] * len(canon)
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    out = []
    for qc, ts in zip(a["query_aligned"], a["target_aligned"]):
        if qc == "-":
            continue
        out.append(None if ts == "-" else ts)
    return out


def load_tag(manifest, split):
    tsv = open(os.path.join(manifest, "%s.tsv" % split), encoding="utf-8").read().splitlines()[1:]
    ltr = open(os.path.join(manifest, "%s.ltr" % split), encoding="utf-8").read().splitlines()
    return dict((os.path.basename(row.split("\t")[0])[:-4], ltr[i].split()) for i, row in enumerate(tsv))


def build(gop_path, manifest, split, calib_json=None):
    """자리 단위 표: 특징(ALL_COLUMNS) + 정답 + 메타(화자·발화·자모·위치·실제 소리·서비스 점수)."""
    TAG = load_tag(manifest, split)
    cal = json.load(open(calib_json, encoding="utf-8")) if calib_json else None
    if cal:
        from mdd.session import phone_context
        from mdd.jamo_to_kog2p import VOWEL
    X, y, spk, utt, ph, pos, act, svc = [], [], [], [], [], [], [], []
    uid = 0
    with open(gop_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            canon, gm = d.get("canon"), d.get("gops_m")
            tag = TAG.get(d["stem"])
            if not canon or not gm or tag is None or len(gm) != len(canon):
                continue
            canon = list(canon)
            pt, pp = proj(canon, tag), proj(canon, list(d.get("pred") or ""))
            if len(pt) != len(canon) or len(pp) != len(canon):
                continue
            mism = [1 if pp[i] != canon[i] else 0 for i in range(len(canon))]
            X += site_features(canon, gm, d.get("gops"), d, mism)
            y += [1 if pt[i] != canon[i] else 0 for i in range(len(canon))]
            spk += [d["stem"][:5]] * len(canon)
            utt += [uid] * len(canon)
            ph += canon
            pos += [position_code(canon, i) for i in range(len(canon))]
            act += [pt[i] if pt[i] is not None else u"∅" for i in range(len(canon))]
            if cal:
                for i in range(len(canon)):
                    c = cal.get(phone_context(canon, i, VOWEL)) or cal["_global"]
                    pe = 1.0 / (1.0 + np.exp(-(c["a"] * float(gm[i]) + c["b"])))
                    svc.append((1.0 - pe) * 100.0)
            uid += 1
    T = {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(y, dtype=np.int8), "spk": np.asarray(spk),
         "utt": np.asarray(utt), "ph": np.asarray(ph), "pos": np.asarray(pos, dtype=np.int8),
         "act": np.asarray(act), "n_utt": uid}
    T["mism"] = T["X"][:, ALL_COLUMNS.index("mismatch")] > 0.5
    T["svc"] = np.asarray(svc, dtype=np.float32) if svc else None
    return T


def subset(T, mask):
    uids = np.unique(T["utt"][mask])
    out = dict((k, (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)) for k, v in T.items())
    out["n_utt"] = int(len(uids))
    return out


def prf(flag, y):
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int(((~flag) & (y == 1)).sum())
    p = tp / float(tp + fp) if tp + fp else 0.0
    r = tp / float(tp + fn) if tp + fn else 0.0
    return {"flagged": int(flag.sum()), "TP": tp, "FP": fp, "FN": fn, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def states_of(p, mism, t_inc, t_unc):
    """MddCalibrator.states 와 같은 식(벡터판). 0 correct · 1 uncertain · 2 incorrect."""
    inc = (p >= t_inc) & mism
    unc = (~inc) & ((p >= t_unc) | (mism & (p >= t_unc * 0.5)))
    return np.where(inc, 2, np.where(unc, 1, 0)).astype(np.int8)


def choose_thresholds(p, y, mism, target_precision, cover):
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    t_inc = None
    for t in grid:
        if prf((p >= t) & mism, y)["precision"] >= target_precision:
            t_inc = float(t)
            break
    if t_inc is None:
        t_inc = float(max(grid, key=lambda t: prf((p >= t) & mism, y)["precision"]))
    t_unc = None
    for t in np.round(np.arange(t_inc, 0.0, -0.01), 2):          # 판정식 그대로 커버리지를 잰다
        st = states_of(p, mism, t_inc, float(t))
        if prf(st > 0, y)["recall"] >= cover:
            t_unc = float(t)
            break
    if t_unc is None:
        t_unc = round(max(0.02, t_inc / 2.0), 2)
    return t_inc, t_unc


def summarize(p, T, t_inc, t_unc, ratios):
    y, mism, utt = T["y"], T["mism"], T["utt"]
    st = states_of(p, mism, t_inc, t_unc)
    n = float(len(y))
    tot_err = max(1, int(y.sum()))
    res = {"sites": int(len(y)), "utterances": T["n_utt"], "error_rate": round(float(y.mean()), 4),
           "incorrect": prf(st == 2, y), "states": {}}
    for code, name in ((2, "incorrect"), (1, "uncertain"), (0, "correct")):
        m = st == code
        res["states"][name] = {"share": round(float(m.mean()), 4),
                               "error_rate_inside": round(float((m & (y == 1)).sum()) / max(1, int(m.sum())), 4),
                               "share_of_true_errors": round(float((m & (y == 1)).sum()) / tot_err, 4)}
    # 발화 단위 — 재녹음 비율 기준 비교
    _, inv = np.unique(utt, return_inverse=True)
    nu = inv.max() + 1
    n_site = np.bincount(inv, minlength=nu).astype(float)
    n_unc = np.bincount(inv, weights=(st == 1).astype(float), minlength=nu)
    n_inc = np.bincount(inv, weights=(st == 2).astype(float), minlength=nu)
    n_true = np.bincount(inv, weights=y.astype(float), minlength=nu)
    n_fp = np.bincount(inv, weights=((st == 2) & (y == 0)).astype(float), minlength=nu)
    clean = n_true == 0
    res["utterance"] = []
    for rr in ratios:
        rer = (n_unc / np.maximum(1, n_site)) >= rr
        fb = (~rer) & (n_inc > 0)
        res["utterance"].append({
            "rerecord_ratio": rr, "rerecord_all": round(float(rer.mean()), 4),
            "rerecord_clean": round(float(rer[clean].mean()), 4) if clean.any() else None,
            "rerecord_with_error": round(float(rer[~clean].mean()), 4) if (~clean).any() else None,
            "feedback_fp_clean": round(float((fb & clean & (n_fp > 0)).sum()) / max(1, int(clean.sum())), 4),
            "feedback_on_error_utts": round(float((fb & ~clean).sum()) / max(1, int((~clean).sum())), 4),
            "pass_clean": round(float(((~rer) & (~fb))[clean].mean()), 4) if clean.any() else None})
    return res, st


def cmd_select(a):
    t0 = time.time()
    T = build(a.gop, a.manifest, a.split, a.calib_json)
    spks = np.unique(T["spk"]); rng = np.random.RandomState(a.seed); rng.shuffle(spks)
    val_spk = set(spks[:int(len(spks) * a.valid_frac)])
    mv = np.array([s in val_spk for s in T["spk"]])
    TR, VA = subset(T, ~mv), subset(T, mv)
    print("[split] calibrator_train 화자 %d · 발화 %d · 자리 %d / calibrator_valid 화자 %d · 발화 %d · 자리 %d (%.0fs)"
          % (len(spks) - len(val_spk), TR["n_utt"], len(TR["y"]), len(val_spk), VA["n_utt"], len(VA["y"]),
             time.time() - t0), flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    os.makedirs(a.out_dir, exist_ok=True)
    report = {"protocol": "select — test 미사용. 학습=calibrator_train, 선택=calibrator_valid",
              "source": {"gop": os.path.abspath(a.gop), "manifest": os.path.abspath(a.manifest), "split": a.split},
              "split": {"seed": a.seed, "valid_frac": a.valid_frac,
                        "train": {"speakers": len(spks) - len(val_spk), "utterances": TR["n_utt"], "sites": int(len(TR["y"]))},
                        "valid": {"speakers": len(val_spk), "utterances": VA["n_utt"], "sites": int(len(VA["y"]))}},
              "targets": {"incorrect_precision": a.target_precision, "coverage": a.coverage},
              "valid_baselines": {"transcription_only": prf(VA["mism"], VA["y"])}, "variants": {}}
    if VA["svc"] is not None:
        report["valid_baselines"]["service_rule"] = prf(VA["mism"] & (VA["svc"] < SERVICE_FLAG_MAX), VA["y"])
    for name in a.variants:
        cols = column_index(FEATURE_SETS[name])
        clf = HistGradientBoostingClassifier(max_iter=a.max_iter, learning_rate=0.08, max_depth=6,
                                             min_samples_leaf=200, l2_regularization=1.0,
                                             random_state=a.seed).fit(TR["X"][:, cols], TR["y"])
        p = clf.predict_proba(VA["X"][:, cols])[:, 1]
        t_inc, t_unc = choose_thresholds(p, VA["y"], VA["mism"], a.target_precision, a.coverage)
        summ, _ = summarize(p, VA, t_inc, t_unc, a.rerecord_ratios)
        summ.update({"features": FEATURE_SETS[name], "n_features": len(cols),
                     "auc": round(float(roc_auc_score(VA["y"], p)), 4),
                     "ap": round(float(average_precision_score(VA["y"], p)), 4),
                     "thresholds": {"incorrect": t_inc, "uncertain": t_unc}})
        report["variants"][name] = summ
        bundle = {"model": clf, "feature_set": name, "feature_names": FEATURE_SETS[name],
                  "thresholds": {"incorrect": t_inc, "uncertain": t_unc, "rerecord_ratio": RERECORD_RATIO},
                  "meta": {"protocol": report["protocol"], "split": report["split"], "valid": summ,
                           "source": report["source"]}}
        joblib.dump(bundle, os.path.join(a.out_dir, "calibrator_%s.joblib" % name), compress=3)
        i = summ["incorrect"]
        print("[%-9s] %2d특징 · AUC %.4f AP %.4f · t_inc %.2f t_unc %.2f · incorrect P %.4f R %.4f F1 %.4f (%.0fs)"
              % (name, len(cols), summ["auc"], summ["ap"], t_inc, t_unc, i["precision"], i["recall"], i["f1"],
                 time.time() - t0), flush=True)
    best = max(report["variants"].values(), key=lambda v: v["incorrect"]["f1"])["incorrect"]["f1"]
    ok = [k for k, v in report["variants"].items() if best - v["incorrect"]["f1"] <= a.simplicity_tol]
    chosen = min(ok, key=lambda k: report["variants"][k]["n_features"])
    report["selection"] = {"rule": "valid incorrect F1 이 최고값에서 %.3f 안이면 가장 작은 묶음" % a.simplicity_tol,
                           "best_f1": best, "within_tol": ok, "chosen": chosen}
    json.dump(report, open(os.path.join(a.out_dir, "select_report.json"), "w"), ensure_ascii=False, indent=1)
    import shutil
    shutil.copyfile(os.path.join(a.out_dir, "calibrator_%s.joblib" % chosen),
                    os.path.join(a.out_dir, "mdd_calibrator.joblib"))
    print("[pick ] %s (허용폭 %.3f 안: %s) → mdd_calibrator.joblib 로 잠금" % (chosen, a.simplicity_tol, ok), flush=True)


def cmd_final(a):
    out = os.path.join(a.out_dir, "%s.json" % a.name)
    if os.path.exists(out) and not a.force:
        raise SystemExit("%s.json 이 이미 있다 — test 는 한 번만 본다. 정말 다시 볼 이유가 있으면 --force" % a.name)
    bundle = joblib.load(os.path.join(a.out_dir, "mdd_calibrator.joblib"))
    name, thr = bundle["feature_set"], bundle["thresholds"]
    cols = column_index(FEATURE_SETS[name])
    T = build(a.gop, a.manifest, a.split, a.calib_json)
    p = bundle["model"].predict_proba(T["X"][:, cols])[:, 1]
    summ, st = summarize(p, T, thr["incorrect"], thr["uncertain"], [thr.get("rerecord_ratio", RERECORD_RATIO)])
    y, mism = T["y"], T["mism"]
    inc = st == 2
    rules = {"transcription_only": mism, "calibrator_incorrect": inc}
    if T["svc"] is not None:
        rules["service_rule"] = mism & (T["svc"] < SERVICE_FLAG_MAX)
    res = {"protocol": "final — select 에서 잠근 번들을 test 에 한 번 적용", "feature_set": name,
           "thresholds": thr, "bundle_valid": bundle["meta"].get("valid", {}).get("incorrect"),
           "source": {"gop": os.path.abspath(a.gop), "manifest": os.path.abspath(a.manifest), "split": a.split},
           "summary": summ, "rules": dict((k, prf(v, y)) for k, v in rules.items()), "breakdown": {}}
    # 오류 유형 · 위치 · 약한 쌍 — 규칙마다 재현율/정밀도
    is_del = T["act"] == u"∅"
    groups = {"type:substitution": (y == 1) & ~is_del, "type:deletion": (y == 1) & is_del}
    for code, nm in POS_NAME.items():
        groups["pos:" + nm] = (y == 1) & (T["pos"] == code)
    for c, t in HARD_PAIRS:
        groups[u"pair:%s→%s" % (c, t)] = (y == 1) & (T["ph"] == c) & (T["act"] == t)
    for g, m in groups.items():
        row = {"true_errors": int(m.sum())}
        for k, v in rules.items():
            row["recall_" + k] = round(float((v & m).sum()) / max(1, int(m.sum())), 4)
        res["breakdown"][g] = row
    for code, nm in POS_NAME.items():                     # 위치별 정밀도(그 위치에서 지적한 것 중 진짜)
        m = T["pos"] == code
        res["breakdown"]["pos:" + nm].update(dict(("precision_" + k, prf(v & m, y)["precision"]) for k, v in rules.items()))
    res["insertion"] = "해당 없음 — 삽입은 표준발음에 자리가 없어 자리 단위 보정기가 판정하지 않는다"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    for k, v in res["rules"].items():
        print("[final] %-22s %s" % (k, v))
    print("[final] 상태: %s" % summ["states"])
    print("[final] 발화: %s" % summ["utterance"][0])
    for g in ("type:substitution", "type:deletion", u"pos:중성", u"pos:초성", u"pos:종성") + tuple(
            u"pair:%s→%s" % pr for pr in HARD_PAIRS[:5]):
        r = res["breakdown"][g]
        print("[final] %-16s 오류 %6d · 재현율 전사 %.3f / 서비스 %s / 보정기 %.3f" % (
            g, r["true_errors"], r["recall_transcription_only"], r.get("recall_service_rule"), r["recall_calibrator_incorrect"]))
    print("[saved] %s" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for nm in ("select", "final"):
        s = sub.add_parser(nm)
        s.add_argument("--gop", required=True)
        s.add_argument("--manifest", required=True)
        s.add_argument("--split", required=True)
        s.add_argument("--out-dir", dest="out_dir", default=os.path.join(str(MDD_OUTPUTS), "analysis", "mdd_calibrator"))
        s.add_argument("--calib-json", dest="calib_json",
                       default=os.path.join(str(MDD_OUTPUTS), "score_phoneme", "calibration.json"),
                       help="현재 서비스 규칙(음소점수<60) 재현용 — 읽기만 한다")
    s = sub.choices["select"]
    s.add_argument("--variants", nargs="+", default=["full", "full_mism", "tiny", "minimal"])
    s.add_argument("--valid-frac", dest="valid_frac", type=float, default=0.2)
    s.add_argument("--target-precision", dest="target_precision", type=float, default=0.65)
    s.add_argument("--coverage", type=float, default=0.60, help="incorrect+uncertain 이 덮을 진짜 오류 비율")
    s.add_argument("--rerecord-ratios", dest="rerecord_ratios", nargs="+", type=float, default=[0.2, 0.25, 0.3, 0.4])
    s.add_argument("--simplicity-tol", dest="simplicity_tol", type=float, default=0.005)
    s.add_argument("--max-iter", dest="max_iter", type=int, default=400)
    s.add_argument("--seed", type=int, default=42)
    sub.choices["final"].add_argument("--force", action="store_true")
    sub.choices["final"].add_argument("--name", default="final_test", help="결과 파일 이름(test 가 여러 벌일 때)")
    a = ap.parse_args()
    if a.cmd == "select":
        cmd_select(a)
    elif a.cmd == "final":
        cmd_final(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
