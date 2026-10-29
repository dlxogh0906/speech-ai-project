#!/usr/bin/env bash
# 문항 hold-out Stage 2 실험 하나를 끝까지 돌린다: 학습 → valid 로 step 선택 → test 두 벌 채점
#   → 새 인코더로 margin 추출 → 보정기 select/final (A/B/C/D 의 C·D).
#
#   bash tasks/pronunciation-assessment/scripts/train/run_holdout_experiment.sh H0
#   bash tasks/pronunciation-assessment/scripts/train/run_holdout_experiment.sh H4 0.5
#
# H0 = 가중 없음(가중치 전부 1 · 같은 loss 경로) · H1 삽입 · H2 약한 자음쌍 · H3 모음 · H4 셋 다.
# H0 이외는 H0 의 선택 결과를 기준으로 정상자리오류 제약을 건다 → H0 을 먼저 끝낼 것.
# 모든 단계는 산출물이 있으면 건너뛴다(중간에 끊겨도 다시 실행하면 이어서 간다).
# ⚠️ GPU 0 을 쓴다 — 같은 GPU 의 서비스(8932 등) 응답이 느려질 수 있다. 서버는 건드리지 않는다.
set -u
H=${1:?"H0|H1|H2|H3|H4"}; B=${2:-0.5}
cd "$(dirname "$0")/../../../.." || exit 1
source tools/env.sh >/dev/null
export HF_HOME=/data/speech-ai/corpus/weights/external PYTHONPATH=/data/speech-ai/tools/hf_py38:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
T=tasks/pronunciation-assessment
M=$MDD_OUTPUTS/manifest_holdout_v1
if [[ $H == H0 ]]; then NAME=holdout_v1_H0; W=$M/weights_b0.5/weights_H0.txt; REF=""
else NAME=holdout_v1_${H}_b$B; W=$M/weights_b$B/weights_$H.txt
     REF="--reference $MDD_OUTPUTS/experiments/holdout_v1/holdout_v1_H0/selection.json"; fi
CK=$SPEECH_AI_DATA/checkpoints/$NAME
R=$MDD_OUTPUTS/experiments/holdout_v1/$NAME
V=$M/_views
mkdir -p "$R/sel" "$R/test" "$R/margin" "$R/calib"
log(){ echo "[$(date '+%F %T')] [$NAME] $*"; }
[[ -f $W ]] || { log "가중치 파일 없음: $W"; exit 1; }
[[ -n $REF && ! -f $MDD_OUTPUTS/experiments/holdout_v1/holdout_v1_H0/selection.json ]] && { log "H0 선택 결과가 없다 — H0 먼저"; exit 1; }

# 0) gop 추출기용 보기 매니페스트 — 추출기(mdd.gop_feats_v2, API 가 import 하는 파일이라 건드리지 않는다)는
#    tsv 첫 줄 루트를 무시하고 "/" + 경로로 읽고, <split>.canon 을 요구한다. 그 형식에 맞춰 새로 쓴다.
if [[ ! -f $V/done ]]; then
  $PY - "$M" "$V" "$MDD_OUTPUTS/cache/foreign_tag_lab_parsed.jsonl.gz" <<'PY'
import gzip, json, os, sys
M, V, CACHE = sys.argv[1:4]
def rows_abs(split):
    lines = open(os.path.join(M, "%s.tsv" % split), encoding="utf-8").read().splitlines()
    root = lines[0].strip()
    out = []
    for l in lines[1:]:
        rel, fr = l.split("\t")
        out.append((os.path.join(root, rel).lstrip("/"), fr))
    return out
def write(view, rows, ltr, canon, dst="test"):
    d = os.path.join(V, view); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s.tsv" % dst), "w", encoding="utf-8") as f:
        f.write("/\n")
        for r, fr in rows: f.write("%s\t%s\n" % (r, fr))
    open(os.path.join(d, "%s.ltr" % dst), "w", encoding="utf-8").write("".join(x + "\n" for x in ltr))
    open(os.path.join(d, "%s.canon" % dst), "w", encoding="utf-8").write("".join(x + "\n" for x in canon))
    t = os.path.join(d, "dict.ltr.txt")
    if not os.path.exists(t): os.symlink(os.path.join(M, "dict.ltr.txt"), t)
    return len(rows)
rd = lambda sp, ext: open(os.path.join(M, "%s.%s" % (sp, ext)), encoding="utf-8").read().splitlines()
n1 = write("test_seen", rows_abs("test"), rd("test", "ltr"), rd("test", "canon"))
n2 = write("test_unseen", rows_abs("test_unseen"), rd("test_unseen", "ltr"), rd("test_unseen", "canon"))
# 학습 margin 용 — 외국인 행만 · 발화 중복 제거 · canonical 은 캐시에서
rows, ltr, src = rows_abs("train"), rd("train", "ltr"), rd("train", "source")
keep, seen = [], set()
for (r, fr), l, so in zip(rows, ltr, src):
    if so == "native" or r in seen: continue
    seen.add(r); keep.append((r, fr, l))
stems = set(os.path.basename(r)[:-4] for r, _, _ in keep)
canon = {}
with gzip.open(CACHE, "rt", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["stem"] in stems: canon[d["stem"]] = " ".join(d.get("canon") or [])
keep = [k for k in keep if canon.get(os.path.basename(k[0])[:-4])]
n3 = write("train_foreign", [(r, fr) for r, fr, _ in keep], [l for _, _, l in keep],
           [canon[os.path.basename(r)[:-4]] for r, _, _ in keep], dst="train")
open(os.path.join(V, "done"), "w").write("ok\n")
print("[views] test_seen %d · test_unseen %d · train_foreign %d" % (n1, n2, n3))
PY
fi

# 1) 학습 — 원래 Stage 2 와 같은 설정. 바뀐 것은 매니페스트(hold-out)와 --weights 뿐
if [[ ! -d $CK/step_40000 ]]; then
  log "학습 시작 (weights=$W)"
  $PY $T/scripts/train/train_w2vbert_ctc.py --manifest $M --valid-split valid_unseen \
    --model $SPEECH_AI_DATA/checkpoints/w2vbert_stage1/step_10000 --save-dir $CK \
    --max-update 40000 --lr 1e-5 --freeze-updates 0 --save-interval 2000 --valid-n 10000 --weights $W \
    2>&1 | grep -v -E "Warning|warn|conv1d"
fi
[[ -d $CK/step_40000 ]] || { log "학습 실패 — 중단"; exit 1; }

# 2) step 채점 (valid_seen · valid_unseen) — 뒤쪽 절반만, 4k 간격
for s in 16000 20000 24000 28000 32000 36000 40000; do
  for sp in valid valid_unseen; do
    o=$R/sel/step_${s}_$sp
    [[ -f $o/summary.json ]] && continue
    [[ -f $R/sel/step_${s}_$sp.preds.jsonl ]] || $PY $T/scripts/eval/decode_hf_ctc.py --model $CK/step_$s \
        --manifest $M --split $sp --out $R/sel/step_${s}_$sp 2>&1 | grep -E "^\[preds\]"
    $PY $T/scripts/eval/mdd_error_analysis.py --manifest $M --split $sp --preds $R/sel/step_${s}_$sp.preds.jsonl \
        --out $R/sel --name step_${s}_$sp --no-plots 2>&1 | grep -E "baseline"
  done
done
[[ -f $R/selection.json ]] || $PY $T/scripts/eval/select_holdout_ckpt.py --sel-dir $R/sel --out $R/selection.json $REF
S=$($PY -c "import json;print(json.load(open('$R/selection.json'))['chosen']['step'])")
FD=$CK/step_$S; log "선택 step $S"

# 3) test 두 벌 (seen 문항 · unseen 문항) — 선택 후 한 번
for sp in test test_unseen; do
  [[ -f $R/test/$sp.preds.jsonl ]] || $PY $T/scripts/eval/decode_hf_ctc.py --model $FD --manifest $M --split $sp \
      --out $R/test/$sp 2>&1 | grep -E "^\[preds\]"
  [[ -f $R/test/$sp/summary.json ]] || $PY $T/scripts/eval/mdd_error_analysis.py --manifest $M --split $sp \
      --preds $R/test/$sp.preds.jsonl --out $R/test --name $sp 2>&1 | grep -E "baseline|done"
done

# 4) 새 인코더의 margin — 보정기용 (학습 20만 · test 두 벌)
[[ -f $R/margin/train200k.jsonl ]] || $PY $T/scripts/score/extract_gop_feats.py --split train --n 200000 \
    --hf-ckpt $FD --manifest $V/train_foreign --out $R/margin/train200k.jsonl 2>&1 | grep -E "^\[done\]|Error"
for sp in test_seen test_unseen; do
  [[ -f $R/margin/$sp.jsonl ]] || $PY $T/scripts/score/extract_gop_feats.py --split test --n 1000000 \
      --hf-ckpt $FD --manifest $V/$sp --out $R/margin/$sp.jsonl 2>&1 | grep -E "^\[done\]|Error"
done

# 5) 보정기 — 이 인코더로 다시 학습(select: valid 만) → test 두 벌에 한 번씩 (D)
[[ -f $R/calib/mdd_calibrator.joblib ]] || $PY $T/scripts/score/train_mdd_calibrator.py select \
    --gop $R/margin/train200k.jsonl --manifest $V/train_foreign --split train --out-dir $R/calib 2>&1 | grep -E "^\[(split|full|tiny|minimal|full_mism|pick)"
for sp in test_seen test_unseen; do
  [[ -f $R/calib/final_$sp.json ]] || $PY $T/scripts/score/train_mdd_calibrator.py final \
      --gop $R/margin/$sp.jsonl --manifest $V/$sp --split test --out-dir $R/calib --name final_$sp 2>&1 | grep -E "^\[final\] (transcription|calibrator|service)"
done
log "완료 → $R"
