#!/usr/bin/env bash
# CTC 실험 실행기 — 재현 정보를 먼저 남기고 fairseq-hydra-train 을 포그라운드로 돌린다 (tmux 안에서 실행).
#
#   bash tasks/pronunciation-assessment/scripts/train/run_ctc_experiment.sh <실험이름> <config-name> <GPU id> [hydra override ...]
#   예) bash .../run_ctc_experiment.sh e1_foreign_clean e1_foreign_clean 0
#
# 기록: $MDD_OUTPUTS/experiments/<실험이름>/{run_info.json, train.log, config.yaml}
#   git commit·dirty 여부 · seed · manifest 경로·sha256 · config · 초기화 체크포인트 · GPU·CUDA·PyTorch·fairseq · 시작/종료 시각
# 이미 기록 폴더가 있으면 멈춘다(덮어쓰기 금지) — 이어서 돌리려면 RESUME=1.

set -eo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 3 ]]; then sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
NAME="$1"; CFG="$2"; GPU="$3"; shift 3

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
T="$ROOT/tasks/pronunciation-assessment"
EXP="$MDD_OUTPUTS/experiments/$NAME"
[[ -f "$T/configs/$CFG.yaml" ]] || { echo "config 없음: $T/configs/$CFG.yaml" >&2; exit 1; }
if [[ -e "$EXP/run_info.json" && "${RESUME:-0}" != 1 ]]; then echo "이미 있다(덮어쓰기 금지): $EXP  — 이어서 돌리려면 RESUME=1" >&2; exit 1; fi
mkdir -p "$EXP"

# GPU 점유 확인 — 다른 프로세스가 쓰는 메모리를 보여주고 기록한다
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -i "$GPU" | tee "$EXP/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >> "$EXP/gpu_before.csv" || true

export CUDA_VISIBLE_DEVICES="$GPU"
cp "$T/configs/$CFG.yaml" "$EXP/config.yaml"

DATA=$(grep -E '^\s*data:' "$T/configs/$CFG.yaml" | head -1 | sed -E 's/.*data:\s*//; s/\s*#.*//')
DATA=$(eval echo "$(echo "$DATA" | sed -E 's/\$\{env:([A-Z_]+)\}/${\1}/g')")
for o in "$@"; do [[ "$o" == task.data=* ]] && DATA="${o#task.data=}"; done
MANI_SHA=$(cat "$DATA"/train.tsv "$DATA"/train.ltr "$DATA"/valid.tsv "$DATA"/valid.ltr | sha256sum | cut -d' ' -f1)

$PY - "$EXP/run_info.json" "$NAME" "$CFG" "$GPU" "$DATA" "$MANI_SHA" "$ROOT" "$FAIRSEQ_ROOT" "$@" <<'EOF'
import json, subprocess, sys, time, torch, fairseq, os
out, name, cfg, gpu, data, sha, root, fs = sys.argv[1:9]
over = sys.argv[9:]
g = lambda *c, cwd=None: subprocess.run(c, cwd=cwd, capture_output=True, text=True).stdout.strip()
info = {
    "experiment": name, "config_name": cfg, "overrides": over, "cuda_visible_devices": gpu,
    "git_commit": g("git", "rev-parse", "HEAD", cwd=root), "git_dirty_files": len(g("git", "status", "--short", cwd=root).splitlines()),
    "manifest": data, "manifest_train_valid_sha256": sha,
    "manifest_stats": json.load(open(os.path.join(data, "stats.json"))) .get("splits", {}).get("train") if os.path.exists(os.path.join(data, "stats.json")) else None,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    "fairseq": fairseq.__version__, "fairseq_commit": g("git", "rev-parse", "HEAD", cwd=fs),
    "start": time.strftime("%Y-%m-%d %H:%M:%S"),
}
if info["manifest_stats"]:
    info["manifest_stats"].pop("label_token_dist", None); info["manifest_stats"].pop("language", None)
json.dump(info, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "manifest_stats"}, ensure_ascii=False))
EOF

cd "$EXP"
# conda env 를 activate 하지 않고 $PY 절대경로만 쓰므로 PATH 에 fairseq-hydra-train 이 없다 — env 의 bin 에서 직접 부른다
HYDRA_TRAIN="$(dirname "$PY")/fairseq-hydra-train"
[[ -x "$HYDRA_TRAIN" ]] || { echo "fairseq-hydra-train 없음: $HYDRA_TRAIN" >&2; exit 1; }
set +e
"$HYDRA_TRAIN" --config-dir "$T/configs" --config-name "$CFG" hydra.run.dir="$EXP/hydra" "$@" 2>&1 | tee -a "$EXP/train.log"
RC=${PIPESTATUS[0]}
set -e
$PY - "$EXP/run_info.json" "$RC" <<'EOF'
import json, sys, time
p, rc = sys.argv[1], int(sys.argv[2])
info = json.load(open(p)); info["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); info["exit_code"] = rc
json.dump(info, open(p, "w"), ensure_ascii=False, indent=1)
EOF
echo "rc=$RC"
exit $RC
