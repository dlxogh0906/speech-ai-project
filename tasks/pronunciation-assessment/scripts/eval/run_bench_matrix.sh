#!/usr/bin/env bash
# 발음 진단 인코더 — 경량화 변형 × 장비 조건 속도 매트릭스.
#
# 사용법:
#   bash tasks/pronunciation-assessment/scripts/eval/run_bench_matrix.sh <출력폴더> <onnx폴더> [<quant폴더>]
#
#   <onnx폴더>/mdd_fp32.onnx 와 <quant폴더>/mdd_{dyn_matmul,dyn_matmul_pc,static_qdq,fp16}.onnx 중 있는 것만 잰다.
#
# 조건
#   CPU  코어 16 / 8 / 4 / 2 — `taskset` 으로 코어를 묶어 작은 인스턴스를 흉내 낸다(스레드 수 = 코어 수)
#        torch fp32(서빙 경로) · ORT fp32 · ORT int8 변형들
#   GPU  torch fp16(서빙 경로) · ORT CUDA fp32 · ORT CUDA fp16
#
# ⚠️ 단독 실행 전제(docs/ENVIRONMENT.md 함정 9). 시작할 때 1분 load average 가 IDLE_LOAD(기본 2) 를
#    넘으면 멈춘다. 이미 떠 있는 서비스(GPU 공유 등)는 결과 summary 에 그대로 적는다.
#    환경변수: CPU_LIST="16 8 4 2"  DURATIONS=3,6,12  REPS=7  SKIP_GPU=1  SKIP_CPU=1
#              TORCH_CKPT=<체크포인트>  (torch 측정 대상. 기본은 .env 의 MDD_SERVE_CKPT — 학생 모델을 잴 때 지정)
set -uo pipefail

case "${1:-}" in -h|--help|"") sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
OUT="$1"; ONNX_DIR="${2:?onnx 폴더}"; QDIR="${3:-$ONNX_DIR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$ROOT/tools/env.sh" >/dev/null
B="$PY $ROOT/tasks/pronunciation-assessment/scripts/eval/bench_backends.py"
DUR="${DURATIONS:-3,6,12}"; REPS="${REPS:-7}"
mkdir -p "$OUT"

load1=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$load1" -v m="${IDLE_LOAD:-2}" 'BEGIN{exit !(l>m)}'; then
    echo "[stop] load average $load1 > ${IDLE_LOAD:-2} — 다른 작업이 돈다. 끝난 뒤 다시 실행할 것" >&2; exit 1
fi
{ date; uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null; } > "$OUT/env.txt"

run() {  # run <이름> <taskset 코어 or -> <인자...>
    local name="$1" cores="$2"; shift 2
    if [ -s "$OUT/$name.json" ]; then echo "[skip] $name"; return; fi
    echo "── $name"
    if [ "$cores" = "-" ]; then
        $B "$@" --durations "$DUR" --reps "$REPS" --tag "$name" --out "$OUT/$name.json" 2>&1 | grep -E "초 |적재|Error|error|stop" || true
    else
        taskset -c "$cores" $B "$@" --durations "$DUR" --reps "$REPS" --tag "$name" --out "$OUT/$name.json" 2>&1 | grep -E "초 |적재|Error|error|stop" || true
    fi
}

if [ "${SKIP_CPU:-0}" != 1 ]; then
    for th in ${CPU_LIST:-16 8 4 2}; do
        cores="0-$((th - 1))"
        run "cpu${th}_torch_fp32" "$cores" --backend torch --device cpu --threads "$th" ${TORCH_CKPT:+--ckpt "$TORCH_CKPT"}
        [ -f "$ONNX_DIR/mdd_fp32.onnx" ] && run "cpu${th}_ort_fp32" "$cores" --backend ort --device cpu --threads "$th" --onnx "$ONNX_DIR/mdd_fp32.onnx"
        for v in dyn_matmul dyn_matmul_pc static_qdq; do
            [ -f "$QDIR/mdd_$v.onnx" ] && run "cpu${th}_ort_$v" "$cores" --backend ort --device cpu --threads "$th" --onnx "$QDIR/mdd_$v.onnx"
        done
    done
fi
if [ "${SKIP_GPU:-0}" != 1 ]; then
    run "gpu_torch_fp16" - --backend torch --device cuda ${TORCH_CKPT:+--ckpt "$TORCH_CKPT"}
    # onnxruntime-gpu 가 CUDA 11 라이브러리(libcurand.so.10 등)를 못 찾는 머신이 있다 — 그 폴더를 ORT_CUDA_LIBS 로 준다.
    # 머신마다 경로가 달라 여기 박지 않는다. 지정하면 ORT GPU 측정에만 LD_LIBRARY_PATH 로 건다.
    if [ -n "${ORT_CUDA_LIBS:-}" ]; then _OLD_LD="${LD_LIBRARY_PATH:-}"; export LD_LIBRARY_PATH="$ORT_CUDA_LIBS${_OLD_LD:+:$_OLD_LD}"; fi
    [ -f "$ONNX_DIR/mdd_fp32.onnx" ] && run "gpu_ort_fp32" - --backend ort --device cuda --onnx "$ONNX_DIR/mdd_fp32.onnx"
    [ -f "$QDIR/mdd_fp16.onnx" ] && run "gpu_ort_fp16" - --backend ort --device cuda --onnx "$QDIR/mdd_fp16.onnx"
    if [ -n "${ORT_CUDA_LIBS:-}" ]; then export LD_LIBRARY_PATH="$_OLD_LD"; fi
fi

"$PY" - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "*.json")))]
durs = sorted({r["sec"] for x in rows for r in x["rows"]})
lines = ["| 조건 | 파일 MiB | RSS MiB | " + " | ".join(f"{d:g}초 ms" for d in durs) + " | 6초 실시간배수 |",
         "|---|--:|--:|" + "--:|" * len(durs) + "--:|"]
for x in rows:
    by = {r["sec"]: r for r in x["rows"]}
    six = by.get(6.0, {}).get("rtf_x", "")
    lines.append(f"| {x['tag']} | {x['model_bytes']/2**20:.0f} | {x['rss_mb']['peak']} | "
                 + " | ".join(str(by[d]["median_ms"]) if d in by else "" for d in durs) + f" | {six} |")
md = "\n".join(lines)
open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(md + "\n\n측정 환경: env.txt\n")
print(md)
PY
