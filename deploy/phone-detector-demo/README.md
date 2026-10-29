# 음소 판별기 데모 — "어떤 문장이든 — 발음 진단"

문장을 외우지 않는 음소 판별기(`mdd.phone_detector`, results/15)의 웹 데모. API 는 `app.py` 머리말 참고.

## 다시 띄우기 (GPU 서버 btdevgpu02 기준)

2026-09-21 ~ 10-01 에 tmux `nv-demo` 에서 8921 로 돌던 명령 그대로다(10-01 사용 없음으로 내림).

```bash
tmux new-session -d -s nv-demo "cd <저장소 루트> && source tools/env.sh && \
  export PD_ENSEMBLE_JSON=/data/speech-ai/out/novelty/detector/ensemble_full.json \
         PD_SAMPLES_JSON=/data/speech-ai/out/novelty/demo/samples.json && \
  \$PY -W ignore deploy/phone-detector-demo/app.py --port 8921 2>&1 \
  | grep --line-buffered -v tensorboardX | tee /data/speech-ai/out/novelty/jobs/demo.log"
```

- GPU 메모리 약 2 GB. 0.0.0.0:8921 로 열린다(사내망 · tailnet). funnel 에는 걸려 있지 않았다.
- 로그 `/data/speech-ai/out/novelty/jobs/demo.log` 는 `tee`(덮어쓰기)라 다시 띄우면 이전 로그가 지워진다 — 필요하면 먼저 옮겨 둘 것.
