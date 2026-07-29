# 🚫 이 디렉터리는 비어 있다 — 의도적이다

**국적 분류 추론 API 는 다른 사람이 사용 중이라 이 저장소로 옮기지 않았다.**

- 원본: `$HOME/xlsr_infer_api/`
- 포트 8900 · Tailscale Funnel `/` 로 공개 · 17일 이상 무중단 가동 중
- 구성: `infer_api.py` + `encoder_l17.onnx`(915 MB) + `head_np.npz` + `silero_vad.onnx`
- 무음/소음 입력이면 `top1` 과 `predictions` 를 둘 다 `null` 로 반환한다
  (에너지 게이트 + torch 없는 silero-vad)

## 하지 말 것

- 파일 이동·수정·삭제
- 프로세스 재기동
- `$HOME/xlsr_infer_api.tar.gz` 삭제 (소유자 확인 전까지 보류)

## 해도 되는 것

- 읽기 (`cat`, `less`)
- `curl localhost:8900/...` 로 동작 확인

옮겨야 할 상황이 오면 소유자와 먼저 합의하고, `deploy/README.md` 의 "저장소 사본으로
운영을 옮기는 절차"를 따를 것.
