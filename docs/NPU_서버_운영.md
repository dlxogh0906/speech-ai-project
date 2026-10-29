# NPU 서버 운영 메모 (내부용)

작성 2026-09-27 · 갱신 2026-10-10 · **외부 개발자에게 보내지 않는다.** 개발자용 안내는 `docs/NPU_서버_API_안내.md`.

## 2026-10-28 사투리 분류 API 추가

- systemd `dialect-api`(127.0.0.1:8907) — EAT Large + XLS-R 2B 앙상블을 NPU 5번에서 돈다. 번들 `/root/dialect_serve/bundle`, 설명은 저장소 `deploy/dialect-api/npu/README_NPU.md`.
- 입력·결과 저장은 국적·발음과 같은 공통 기능이다: `/data/speech-logs/captures/dialect/`, 로그 `/data/speech-logs/runtime/dialect-api.log`, 배포 기록 `/data/speech-logs/deployments/dialect-npu-20261028.json`.
- Caddy 에 세 번째 주소 이름(사투리 → 8907)을 더했다. 토큰은 앱이 확인한다 — 값은 `/root/dialect_serve/api.env`(600). 백업 `Caddyfile.bak-20261028-1700`.
- NPU 5번 메모리 6.4 → 11.4 GiB. 같은 장치의 국적 워커 하나는 동시에 요청이 오면 2B 계산(약 0.2초)이 끝나길 기다린다.

## 2026-10-10 전체 요청 저장 적용

- 추가 블록스토리지 `/dev/vdb`(300GiB)를 XFS로 포맷해 `/data/speech-logs`에 마운트했다. `/etc/fstab`에는 UUID로 등록했다. 두 API의 systemd 설정에 `RequiresMountsFor=/data/speech-logs`를 넣어 저장 디스크가 준비된 뒤 시작한다.
- 웹·앱·데모를 구분하지 않고 **NPU API에 도달하는 모든 HTTP 요청**을 공통 ASGI 저장 기능으로 기록한다. 인증 토큰, 쿠키, 서명은 메타데이터에서 제거한다. HTML 페이지의 응답 본문은 저장하지 않는다.
- 적용 경로: 국적 `/predict`, `/predict_long`, `/predict_multi`, `/kd/*`와 기타 API 요청; 발음 `/speak`, `/sent.eval.kr`, `/word.judge`, `/session` 등. 성공뿐 아니라 400·401·422 및 서버가 반환하는 500도 기록한다. 입구(Caddy)가 API에 전달하기 전에 거절한 요청은 기존 Caddy 접근 로그에만 남는다.
- **원본 음성은 서버가 받은 업로드 파일의 바이트 그대로** 저장한다. 원래 파일명, 크기, SHA256, WAV 형식 정보도 남긴다. 국적 API가 8MiB 초과로 거절한 파일도 원본을 저장한다. 파일을 전송하지 않은 요청에는 메타데이터·응답만 있다.
- 전체 API 응답을 `result`로 보관한다. 국적은 각 구간·각 인코더의 **41개 라벨 확률**과 게이트 결과·합산 결과를, 문장 발음 평가는 음소 정렬·margin·점수·VAD·채점 특징을 추가로 기록한다.
- 새 저장 위치:
  - `/data/speech-logs/captures/nationality/YYYYMMDD/<rid>/request.json` 및 `audio-00.<확장자>`(여러 파일이면 01, 02…)
  - `/data/speech-logs/captures/pronunciation/YYYYMMDD/<rid>/request.json` 및 원본 파일
  - 각 날짜 폴더의 `captures.jsonl`: 완료 요청 목록. `request.json`을 처리 전에 써 두므로 프로세스가 종료돼도 진행 중이던 요청을 찾을 수 있다.
  - `/data/speech-logs/runtime/{nat-api,pron-api}.log`: 접근·실행·예외·저장 실패 로그. 기존 `/root/*_serve/serve.log`는 이 파일로 연결된다.
  - `/data/speech-logs/deployments/<deployment_id>.json`: 사용한 모델·점수 파일 SHA256과 서비스 설정; 같은 이름의 `-verification.json`: 적용 확인 결과.
- 응답 헤더 `X-Capture-ID`로 요청과 기록을 연결한다. 국적 `/predict`·`/predict_long`의 JSON `rid`도 같은 ID다. 발음 응답의 SpeechSuper 형식은 유지한다. `tokenId`/`recordId`, `userId`, 기준 문장, 전달된 세션·기기 정보를 함께 남긴다.
- 예전 자료는 `/data/speech-logs/legacy/{nationality-captures,nationality-uploads,pronunciation-uploads}`로 복사하고 기존 저장 폴더를 심볼릭 링크로 연결했다. 이전 코드와 저장 기능 확대본 `app_new.py`는 별개이며, 이번 적용은 실행 중이던 코드에 공통 수집 기능을 붙인 것이다.
- 운영 설정: `/etc/systemd/system/{nat-api,pron-api}.service.d/capture.conf`. `SPEECH_CAPTURE_DIR`와 `SPEECH_DEPLOYMENT_ID`를 설정한다. 코드: 저장소 `shared/request_capture.py` 및 두 서비스 번들의 `shared/request_capture.py`.
- 확인: 6개 추론 경로에서 적용 전후 판정 동일; 성공·오류 14건에서 원본 바이트·응답 일치·토큰 제거 확인. 500 기록은 모델을 중단시키지 않고 별도 ASGI 확인으로 검증했다. 모델 파일은 교체하지 않았다.
- 자동 삭제는 적용하지 않았다. 디스크 사용량은 `df -h /data/speech-logs`로 확인한다.

분석 요약(운영 확인 요청은 기본 제외):

```bash
/root/nat_serve/venv/bin/python /data/speech-logs/backups/speech_capture_report.py /data/speech-logs/captures --day 20261010
# Philippines 판정만 집계
/root/nat_serve/venv/bin/python /data/speech-logs/backups/speech_capture_report.py /data/speech-logs/captures --day 20261010 --label Philippines
```

아래의 10-09 `KD_CAPTURE_DIR` 설명은 예전 자료 형식이다. 현재의 새 요청은 위 저장 위치를 사용한다.

## 1. 서비스와 재시작

접속: Tailscale 로 `ssh root@<NPU-tailscale-IP>`.

**2026-10-05 부터 국적 분류 · 발음 평가 · 공개 입구(Caddy)는 systemd 서비스다.** 재부팅하거나 비정상 종료되면 자동으로 다시 뜬다. 아래 명령은 손으로 다시 띄울 때 쓴다(내부에서 `systemctl restart` 를 부른다).

| 포트 | 서비스 | 구성 | 다시 띄우기 (systemd 서비스) | 로그 · 녹음 |
|---|---|---|---|---|
| 8905 | 국적 분류 **운영** | 12초 고정 모델 · 프로세스 8개 · NPU 0~7 · `/predict_long` (10-05) | `bash /root/nat_serve/restart.sh` (`nat-api`) | `/root/nat_serve/serve.log` · **`/root/nat_serve/captures`(10-09~, 들어온 녹음 전부 + 판정 · 품질 지표)** · `/root/nat_serve/uploads`(예전 `/predict` 저장) |
| 8906 | 국적 분류 **실험**(길이 버킷) | 4·8·12초 버킷 · 프로세스 16개 · NPU 0~7 | `bash /root/nat_serve/start_bucket.sh 16` (수동 · 자동 시작 아님) | `/root/nat_serve/serve_b.log` · 저장 안 함 |
| 8932 | 발음 평가 | 3버킷(3.8·9.0·20.5초) · 프로세스 1개 · NPU 7 · 토큰 인증 · 데모 페이지는 토큰을 브라우저에서 입력받음(`--ask-token`) · **2026-10-17 부터 섞은 모델(`mix_2026-10-16/`) + 표준발음 보정(`--canon-fix`)** — `docs/한국어_발음평가_개선_NPU_배포.md` · **10-18: 짧은 문장 말소리 기준(음절당 0.08초, 18:29) · 예사→된·거센소리 화면 지적 예외(20:16)** — 백업 `backup_2026-10-18/` · `backup_2026-10-18_rule/` · **10-21 01:58: 100% 모델(`ft3000_2026-10-19/`) + 빠짐은 강제정렬 점수 50 미만만 지적(`--del-score-max 50`) + ㄴ↔ㄹ 예외(`--liquid-rule`)** — 백업 `backup_2026-10-20/` · `docs/발음평가_서비스_점검_2026-10-19.md` §8 · **10-22 01:18: 글자 판정 결함 수정(`speechsuper.py`, 통째로 안 들린 글자를 글자 단위로 판정)** — 백업 `backup_2026-10-22/` · `docs/발음평가_개선_반영_2026-10-21.md` §10 | `bash /root/pron_serve/start_pron.sh` (`pron-api`, 실행 내용 `run_8932.sh`) | `/root/pron_serve/serve.log` · `/root/pron_serve/uploads` |
| 8907 | 사투리 분류 (10-28) | EAT + 2B 12초 고정 `.rbln` · 프로세스 1개 · NPU 5 · 토큰 인증(앱) | `systemctl restart dialect-api` | `/data/speech-logs/runtime/dialect-api.log` · `/data/speech-logs/captures/dialect` |
| 443 | **공개 입구(서울 직통)** · Caddy | 주소 이름 3개: 발음 평가(10-04) → 8932 · 국적 분류(10-05) → 8905 · 사투리 분류(10-28) → 8907. 국적은 토큰 확인을 Caddy 가 한다 | 설정만 바꿨으면 `systemctl reload caddy-npu`(끊김 없음) · `bash /root/caddy/start_caddy.sh` (`caddy-npu`) | `/root/caddy/caddy.log` · `access.log`(발음) · `nat_access.log`(국적, 토큰 헤더는 지움) |

- 공개 주소 ① 서울 직통(Caddy 443): 웹 서버가 쓰는 주 경로다. **주소 값은 `/root/caddy/Caddyfile` 에만 두고 문서·저장소에 적지 않는다**(웹 요청). 웹은 환경변수(`NATIONALITY_URL` 등)로만 쓴다.
- 공개 주소 ② Tailscale Funnel(예비): `https://<NPU-funnel-주소>` → `127.0.0.1:8905`(국적 분류, 토큰 없음),
  `https://<NPU-funnel-주소>:8443` → `<NPU-tailscale-IP>:8932`(발음 평가)로 연결한다(`tailscale funnel status`). **`tailscale serve` 를 쓰면 funnel 이 꺼진다.**
- 발음 평가 토큰은 `/root/pron_serve/.token`(권한 600)에 있다. 바꾸려면 파일을 고치고 `start_pron.sh` 를 다시 실행한다.
- 국적 분류 직통 입구 토큰은 `/root/caddy/nat_api_token`(600)에 있고, Caddy 는 같은 값을 `/root/caddy/nat_auth.caddy`(600)로 읽는다. 바꾸려면 두 파일을 같이 고치고 `systemctl reload caddy-npu`. 헤더는 `X-API-Key: <토큰>` 또는 `Authorization: Bearer <토큰>`, `/health` 만 토큰 없이 열린다.
- 국적 분류 웹 브라우저 직접 호출(CORS) 허용 도메인은 `/root/nat_serve/cors_origins`(한 줄에 하나) — 고친 뒤 `restart.sh`.
- 국적 분류 NPU 번들의 첫 화면(`/root/nat_serve/bundle*/app/index.html`)에는 AI 허브 출처 줄을 넣었다. 저장소 사본에는 없다
  (GPU 서버의 기존 8905 서비스가 저장소 파일을 그대로 읽기 때문). `/read` 는 저장소·NPU 모두 들어 있다.
- 공인 IP(<NPU-공인-IP>)로는 443(Caddy)만 열려 있다(클라우드 보안그룹 · 80 막힘 → 인증서는 TLS-ALPN 으로만 받는다). 서비스 포트(8905·8906·8932)와 SSH(22)는 닫혀 있다. 그 밖의 들어오는 길은 funnel 과 Tailscale 이다.
- NPU 메모리(장당 15.7 GiB): 0~4번 · 6번 6.4 GiB · 5번 11.4 GiB(사투리 5.0 GiB 포함, 10-28) · 7번 11.4 GiB(발음 평가 5.0 GiB 포함).
- 파이썬 환경은 `/root/nat_serve/venv` (Python 3.12 · rebel 0.10.2) 하나를 두 서비스가 같이 쓴다.
- `/root/beaver` · `/root/nara` 는 다른 사람 것이다. 건드리지 않는다.
- 상태 확인: `curl -s localhost:8905/health` · `curl -s <NPU-tailscale-IP>:8932/health` · `rbln-stat`(NPU 사용량)
- **국적 API 수집분(2026-10-09~)**: `KD_CAPTURE_DIR=/root/nat_serve/captures`(`run_8905.sh`).
  - 날짜 폴더(700)에 `<rid>.<확장자>`(보낸 파일 그대로)와 `captures.jsonl` 이 쌓인다(600).
  - 기록에는 판정 · 녹음 길이 · 말소리 길이 · 음량 · 스펙트럼 평탄도 · 클라이언트 종류가 있고, IP 는 없다.
  - 요약은 `python /root/nat_serve/captures_report.py [국가]`(기본 Philippines). 확인용 요청은 User-Agent `npu-ops-test` 로 보내면 요약에서 빠진다.
  - **보관 기간은 아직 정하지 않았다(자동 삭제 없음).**

## 2. 운영 전에 할 일

| 항목 | 지금 | 필요한 것 |
|---|---|---|
| 인증 | 발음 평가는 토큰 필요 · 국적 분류는 서울 직통 입구에만 토큰(10-05). Funnel 예비 주소는 아직 토큰 없음 | 웹이 직통으로 옮긴 뒤 Funnel 예비 주소에도 토큰(또는 닫기) |
| 녹음 저장 | 국적 분류는 10-09 부터 들어온 녹음 전부를 품질 분석용으로 저장(`captures`) · 발음 평가는 전부 저장 | **보관 기간과 자동 삭제** 정하기, 사용자 안내(처리방침)에 반영 |
| 공개 경로 | 서울 직통(Caddy 443 · sslip.io 주소) + Funnel 예비 | 회사 도메인으로 바꾸기 |
| 브라우저 호출 | 국적 분류는 `cors_origins` 의 도메인만 허용(지금 로컬 개발 주소뿐) · 발음 평가는 CORS 없음 | 백엔드 경유, 또는 운영 웹 도메인 등록 |
| 장애 대응 | systemd 자동 시작·재시작(10-05) · 서버 1대 | 모니터링(상태 알림) |
| 발음 평가 동시 처리 | 1개 프로세스, 초당 약 16건 | 프로세스 여러 개 + 세션 공유 저장소 |
| 출처 표기 | 국적 분류 공개 페이지에만 들어 있다 | 앱·웹 안내 화면에 AI 허브 출처 표기(개발자용 안내 4절) |

성능 수치의 근거는 `docs/NPU_GPU_성능비교.md` 에 있다.
