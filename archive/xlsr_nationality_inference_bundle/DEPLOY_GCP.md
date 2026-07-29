# GCP 배포 가이드 (초보자용) — XLS-R 국적 분류 (policy31, 31-class)

목표: 이 번들을 **GCP의 리눅스 서버(VM) 하나**에 올려서, 웹에서 음성을 보내면
국적 top-5를 돌려주는 **HTTP API**로 띄운다. (서버리스 등 복잡한 건 나중에)

전체 그림:
```
브라우저/웹 --(음성 파일)--> [GCP VM: FastAPI(app.py) + XLS-R 2B] --(top-5 JSON)--> 응답
```

> ⚠️ 모델이 큼(XLS-R 2B). **GPU 권장**(요청당 ~1초). CPU도 되지만 ~5~15초/요청(데모용).

---

## 0. 준비물
- 구글 계정 + **결제 등록된 GCP 프로젝트** (신규 가입 시 무료 크레딧 $300 있음).
- 명령은 전부 **브라우저의 Cloud Shell**(GCP 콘솔 우상단 `>_` 아이콘)에서 실행하면 됨 — 로컬 설치 불필요.

## 1. 프로젝트/결제/API 활성화 (콘솔에서 클릭)
1. console.cloud.google.com → 프로젝트 만들기 (예: `nationality-app`)
2. 결제(Billing) 연결
3. Compute Engine API 활성화: 콘솔 검색창 "Compute Engine API" → **사용 설정**

## 2. (GPU 쓸 경우) GPU 할당량 요청 — 신규 계정은 0이라 필수
- 콘솔: **IAM 및 관리자 → 할당량(Quotas)** → 필터 `GPUs (all regions)` → 한도 1로 증가 요청.
- 보통 몇 분~하루. **귀찮으면 일단 CPU로 시작**(아래 4-B)하고 나중에 GPU 전환.

## 3. Cloud Shell 열고 변수 설정
콘솔 우상단 `>_` (Cloud Shell) 클릭 후:
```bash
gcloud config set project <프로젝트ID>
export ZONE=asia-northeast3-b   # 서울 리전
```

## 4-A. VM 만들기 — **GPU(T4) + 딥러닝 이미지** (권장)
PyTorch+CUDA가 미리 깔린 Deep Learning VM 이미지라 드라이버 고생 없음:
```bash
gcloud compute instances create xlsr-api \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu121-debian-11 --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=120GB \
  --metadata="install-nvidia-driver=True" \
  --tags=http-server
```

## 4-B. VM 만들기 — **CPU만** (할당량 불필요, 느림)
```bash
gcloud compute instances create xlsr-api \
  --zone=$ZONE --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=60GB --tags=http-server
```

## 5. 방화벽 열기 (포트 8000)
```bash
gcloud compute firewall-rules create allow-8000 \
  --allow=tcp:8000 --target-tags=http-server --source-ranges=0.0.0.0/0
```

## 6. 번들을 VM으로 복사
이 번들 폴더(`xlsr_nationality_inference_bundle`, app.py 포함)를 VM으로:
```bash
# (로컬/이 서버에서) 폴더를 GCS 거치거나 직접 scp
gcloud compute scp --recurse --zone=$ZONE \
  $NATIONALITY_OUTPUTS/xlsr_nationality_inference_bundle xlsr-api:~/bundle
```

## 7. VM 접속해서 의존성 설치 + 서버 실행
```bash
gcloud compute ssh xlsr-api --zone=$ZONE
# --- 이하 VM 안에서 ---
cd ~/bundle/xlsr_nationality_inference_bundle
# (GPU 딥러닝 이미지면 torch/torchaudio 이미 있음. CPU VM이면 torch도 설치)
pip install --user fastapi "uvicorn[standard]" python-multipart \
    transformers accelerate safetensors soundfile torchaudio
# 첫 실행 때 XLS-R 2B(~8GB) 자동 다운로드 → 수 분
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
`[startup] ready ...` 가 뜨면 준비 완료.

## 8. 테스트 (다른 터미널/Cloud Shell에서)
```bash
IP=$(gcloud compute instances describe xlsr-api --zone=$ZONE \
     --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
curl -F "audio=@sample.wav" http://$IP:8000/predict
# -> {"top":[{"label":"중국어권","prob":0.83}, ...], "device":"cuda", "took_sec":1.1}
curl http://$IP:8000/health
```

## 9. 항상 켜두기 (SSH 끊겨도 유지) — systemd
VM 안에서:
```bash
sudo tee /etc/systemd/system/xlsr.service >/dev/null <<EOF
[Unit]
Description=XLS-R nationality API
[Service]
User=$USER
WorkingDirectory=$HOME/bundle/xlsr_nationality_inference_bundle
ExecStart=$(which uvicorn) app:app --host 0.0.0.0 --port 8000
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now xlsr.service
```

---

## 비용 / 운영 팁
- **T4 VM ≈ 시간당 $0.35~0.5** (24시간이면 월 ~$250~350). **안 쓸 땐 `gcloud compute instances stop xlsr-api`** 로 정지(과금 멈춤, 재시작 빠름).
- CPU(e2-standard-4) ≈ 월 ~$100, 느리지만 데모엔 충분.
- 처음엔 **GPU VM 1대로 시작 → 트래픽 늘면** Cloud Run(GPU)·Vertex AI·로드밸런서로 확장.

## 자주 막히는 지점 (gotcha)
1. **브라우저 녹음은 보통 webm/opus** → `soundfile`이 못 읽음. 해결: 프론트에서 **WAV로 보내기**(JS 라이브러리) 또는 서버에 `ffmpeg`/`pydub` 추가해 변환.
2. **HTTPS/도메인**: 지금은 `http://IP:8000`. 실서비스는 nginx + certbot(무료 SSL) 또는 GCP 로드밸런서로 https 붙이기.
3. **모델 다운로드 8GB**: 첫 실행만. 디스크 120GB 권장(모델 캐시 `~/.cache/huggingface`).
4. **CORS**: 웹 프론트에서 직접 호출하면 `app.py`에 `fastapi.middleware.cors` 추가 필요.
5. 라벨은 **31개 언어권**(`labels/class_labels.csv`). 한국어 클래스 포함 버전(policy32)을 원하면 그 체크포인트로 교체.
