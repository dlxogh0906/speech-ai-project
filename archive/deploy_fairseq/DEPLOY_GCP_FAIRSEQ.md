# GCP 배포 가이드 — XLS-R 국적 분류 (fairseq 인코더, 62-class = 61 외국어 + 한국어_모국어)

HuggingFace 버전이 아니라 **우리 로컬 fairseq XLS-R 2B 체크포인트**로 추론한다.
헤드(linear probe)가 fairseq feature로 학습됐기 때문에 인코더도 fairseq여야 정확도가 맞는다.

```
브라우저 --(음성)--> [GCP GPU VM: FastAPI(app.py) + fairseq XLS-R 2B] --(top-5 JSON)--> 응답
```

## 무엇을 올리나 (HF 버전과 다른 점)
HF 버전은 `pip install transformers` 후 모델을 인터넷에서 자동 다운로드했다.
fairseq 버전은 **모델 파일과 코드를 우리가 직접 올려야** 한다:

| 항목 | 파일 | 크기 | 비고 |
|---|---|---|---|
| 인코더 | `$SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt` | **8.1 GB** | 옵티마이저 제거판(원본 26GB 아님) |
| 분류 헤드 | `taeho/deploy_fairseq/head_fine62_16s.pt` | 0.5 MB | `{head, mean, std, labels}`, 62-class(16s 학습) |
| 코드 | `country_classifier/`, `fairseq/`(88MB), `app.py` | ~90 MB | git repo 그대로 |
| 라벨 | `class_labels.csv` | - | 프론트 표시용 |

> **GPU 필수.** 2B 인코더라 CPU면 요청당 수십 초~분 단위. T4(16GB) 이상 권장. 요청당 ~1초.

## 0. 준비물
- 결제 등록된 GCP 프로젝트(신규 $300 크레딧). 명령은 콘솔 우상단 **Cloud Shell**(`>_`)에서.
- GPU 할당량: **IAM → 할당량**에서 `GPUs (all regions)` 1 이상 요청(신규 계정은 0). 몇 분~하루.

## 1. 코드+헤드 묶기 (이 서버에서)
```bash
bash $NATIONALITY_OUTPUTS/deploy_fairseq/package.sh
# -> $NATIONALITY_OUTPUTS/deploy_fairseq/deploy_fairseq_code.tar.gz  (~90MB, 인코더 미포함)
```

## 2. GPU VM 생성 (딥러닝 이미지 = torch+CUDA+conda 내장)
```bash
gcloud config set project <프로젝트ID>
export ZONE=asia-northeast3-b   # 서울
gcloud compute instances create xlsr-api \
  --zone=$ZONE --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu121-debian-11 --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE --boot-disk-size=120GB \
  --metadata="install-nvidia-driver=True" --tags=http-server
gcloud compute firewall-rules create allow-8000 \
  --allow=tcp:8000 --target-tags=http-server --source-ranges=0.0.0.0/0
```

## 3. 파일 전송 (코드 + 8GB 인코더)
```bash
# 코드 묶음
gcloud compute scp --zone=$ZONE \
  $NATIONALITY_OUTPUTS/deploy_fairseq/deploy_fairseq_code.tar.gz xlsr-api:~/
# 인코더 8GB — scp가 느리면 GCS 경유 권장:
#   gsutil cp $SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt gs://<버킷>/  (이 서버)
#   gsutil cp gs://<버킷>/xlsr2_2B_1000k_inference.pt ~/   (VM 안)
gcloud compute scp --zone=$ZONE \
  $SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt xlsr-api:~/
```

## 4. VM 접속 → 풀기 → 환경 구성
```bash
gcloud compute ssh xlsr-api --zone=$ZONE
# --- 이하 VM 안 ---
mkdir -p ~/app && tar -xzf ~/deploy_fairseq_code.tar.gz -C ~/app
cd ~/app

# fairseq 전용 conda 환경 생성 (torch 2.1.2+cu118 + fairseq editable + omegaconf 핀)
# repo의 create_env.sh를 그대로 쓰되 fairseq 경로만 VM 기준으로 지정:
FAIRSEQ_DIR=~/app/fairseq bash scripts/create_env.sh
conda activate fairseq_xlsr

# 웹서버 의존성 (env에 없던 것만)
pip install fastapi "uvicorn[standard]" python-multipart soundfile
```
> `create_env.sh`가 실패하면 로그의 Fallback 안내(`--no-build-isolation` 또는
> `PYTHONPATH=~/app/fairseq`)를 따른다. `tiny` 백엔드는 fairseq 없이도 되지만 우리는
> XLS-R을 써야 하므로 fairseq import가 반드시 돼야 한다.

## 5. 서버 실행
```bash
cd ~/app/taeho/deploy_fairseq
export REPO_ROOT=~/app                         # country_classifier/ + fairseq/ 위치
export XLSR_CKPT=~/xlsr2_2B_1000k_inference.pt  # 위에서 올린 8GB
export HEAD_PT=~/app/taeho/deploy_fairseq/head_fine62_16s.pt
export MAX_SEC=16 TOP_K=5   # 헤드가 16s 기준으로 학습됨
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
`[startup] ready in ..s (62 classes)` 가 뜨면 완료. (인코더 8GB 로드에 수십 초)

## 6. 테스트 (다른 터미널/Cloud Shell)
```bash
IP=$(gcloud compute instances describe xlsr-api --zone=$ZONE \
     --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
curl http://$IP:8000/health
curl -F "audio=@sample.wav" http://$IP:8000/predict
# -> {"top":[{"label":"중국어","prob":0.83}, ...], "timing":{...}, "gpu":{...}}
```
**웹페이지**: 브라우저에서 `http://$IP:8000/` 접속 → 음성 파일 업로드하면
top-5 · 추론 속도(인코더/헤드/실시간 배속) · GPU 메모리(peak/reserved/전체)를 화면에 표시.

## 7. 상시 가동 (systemd)
```bash
ENV=$(conda info --base)/envs/fairseq_xlsr/bin
sudo tee /etc/systemd/system/xlsr.service >/dev/null <<EOF
[Unit]
Description=XLS-R nationality API (fairseq)
[Service]
User=$USER
WorkingDirectory=$HOME/app/taeho/deploy_fairseq
Environment=REPO_ROOT=$HOME/app
Environment=XLSR_CKPT=$HOME/xlsr2_2B_1000k_inference.pt
Environment=HEAD_PT=$HOME/app/taeho/deploy_fairseq/head_fine62_16s.pt
Environment=MAX_SEC=16
Environment=TOP_K=5
ExecStart=$ENV/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now xlsr.service
journalctl -u xlsr -f   # 로그 보기
```

## 비용 / 운영
- T4 VM ≈ 시간당 **$0.35~0.5** (24h면 월 ~$250~350). 안 쓸 땐 `gcloud compute instances stop xlsr-api` (과금 멈춤).
- 인코더 8GB는 부팅 디스크에 두면 재시작해도 유지. GCS에 백업해두면 VM 재생성 시 편함.

## 자주 막히는 지점
1. **`import fairseq` 실패**: `app.py`는 `country_classifier.model`이 fairseq_root를
   sys.path 맨 앞에 넣어 해결한다. `REPO_ROOT`가 `country_classifier/`와 `fairseq/`를
   **둘 다** 포함하는 디렉터리인지 확인. CWD를 `~/app`(repo 루트)로 두지 말고 위처럼
   `deploy_fairseq` 안에서 실행.
2. **브라우저 녹음(webm/opus)**: `soundfile`이 못 읽음 → 프론트에서 **WAV로** 보내거나
   서버에 `ffmpeg`/`pydub` 추가해 변환.
3. **GPU OOM**: T4 16GB면 fp16(autocast)로 충분. `MAX_SEC` 줄이면 메모리/지연 감소.
4. **HTTPS/도메인**: 지금은 `http://IP:8000`. 실서비스는 nginx + certbot 또는 GCP LB로 https.
5. CORS는 `app.py`에서 `allow_origins=["*"]`로 이미 열어둠(데모용). 운영 시 도메인 제한.

## 라벨 / 모델 메모
- 62-class = 외국어 모국어 61종 + `한국어_모국어`. 16s linear probe, test **top-1 ≈ 0.795,
  top-5 ≈ 0.932** (`head_fine62_16s_test_metrics.json`). 한국어_모국어 클래스는 top-1 ≈ 0.9998로
  원어민/외국인 구분이 매우 정확. 외국어 61-way만 보면 top-1 ≈ 0.760. 롱테일 희소 클래스는 top-1이 낮다.
- 더 좋은 헤드로 교체하려면 같은 `{head, mu, sd, labels}` 포맷이고 **frozen feature로 학습된**
  것이어야 한다. finetune된 인코더용 헤드(`ft_*`)는 이 frozen 인코더와 호환되지 않는다.
