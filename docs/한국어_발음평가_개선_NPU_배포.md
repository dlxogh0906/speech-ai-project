# 한국어 발음 평가 개선 — NPU 배포 묶음 (2026-10-16)

NPU 서버의 발음 평가 서비스(`pron-api` · :8932 · `/root/pron_serve`)에 올릴 개선 두 가지를 묶었다.

> **반영 완료 (2026-10-17 21:25 KST).** 1·2단계를 한 번에 올렸다.
> - 백업: `/root/pron_serve/backup_2026-10-16/`
> - 컴파일 검사: NPU 프레임 일치 99.72% · 완전 일치 34/40 · 지연 29/94/419 ms
> - 시험 포트 비교: 오류 응답 차이 0 · 틀림 표시 3 → 3
> - 재시작: 6초
**두 단계로 나눠 올린다.** 1단계만 올려도 효과의 대부분이 나온다.

| 단계 | 바뀌는 것 | 효과 (아래 "수치 읽는 법") | 지금 |
|---|---|---|---|
| **1. 표준발음 보정** | 코드 파일 2개 + 실행 인자 `--canon-fix`. 모델은 그대로 | 헛지적 발화 **42.9% → 18.5%** · 재현율 변화 없음(+1.4%p, 유의하지 않음) | **올려도 됨** |
| **2. 섞은 모델** | 모델 교체 + NPU 재컴파일(약 3분) | 1+2단계: 헛지적 발화 **20.3%** · 재현율 **25.6% → 30.3%** | 점수 확인 통과(4-0절). 1단계 뒤 NPU 컴파일 검사(4-1절)를 통과하면 |

- 1단계는 끄면(`--canon-fix` 를 빼면) 예전과 똑같이 돈다. 되돌리기는 인자 하나를 지우고 재시작하면 된다.
- 응답 형식(JSON 필드)은 그대로다. 표준발음이 바뀌는 문장에서만 정답 자모(`alpha_jamo` · 단어 판정의 `canon_jamo`)와
  그에 따른 지적·점수가 달라진다. 점수는 평균 +0.5점(0~100) 정도 오른다(4-0절).
- 처리 시간: 보정은 CPU 에서 돌고 요청당 약 +4 ms 걸린다(GPU 서버에서 잰 값). 2단계는 같은 구조의 모델이라 NPU 시간이 같다.

### 수치 읽는 법

- 대상: 학습에 쓴 문장 224문항, 테스트 발화 6,720개 중 **API 가 받아 주는 발화**(짧은 단어는 API 가 거절한다).
- **헛지적 발화**: 맞게 읽은 자리를 "틀렸다"고 화면에 띄운 곳이 하나라도 있는 발화의 비율. 낮을수록 좋다.
- **재현율**: 실제로 틀린 자리 중 화면에 "틀렸다"고 띄운 비율. 높을수록 좋다.
- 화면 기준이다. 지금 서비스와 같은 규칙(점수 60 미만만 지적)으로 쟀다.
- 문항 단위 부트스트랩 95% 구간:
  - 1단계의 헛지적은 −24.4%p (−32.2 ~ −17.9)이다.
  - 1+2단계의 헛지적은 −22.6%p (−30.0 ~ −16.2), 재현율은 +4.8%p (+2.7 ~ +7.0)이다.
- 반영 설정(변이 margin 1.0)으로 쟀다(2026-10-17 재측정). 10-15 측정값(17.6% · 18.9% · 29.9%)은 margin 0 이었다.

## 1. 묶음 구성

```
pron_improve_2026-10-16/                    (2.2 GB — 거의 전부 2단계 모델)
├── README.md                               이 문서
├── SHA256SUMS                              전 파일 체크섬
├── stage1_canonical_fix/                   (약 2 MB)
│   ├── files/                              /root/pron_serve/app 와 같은 폴더 구조 — 그대로 덮어 넣는다
│   │   ├── deploy/pronunciation-demo/app.py                         보정 인자를 넣은 NPU 판 앱
│   │   ├── tasks/pronunciation-assessment/src/mdd/canonical_fix.py  보정 라이브러리(새 파일)
│   │   ├── tasks/pronunciation-assessment/scripts/eval/check_canonical_fix.py   회귀 검사
│   │   └── tasks/pronunciation-assessment/evalsets/g2p_regression_v1.tsv        회귀 검사 문장 83개
│   ├── app.patch                           위 app.py 의 변경분만(patch -p1 용)
│   ├── app.npu_original_2026-10-16.py      이 묶음을 만들 때 기준으로 삼은 NPU 판 app.py 원본
│   ├── verify_side_port.py                 운영 포트 ↔ 시험 포트 응답 비교(표준 라이브러리만)
│   └── test_audio/                         합성 녹음 32개 + items.json (사람 녹음 아님)
└── stage2_mix_model/
    ├── model_hf/                           섞은 모델(HF 형식, fp32) — config · jamo_vocab · preprocessor · model.safetensors
    ├── ref/                                컴파일 검사 기준 40개(GPU 에서 이 모델로 뽑은 로그확률) + meta.json
    └── npu_compile_check.py                컴파일 + 기준 비교 + 버킷 지연 (/root/pron_serve 의 것과 같은 스크립트)
```

## 2. 옮기기

GPU 서버에서 보낸다. 1단계만 먼저 보내도 된다.

```bash
# GPU 서버
B=/data/speech-ai/out/mdd/deploy_bundles/pron_improve_2026-10-16
ssh npu 'mkdir -p /root/pron_serve/incoming'           # ssh 설정의 Host npu, 또는 Tailscale root@<NPU-tailscale-IP>
scp -r "$B" npu:/root/pron_serve/incoming/              # 1단계만: README.md SHA256SUMS stage1_canonical_fix

# NPU 서버
B=/root/pron_serve/incoming/pron_improve_2026-10-16
cd "$B" && sha256sum -c SHA256SUMS | grep -v ': OK$'    # 아무것도 안 나오면 정상(1단계만 보냈으면 stage2 줄만 없다고 나온다)
```

## 3. 1단계 — 표준발음 보정

### 무엇을 고치나

지금 서비스는 KoG2P 가 만든 표준발음을 정답으로 쓴다. KoG2P 가 틀리는 곳에서는 맞게 읽어도 "틀렸다"고 나온다.
보정은 두 가지를 한다.

1. **KoG2P 오류 고치기**(표준발음법 규칙 + 낱말 목록). 예:
   - 먹을 수[머글**쑤**], 볼게요[볼**께**요] — 관형형 뒤 된소리
   - 사건[사**껀**], 여권[여**꿘**] — 낱말 된소리
   - 필요[**피료**] — ㄴ첨가 과다
   - 별일[**별릴**] (KoG2P 는 [벼릴]) — ㄴ첨가 누락
   - 희망[**히**망], 다쳐요[다**처**요]
2. **둘 다 맞는 곳은 녹음에 맞춰 고르기.** 단어 경계의 된소리처럼 표준발음이 두 가지인 곳은 모델이 들은 쪽을 정답으로 고른다.
   - 기준: CTC 로그우도가 기본 발음보다 1.0 넘게 높을 때만 바꾼다.

- 회귀 검사: 이번 224문항 **밖**의 문장 80개(지켜야 할 것 40 + 고쳐야 할 것 40)에서 보정은 80/80 을 맞힌다. KoG2P 혼자서는 40/80 이다.
- 알면서 못 고친 3개는 `g2p_regression_v1.tsv` 의 `known` 줄에 있다.

### 3-1. 백업 · 원본 확인

```bash
A=/root/pron_serve/app
BK=/root/pron_serve/backup_2026-10-16 && mkdir -p "$BK"
cp -p "$A/deploy/pronunciation-demo/app.py" "$BK/app.py"
cp -p /root/pron_serve/run_8932.sh "$BK/run_8932.sh"
cmp "$A/deploy/pronunciation-demo/app.py" "$B/stage1_canonical_fix/app.npu_original_2026-10-16.py" && echo 원본과-같음
```

### 3-2. 파일 넣기

- `원본과-같음` 이 나왔으면 통째로 덮어쓴다.

  ```bash
  cp -r "$B/stage1_canonical_fix/files/." "$A/"
  ```

- 다르면(그 사이 NPU 쪽 app.py 가 바뀌었으면) 변경분만 얹는다. 그다음 다른 세 파일만 복사한다.

  ```bash
  cd "$A" && patch -p1 --dry-run < "$B/stage1_canonical_fix/app.patch" && patch -p1 < "$B/stage1_canonical_fix/app.patch"
  cd "$B/stage1_canonical_fix/files" && cp --parents tasks/pronunciation-assessment/src/mdd/canonical_fix.py \
      tasks/pronunciation-assessment/scripts/eval/check_canonical_fix.py tasks/pronunciation-assessment/evalsets/g2p_regression_v1.tsv "$A/"
  ```

이 시점에는 운영(8932)에 아무 변화가 없다. 돌고 있는 프로세스는 옛 코드를 메모리에 들고 있고, 재시작해도 `--canon-fix` 가 없으면 예전과 같다.

### 3-3. 회귀 검사

```bash
cd "$A" && /root/nat_serve/venv/bin/python tasks/pronunciation-assessment/scripts/eval/check_canonical_fix.py | tail -3
# 기대: 마지막 줄 "실패(guard·fix) 0건"
```

### 3-4. 시험 포트(8942)에 띄우기

운영과 같은 인자에 보정만 켜서 **다른 포트·다른 NPU 장치**에 띄운다. 묶음의 `run_8942_test.sh` 를 그대로 쓴다.
이 스크립트는 2026-10-16 의 `run_8932.sh` 를 본떠 만들었다. 운영과 다른 점은 아래뿐이다.

| 항목 | 운영 8932 | 시험 8942 | 이유 |
|---|---|---|---|
| NPU | 7번 | 6번 | 7번은 운영이 쓰고 있어 자리가 없다(11.4 / 15.7 GiB). 6번은 6.4 GiB 사용 |
| 주소 | <NPU-tailscale-IP> | 127.0.0.1 | 밖에 열지 않는다 |
| 녹음 저장 | `--save-dir uploads` | 안 함 | 시험 녹음이 사용자 녹음 폴더에 섞이지 않게 |
| 요청 수집 | systemd 가 켠다 | 안 함 | systemd 밖에서 돈다 |
| 표준발음 보정 | 없음 | `--canon-fix` | |

```bash
rbln-stat                                    # 6번에 5.1 GiB 이상 비었는지 확인
cp "$B/stage1_canonical_fix/run_8942_test.sh" /root/pron_serve/
cd /root/pron_serve && setsid nohup bash run_8942_test.sh > test_8942.log 2>&1 < /dev/null & echo $! > /root/pron_serve/test_8942.pid   # NPU 서버에는 tmux 가 없다
# 모델 적재가 끝나면
grep "표준발음 보정 켬" /root/pron_serve/test_8942.log && curl -s 127.0.0.1:8942/health
```

### 3-5. 응답 비교

같은 합성 녹음 32개를 운영(8932)과 시험(8942)에 보내 비교한다. 토큰은 파일에서 바로 읽어 넘긴다(화면에 찍지 않는다).

```bash
cd "$B/stage1_canonical_fix"
MDD_API_TOKEN="$(cat /root/pron_serve/.token)" /root/nat_serve/venv/bin/python verify_side_port.py \
    --a http://<NPU-tailscale-IP>:8932 --b http://127.0.0.1:8942
```

GPU 서버에서 같은 앱(이 묶음의 app.py, 보정 끔/켬)으로 미리 돌린 결과다. 2026-10-16 NPU 서버 실측도 똑같이 나왔다:

```
같음 21 · 정답/판정 다름 11 · 상태 코드 다름 0 (전체 32)
틀림 표시 합계: 지금 3 → 보정 3
```

- 비교 요청 32개는 운영(8932)에도 들어가므로 운영 요청 수집(`/data/speech-logs/captures/pronunciation/<날짜>/`)에 남는다.
  합성 녹음이고 `form.tokenId` 가 `verify` 라 골라낼 수 있다. 수집 통계를 낼 때 뺀다.
- **상태 코드 다름은 반드시 0** 이어야 한다. 0 이 아니면 운영에 넣지 말 것(스크립트가 종료 코드 1 을 낸다).
- 다른 11개는 표준발음이 바뀌는 문장이다(먹을 수 · 필요 · 별일 · 일을 누가 · 밥 같이 · 사건 · 결정 …).
  - 틀림 표시가 줄어든 곳: 일을 누가 · 별일 · 밥 같이. 보정 전에는 KoG2P 오류 때문에 헛지적이 나던 곳이다.
  - 늘어난 곳: 사건 · 결정. 합성 음성이 [사건] · [결정] 처럼 예사소리로 읽은 것으로 보인다.
    표준발음 [사껀] · [결쩡] 기준으로는 맞는 지적이다.
- NPU 는 GPU 와 계산값이 아주 조금 달라서(프레임 1위 일치 99.74%) 위 개수가 1~2개 다를 수 있다.
  "바뀌면 안 되는 문장"(동료가 와요 · 서울 거예요 …)의 정답이 같은지 눈으로 확인한다.

### 3-6. 운영에 반영

```bash
sed -i 's/--port \$P$/--port $P --canon-fix/' /root/pron_serve/run_8932.sh     # 마지막 줄 끝에 붙인다
grep -c -- "--canon-fix" /root/pron_serve/run_8932.sh                            # 1 이어야 한다
bash /root/pron_serve/start_pron.sh          # = systemctl restart pron-api. 모델 적재 동안(수십 초) 8932 가 응답하지 않는다
grep "표준발음 보정 켬" /root/pron_serve/serve.log | tail -1 && curl -s <NPU-tailscale-IP>:8932/health
kill $(cat /root/pron_serve/test_8942.pid)  # 시험 포트 정리 (run_8942_test.sh 는 2단계에서 다시 쓴다)
```

- 사용이 적은 시간에 한다.
- 공개 입구(Caddy 443 · funnel 8443)는 8932 를 그대로 가리키므로 손댈 것이 없다.

### 3-7. 되돌리기

| 상황 | 방법 |
|---|---|
| 동작만 되돌릴 때 | `run_8932.sh` 에서 `--canon-fix` 를 지우고 `bash /root/pron_serve/start_pron.sh` |
| 코드까지 되돌릴 때 | `cp -p "$BK/app.py" "$A/deploy/pronunciation-demo/app.py"` · `cp -p "$BK/run_8932.sh" /root/pron_serve/` 뒤 재시작 |

`canonical_fix.py` 는 남겨 둬도 아무 데서도 안 불린다.

### 3-8. 웹·앱 쪽에 알릴 것

- 응답 필드는 그대로다.
- 일부 문장에서 정답 발음이 바뀐다.
  - 먹을 수 있어요: [머글수] → [머글쑤]
  - 필요해요: [필료] → [피료]
  - 그 결과 그 문장의 지적·점수가 달라진다. 전체로는 종합 점수가 평균 +0.5점 정도 오른다.
- 웹에서 정답 발음을 따로 들고 있다면(캐시·고정 문구) 맞춰야 한다.

## 4. 2단계 — 섞은 모델

### 무엇이 바뀌나

모델이 기존 서비스 모델(w2v-BERT `step_38000`)과 합성 최소대립쌍으로 더 학습한 모델을 반반 섞은 것으로 바뀐다(WiSE-FT, `interp_a0.5_step3000`).

- 효과: 된소리·거센소리를 예사소리로 낸 오류를 더 잡는다.
- 효과의 크기: 1단계 위에서 재현율이 +4%p 정도 오른다.
- 헛지적: 1단계와 비슷하다(18.5 → 20.3%). 원어민 발화의 자모 오류율도 7.16 → 7.12% 로 거의 같다.

### 4-0. 점수가 달라지나 — 확인함 (2026-10-16)

점수 3종(발음·유창성·이해도)을 내는 점수 모델과 화면 점수 보정 파일은 기존 모델 출력에 맞춰 만든 것이다.
그래서 모델을 바꾸면 점수가 어긋나는지 따로 쟀다.

- 측정 대상: 테스트 화자 800명의 문장 녹음 26,390개(API 가 받는 발화만)
- 방법: 같은 녹음을 구성마다 API 채점 경로 그대로 채점했다.

| 비교 (지금 서비스 대비) | 발화 점수(1~5) 평균 차 | 화면 종합(0~100) 평균 차 | 화자 점수 ↔ 사람 점수 상관 |
|---|--:|--:|---|
| 모델만 바꿈 | −0.01 | −0.07 | 같음 (발음 0.664 → 0.668) |
| 1단계(보정) | +0.11 | +0.5 | 오름 (발음 0.664 → 0.678) |
| 1+2단계 | +0.10 | +0.4 | 오름 (발음 0.664 → 0.681, 95% 구간 +0.011 ~ +0.022) |

- **모델 교체는 점수를 거의 안 바꾼다.** 기존 점수와 상관 0.99, 화자 순위 상관 0.999 다. 그래서 점수 모델을 다시 만들 필요가 없다.
- 점수를 바꾸는 건 1단계 보정이다.
  - 점수가 조금 오른다. 맞게 읽었는데 깎이던 자리가 줄어서다.
  - 사람 점수와의 상관도 세 점수 모두 오른다.
  - 점수 모델은 사람 라벨 표준발음으로 학습했다. 보정한 표준발음이 그쪽에 더 가깝다.
- 화면 유창성(속도·쉼 규칙)은 모델을 바꾸면 발화의 8% 가 5점 이상 움직인다(최대 7점). 평균과 사람 점수 상관은 같다.
- 결과: `/data/speech-ai/out/mdd/analysis/score_shift_2026-10-16/summary.md`. 저장소 스크립트는 `scripts/eval/eval_score_shift.py`.

### 4-1. 컴파일 (운영 파일을 덮지 않게 별도 폴더에서)

`npu_compile_check.py` 는 **자기가 있는 폴더**에 `.rbln` 과 `check_result.json` 을 쓴다.
`/root/pron_serve` 에서 그냥 돌리면 운영 중인 `w2vbert_ctc_f192_448_1024.rbln` 을 덮어쓴다. 반드시 새 폴더에서 돌린다.

```bash
S=/root/pron_serve/mix_2026-10-16 && mkdir -p "$S"
cp -r "$B/stage2_mix_model/model_hf" "$B/stage2_mix_model/ref" "$B/stage2_mix_model/npu_compile_check.py" "$S/"
cp "$A/tasks/pronunciation-assessment/src/mdd/npu_w2vbert.py" "$S/"     # 운영이 쓰는 것과 같은 감싼 모듈
cd "$S" && MODEL="$S/model_hf" DEV=6 /root/nat_serve/venv/bin/python npu_compile_check.py both 192,448,1024 2>&1 | tee compile.log
```

- `DEV=6` 을 꼭 준다. 기본값 7번은 자리가 없다.
- 시험 포트(`test_8942.pid`)는 내려 둔다. 같은 6번에 둘이 같이 올라가면 자리가 모자란다(6.4 + 5.0 + 5.0 > 15.7 GiB).
- 결과 파일은 `$S/w2vbert_ctc_f192_448_1024.rbln`(약 3.6 GB · NPU 점유 5.0 GiB)이다.
- **통과 기준**: 기존 모델의 3버킷 기록과 비슷해야 한다.

  | 항목 | 기존 모델 기록 | 통과 기준 |
  |---|---|---|
  | NPU 프레임 1위 일치 | 99.74% | 99.5% 이상 |
  | 자모열 완전 일치 | 35/40 | 33/40 이상 |

  CPU 줄은 거의 100% 여야 한다(감싼 모듈이 맞다는 뜻).

### 4-2. 시험 포트

같은 `run_8942_test.sh` 에 모델 경로만 환경변수로 바꿔 띄운다(보정은 켠 채).

```bash
cd /root/pron_serve && HF_CKPT=$S/model_hf RBLN=$S/w2vbert_ctc_f192_448_1024.rbln setsid nohup bash run_8942_test.sh > test_8942.log 2>&1 < /dev/null & echo $! > /root/pron_serve/test_8942.pid
cd "$B/stage1_canonical_fix" && MDD_API_TOKEN="$(cat /root/pron_serve/.token)" /root/nat_serve/venv/bin/python verify_side_port.py \
    --a http://<NPU-tailscale-IP>:8932 --b http://127.0.0.1:8942
```

- 이번에는 모델도 달라서 "다름"이 1단계보다 많은 게 정상이다.
- **상태 코드 다름 0** 과 **틀림 표시 합계가 크게 늘지 않는지**만 본다.

### 4-3. 운영에 반영 · 되돌리기

```bash
kill $(cat /root/pron_serve/test_8942.pid)
cp -p /root/pron_serve/run_8932.sh /root/pron_serve/backup_2026-10-16/run_8932.stage1.sh
sed -i -e "s#--hf-ckpt /root/pron_serve/model_fp32#--hf-ckpt $S/model_hf#" \
       -e "s#--rbln /root/pron_serve/w2vbert_ctc_f192_448_1024.rbln#--rbln $S/w2vbert_ctc_f192_448_1024.rbln#" /root/pron_serve/run_8932.sh
grep -o mix_2026-10-16 /root/pron_serve/run_8932.sh | wc -l                     # 2 여야 한다
bash /root/pron_serve/start_pron.sh && curl -s <NPU-tailscale-IP>:8932/health
```

- 되돌리기: `cp -p /root/pron_serve/backup_2026-10-16/run_8932.stage1.sh /root/pron_serve/run_8932.sh` 뒤 재시작(1단계 상태로 돌아간다).
- 원래 `.rbln` 과 `model_fp32/` 는 손대지 않았으므로 그대로 남아 있다.

## 5. 한계 — 같이 알고 있어야 할 것

- **사람 녹음 검증이 없다.**
  - 수치는 이미 있는 테스트 발화를 다시 채점한 것이다.
  - 일부러 틀리게 읽은 사람 녹음은 아직 받지 못했다(`docs/일부러_틀린_발음_녹음_안내.md`).
- **학습에 쓴 문장 기준이다.** 안 본 문장에서도 헛지적은 줄었다. 다만 절대값이 높아 모델끼리 비교에만 쓴다.
- **보정 사전의 일부는 이 224문항을 보고 맞췄다.**
  - 대상 낱말: 여권 · 용돈 · 별일 · 웬일 등.
  - 그래서 1단계 이득에는 낙관적인 부분이 있다.
  - 사전과 무관한 단어 경계 고르기는 일반적인 규칙이다.
  - 사전이 안 건드리는 문항에서는 재현율이 약간 내려간다(−1.1%p).
- **외운 단어는 그대로다.** 한국어를 [한쿡]으로 읽어도 [한구거]로 받아 적는 문제는 2단계도 일부만 고친다.
  예전 XLS-R 모델이 이 점은 더 낫다.
- **점수가 조금 오른다(1단계).** 같은 녹음에서 종합 점수가 평균 +0.5점 오른다(4-0절). 기존 기록과 나란히 비교할 때는 이걸 감안한다.
- **화면 표시 규칙(점수 60 미만만 지적)은 그대로다.**
  - 모델이 [ㅋ]로 들었어도 점수가 60 이상이면 "애매"로 숨긴다.
  - 들린 자모(`actual_jamo`)는 응답에 이미 들어 있다. 화면에 보여 줄지는 웹 쪽이 정한다.

## 6. 근거 (GPU 서버)

| 무엇 | 어디 |
|---|---|
| 1단계 화면 기준 측정 | `/data/speech-ai/out/mdd/analysis/canonical_source_2026-10-11/summary_fix2026_10_15.md` |
| 1+2단계 화면 기준 측정 | 같은 폴더 `summary_mix_fix2026_10_15.md` |
| 점수 이동 · 사람 점수 상관 | `/data/speech-ai/out/mdd/analysis/score_shift_2026-10-16/summary.md` |
| 보정 라이브러리 · 회귀 세트 · 측정 스크립트 | 저장소 `tasks/pronunciation-assessment/src/mdd/canonical_fix.py` · `evalsets/g2p_regression_v1.tsv` · `scripts/eval/eval_canonical_source.py` |
| 섞은 모델 원본 | `/data/speech-ai/corpus/checkpoints/w2vbert_synth_tense_v2/interp_a0.5_step3000` |
| 기존 모델 NPU 이식 기록(통과 기준 출처) | 저장소 `tasks/pronunciation-assessment/results/17_NPU_이식_검토.md` |
| NPU 서버 운영 일반 | 저장소 `docs/NPU_서버_운영.md` |
