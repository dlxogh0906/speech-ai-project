# 발음 챌린지 데모 (CTC)

accent.beavertalk.im 의 발음 챌린지(30초 라운드 · 약 2초에 단어 하나 · 즉시 PERFECT/MISS)를
발음평가 모델(w2v-BERT CTC `step_38000`)로 바꿨을 때의 **속도와 판정**을 체감하는 데모.
배경 측정은 `tasks/pronunciation-assessment/results/20_발음챌린지_처리속도_2026-10-01.md`.

- 발음평가 API(8932)와 **같은 모델**이지만 별도 프로세스다. API 서버는 건드리지 않는다.
- 판정 로직은 라이브러리 `mdd.word_judge`(서비스에 붙일 때 그대로 쓸 수 있다). 이 폴더는 페이지·HTTP 만.
- 녹음은 저장하지 않는다.

## 실행

```bash
source tools/env.sh
export PYTHONPATH=/data/speech-ai/tools/hf_py38${PYTHONPATH:+:$PYTHONPATH}   # transformers(HF) — 발음 API 와 같은 경로
OMP_NUM_THREADS=4 $PY deploy/pronunciation-challenge/app.py \
    --hf-ckpt $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 --port 8946
```

- `http://127.0.0.1:8946/` — 마이크는 **localhost 나 HTTPS** 에서만 열린다. 원격 서버면 VS Code 포트 전달로 localhost 로 열거나
  HTTPS 프록시(tailscale funnel `--set-path`) 뒤에 둔다. 페이지는 상대 경로로 API 를 불러 하위 경로에 걸어도 된다.
- `OMP_NUM_THREADS=4` 를 빼면 CPU 판정이 느려지고 GPU 쪽도 흔들린다(results/20 §2).
- `--devices cuda,cpu`(기본) — 둘 다 올려 두고 페이지에서 고른다. CPU 모델은 RAM 약 2.4 GB.
- 학습 단어 목록은 `$MDD_OUTPUTS/manifest_holdout_v1` 메타에서 읽는다(저장소에 두지 않는다).

## 동작

| 단계 | 어디서 | 내용 |
|---|---|---|
| 녹음 | 브라우저 | 마이크 → 16 kHz 다운샘플 → 30 ms 프레임 |
| 말끝 감지 | 브라우저 | 에너지 VAD. 말 시작 0.06초 연속 → 녹음, 무음 0.3/0.4/0.6초(선택) → 끊어서 전송. 최대 3.5초 |
| 판정 | 서버 `POST /judge` | 16 kHz WAV + 단어 → 무음 다듬기 → 모델 → 자모열 → 표준발음과 편집거리 |
| 등급 | 서버 | 0 PERFECT · 1 GOOD · 2 이상 MISS (페이지에서 "엄격"을 고르면 1 도 MISS) |

말소리가 0.12초 미만이면 채점하지 않고 다시 말하게 한다. 문장용 API 기준(1.0초)은 단어의 95% 를, 0.2초는 한 글자 단어의 7.4% 를 거절했다(0.12초면 0.2%).
단어가 뜬 뒤 5초 안에 말을 시작하지 않으면 시간 초과 MISS.

## API

| 경로 | 설명 |
|---|---|
| `GET /config` | 단어 세트(학습 단어 · 처음 보는 단어)와 올라간 장치 |
| `GET /tags?w=사과,고양이` | 단어별 태그(`seen` 학습 단어 · `partial` 학습 문항 일부 · `new` 처음 보는 단어)와 판정 가능 여부 |
| `POST /judge` | multipart `audio`(16 kHz mono 16bit WAV) · `word` · `device`(`cuda`/`cpu`) → 등급 · 편집거리 · 표준/들린 자모 · 자리별 비교 · 처리 시간 |

## 알아 둘 것

- **처음 보는 단어는 맞게 읽어도 MISS 가 잦다** — 사전 측정(results/20 §4): 학습 단어 MISS 2.2% → 처음 보는 단어 22%
  ("엄격" 판정이면 12% → 69%). 데모에서 단어 세트를 바꿔 직접 비교할 수 있게 해 두었다.
- 페이지가 보여 주는 "말 끝 → 판정" 은 말끝 감지 대기 + 업로드·서버 처리·응답이다. 모델 계산은 그중 일부다.
- 같은 서버에서 다른 GPU·CPU 작업이 돌면 처리 시간이 부풀려진다.
