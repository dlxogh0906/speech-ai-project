# 영어 발음 평가 서버 (`pronunciation-en`)

한국인 학습자의 영어 문장 발음을 채점한다. 요청과 응답 형식은 한국어 API(`deploy/pronunciation-demo`, 8932)와 같다.
규격은 [API.md](API.md), 근거 수치는 `tasks/pronunciation-assessment/results/23_영어발음평가_공개모델_구축_2026-10-04.md`에 있다.

**한국어 API(8932)와는 별개 서버다.** 코드와 가중치를 공유하지 않고 서로 import하지 않는다.

## 띄우기

```bash
source tools/env.sh
NLTK_DATA=/data/speech-ai/tools/nltk_data \
PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/en_py38 \
HF_HOME=/data/speech-ai/corpus/weights/external HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
$PY deploy/pronunciation-en/app.py --port 8952 \
    --scorer-dir $MDD_OUTPUTS/english/scorer/synth-d-ko037 --display-scale gentle    # tmux 세션 pron-en
```

- `--display-scale gentle`(2026-10-14~ 8952 기본 운영): 발음·운율 표시를 `40 + 0.6 × 기존 점수`로 올려 보인다.
  사람이 조음 3·4·5점 준 발화의 평균이 80 · 87 · 93(기존 66 · 79 · 89). 순서·판별력은 같고 단어·음소 점수와 지적은 바뀌지 않는다.
  `overall`은 바뀐 발음·운율로 다시 계산되고 `fluency`는 그대로다. 빼면(`human`) 예전 눈금이다.

- `--scorer-dir …/synth-d-ko037`은 **문장 점수를 한국인 학습자 사람 점수(AIHub 037)에 맞춘** 점수기다(2026-10-10).
  빼면 예전 점수기(`scorer/synth-d`, speechocean762 기준. 한국인에게 발음 점수가 96±3으로 몰린다)로 돈다.
  단어·음소 판정과 지적은 둘 다 같다. 점수기 폴더에 `sent_scorer_ko.joblib`이 있으면 새 점수를 쓴다.

- 사내망에 열려면 `--host 0.0.0.0 --token <열쇠>`를 준다. 토큰 없이 127.0.0.1 밖으로 열면 서버가 띄우기를 거부한다.
  환경변수 `PRON_EN_TOKEN`도 된다.
- `--api-only`를 주면 확인용 화면(`/`)을 끈다.
- `--device cpu`를 주면 CPU로 돈다(약 0.7초/문장).
- 기본 모델은 `synth-d`다(사람 전사에 가장 가깝다, results/23 §10). `--model synth-a`는 대표 오류를 더 많이 잡지만 오지적이 많다.
  `--model lv60-espeak`는 공개 모델 그대로다.
  `xlsr53-espeak`·`charsiu-ctc`는 **라이선스 확인 전에는 쓰지 말 것**(results/23 §2).

## 필요한 것 (저장소 밖)

| 무엇 | 어디 | 만드는 법 |
|---|---|---|
| 음소 인식기 | `$SPEECH_AI_DATA/checkpoints/en_ctc_synth_v2d/step_3000`(synth-d) | results/23 §9·§10 순서로 학습. 출발점 `facebook/wav2vec2-lv-60-espeak-cv-ft`(Apache-2.0) |
| 합성 TTS(학습 데이터 만들 때만) | `/data/speech-ai/envs/kokoro` | `tts` 환경 위 `--system-site-packages` venv + `pip install kokoro`(0.9.4, Apache-2.0) |
| 점수기 | `$MDD_OUTPUTS/english/scorer/<모델>/{phone_scorer,sent_scorer}.joblib` | 아래 "다시 만들기"(모델 이름만 바꿔서) |
| 한국인 문장 점수기 | `$MDD_OUTPUTS/english/scorer/synth-d-ko037/sent_scorer_ko.joblib`(+ 위 파일 사본) | 아래 "한국인 기준 문장 점수" |
| g2p_en, nltk, inflect | `/data/speech-ai/tools/en_py38`(덧씌우기 폴더) | `pip install --no-deps --target …`. g2p_en 2.1.0 · nltk 3.8.1 · inflect 5.6.2 |
| nltk 데이터 | `/data/speech-ai/tools/nltk_data` | `nltk.download('averaged_perceptron_tagger')`, `nltk.download('cmudict')` |
| transformers | `/data/speech-ai/tools/hf_py38` | 한국어 서버와 같은 덧씌우기 |

- 고정된 파이썬 환경(`$PY`)에는 아무것도 설치하지 않았다. omegaconf, hydra 핀이 깨지지 않게 하기 위해서다.
- g2p_en이 의존성으로 끌어오는 `distance`(GPL)는 실제로 import되지 않아 **설치하지 않았다.**
  `--no-deps`로 깔 것.

## 다시 만들기 (점수기)

```bash
ENV="NLTK_DATA=/data/speech-ai/tools/nltk_data PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/en_py38 HF_HOME=/data/speech-ai/corpus/weights/external HF_HUB_OFFLINE=1"
# 1) 인식기 출력 캐시 + 인식 기반 오류 판별 비교 (speechocean762 test/train, L2-ARCTIC 한국인)
env $ENV $PY tasks/pronunciation-assessment/scripts/eval/en_phone_bakeoff.py --models lv60-espeak --data l2arctic,so762,so762train
# 2) 음소 점수기 (사람 음소 점수 회귀) + 지적 문턱
env $ENV $PY tasks/pronunciation-assessment/scripts/score/en_fit_phone_scorer.py --model lv60-espeak
# 3) 문장 점수기 (정확도·유창성·운율·종합)
env $ENV $PY tasks/pronunciation-assessment/scripts/score/en_fit_sentence_scorer.py --model lv60-espeak
# 4) 서버와 같은 경로로 평가
env $ENV $PY tasks/pronunciation-assessment/scripts/eval/en_eval_engine.py --model lv60-espeak
```

## 한국인 기준 문장 점수 (AIHub 037)

```bash
# 1) 037 낭독(PRN) 발화를 서버와 같은 경로로 채점 → 특징표 (화자당 30발화 · GPU 1장에 3조각, 약 27분)
for i in 0 1 2; do env $ENV $PY tasks/pronunciation-assessment/scripts/eval/en_extract_037_feats.py --model synth-d --shard $i/3 & done; wait
# 2) 회귀 + 화자 5겹 · 화자×문장 분리 평가 + 눈금 → scorer/synth-d-ko037/ (CPU 몇 분)
env $ENV $PY tasks/pronunciation-assessment/scripts/score/en_fit_sentence_scorer_037.py --model synth-d
# 3) 잠근 외부 검증: 037 SPK(말하기 평가, PRN 에 없는 화자)
env $ENV $PY tasks/pronunciation-assessment/scripts/data/build_037_spk_eval.py
for i in 0 1 2; do env $ENV $PY tasks/pronunciation-assessment/scripts/eval/en_extract_037_feats.py --model synth-d \
    --list $MDD_OUTPUTS/english/aihub037_spk/test.jsonl --carry rating,task --per-spk 6 --shard $i/3 \
    --out $MDD_OUTPUTS/english/score037/synth-d-spk & done; wait
env $ENV $PY tasks/pronunciation-assessment/scripts/eval/en_eval_score_spk037.py \
    --feats $MDD_OUTPUTS/english/score037/synth-d-spk --scorer-dir $MDD_OUTPUTS/english/scorer/synth-d-ko037
```

- 037 에서 읽는 칸은 녹음 경로·문장·화자 번호(분할용)·사람 점수뿐이다. 나이·수준 칸은 읽지 않는다.
- 수치는 `scorer/synth-d-ko037/sent_scorer_ko.{md,json}` · `sent_scorer_ko_spk.{md,json}`, 해설은 results/23 §12.

데이터 위치:
- `$SPEECH_AI_DATA/english/speechocean762`: OpenSLR 101, CC BY 4.0
- `$SPEECH_AI_DATA/english/l2arctic_v5`: 한국인 4명, **CC BY-NC**. 평가에만 쓰며 학습과 문턱 결정에는 쓰지 않는다.

## 주의

- **AIHub 오디오나 L2-ARCTIC 오디오를 SpeechSuper 같은 외부 API로 보내지 말 것.**
  SpeechSuper 비교에는 동의를 받은 우리 녹음만 쓴다.
- 확인용 화면의 마이크는 보안 컨텍스트(127.0.0.1 또는 HTTPS)에서만 열린다.

## 외부 공개 (2026-10-14~)

> 2026-10-16: 내부용 8952 는 닫았다(요청). 지금 떠 있는 것은 공개용 **8954** 하나다 — 데모 화면도 이쪽(`/pron-en/`)에서 연다.

| 항목 | 값 |
|---|---|
| 공개 주소 | `https://<GPU-funnel-주소>/pron-en/` (Funnel 경로 `/pron-en` → `127.0.0.1:8954`) |
| 서버 | 공개용 인스턴스 **8954** · tmux `pron-en-public` · 8952(내부용)와 같은 모델·점수기·`--display-scale gentle` |
| 토큰 | `~/.config/pron-en/public.env`(600, `PRON_EN_TOKEN=`) — 저장소·문서에 적지 않는다. 띄울 때 `set -a; . ~/.config/pron-en/public.env; set +a` |
| 외부 개발자 안내 | `docs/영어_발음평가_API_안내.md` |
| 녹음 저장 | 하지 않음(`--save-dir` 없음) |

- **속도**(서울에서 공개 경로로 잼, 4초 문장 16 kHz WAV): 연결 재사용 0.35~0.46초 · 새 연결 1.35~1.46초. 서버 채점은 0.06초이고 나머지는 Funnel 중계(홍콩·도쿄)다.
  이 서버는 바깥에서 들어오는 포트가 없고(삼성SDS 망, 외부 점검 80·443·22 모두 시간 초과) UDP 도 밖으로 못 나가서(`tailscale netcheck` UDP false) 직통 입구를 못 만든다.
  빠르게 하려면 ① 이 VM 에 공인 IP + 80·443 허용(서버 제공 쪽) 또는 ② NPU 서버 Caddy 443 으로 TCP 터널(NPU 담당)이 필요하다.
- **재부팅하면 자동으로 안 뜬다**(tmux). 다시 띄울 때 위 실행 명령에서 포트만 8954 로, 토큰 환경변수를 넣는다.
- 끄기: `tailscale funnel --set-path /pron-en off`(다른 경로는 그대로) 뒤 `tailscale funnel status` 로 확인. **`tailscale serve` 를 쓰면 funnel 이 꺼진다.**
