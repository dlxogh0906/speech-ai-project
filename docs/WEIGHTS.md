# 가중치 — 어느 파일이 무엇인가

> **발음평가 범위 갱신(2026-10-29):** 아래는 8월 XLS-R 번들 안내다. NPU 운영 발음평가는 **M3 `w2vbert_synth_tense_v2/step_3000`**이며, 마지막 원격 확인은 10월 25일이다. M3의 HF·RBLN 경로와 배포 이력은 [현재 서비스 모델](발음평가_현재_서비스_모델.md)을 따른다. 이 문서의 `tag_all_native`는 기존 CLI·`deploy/mdd-api`용이다.

파일명이 서로 비슷해서 **잘못 골라도 에러 없이 답만 틀리는** 조합이 있다. 옮기거나
교체하기 전에 이 문서를 볼 것.

기준 번들: 26 GB · 4개 폴더 (2026-08-04 내보내기 + `mdd-full` 원본).
사실 확인은 전부 실측이다(2026-08-08, 파일을 열어 파라미터를 대조).

---

## 1. 기존 번들 추론에 필요한 파일 5개

전부 `final/` 안에 있다. 이 2.1 GB는 기존 번들 CLI·API의 추론 자산이며 M3 NPU 배포 자산은 포함하지 않는다.

| 용도 | 파일 |
|---|---|
| 발음 | `final/pronunciation/mdd_tag_all_native.pt` · `dict.ltr.txt` · `score_models_v2_native.joblib` |
| 국적 | `final/nationality/encoder_kd_l17.onnx` · `head_np.npz` |

나머지 24 GB 는 실험본과 **학습 재개용** 원본이다. 추론에는 쓰지 않는다.
사전학습 백본(`xlsr2_300m.pt` · `xlsr2_2B_1000k.pt`)은 번들에 없고, **추론에는 필요 없다** —
`.pt` 안에 인코더가 통째로 들어 있고 `.onnx` 는 자체 완결이다.

```
speech-ai-weights/
├── final/                  ← 기존 번들 추론용 (2.1 GB)
│   ├── pronunciation/  mdd_tag_all_native.pt · dict.ltr.txt · score_models_v2_native.joblib
│   └── nationality/    encoder_kd_l17.onnx · head_np.npz  (+ 구버전 encoder_l17.onnx · head.pt)
├── mdd-full/               ← 발음 모델 6종 원본 (21 GB · 재학습용)
├── nationality/            ← 국적 헤드 3종 실험본
└── score/                  ← 점수 회귀기 2종
```

---

## 2. 헷갈리는 것 셋

### ① 발음 모델 6개가 **전부 파일명이 같다**

```
mdd-full/mdd_ckpt_tag_all/checkpoint_best.pt          본학습 (TAG 143만 발화)
mdd-full/mdd_ckpt_tag_all_lar/checkpoint_best.pt      ↑ + LAR 도메인 적응 — 과거 평가 기준
mdd-full/mdd_ckpt_tag_all_native/checkpoint_best.pt   ↑ + 원어민 — 기존 CLI/mdd-api 기본 모델
mdd-full/mdd_ckpt_errup/checkpoint_best.pt            실험 (오류 가중)
mdd-full/mdd_ckpt_omniw2v/checkpoint_best.pt          실험 (백본 교체)
mdd-full/mdd_ckpt_diverse50k/checkpoint_best.pt       실험 (문장 다양성)
```

전부 `checkpoint_best.pt` 이고 전부 3.53 GB 다. **구분은 상위 폴더 이름뿐이다.**
파일만 떼어 옮기면 복구할 방법이 없다. 옮길 때는 폴더째 옮길 것.

### ② 국적: 인코더·헤드가 각각 두 벌이고, **짝이 정해져 있다**

같은 `final/nationality/` 안에 두 세트가 섞여 있다.

| 세트 | 인코더 | 헤드 | top-1 | top-5 |
|---|---|---|--:|--:|
| **KD (현재 배포)** | `encoder_kd_l17.onnx` | `head_np.npz` | **0.8245** | 0.9584 |
| 구버전 | `encoder_l17.onnx` | `head.pt` | 0.7510 | 0.9331 |

- 두 `.onnx` 는 **크기가 16 바이트 차이**다(959,943,599 vs 959,943,615). 눈으로 구분 못 한다.
- 두 헤드는 shape `(41,1024)` 도 같고 라벨 41종도 같은데 **가중치가 다르다.**
  그래서 짝을 섞어도 **예외가 안 나고 확률만 틀리게 나온다.**
- 근거: `scripts/export/export_onnx_kd_encoder.py` 가 `encoder_kd_l17.onnx` 와
  `head_np.npz` 를 **같은 `deploy/` 폴더에 함께** 써낸다. 한 세트로 생성된다.
- `deploy/nationality-kd-demo/app.py` 가 읽는 것도 이 두 개다(`KD_MODEL_DIR`).

> ⚠️ **`final/nationality/metrics.json` 과 번들 `final/README.md` 는 구버전(0.751)만 기술한다.**
> KD 세트는 그 문서들이 만들어진 뒤에 들어왔다. 문서만 보고 0.751 을 인용하면
> 실제 서비스 성능(0.8245)과 어긋난다.

### ③ `_native` 접미사 — 발음 모델과 점수 회귀기는 **짝이다**

| 체크포인트 | 짝이 되는 회귀기 |
|---|---|
| `mdd_tag_all_native` (기존 CLI/mdd-api) | `score_models_v2_native.joblib` |
| `mdd_tag_all_lar` (과거 평가 모델) | `score_models_v2.joblib` |

회귀기는 그 체크포인트가 뽑은 특징 41종으로 학습됐다. 짝이 어긋나면 **에러 없이
점수만 조용히 틀린다.** 위 번들은 해당 짝으로 사용한다. 다른 모델로 교체할 때는 특징·보정 정합성을 검증해야 한다. M3 운영 교체는 기존 회귀기를 유지하고 별도 검증한 사례다([10월 21일 기록](발음평가_개선_반영_2026-10-21.md)).

---

## 3. 이름이 달라도 같은 파일

### 슬림본과 원본은 **같은 모델**이다

`final/pronunciation/mdd_tag_all_native.pt`(1.18 GB) 와
`mdd-full/mdd_ckpt_tag_all_native/checkpoint_best.pt`(3.53 GB) 를 열어 대조했다:

> **모델 파라미터 424개가 전부 동일.** 차이는 최상위 키 `last_optimizer_state` 하나뿐이다.

즉 추론 결과는 완전히 같고, 큰 쪽은 **학습 재개용** 옵티마이저 상태만 더 들고 있다.
서비스에는 슬림본을 쓰고, 이어학습을 할 때만 원본을 쓴다.

### 내용이 바이트 단위로 같은 중복 3쌍

`final/` 은 실험 폴더에서 골라 복사한 큐레이션 사본이라 원본이 따로 남아 있다.

```
final/pronunciation/score_models_v2_native.joblib  ==  score/score_models_v2_native.joblib
final/nationality/head.pt                          ==  nationality/0705_country41_sent043_300m/head.pt
final/nationality/encoder_l17.onnx                 ==  nationality/0705_country41_sent043_300m/encoder_l17.onnx
```

---

## 4. `head.pt` 가 3개인데 이름이 같다 — 구분법

파일 안에 메타데이터가 들어 있어 **열어보면 구분된다.**

```python
import torch
h = torch.load("head.pt", map_location="cpu")
print(h["backbone"], h["layer"], h["feat_dim"], h["head"]["weight"].shape)
```

| 파일 | layer | feat_dim | shape | 지금 쓸 수 있나 |
|---|--:|--:|---|---|
| `final/nationality/head.pt` (= `0705_*_300m/`) | 17 | 1024 | (41,1024) | ✅ 300m |
| `nationality/0702_country41_sent_best/head.pt` | 31 | 1920 | (41,1920) | ❌ 2B 백본(25 GB) 필요 |
| `nationality/0704_country41_sent043_en/head.pt` | 31 | 1920 | (41,1920) | ❌ 2B 백본(25 GB) 필요 |

---

## 5. "국적 분류 정확도"가 문서마다 다른 이유

셋 다 사실이지만 **서로 다른 모델**이다. 인용할 때 어느 것인지 반드시 밝힐 것.

| 출처 | top-1 | top-5 | 모델 |
|---|--:|--:|---|
| `CLAUDE.md` 헤드라인 | 0.798 | 0.950 | 2B layer31 (`0702`) — 백본이 없으면 실행 불가 |
| 번들 `final/README.md` | 0.751 | 0.933 | 300m 비KD (구버전 세트) |
| **`deploy/infer_nationality.sh` 실제** | **0.8245** | **0.9584** | 300m KD (현재 배포 세트) |

---

## 6. 무결성 확인

번들에 체크섬이 세 벌 있다. 전송 후 반드시 돌릴 것.

```bash
cd <가중치>/final    && sha256sum -c SHA256SUMS.txt
cd <가중치>/mdd-full && sha256sum -c ../SHA256SUMS_mdd_full.txt
cd <가중치>          && sha256sum -c --ignore-missing SHA256SUMS.txt
```

`tools/setup.sh` 는 `final/` 체크섬을 자동으로 대조한다.

> `encoder_kd_l17.onnx` 와 `head_np.npz` 는 **어느 매니페스트에도 없다**(나중에 추가됨).
> 정작 서비스가 쓰는 두 파일이라 해시 보증이 없는 지점이니, 다음 내보내기 때
> 매니페스트에 포함시킬 것.
