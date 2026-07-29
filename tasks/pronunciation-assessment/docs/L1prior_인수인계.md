# 인수인계 — Speech_lab MDD → L1 prior 구축

_최종 업데이트: 2026-07-06 · 작성 근거: `발음오류_데이터구축_작업지시서.md`, `비버톡_AI개발로드맵_발표자료.html`_

---

## 0. 한 줄 요약

외국인 한국어 학습자의 **Speech_lab(들리는 대로의 한글 전사)**을 KoG2P로 음소화하고
정답열과 **MDD(edit-distance) 정렬**해, 국적(L1)별 발음 오류 패턴(**= L1 prior**)을 만들었다.
소량 gold(**pronunciation_lab error_tags**)로 교차검증까지 마쳤다.
이는 로드맵 v2의 **"갈래 A · MDD 즉시 분석(Sprint 1-A)"**의 실행 산출물이다.

---

## 1. 목표와 방향 (검증 완료)

- 작업지시서 = 로드맵 갈래 A(Sprint 1-A)의 상세 실행 스펙. **방향 정합 확인됨.**
- **Speech_lab = L1 prior 본체**(대량·약라벨, 실현형 有), **pronunciation_lab = 감독자·벤치마크**(소량·강라벨, 위치/유형/타임스탬프 有).
- 인식기(XLS-R+CTC) **없이** 순수 알고리즘으로 착수 가능한 범위 전체를 구현했다.

---

## 2. G2P 도구 (다운로드·검증 완료)

| 도구 | 위치 | 채택 | 비고 |
|---|---|---|---|
| **KoG2P** | `_g2p_tools/KoG2P/` | ✅ 주 | 의존성 0, 출력이 스키마 음소기호(`k0 aa nf`)와 일치. 규칙 전체 적용. |
| g2pK | `_g2p_tools/g2pK/` | 보조 | mecab-ko/konlpy/nltk/Java 의존성 무겁고 출력이 한글 발음형. 교차검증용 보관만. |

> KoG2P는 `graph2prono`로 음운규칙(연음·격음화·경음화 등)을 **양방향(prompt·orthographic 모두)** 적용한다.
> ⚠️ 이 때문에 **표준 경음화·비음화는 양쪽에서 상쇄**되어 Speech_lab MDD로는 과소검출된다(4절 교차검증 참조).

---

## 3. 파이프라인 (스크립트, 프로젝트 루트)

경로: `D:\교육용 언어별 모국어 사용자의 한국어 음성 데이터\`

```
s0_kog2p_wrapper.py   한글 -> KoG2P 음소기호(리스트). 규칙 전체 적용 + lru_cache.
s0_symbol_map.py      pronunciation_lab 호환자모(ㄱ,ㅉ…) -> KoG2P 기호(위치 추론) + UI 역매핑.
s0_taxonomy.py        gold 17종 -> 음소 자질 편차 규칙(classify_op). 두 데이터의 '다리'.
      ▼
s1_speechlab_mdd.py   레코드 1건 처리: 정제 -> 양방향 G2P -> NW 정렬 -> 자동태깅 -> 품질필터.
      │   핵심 함수: clean_orthographic(), build_record()
      ▼
s1_run_batch.py       LAR 파일 대량 처리(멀티프로세싱) -> records.jsonl + 집계 CSV.
      ▼
s4_l1_prior.py        L1 prior 잔차표 + 대치쌍 + gold 교차검증.
```

각 s0 스크립트는 `python scripts/sX.py`로 자가검증 실행 가능.

### 3-A. 처리 대상 = **파일명에 `LAR` 포함** 레코드만
- LAR = 읽기 과제(고정 프롬프트) → 정답열이 깨끗해 MDD에 적합. (ATQ/SPT/PDT 등 자유발화 과제는 제외)
- 4개 언어군 Train+Val 합계 **LAR 268,533건**.

### 3-B. orthographic 정제 (MDD 이전) — `clean_orthographic()`
1. **간투사 제거**: 사전(`아·어·음·그·저·으·에·응·엄·뭐` + 늘임형) 매칭.
   - **prompt와 어절정렬(word-NW) 후 '삽입'된 것만** 제거 → "그 사람은"의 지시관형사 '그' **보존**.
   - **띄어쓰기 가드**: 인접어절과 합쳐 prompt 단어를 이루면 보존 → "저한테"를 "저 한테"로 발화한 '저' 보존.
2. **더듬음/반복 병합**: 인접 유지어절과 동일·접두면 삭제(예: "키 키울"→"키울", "아이 아이가"→"아이가").
3. 제거·병합 위치는 `disfluencies`(type=filler|repetition, ortho_pos)로 **기록만 하고 삭제**.

### 3-C. 품질 플래그 (옵션 A 채택) — 집계 배제 신호
- `collapsed` : err_ratio ≥ 0.5 (문장·어휘 이탈)
- `restart_suspect` : 같은 target 위치 연속 삽입 ≥3 (재시작 조각, 예 "그 음식 그 식당은…")
- `ok` : 위 아님 → `usable_for_aggregation=True` (집계 대상)

---

## 4. 대량 실행 결과 (2026-07-06)

실행: `python scripts/s1_run_batch.py --groups 040 041 042 043 --split both --workers 10`

- 처리 **268,533건** → **ok 244,149 (90.9%)** / collapsed 16,972 (6.3%) / restart 7,412 (2.8%)
- **L1 112개국**, 대치쌍 **34,200종**
- 물량 상위: 인도네시아(34.5k) · 베트남(24.9k) · 중국(22.3k) · 필리핀(17.7k) · 일본(16.6k)

### 4-A. L1 prior 검증 (언어학적 타당성)
| L1 | 대표 오류(양의 잔차) | 대표 대치쌍 |
|---|---|---|
| Japan | 종성 +24, 모음삽입 +11 | ㅗ↔ㅓ, 종성 삽입 (모라구조 간섭) |
| Philippines | 모음오류 +39 | (기존 gold와 동일 — 필리핀 모음 취약) |
| Spain | 격음화 +22, 모음오류 +18 | ㅡ>ㅜ (스페인어 5모음 → ㅡ 부재) |
| Vietnam | 비음화 +60, 격음화 +20 | ㄹ>ㄴ |
→ **알려진 L1 간섭 + 기존 pronunciation_lab 발견과 일치.**

### 4-B. gold 교차검증 (`s4` crossval)
Speech_lab 잔차 vs pronunciation_lab gold 잔차(공통 44개국 × 10유형):
- **전체 Pearson r = 0.285 (p = 1.2e-9)**
- 강한 일치: **모음오류 0.66 · 종성 0.55 · 모음삽입 0.54 · 유음화 0.42** (전사로 잘 잡히는 유형)
- near-zero: 경음화 −0.05 · 비음화 0.00 · 초성 −0.01 (KoG2P 규칙 양방향 상쇄 / 정의 불일치)
- **해석**: L1 prior는 모음·종성·삽입·유음 계열에서 신뢰 가능. 규칙기반(경음화·비음화·연음·구개음화)·운율(강세·억양)은 **gold가 권위** → 상보 관계 재확인.

---

## 5. 산출물 위치

### Speech_lab MDD — `_mdd_analysis\040_041_042_043_both\`
| 파일 | 내용 |
|---|---|
| `records.jsonl` (526MB) | 통합 스키마 전량 26.8만건(품질플래그·disfluencies 포함) |
| `substitution_pairs.csv` | L1 × (정답>실현) 대치쌍 = **L1 prior 원천** |
| `l1_errortype_counts.csv` / `_rate.csv` | L1 × 오류유형 카운트/비율 |
| `l1_summary.csv` | L1별 records/usable/오류/간투사 |
| `quality_report.csv` | 품질플래그 분포 |
| `l1_prior\l1_prior_residuals.csv` | **L1 prior 본체**(44개국 표준화잔차) |
| `l1_prior\l1_prior_top_substitutions.csv` | L1별 상위 대치쌍(한글 병기) |
| `l1_prior\crossval_by_errortype.csv` | gold 교차검증 상관 |
| `l1_prior\l1_prior_report.md` | 요약 보고서 |

### 기존 pronunciation_lab 분석(선행 작업, 보존) — `_error_tag_analysis\combined\`
- `standardized_residuals_core.csv` (교차검증 기준), `nationality_pronunciation_feedback.json`(34개국), 보고서·히트맵 등.
- 상세는 `_error_tag_analysis\combined\HANDOFF.md` 참조.

---

## 6. 통합 오류 스키마 (records.jsonl 1건)

```json
{
  "utterance_id": "...", "file": "...LAR...json", "task": "LAR",
  "L1": "Japan", "proficiency": "...", "recording_level": "...",
  "prompt": "...", "orthographic_raw": "...", "orthographic_clean": "...",
  "disfluencies": [{"type":"filler","word":"그","ortho_pos":0}],
  "target_phones": ["..."], "actual_phones": ["..."],
  "n_errors": 3, "err_ratio": 0.09, "max_ins_run": 1,
  "quality_flag": "ok", "usable_for_aggregation": true,
  "errors": [{"pos":7,"target_phone":"t0","actual_phone":"tt","op":"sub",
              "error_type":"경음화","confidence":"high","source":"speech_lab", ...}],
  "eval": {"Delivery":4,"LanguageUse":0,"Content":5,"PronunProf":null,"Fluency":null},
  "source": "speech_lab"
}
```
- 소스별 못 채우는 필드는 `null`. `source`로 두 출처 구분(현재 전부 speech_lab).

---

## 7. 재현 방법

```powershell
# (별도 PowerShell 창, 프로젝트 루트)
cd "D:\교육용 언어별 모국어 사용자의 한국어 음성 데이터"
$env:PYTHONUTF8=1
# 대량 처리(약 10~30분, 12코어 기준 workers 10)
python scripts/s1_run_batch.py --groups 040 041 042 043 --split both --workers 10
# L1 prior + 교차검증
python scripts/s4_l1_prior.py --tag 040_041_042_043_both --min-usable 500
```
- Windows 콘솔 한글: `PYTHONUTF8=1`. 세션 내 `!` 프리픽스는 bash라 PowerShell 문법 불가.
- 무거운 스캔은 사용자 별도 실행 권장(토큰 절약). 재실행 시 출력 덮어씀(스트리밍 저장이라 중단 안전).

---

## 8. 알려진 한계 & 다음 단계

### 한계
1. **KoG2P 규칙 양방향 상쇄** → 표준 경음화·비음화 과소검출. 이 유형은 gold 사용.
2. **재시작/false-start** 완전 처리 못함(옵션 A로 `restart_suspect` 배제 중). 필요 시 s1에 연속삽입 블록 병합(옵션 B) 추가.
3. **어절 경계(#)** 규칙 적용 후 소실 → 현재 문장 단위 MDD. 어절 단위 정밀화는 미구현.
4. `연음규칙·구개음화·강세·경계억양` = Speech_lab MDD로 미검출/약검출 → pronunciation_lab 전용.

### 다음 단계 후보 (미착수)
- **갈래 B(인식기)**: XLS-R+CTC 음소 인식기 학습 → 진짜 phone 단위 MDD F1 + 발화별 위치 확정(GPU 필요, 코드 트랙 밖).
- **Sprint 2 GOP/UI**: 인식기 사후확률 → GOP 점수, 기호→한글 UI(재료 `SYMBOL_TO_JAMO` 확보).
- **taxonomy 정밀화**: 규칙 상쇄 보정(경음화·비음화를 원문 자소기반 별도 경로로 재검출) 검토.

---

## 9. 신규 완료 — 통합 적재 · 벤치마크 · 제품화 · 시각화 (2026-07-06)

### 9-A. 파이프라인 확장 스크립트
```
s2_pronlab_to_schema.py  gold(pronunciation_lab) → 통합 스키마.
     error_tag 시간창 ↔ phones[] 타임라인 겹침으로 pos 매핑. 자모→KoG2P.
     actual_phone=null(gold는 실현형 미기록·표준음 확인). source=pronun_lab.
     출력: records_pronlab.jsonl, l1_utterance_labels.csv(B1 입력), 집계 CSV
s3_merge_schema.py       speech(실현형)·gold(위치·유형) 소스별 비교매트릭스로 병합.
     ※ 측정대상이 달라 단순합산 금지 → l1_source_rate_wide.csv 병렬
s5_benchmark.py          L1 prior 예측력 벤치마크(P/R/F1·AP·FAR·FRR·DER + L1 lift).
     인식기 부재로 phone단위 MDD F1은 불가 → 그 대체(제품가치 직접근거).
s6_l1_prior_product.py   잔차·율 → 진단 prior 가중치(81개국) + 정성피드백 확장.
s7_visualize.py          자체완결 HTML 통계 리포트(그림 7종).
```

### 9-B. 핵심 산출물
| 파일 | 내용 |
|---|---|
| `_mdd_analysis/l1_prior_product/l1_prior_weights.json` | **진단 prior 가중치(제품)** — `priors[L1].weights[type].prior_weight` |
| `_mdd_analysis/l1_prior_product/nationality_feedback_enriched.json` | 기존 정성피드백 + 기계 가중치(비파괴) |
| `_mdd_analysis/<pron_tag>/benchmark/benchmark_*` | B1 벤치마크(유형별/국적별 AP·lift·P/R/F1) |
| `_mdd_analysis/_report/L1prior_통계리포트.html` | **상세 시각화 리포트**(전체 데이터, 그림 7종) |

### 9-C. 전체 재현 (무거운 스캔은 사용자 실행)
```powershell
cd "D:\교육용 언어별 모국어 사용자의 한국어 음성 데이터"
$env:PYTHONUTF8=1
# (1) gold 통합 적재 — 무거움(≈188만 파일). 사용자 실행
python scripts/s2_pronlab_to_schema.py --groups 040 041 042 043 --split both --workers 10
# (2) 소스 병합(경량)
python scripts/s3_merge_schema.py --speech-tag 040_041_042_043_both --pron-tag 040_041_042_043_both_pronlab
# (3) 벤치마크 전체(경량, s2 산출 소비)
python scripts/s5_benchmark.py --pron-tag 040_041_042_043_both_pronlab --min-utt 300
# (4) 제품 가중치(경량, 기존 gold 집계 소비 — 이미 실행됨)
python scripts/s6_l1_prior_product.py --min-files 500
# (5) 리포트 갱신(경량) — bench-tag를 전체로
python scripts/s7_visualize.py --speech-tag 040_041_042_043_both --bench-tag 040_041_042_043_both_pronlab
```
> ①만 무겁다(다른 단계는 수초~수분). ①·③·⑤ 실행 후 리포트의 B1 벤치마크가 전체 수치로 갱신됨.
> ②④는 이미 검증·실행됨. 현재 리포트의 ①②③④⑥⑦은 전체 데이터, ⑤만 영어군 검증표본.

> 새 작업은 사용자 확인 후 진행. 무거운 스캔은 코드만 전달하고 사용자가 실행하는 원칙 유지.

---

## 10. 최종 현황 · 다음 착수 (2026-07-06 갱신)

### 10-A. 신규 산출물 (9절 이후 추가)
| 스크립트 | 산출물 | 내용 |
|---|---|---|
| `s8_error_catalog.py` | `_mdd_analysis/.../error_catalog/L1_발음오류_상세카탈로그.md` (44국) | **speech 실현형**: 초성/중성/종성별 구체 대치(예: 초성 k0=ㄱ→kh=ㅋ 격음화) |
| `s9_gold_catalog.py` | `_mdd_analysis/.../error_catalog_gold/L1_발음오류_gold카탈로그.md` (81국) | **gold 위치·유형**: 어느 음소에 어떤 유형이 걸리나(예 베트남 종성 ㄹ받침 39%, 경음화 ㄲ 43%) |

> 두 카탈로그는 **상보적**: speech가 규칙상쇄로 과소검출하는 경음화·비음화를 gold가 상세 제공.

### 10-B. 전체 배치 확정 수치 (s2 전량 실행 완료)
- gold **1,881,760건** 처리(0 스킵/0 에러) · has_error 513,161(27.3%) · **115개국**.
- B1 벤치마크(전체): 83개국·test **56.4만 발화** · 미시 AP lift **×1.18** · 매크로 **×1.36** · **전 유형 lift>1**.
- DER 0.783 → prior만으론 발화별 정확 지목 불가 = **인식기(갈래 B) 필요성의 정량 근거**.

### 10-C. 문서 인덱스 (읽는 순서)
1. `_mdd_analysis/00_산출물_인덱스.md` — 전체 산출물 지도(스크립트·CSV·리포트 위치).
2. `실행체크리스트_현황점검_2026-07-06.md` — 착수 전 체크리스트 대비 현황(STEP 0~6).
3. `GPU작업_상세_및_전체체크리스트.md` — GPU 필요 작업 상세.
4. `발표자료_음소인식기_1A_1B.html` — 다음 착수(backbone 벤치 1-A + XLS-R+CTC 인식기 1-B) 설계 발표자료.
5. `처리보고서_전체배치_2026-07-06.md` — 이번 세션 처리 요약.
6. `_report/L1prior_통계리포트.html` — 그림 7종 시각화 리포트.

### 10-D. 현재 상태 한 줄
> **로드맵 갈래 A(알고리즘 트랙) 코드 작업 완결.** 국적별 발음오류 패턴(speech·gold 2소스)·L1 prior·진단 가중치(81국)·벤치마크·카탈로그·시각화 모두 **전체 데이터로 산출**됨. 남은 실질 과제는 전부 **GPU 트랙(코드 트랙 밖)**.

### 10-E. 다음 착수 순서 (사용자·조직 액션)
| 순서 | 할 일 | 담당 | 게이트 |
|---|---|---|---|
| **1** | **GPU 예산·환경 결정**(자체 vs 클라우드, 상한) | 🏢 이윤구 | **이게 전부의 선행조건** |
| 2 | AI Hub·데이터바우처 라이선스(학습 사용 범위) 확인 | 🏢 이윤구 | 파인튜닝 전 |
| 3 | STEP 1-4 backbone 벤치(XLS-R 300M/1B/MMS PER·지연·비용→1개 확정) | 🔴 코드(GPU) | 1 완료 후 |
| 4 | STEP 3 XLS-R+CTC 인식기 학습 → phone 단위 MDD F1 + 발화별 위치 확정 | 🔴 코드(GPU) | 3 완료 후 |
| 5 | STEP 4-1 GOP 점수 · 4-2 UI(기호→한글, 재료 확보됨) | 🔴/코드 | 4 완료 후 |
| 병행 | 인스타 2만 검증 · IR/과제 문서화(산출 리포트=실증 근거) | 🏢 | GPU 무관, 지금 가능 |

> **결론: 지금 유일한 병목은 "GPU 결정"(순서 1).** 결정되면 1-A→1-B가 도미노로 이어지고 코드는 착수 준비 완료. 그 전까지 GPU 없이 가능한 실작업은 병행 항목(인스타 검증·IR 문서)뿐.
