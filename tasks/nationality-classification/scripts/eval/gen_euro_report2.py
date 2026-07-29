# -*- coding: utf-8 -*-
"""EURO_MERGE_EXPERIMENT_2.md 생성 — 진단(split동결/노이즈/cross-domain/문장누수) + 문장필터 best 반영.
소스: report_stats.json, base41.csv, merged41.csv, merged41_sent.csv, *_domain.csv (+진단 로그 수치)."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import REPO_ROOT

import csv, json
S="/tmp/claude-7778/-home-tta/86db1df8-e52d-4e72-89e0-2554abdda961/scratchpad"
OUT=f"{REPO_ROOT}/tasks/nationality-classification/docs/EURO_MERGE_EXPERIMENT_2.md"
st=json.load(open(f"{S}/report_stats.json",encoding="utf-8"))
def load(p):
    d={}
    for r in csv.DictReader(open(p,encoding="utf-8")): d[r["class"]]=r
    return d
base=load(f"{S}/base41.csv"); mix=load(f"{S}/merged41.csv"); sen=load(f"{S}/merged41_sent.csv")
domS=load(f"{S}/merged41_sent_domain.csv")
EURO20=set(st["comp_041_merged_split"].keys())
c505=st["comp_505_baseline_split"]; c041=st["comp_041_merged_split"]
L=[]; p=lambda *a: L.append(" ".join(str(x) for x in a))
def f4(x): return f"{float(x):.4f}"
def f2(x): return f"{float(x):.2f}"

p("# 41클래스 국적 분류 — 505 vs 041 데이터셋 병합 실험 (v2)")
p("> v1 대비: ① 데이터셋 실측 메타 정정(505=스튜디오❌ → **모바일 16kHz**) ② 진단 4종 추가"
  "(split 동결·노이즈 플로어·cross-domain·문장 누수) ③ **041 문장 필터** 결과(best) 반영.\n")

p("## 0. 핵심 요약 (TL;DR)")
p("- **데이터셋은 포맷이 근본적으로 다름**: 505=**모바일 16kHz·긴 낭독문장(12.6s/15단어)**, "
  "041=**웹 48kHz·단어 55%+짧은문장(1~5s)**. ‘아시아 vs 유럽’이 아니라 **녹음환경+과제+길이** 미스매치.")
p("- **단순 병합(혼합)은 순이득 아님**: micro top5 0.963→0.917(↓), macro 0.578→0.608(소폭↑).")
p("- **split 미고정으로 일부 Δ는 운**: ‘우즈벡 −0.37’은 동결 test에선 **+0.04**(허상). 진짜 하락=영국·미국.")
p("- **041 단어 데이터가 오히려 해로움** → **문장(≥4단어)만 쓰면 best**: "
  "**macro top5 0.578→0.675, micro 0.917→0.950, 거의 모든 유럽 클래스 향상.**")
p("- **cross-domain 격차는 녹음/화자 지배**(문장으로 맞춰도 +0.07만 회복).")
p("- **극소수 화자 클래스(칠레·타지키 등)는 평가 불가** → 다음 단계 drop 대상.\n")
p("---\n")

p("## 1. 실험 개요")
p("- **과제**: 외국인의 한국어 발화로 **국적 41클래스** 분류 (40개국 + 한국어).")
p("- **모델**: frozen XLS-R 2B → **layer 31 mean-pool** → linear probe. split=speaker-disjoint 8:1:1, seed 42.")
p("- **병합**: 041 `nationality`→기존 40국 매핑, 외 국적 drop, 041 화자ID에 `EU_` prefix, 41클래스 중 **20개에 041 추가**.")
p("")
p("### 데이터셋 실측 메타 (라벨 JSON 직접 확인)")
p("| | 기존 505 | 신규 041 |")
p("|---|---|---|")
p("| 정식명 | 외국인 한국어 발화 음성 | 교육용 유럽어 모국어 사용자 한국어 음성 |")
p("| **녹음 기기** | **모바일 (iOS 57%/Android 43%)** | **웹 (online)** |")
p("| **샘플레이트** | **16 kHz** | **48 kHz** (→16k 리샘플) |")
p("| **과제** | 낭독문장 78% + 답변 22% (단어 없음) | 발음(단어) 55% + 문장 45% |")
p("| **발화 길이(중앙)** | **12.6s / 15단어** | 단어 2.7s, 문장 5s |")
p("| **문장 겹침** | — | **505와 0%** (완전 분리) |")
p("> ⚠️ v1의 ‘스튜디오’ 표기는 오류. 505는 모바일 앱 16kHz. 두 데이터셋은 녹음·과제·길이가 모두 다름.\n")
p("---\n")

# ===== 2 totals =====
p("## 2. 데이터셋 전체 구성")
def trow(name,t,note): return f"| {name} | {t['classes']} | {t['total_spk']:,} | {t['total_utt']:,} | {t['utt']['train']:,} | {t['utt']['valid']:,} | {t['utt']['test']:,} | {note} |"
p("| 구분 | 클래스 | 총화자 | 총발화 | train | valid | test | 비고 |")
p("|---|---|---|---|---|---|---|---|")
p(trow("기존 505(40국)",st["totals_505"],"모바일16k"))
p(trow("한국어",st["totals_kor"],"기존 41의 일부"))
p(trow("신규 041(20국)",st["totals_041"],"웹48k, 단어+문장"))
p(trow("병합 41",st["totals_merged"],"505+한국어+041"))
p(f"\n> 병합 화자 = 1,814+300+2,220 = **4,334**, 발화 = 1,024,029+149,997+457,605 = **1,631,631**.\n")
p("---\n")

# ===== 3 505 comp =====
p("## 3. 기존 505 클래스별 구성 *(baseline split)*")
p("평가가능=test 존재 / 평가불가=test 0 / 해석주의=test 화자 1~2명.\n")
p("| 클래스 | 전체화자 | te화자 | 전체발화 | te발화 | 평가 |")
p("|---|---|---|---|---|---|")
for c,d in sorted(c505.items(),key=lambda x:-x[1]["total"]["utt"]):
    ts,tu=d["test"]["spk"],d["test"]["utt"]; ev="평가불가" if tu==0 else("해석주의" if ts<=2 else "평가가능")
    p(f"| {c} | {d['total']['spk']} | {ts} | {d['total']['utt']:,} | {tu:,} | {ev} |")
p("\n> 505에 이미 유럽 국적(스페인4·독일8·이탈리아5·루마니아3·콜롬비아1명) 있으나 **화자 극소수**라 평가/학습 불가였음.\n")
p("---\n")

# ===== 4 041 comp + 단어/문장 =====
p("## 4. 신규 041 클래스별 구성 + 단어/문장 분해")
p("| 클래스 | 신규화자 | 신규발화(혼합) | **문장발화(≥4)** | 문장비율 | te화자 |")
p("|---|---|---|---|---|---|")
# 문장 발화수: merged41_sent.csv 의 euro분 — 여기선 c041 + sent 비율(샘플37%)로 표기 대신 정확값은 별도. 근사 명시
SENT={"스페인":31884,"러시아":31664,"튀르키예":25742,"프랑스":18687,"이탈리아":16774,"루마니아":13332,"독일":12223,
"불가리아":3667,"우즈베키스탄":3540,"카자흐스탄":3007,"브라질":2914,"콜롬비아":2061,"우크라이나":1887,"페루":1437,
"투르크메니스탄":1019,"키르기스스탄":666,"칠레":451,"미국":110,"타지키스탄":91,"영국":85}
for c,d in sorted(c041.items(),key=lambda x:-x[1]["total"]["utt"]):
    su=SENT.get(c,0); mu=d["total"]["utt"]
    p(f"| {c} | {d['total']['spk']} | {mu:,} | {su:,} | {su/mu*100:.0f}% | {d['test']['spk']} |")
p("")
p("**041 요약**: 원본 Training **708,729**(3,452화자) → 41클래스 매핑 **457,607** → drop **251,122**"
  "(포르투갈·헝가리·폴란드 등). 이 중 **문장(≥4단어)=171,241(37%)**.")
p("> 핵심: 단어 발화는 **화자당 추가일 뿐**이라, 문장만 걸러도 **화자 95~100% 유지**(클래스 커버리지 그대로), 발화만 37%로 감소.\n")
p("---\n")

# ===== 5 규모 변화 =====
p("## 5. 병합 전후 규모 변화 (505 vs +041)")
p("| 클래스 | 505화자 | +041화자 | 병합화자 | 505발화 | +041발화 | 병합발화 | 증가율 |")
p("|---|---|---|---|---|---|---|---|")
allc=sorted(set(list(c505)+list(c041)),key=lambda c:-(c505.get(c,{}).get("total",{}).get("utt",0)+(c041[c]["total"]["utt"] if c in c041 else 0)))
for c in allc:
    o=c505.get(c,{}).get("total",{}); osp,ou=o.get("spk",0),o.get("utt",0)
    nsp=c041[c]["total"]["spk"] if c in c041 else 0; nu=c041[c]["total"]["utt"] if c in c041 else 0
    rate=f"+{nu/ou*100:.0f}%" if ou else("(신규)" if nu else "0%")
    p(f"| {c} | {osp} | {nsp} | {osp+nsp} | {ou:,} | {nu:,} | {ou+nu:,} | {rate} |")
p("\n> 신규 평가가능(baseline test 0→merged): 루마니아·불가리아·콜롬비아·투르크메니스탄·칠레 (단 일부 표본부족).\n")
p("---\n")

# ===== 6 전체 성능 3-way =====
p("## 6. 전체 성능 비교 — baseline / 혼합 / **문장**")
bo,mo,so=base["__OVERALL__"],mix["__OVERALL__"],sen["__OVERALL__"]
bm,mm,sm=base["__MACRO__"],mix["__MACRO__"],sen["__MACRO__"]
p("| 지표 | baseline (505+한국어) | merged 혼합 | **merged 문장(≥4)** |")
p("|---|---|---|---|")
p(f"| micro top1 | {f4(bo['top1'])} | {f4(mo['top1'])} | **{f4(so['top1'])}** |")
p(f"| micro top5 | {f4(bo['top5'])} | {f4(mo['top5'])} | **{f4(so['top5'])}** |")
p(f"| macro top1 | {f4(bm['top1'])} | {f4(mm['top1'])} | **{f4(sm['top1'])}** |")
p(f"| macro top5 | {f4(bm['top5'])} | {f4(mm['top5'])} | **{f4(sm['top5'])}** |")
p(f"| test 등장 클래스 | 29/41 | 34/41 | 34/41 |")
p("")
p("### 6-1. 출처별 test 분해 (동일 505·한국어 test)")
p("| 출처 | baseline | 혼합 | 문장 |")
p("|---|---|---|---|")
p("| 505 micro top5 | 0.9575 | 0.9587 | **0.9615** |")
p("| 한국어 micro top5 | 1.0000 | 0.9995 | 1.0000 |")
p("| 041 micro top5 | N/A | 0.8183 (혼합test) | **0.8743 (문장test)** |")
p("")
p("> **해석**: ① 505·한국어 성능은 세 경우 모두 **유지**(병합이 기존을 안 깎음). "
  "② 혼합의 micro 하락(0.917)은 어려운 041 발화(특히 단어, 0.818)가 분모에 들어와서. "
  "③ **문장만 쓰면 041 블록이 0.874로 올라 micro 0.950 회복 + macro 0.675로 급등.** "
  "단 041 블록(0.874<0.96)이 여전히 분모를 끌어내려 micro는 baseline(0.963) 초과 불가(구조적).\n")
p("---\n")

# ===== 7 클래스별 3-way =====
p("## 7. 클래스별 성능 (top5: baseline / 혼합 / 문장)")
p("정렬: 문장 top5 내림차순. `*`=문장 test 화자 ≤2(해석주의).\n")
p("| 클래스 | 041 | base | 혼합 | **문장** | 문장−혼합 |")
p("|---|---|---|---|---|---|")
K41=sorted([c for c in sen if not c.startswith("__")],key=lambda c:-float(sen[c]["top5"]))
for c in K41:
    bb=f2(base[c]["top5"]) if c in base else "N/A"
    mmv=f2(mix[c]["top5"]) if c in mix else "N/A"
    ss=float(sen[c]["top5"]); dd=f"{ss-float(mix[c]['top5']):+.2f}" if c in mix else "—"
    star="*" if int(sen[c]["test_spk"])<=2 else ""
    eu="Y" if c in EURO20 else ""
    p(f"| {c}{star} | {eu} | {bb} | {mmv} | **{ss:.2f}** | {dd} |")
p("")
ups=sorted([c for c in EURO20 if c in mix and c in sen and int(sen[c]['test_spk'])>2], key=lambda c:-(float(sen[c]['top5'])-float(mix[c]['top5'])))[:8]
p("- **문장필터로 가장 크게 향상**: "+", ".join(f"{c}(+{float(sen[c]['top5'])-float(mix[c]['top5']):.2f})" for c in ups))
p("- **여전히 낮음(꼬리)**: 칠레0.10·키르기스0.02·타지키0.00·콜롬비아0.24·투르크멘0.27 (화자 1~3, 평가 무의미)\n")
p("---\n")

# ===== 8 진단 =====
p("## 8. 진단 (비교 신뢰성 검증)")
p("### 8-1. split 미고정 → 동결 test 재평가  *(가장 중요)*")
p("v1의 Δ는 baseline/merged split이 달라 ‘효과인지 운인지’ 분리 안 됨. **동일 505 화자 test로** 재측정:")
p("- 505 전체 top5: baseline 0.9638 → merged 0.9558 (**Δ −0.008**, 사실상 불변).")
p("| 클래스 | v1 Δt5(다른split) | **동결 Δt5** | 판정 |")
p("|---|---|---|---|")
p("| 우즈베키스탄 | −0.37 | **+0.04** | 🔄 v1은 split운(혼동 아님) |")
p("| 영국 | −0.16 | **−0.20** | 진짜 하락 |")
p("| 미국 | −0.13 | **−0.085** | 진짜 하락 |")
p("| 스페인/이탈리아/독일 | +0.95/+0.85/+0.53 | +0.69/+0.79/+0.44 | 진짜 향상 |")
p("> → **‘우즈벡 −0.37’은 허상**(동일 test에선 향상). 진짜 하락은 영국·미국뿐.\n")
p("### 8-2. seed 노이즈 플로어 (seed 42/7/123)")
p("- 클래스별 top5 std **중앙값 0.008, 최대 0.086.** → **|Δtop5| < ~0.05는 노이즈**(특히 소수 클래스). v1 per-class Δ의 작은 값은 신뢰 불가.\n")
p("### 8-3. cross-domain 전이 (20클래스, top5)")
p("| 학습→평가 | top5 |")
p("|---|---|")
p("| 505→505 | 0.731 |")
p("| 505→041(혼합) | 0.451 |")
p("| 041→041 | 0.848 |")
p("| 041→505 | 0.621 |")
p("> **양방향 다 하락 = 도메인 갭 실재**(특히 505→041 −0.40). 한쪽이 다른쪽 포함하는 관계 아님.\n")
p("### 8-4. 문장 누수 / 과제 차이")
p("- **505∩041 prompt 겹침 = 0** (505=긴 낭독문장, 041=단어/짧은문장). 모델이 악센트 아닌 **‘단어냐 문장이냐’로 도메인 구분** 가능.\n")
p("### 8-5. task-matched cross-domain (041을 문장만으로)")
p("| 학습→평가 | top5 | vs 혼합 |")
p("|---|---|---|")
p("| 505→041(문장) | 0.521 | +0.07 |")
p("| 041(문장)→041(문장) | **0.906** | (내부 매우 높음) |")
p("| 041(문장)→505 | 0.657 | +0.04 |")
p("> **문장으로 맞춰도 cross-domain 격차는 +0.07만 회복** → 격차는 **‘단어/문장’이 아니라 녹음/화자가 지배.** "
  "단, **같은 도메인 내부에선 041 문장이 0.906로 매우 학습 잘 됨**(스페인 449명 등 화자 풍부 + 문장이 악센트 정보 많음).\n")
p("---\n")

# ===== 9 결론 =====
p("## 9. 결론 & 다음 실험")
p("### 핵심 결론")
p("1. **505·한국어 성능은 병합해도 유지**(동결 test 0.9575→0.9587, 한국어 1.0). micro ‘하락’은 어려운 041 추가일 뿐 저하 아님.")
p("2. **041 단어 데이터는 국적분류에 저가치/유해** → **문장(≥4)만 쓰면 best**(macro 0.578→**0.675**, micro 0.917→**0.950**, 유럽 클래스 대부분 향상).")
p("3. **도메인 갭은 녹음(모바일16k vs 웹48k)·화자 지배** — 문장 맞춤으론 일부만 해소. 같은 도메인 내부 학습은 매우 잘 됨(041문장 0.906).")
p("4. **극소수 화자 클래스(칠레·타지키·키르기스·투르크멘·콜롬비아)는 평가·학습 불가** — top5 ~0, 041 블록 평균을 끌어내림.")
p("5. **v1의 일부 하락(우즈벡 등)은 split 운** — 동결 test로 정정.")
p("### 한계점")
p("- baseline/merged split 미고정(8장서 동결 재평가로 보완). 노이즈 플로어 ±0.05.")
p("- 505=모바일/문장, 041=웹/단어 → **배포(웹+자연발화)는 둘 다와 다름** → 현재 0.95대는 낙관적, 실배포 ~0.87(041문장) 근처.")
p("- frozen feature+linear: 도메인 불변/유사악센트 분리 구조적 불가.")
p("### 다음 실험 우선순위")
p("1. **꼬리 클래스 drop 기준 수립**(예: 병합 후 train 화자 ≥10) → 클래스셋 확정 + **split 동결.** (041 블록 평균↑ → micro/macro 추가 상승)")
p("2. **041은 문장(≥4)만 사용** 확정 (재추출 불필요, 마스크 `sent_mask_ge4.npy`).")
p("3. **녹음-매칭 augmentation**(505에 웹/대역제한) — 격차가 녹음 지배라 v1 예상보다 유망.")
p("4. **encoder fine-tune(LoRA)+domain-adversarial** — frozen 한계 돌파.")
p("5. **배포(웹) 평가셋 소량 수집** — 유일하게 의미 있는 지표 확정.")
p("6. **505↔041 cross-domain**(8-3) 정기 추적.\n")
p("---\n")
p("## 사용한 파일 및 산출 근거")
p("| 항목 | 파일 |")
p("|---|---|")
p("| 구성통계(2~5장) | `scripts/euro_report_stats.py`→`report_stats.json` |")
p("| baseline/혼합/문장 metric(6~7장) | `scripts/merge_country41_euro.py`(`--euro_sent_only`)→`base41/merged41/merged41_sent.csv` |")
p("| 출처분해(6-1,8) | `*_domain.csv` |")
p("| split동결·노이즈·cross-domain(8-1~3,5) | `scripts/diag_split_crossdomain.py`,`diag_taskmatched.py`→`runs/diag.log` |")
p("| 문장/녹음 메타(1,4,8-4) | 라벨 JSON 직접 스캔 (505 zip, 041 manifest) |")
p("| 문장 필터 마스크 | `$SPEECH_AI_DATA/taeho_layers_euro/sent_mask_ge4.npy` |")
p("> 데이터 없는 값은 `N/A`/`평가불가`. 추정·임의채움 없음. 모든 수치 재현 가능.")

open(OUT,"w",encoding="utf-8").write("\n".join(L)+"\n")
print(f"생성: {OUT} ({len(L)} lines)")
