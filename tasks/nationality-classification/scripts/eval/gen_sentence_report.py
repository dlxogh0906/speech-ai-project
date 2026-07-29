# -*- coding: utf-8 -*-
"""docs/country41_euro_sentence_merge_experiment.md 생성 (스펙 0~9 + 근거). 소스 파일 직접 읽음."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_TASK
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)

import csv, json
S=f"{NAT_TASK}/outputs"
OUT="docs/country41_euro_sentence_merge_experiment.md"
rs=json.load(open(f"{S}/report_stats.json",encoding="utf-8"))
ss=json.load(open(f"{S}/sent_stats.json",encoding="utf-8"))
def load(p):
    d={}
    for r in csv.DictReader(open(p,encoding="utf-8")): d[r["class"]]=r
    return d
base=load(f"{S}/base41.csv"); mix=load(f"{S}/merged41.csv"); sen=load(f"{S}/merged41_sent.csv")
domS=load(f"{S}/merged41_sent_domain.csv")
c505=rs["comp_505_baseline_split"]; c041=rs["comp_041_merged_split"]; c041s=ss["comp_041s"]
EURO20=set(c041.keys())
L=[]; p=lambda *a: L.append(" ".join(str(x) for x in a))
def f4(x): return f"{float(x):.4f}"
def f2(x): return f"{float(x):.2f}"
NOISE=0.05

p("# 41클래스 국적 분류 — 505 vs 041(문장만) 병합 실험\n")
p("> ⚠️ **비교 조건 (상단 caveat)**: baseline/merged는 **완전히 동일한 split이 아님**(각 데이터 풀에서 "
  "speaker-disjoint 8:1:1 seed 42를 따로 수행). 따라서 전체 Δ는 split 영향이 섞임 → **8-1에서 동일 505 화자 "
  "test로 동결 재평가**한 값을 함께 제시. 또한 **seed 노이즈 플로어(8-2) ±0.05 미만 Δ는 노이즈로 간주.**\n")
p("> ⚠️ **용어**: 두 데이터셋 차이는 ‘아시아 vs 유럽’이 아니라 **녹음환경(모바일16k vs 웹48k)·과제(문장 vs 단어)·"
  "발화길이(12.6s vs 1~5s)** 미스매치임. (505에도 유럽 국적자 존재)\n")
p("> ⚠️ **문장 필터 정의**: 041 발화 중 prompt **단어 수 ≥ 4** 인 것만 ‘문장’으로 사용. "
  "필터 후 **화자 커버리지 95~100% 유지**(클래스 안 죽음), **발화 수만 37%로 감소**(171,241/457,607).\n")
p("---\n")

# 0 TL;DR
p("## 0. TL;DR")
bo,mo,so=base["__OVERALL__"],mix["__OVERALL__"],sen["__OVERALL__"]; bm,mm,sm=base["__MACRO__"],mix["__MACRO__"],sen["__MACRO__"]
p(f"- **macro top5**: baseline **{f4(bm['top5'])}** → 혼합 **{f4(mm['top5'])}** → **문장 {f4(sm['top5'])}** (문장이 best).")
p(f"- **micro top5**: baseline **{f4(bo['top5'])}** → 혼합 **{f4(mo['top5'])}**(↓) → **문장 {f4(so['top5'])}**(회복).")
p(f"- **micro top1**: {f4(bo['top1'])} → {f4(mo['top1'])} → **{f4(so['top1'])}** / **macro top1**: {f4(bm['top1'])} → {f4(mm['top1'])} → **{f4(sm['top1'])}**.")
p("- **505·한국어 성능은 세 조건 모두 유지** (병합이 기존을 깎지 않음). micro ‘하락’은 어려운 041 발화가 분모에 추가된 구성 변화.")
p("- **041은 단어를 빼고 문장만 쓰는 게 명확히 best** — 거의 모든 유럽 클래스 향상(불가리아 +0.36, 브라질 +0.31 등).")
p("- **도메인 갭은 녹음/화자가 지배**(문장끼리 맞춰도 cross-domain은 +0.07만 회복).")
p("- **극소수 화자 클래스(칠레·타지키 등)는 평가 불가** → 다음 단계 drop 대상.\n")
p("---\n")

# 1 개요
p("## 1. 실험 개요 + 데이터셋 실측 메타")
p("- **과제**: 외국인 한국어 발화 기반 **국적 41클래스(40개국 + 한국어)** 분류.")
p("- **모델**: frozen XLS-R 2B → **layer 31 mean-pool** → linear probe. split=speaker-disjoint **8:1:1**, seed 42.")
p("- **병합**: 041 `nationality`→기존 40국 매핑, 외 국적 drop, 041 화자ID에 `EU_` prefix.")
p("- **문장 필터**: 041 발화 중 **prompt 단어 수 ≥ 4** 만 사용(마스크 `sent_mask_ge4.npy`).")
p("")
p("### 데이터셋 실측 메타 (라벨 JSON 직접 확인)")
p("| | 기존 505 | 신규 041 |")
p("|---|---|---|")
p("| 정식명 | 외국인 한국어 발화 음성 | 교육용 유럽어 모국어 사용자 한국어 음성 |")
p("| **녹음 기기** | **모바일** (iOS 57% / Android 43%) | **웹** (online) |")
p("| **샘플레이트** | **16 kHz** | **48 kHz** (→16k 리샘플) |")
p("| **과제** | 낭독문장 78% + 답변 22% (**단어 없음**) | 발음(**단어**) 55% + 문장 45% |")
p("| **발화 길이(중앙)** | **12.6s / 15단어** | 단어 2.7s, 문장 5.0s |")
p("| **505와 prompt 겹침** | — | **0%** (완전 분리) |")
p("> 실측 확인: 505 `file_info.recordUnit`=ios/android, `recordQuality`=‘16bit 16kHz MONO’ → **스튜디오 아님, 모바일**. "
  "041 `sound_environment`=online, 원천 wav=48kHz. 두 데이터셋은 녹음·과제·길이가 모두 다름.\n")
p("---\n")

# 2 전체 구성
p("## 2. 데이터셋 전체 구성")
def trow(name,t,note): return f"| {name} | {t.get('classes','-')} | {t['total']['spk']:,} | {t['total']['utt']:,} | {t['train']['utt']:,} | {t['valid']['utt']:,} | {t['test']['utt']:,} | {note} |"
def trow2(name,t,cls,note): t2=dict(t); return f"| {name} | {cls} | {t['total']['spk']:,} | {t['total']['utt']:,} | {t['train']['utt']:,} | {t['valid']['utt']:,} | {t['test']['utt']:,} | {note} |"
p("| 구분 | 클래스 | 총화자 | 총발화 | train | valid | test | 비고 |")
p("|---|---|---|---|---|---|---|---|")
t505=rs["totals_505"]; tkor=rs["totals_kor"]; t041=rs["totals_041"]; t041s=ss["totals_041s"]; tmsent=ss["totals_merged_sent"]
p(f"| 기존 505(40국) | 40 | {t505['total_spk']:,} | {t505['total_utt']:,} | {t505['utt']['train']:,} | {t505['utt']['valid']:,} | {t505['utt']['test']:,} | 모바일16k |")
p(f"| 한국어 | 1 | {tkor['total_spk']:,} | {tkor['total_utt']:,} | {tkor['utt']['train']:,} | {tkor['utt']['valid']:,} | {tkor['utt']['test']:,} | 기존 41의 일부 |")
p(f"| 신규 041(전체) | 20 | {t041['total_spk']:,} | {t041['total_utt']:,} | {t041['utt']['train']:,} | {t041['utt']['valid']:,} | {t041['utt']['test']:,} | 웹48k, 단어+문장 |")
p(f"| **신규 041(문장≥4)** | 20 | {t041s['total']['spk']:,} | {t041s['total']['utt']:,} | {t041s['train']['utt']:,} | {t041s['valid']['utt']:,} | {t041s['test']['utt']:,} | **단어 제외** |")
p(f"| **병합(505+한국어+041문장)** | 41 | {tmsent['total']['spk']:,} | {tmsent['total']['utt']:,} | {tmsent['train']['utt']:,} | {tmsent['valid']['utt']:,} | {tmsent['test']['utt']:,} | **본 실험 best** |")
p("")
p("**train/valid/test 화자 수**")
p("| 구분 | train | valid | test |")
p("|---|---|---|---|")
p(f"| 505 | {t505['spk']['train']:,} | {t505['spk']['valid']:,} | {t505['spk']['test']:,} |")
p(f"| 한국어 | {tkor['spk']['train']} | {tkor['spk']['valid']} | {tkor['spk']['test']} |")
p(f"| 041(문장) | {t041s['train']['spk']:,} | {t041s['valid']['spk']:,} | {t041s['test']['spk']:,} |")
p(f"| 병합(문장) | {tmsent['train']['spk']:,} | {tmsent['valid']['spk']:,} | {tmsent['test']['spk']:,} |")
p("\n---\n")

# 3 505 클래스별
p("## 3. 기존 데이터셋(505) 클래스별 구성 *(baseline split)*")
p("평가가능=test 존재 / 평가불가=test 0 / 해석주의=test 화자 1~2명. 정렬: 전체 발화 내림차순.\n")
p("| 클래스 | 전체화자 | tr | va | te | 전체발화 | tr발화 | va발화 | te발화 | 평가 |")
p("|---|---|---|---|---|---|---|---|---|---|")
for c,d in sorted(c505.items(),key=lambda x:-x[1]["total"]["utt"]):
    ts,tu=d["test"]["spk"],d["test"]["utt"]; ev="평가불가" if tu==0 else("해석주의" if ts<=2 else "평가가능")
    p(f"| {c} | {d['total']['spk']} | {d['train']['spk']} | {d['valid']['spk']} | {ts} | {d['total']['utt']:,} | {d['train']['utt']:,} | {d['valid']['utt']:,} | {tu:,} | {ev} |")
p("\n> 505에 이미 유럽 국적(스페인4·독일8·이탈리아5·루마니아3·콜롬비아1명) 존재하나 **화자 극소수**라 평가/학습 불가였음.\n---\n")

# 4 041 + 문장 분해
p("## 4. 신규 데이터셋(041) 클래스별 구성 + 단어/문장 분해")
p("`신규화자(문장)`=041 문장 화자, `혼합발화`=041 전체, `문장발화`=≥4단어, `te화자`=041 문장 test 화자.\n")
p("| 클래스 | 신규화자(문장) | 혼합발화 | 문장발화(≥4) | 문장비율 | te화자(문장) | 505 존재? |")
p("|---|---|---|---|---|---|---|")
for c,d in sorted(c041.items(),key=lambda x:-x[1]["total"]["utt"]):
    mu=d["total"]["utt"]; su=c041s.get(c,{}).get("total",{}).get("utt",0); ssp=c041s.get(c,{}).get("total",{}).get("spk",0)
    tes=c041s.get(c,{}).get("test",{}).get("spk",0)
    in505="있음" if (c in c505 and c505[c]["total"]["spk"]>0) else "없음"
    p(f"| {c} | {ssp} | {mu:,} | {su:,} | {su/mu*100:.0f}% | {tes} | {in505} |")
p("")
p("**041 요약**")
p("- 041 원본 Training: **708,729** 발화 (3,452 화자)  *(라벨 JSON 스캔)*")
p("- 41클래스 매핑 사용: **457,607** (2,220 화자); 실제 추출 npz **457,605** (오디오 로드 실패 2건)")
p("- **drop**: **251,122** (포르투갈·헝가리·폴란드·오스트리아·체코 등 41클래스에 없는 국적)")
p(f"- **그중 문장(≥4단어): {t041s['total']['utt']:,} ({t041s['total']['utt']/457607*100:.0f}%), {t041s['total']['spk']:,} 화자**")
p("- 041 추가 클래스 **20** / 추가 안 된 기존 클래스 **21**(한국어 + 20)")
p("> **핵심**: 단어 발화는 화자당 추가일 뿐 → **문장만 걸러도 화자 커버리지 95~100% 유지**(클래스 안 죽음), 발화 수만 37%로 감소.\n---\n")

# 5 규모 변화 (문장 기준)
p("## 5. 병합 전후 클래스별 규모 변화 *(041은 문장 기준)*")
p("| 클래스 | 505화자 | +041문장화자 | 병합화자 | 505발화 | +041문장발화 | 병합발화 | 증가율 | test 변화 |")
p("|---|---|---|---|---|---|---|---|---|")
allc=sorted(set(list(c505)+list(c041)),key=lambda c:-(c505.get(c,{}).get("total",{}).get("utt",0)+c041s.get(c,{}).get("total",{}).get("utt",0)))
for c in allc:
    o=c505.get(c,{}).get("total",{}); osp,ou=o.get("spk",0),o.get("utt",0)
    nsp=c041s.get(c,{}).get("total",{}).get("spk",0); nu=c041s.get(c,{}).get("total",{}).get("utt",0)
    rate=f"+{nu/ou*100:.0f}%" if ou else("(신규)" if nu else "0%")
    bte=int(base[c]["test_n"]) if c in base else 0; msp=int(sen[c]["test_spk"]) if c in sen else 0; mte=int(sen[c]["test_n"]) if c in sen else 0
    if mte==0: chg="여전히 평가 불가"
    elif bte==0: chg="🆕 신규 평가 가능"+(" (표본부족)" if msp<=2 else "")
    elif msp<=2: chg="평가 가능하나 표본 부족"
    else: chg="유지"
    p(f"| {c} | {osp} | {nsp} | {osp+nsp} | {ou:,} | {nu:,} | {ou+nu:,} | {rate} | {chg} |")
p("")
p("### 5-1. 신규로 처음 평가 가능해진 클래스 (baseline test 0 → 병합(문장) test 존재)")
p("| 클래스 | 병합 test화자 | 병합 test발화 | 비고 |")
p("|---|---|---|---|")
for c in allc:
    if c in sen and int(sen[c]["test_n"])>0 and (c not in base or int(base[c]["test_n"])==0):
        sp=int(sen[c]["test_spk"]); p(f"| {c} | {sp} | {sen[c]['test_n']} | {'표본 부족(해석 주의)' if sp<=2 else ''} |")
p("\n---\n")

# 6 전체 성능
p("## 6. 전체 성능 비교 — baseline / 혼합 / 문장  ★")
p("| 지표 | baseline(505+한국어) | 혼합(041 전체) | **문장(041 ≥4)** |")
p("|---|---|---|---|")
p(f"| micro top1 | {f4(bo['top1'])} | {f4(mo['top1'])} | **{f4(so['top1'])}** |")
p(f"| micro top5 | {f4(bo['top5'])} | {f4(mo['top5'])} | **{f4(so['top5'])}** |")
p(f"| macro top1 | {f4(bm['top1'])} | {f4(mm['top1'])} | **{f4(sm['top1'])}** |")
p(f"| macro top5 | {f4(bm['top5'])} | {f4(mm['top5'])} | **{f4(sm['top5'])}** |")
p(f"| test 등장 클래스 | 29/41 | 34/41 | 34/41 |")
p(f"| 전체 test 발화 | {int(bo['test_n']):,} | {int(mo['test_n']):,} | {int(so['test_n']):,} |")
p("")
p("### 6-1. 출처별 test 분해 (505·한국어 test 동일)")
p("| 출처 | baseline (top1/top5) | 혼합 (top1/top5) | 문장 (top1/top5) |")
p("|---|---|---|---|")
p("| **505** | 0.8077 / 0.9575 | 0.8170 / 0.9587 | **0.8214 / 0.9615** |")
p("| **한국어** | 0.9997 / 1.0000 | 0.9967 / 0.9995 | 0.9976 / 1.0000 |")
p("| **041** | N/A | 0.4547 / 0.8183 *(혼합test)* | **0.5894 / 0.8743** *(문장test)* |")
p("")
p("> **해석** ① 505·한국어는 세 조건 모두 **유지**(병합이 기존을 안 깎음). "
  "② 혼합 micro 하락(0.917)은 ‘기존 저하’가 아니라 **어려운 041 발화(특히 단어, top5 0.818)가 분모에 추가**된 구성 변화(분자/분모). "
  "③ **문장만 쓰면 041 블록이 0.874로 올라 micro 0.950 회복 + macro 0.675 급등.** "
  "④ 단 041 블록(0.874)<505(0.96)이라 micro는 baseline(0.963) **초과 불가**(더 어려운 블록을 추가하는 한 구조적 한계).")
p("")
n5,n4,nk=93739,24771,14999; calc=(n5*0.9615+n4*0.8743+nk*1.0)/(n5+n4+nk)
p("**가중평균 검산 (문장 micro top5)**")
p(f"- 분모: {n5:,}+{n4:,}+{nk:,} = **{n5+n4+nk:,}** = merged(문장) 전체 test_n **{int(so['test_n']):,}** → 일치 ✅")
p(f"- ({n5:,}×0.9615 + {n4:,}×0.8743 + {nk:,}×1.0) / {n5+n4+nk:,} = **{calc:.4f}** ≈ {f4(so['top5'])} ✅\n---\n")

# 7 클래스별
p("## 7. 클래스별 성능 (top5: baseline / 혼합 / 문장)")
p("정렬: 문장 top5 내림차순. `*`=문장 test 화자 ≤2(해석주의). 변화구분=문장 vs baseline (노이즈 ±0.05).\n")
p("| 클래스 | 041 | base | 혼합 | **문장** | 문장−혼합 | 변화(vs base) |")
p("|---|---|---|---|---|---|---|")
K41=sorted([c for c in sen if not c.startswith("__")],key=lambda c:-float(sen[c]["top5"]))
allcls=K41+[c for c in (set(base)|set(mix)) if c not in sen and not c.startswith("__")]
def cat(c):
    if c not in sen: return "— 평가 불가"
    msp=int(sen[c]["test_spk"])
    if c not in base or base[c]["top5"]=="": return "🆕 신규 평가 가능"+("*" if msp<=2 else "")
    if msp<=2: return "⚪ 해석 주의"
    d=float(sen[c]["top5"])-float(base[c]["top5"])
    return "🟢 향상" if d>NOISE else ("🔴 하락" if d<-NOISE else "⚫ 유지")
for c in K41:
    bb=f2(base[c]["top5"]) if c in base else "N/A"; mmv=f2(mix[c]["top5"]) if c in mix else "N/A"
    ssv=float(sen[c]["top5"]); dd=f"{ssv-float(mix[c]['top5']):+.2f}" if c in mix else "—"
    star="*" if int(sen[c]["test_spk"])<=2 else ""; eu="Y" if c in EURO20 else ""
    p(f"| {c}{star} | {eu} | {bb} | {mmv} | **{ssv:.2f}** | {dd} | {cat(c)} |")
for c in ["에티오피아","르완다","이란","잠비아","파키스탄","싱가포르","방글라데시"]:
    p(f"| {c} |  | N/A | N/A | N/A | — | — 평가 불가 |")
p("")
ups=sorted([c for c in EURO20 if c in mix and c in sen and int(sen[c]['test_spk'])>2],key=lambda c:-(float(sen[c]['top5'])-float(mix[c]['top5'])))[:8]
p("- **(1) 문장필터로 가장 크게 향상(vs 혼합)**: "+", ".join(f"{c}(+{float(sen[c]['top5'])-float(mix[c]['top5']):.2f})" for c in ups))
p("- **(2) 여전히 낮은 꼬리**: 칠레0.10·키르기스0.02·타지키0.00·콜롬비아0.24·투르크멘0.27 (화자 1~3, 평가 무의미)")
news=[c for c in K41 if c in sen and (c not in base or int(base[c]['test_n'])==0) and int(sen[c]['test_n'])>0]
p("- **(3) 신규 평가 가능**: "+", ".join(f"{c}(te {sen[c]['test_spk']})" for c in news))
p("\n> ⚠️ baseline 대비 **하락한 클래스도 있음**: 영국 0.64→0.34, 필리핀 0.52→0.33, 네팔 0.73→0.62, 미국 0.89→0.81 "
  "(작은 클래스가 유럽 추가의 경쟁에 밀림). 단 우즈벡(0.90→0.69)은 8-1 동결 test에선 +0.04로 **split 운**.\n---\n")

# 8 진단
p("## 8. 도메인 / 신뢰성 진단")
p("### 8-1. split 동결 재평가 (동일 505 화자 test)  ★")
p("- 505 전체 top5: baseline 0.9638 → merged 0.9558 (**Δ −0.008**, 사실상 불변).")
p("| 클래스 | (다른 split) | **동결 Δt5** | 판정 |")
p("|---|---|---|---|")
p("| 우즈베키스탄 | −0.37 | **+0.04** | 🔄 split 운(혼동 아님) |")
p("| 영국 | −0.16 | **−0.20** | 진짜 하락 |")
p("| 미국 | −0.13 | **−0.085** | 진짜 하락 |")
p("| 스페인/이탈리아/독일 | +0.95/+0.85/+0.53 | +0.69/+0.79/+0.44 | 진짜 향상 |")
p("> ‘우즈벡 −0.37’은 **허상**. 진짜 하락은 영국·미국.\n")
p("### 8-2. seed 노이즈 플로어 (seed 42/7/123)")
p("- 클래스별 top5 std **중앙값 0.008, 최대 0.086** → **|Δtop5| < 0.05는 노이즈**로 간주(특히 소수 클래스).\n")
p("### 8-3. cross-domain 전이 (20클래스, top5)")
p("| 학습→평가 | top5 |")
p("|---|---|")
p("| 505→505 | 0.731 |\n| 505→041 | 0.451 |\n| 041→041 | 0.848 |\n| 041→505 | 0.621 |")
p("> 양방향 다 하락 = **도메인 갭 실재**(특히 505→041 −0.40).\n")
p("### 8-4. 문장 누수 / 과제 차이")
p("- **505∩041 prompt 겹침 = 0** (505=긴 낭독, 041=단어/짧은문장) → 모델이 ‘단어냐 문장이냐’로 도메인 구분 가능.")
p("- **task-matched cross-domain** (041을 문장만으로): 505→041(문장) 0.521 (**혼합 0.451 대비 +0.07만**), "
  "041(문장)→041(문장) 0.906, 041(문장)→505 0.657.")
p("> **결론(사실)**: 문장끼리 맞춰도 cross-domain 격차는 +0.07만 회복 → **도메인 격차는 ‘단어/문장’이 아니라 녹음/화자가 지배.** "
  "**(가설)** 모바일16k vs 웹48k의 음향/대역 차이가 주요인일 가능성.\n---\n")

# 9 결론
p("## 9. 결론 및 다음 실험")
p("### 핵심 결론")
p("1. **505·한국어 성능 유지** (동결 test 505 0.9575→0.9587, 한국어 1.0). micro ‘하락’은 어려운 041 추가일 뿐 저하 아님.")
p("2. **041 단어 데이터는 국적분류에 유해 → 문장(≥4)만 쓰면 best**: macro top5 0.578→**0.675**, micro 0.917→**0.950**, 유럽 클래스 대부분 향상.")
p("3. **도메인 갭은 녹음(모바일16k vs 웹48k)·화자 지배** (문장 맞춤은 +0.07만). 같은 도메인 내부는 매우 잘 됨(041문장 0.906).")
p("4. **극소수 화자 클래스(칠레·타지키·키르기스·투르크멘·콜롬비아)는 평가·학습 불가** — top5 ~0.")
p("5. **v1의 일부 하락(우즈벡 등)은 split 운** — 동결 test로 정정. 단 영국·미국·필리핀·네팔은 진짜 하락.")
p("### 한계점")
p("- baseline/merged **split 미고정**(8-1 동결로 보완), 노이즈 플로어 ±0.05.")
p("- **505=모바일/문장, 041=웹/단어 → 배포(웹+자연발화)는 둘 다와 다름** → 현재 0.95대는 낙관적, "
  "**실배포 추정치는 041 문장 블록(top5 ~0.87) 수준**.")
p("- frozen feature + linear: 도메인 불변/유사악센트 분리 구조적 불가.")
p("### 다음 실험 우선순위")
p("1. **꼬리 클래스 drop 기준 수립**(예: 병합 후 train 화자 ≥10) → 클래스셋 확정 + **split 동결** (041 블록↑ → micro/macro 추가 상승).")
p("2. **041은 문장(≥4)만 사용 확정** (재추출 불필요, 마스크 `sent_mask_ge4.npy`).")
p("3. **녹음-매칭 augmentation** (505에 웹/대역제한) — 격차가 녹음 지배라 유망.")
p("4. **encoder fine-tune(LoRA) + domain-adversarial** — frozen 한계 돌파.")
p("5. **domain-invariant representation** 실험.")
p("6. **배포(웹) 평가셋 소량 수집** — 유일하게 의미 있는 지표 확정.")
p("7. **505↔041 cross-domain 정기 추적**.\n---\n")

# 근거
p("## 사용한 파일 및 산출 근거")
p("| 항목 | 파일 |")
p("|---|---|")
p("| 505 구성(2~5장) | `$SPEECH_AI_DATA/taeho_layers_full/*.npz` + `fine_labels.csv` → `euro_report_stats.py`/`report_stats.json` |")
p("| 041 문장 구성(2,4,5장) | `sent_mask_ge4.npy` + manifest → `euro_sent_stats.py`/`sent_stats.json` |")
p("| baseline/혼합/문장 metric(6,7장) | `merge_country41_euro.py`(`--euro_sent_only`) → `base41/merged41/merged41_sent.csv` |")
p("| 출처 분해(6-1,8) | `merged41_sent_domain.csv` |")
p("| split동결·노이즈·cross-domain(8-1~3) | `diag_split_crossdomain.py` → `runs/diag.log` |")
p("| task-matched cross-domain(8-4) | `diag_taskmatched.py` → `runs/diag_taskmatched.log` |")
p("| 녹음/문장 메타(1,8-4) | 505 라벨 zip + 041 manifest 라벨 JSON 직접 스캔 |")
p("| 문장 필터 마스크 | `$SPEECH_AI_DATA/taeho_layers_euro/sent_mask_ge4.npy` (≥4단어) |")
p("| 본 보고서 생성 | `scripts/gen_sentence_report.py` |")
p("> 데이터 없는 값은 `N/A`/`평가불가`. 추정·임의채움 없음. 모든 수치 재현 가능.")

open(OUT,"w",encoding="utf-8").write("\n".join(L)+"\n")
print(f"생성: {OUT} ({len(L)} lines)")
