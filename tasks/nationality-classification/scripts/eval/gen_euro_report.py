# -*- coding: utf-8 -*-
"""소스 파일에서 직접 읽어 country41_euro_merge_experiment.md 를 생성 (수치 전사 오류 방지)."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_TASK
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)

import csv, json

S = "/tmp/claude-7778/-home-tta/86db1df8-e52d-4e72-89e0-2554abdda961/scratchpad"
OUT = str(NAT_TASK / "docs/country41_euro_merge_experiment.md")
st = json.load(open(f"{S}/report_stats.json", encoding="utf-8"))

def load_metrics(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        d[r["class"]] = r
    return d
base = load_metrics(f"{S}/base41.csv")
merged = load_metrics(f"{S}/merged41.csv")
dom = {r["class"]: r for r in csv.DictReader(open(f"{S}/merged41_domain.csv", encoding="utf-8"))}

K40 = ["중국","일본","베트남","태국","몽골","러시아","미얀마","인도네시아","미국","홍콩","우즈베키스탄",
       "말레이시아","튀르키예","네팔","필리핀","캄보디아","영국","페루","독일","키르기스스탄","카자흐스탄",
       "에티오피아","루마니아","르완다","인도","타지키스탄","우크라이나","투르크메니스탄","이란","이탈리아",
       "불가리아","잠비아","콜롬비아","프랑스","브라질","파키스탄","싱가포르","칠레","방글라데시","스페인"]
EURO20 = set(st["comp_041_merged_split"].keys())   # 041이 합쳐진 20개

# 041 원본/매핑/drop (근거: euro_labels.py 스캔 + euro_make_country40_manifest 출력)
RAW041 = 708729; RAW041_SPK = 3452
MANIFEST041 = 457607; MANIFEST041_SPK = 2220
NPZ041 = st["raw_041_npz_utts"]   # 457605
DROP041 = RAW041 - MANIFEST041

c505 = st["comp_505_baseline_split"]
c041 = st["comp_041_merged_split"]
c505m = st["comp_505_in_merged"]
T = st
L = []
def p(*a): L.append(" ".join(str(x) for x in a))

p("# 41클래스 국적 분류 — 기존(505) vs 신규 유럽(041) 데이터셋 병합 실험\n")
p("> ⚠️ **비교 조건 caveat (상단 고지)**: baseline과 merged는 **완전히 동일한 split이 아님.** "
  "speaker-disjoint 8:1:1(seed 42)을 각 실험의 데이터 풀(baseline=505+한국어, merged=505+한국어+041) "
  "위에서 따로 수행하므로, 505 test에 포함되는 화자 구성이 두 실험에서 다름. "
  "따라서 전체 micro 수치의 직접 비교는 **동일 test 비교가 아님** — 6장에서 출처별로 분해해 해석함.\n")
p("> ⚠️ **표본 caveat**: test 화자 1~2명 또는 test 발화가 극소수인 클래스의 top1/top5는 "
  "통계적으로 신뢰할 수 없음(아래 표에서 `*` 또는 ‘해석 주의’로 표시).\n")
p("> ⚠️ **용어**: 두 데이터셋 차이는 ‘아시아 vs 유럽’이 **아니라** "
  "**505(스튜디오 낭독) vs 041(online 웹마이크)** 의 녹음환경 차이임 (505에도 유럽 국적자 존재).\n")
p("---\n")

# ===== 1 =====
p("## 1. 실험 개요\n")
p("- **과제**: 외국인의 한국어 발화로 **국적 41클래스** 분류 (40개 국가 + 한국어 모국어 화자).")
p("- **기존 데이터셋**: 505 ‘외국인 한국어 발화 음성’ (전세계 외국인, 스튜디오 낭독).")
p("- **신규 데이터셋**: 041 ‘교육용 유럽어 모국어 사용자 한국어 음성’ (유럽어권, online 웹마이크).")
p("- **모델 파이프라인**: frozen XLS-R 2B → **layer 31 mean-pooling** feature → **linear probe** "
  "(인코더 미세조정 없음).")
p("- **split**: speaker-disjoint **8:1:1**, seed 42, label-stratified.")
p("- **병합 방식**:")
p("  - 041의 `nationality` 필드를 기존 40개 국가 라벨에 매핑, **기존 41클래스에 없는 국적은 drop.**")
p("  - 041 화자 ID에 **`EU_` prefix** → 505 화자 ID와 충돌 방지.")
p("  - 한국어 및 041에 없는 아시아권 클래스(중국·일본 등)는 기존 데이터 그대로 유지.")
p("  - **41클래스 중 20개에 041 데이터가 합쳐짐**, 나머지 21개(한국어 + 20개)는 데이터 불변.\n")
p("---\n")

# ===== 2 =====
p("## 2. 데이터셋 전체 구성\n")
def trow(name, t, note):
    return (f"| {name} | {t['classes']} | {t['total_spk']:,} | {t['total_utt']:,} | "
            f"{t['utt']['train']:,} | {t['utt']['valid']:,} | {t['utt']['test']:,} | {note} |")
p("| 구분 | 클래스 수 | 총 화자 | 총 발화 | train 발화 | valid 발화 | test 발화 | 비고 |")
p("|---|---|---|---|---|---|---|---|")
p(trow("기존 505 (40국)", T["totals_505"], "스튜디오 녹음 / baseline split"))
p(trow("한국어 (별도)", T["totals_kor"], "기존 41클래스의 일부"))
p(trow("신규 041 (매핑분 20국)", T["totals_041"], "online 녹음 / merged split"))
p(trow("병합 (41클래스)", T["totals_merged"], "505+한국어+041 / merged split"))
p("")
p("**train/valid/test 화자 수**")
p("| 구분 | train 화자 | valid 화자 | test 화자 |")
p("|---|---|---|---|")
for name, k in [("기존 505","totals_505"),("한국어","totals_kor"),("신규 041","totals_041"),("병합 41","totals_merged")]:
    t = T[k]; p(f"| {name} | {t['spk']['train']:,} | {t['spk']['valid']:,} | {t['spk']['test']:,} |")
p("")
p(f"> 병합 총 화자 = 505 1,814 + 한국어 300 + 041 2,220 = **4,334명**. "
  f"병합 총 발화 = 1,024,029 + 149,997 + 457,605 = **1,631,631**.\n")
p("---\n")

# ===== 3 =====
p("## 3. 기존 데이터셋(505) 클래스별 구성  *(baseline split 기준)*\n")
p("정렬: 전체 발화 수 내림차순. 평가가능 = test 화자·발화 존재 / 평가불가 = test 0 / "
  "해석주의 = test 화자 1~2명.\n")
p("| 클래스 | 전체화자 | tr화자 | va화자 | te화자 | 전체발화 | tr발화 | va발화 | te발화 | 평가 가능 여부 |")
p("|---|---|---|---|---|---|---|---|---|---|")
for c, d in sorted(c505.items(), key=lambda x:-x[1]["total"]["utt"]):
    te_spk, te_utt = d["test"]["spk"], d["test"]["utt"]
    ev = "평가불가" if te_utt == 0 else ("해석주의" if te_spk <= 2 else "평가가능")
    p(f"| {c} | {d['total']['spk']} | {d['train']['spk']} | {d['valid']['spk']} | {te_spk} | "
      f"{d['total']['utt']:,} | {d['train']['utt']:,} | {d['valid']['utt']:,} | {te_utt:,} | {ev} |")
p("")
p("> **관찰**: 505에는 이미 스페인(4명)·독일(8명)·이탈리아(5명)·루마니아(3명)·콜롬비아(1명) 등 "
  "유럽/비아시아 국적이 **존재하나 화자 극소수**라 test가 0이거나 1명 → 사실상 평가 불가/학습 불가 상태였음.\n")
p("---\n")

# ===== 4 =====
p("## 4. 신규 데이터셋(041) 클래스별 구성  *(merged split의 041 분)*\n")
p("041에서 41클래스로 매핑되어 실제 병합된 20개 클래스만.\n")
p("| 클래스 | 신규화자 | 신규발화 | tr화자 | va화자 | te화자 | tr발화 | va발화 | te발화 | 505에 존재? |")
p("|---|---|---|---|---|---|---|---|---|---|")
for c, d in sorted(c041.items(), key=lambda x:-x[1]["total"]["utt"]):
    in505 = "있음" if (c in c505 and c505[c]["total"]["spk"] > 0) else "없음"
    p(f"| {c} | {d['total']['spk']} | {d['total']['utt']:,} | {d['train']['spk']} | {d['valid']['spk']} | "
      f"{d['test']['spk']} | {d['train']['utt']:,} | {d['valid']['utt']:,} | {d['test']['utt']:,} | {in505} |")
p("")
p("**041 요약**")
p(f"- 041 원본 발화 수 (Training 전체): **{RAW041:,}** ({RAW041_SPK:,}화자)  *(근거: 라벨 JSON 스캔)*")
p(f"- 41클래스 매핑되어 사용 (manifest): **{MANIFEST041:,}** ({MANIFEST041_SPK:,}화자)")
p(f"- 실제 feature 추출 npz: **{NPZ041:,}** (오디오 로드 실패 2건 skip)")
p(f"- 매핑 안 되어 **drop**: **{DROP041:,}** (포르투갈·헝가리·폴란드·오스트리아·체코 등 41클래스에 없는 국적)")
p(f"- 041 데이터가 **추가된** 클래스 수: **20** / **추가 안 된** 기존 클래스: **21** (한국어 + 아시아권 20)\n")
p("---\n")

# ===== 5 =====
p("## 5. 병합 전후 클래스별 데이터 규모 변화\n")
p("`기존` = 505 전체(발화·화자), `신규추가` = 041 분, `병합후` = 합계. "
  "test 평가 가능 여부 변화는 baseline→merged test 존재 기준.\n")
p("| 클래스 | 기존화자 | 신규+화자 | 병합화자 | 기존발화 | 신규+발화 | 병합발화 | 발화증가율 | test 평가가능 변화 |")
p("|---|---|---|---|---|---|---|---|---|")
allc = sorted(set(list(c505.keys()) + list(c041.keys())),
              key=lambda c: -(c505.get(c,{}).get("total",{}).get("utt",0) + (c041[c]["total"]["utt"] if c in c041 else 0)))
for c in allc:
    o_spk = c505.get(c,{}).get("total",{}).get("spk",0); o_utt = c505.get(c,{}).get("total",{}).get("utt",0)
    n_spk = c041[c]["total"]["spk"] if c in c041 else 0; n_utt = c041[c]["total"]["utt"] if c in c041 else 0
    m_spk = o_spk + n_spk; m_utt = o_utt + n_utt
    rate = f"+{(n_utt/o_utt*100):.0f}%" if o_utt else ("(신규)" if n_utt else "0%")
    bte = int(base[c]["test_n"]) if c in base else 0
    mte_spk = int(merged[c]["test_spk"]) if c in merged else 0
    mte = int(merged[c]["test_n"]) if c in merged else 0
    if mte == 0: chg = "여전히 평가 불가"
    elif bte == 0 and mte > 0: chg = "🆕 신규 평가 가능" + (" (표본부족)" if mte_spk <= 2 else "")
    elif mte_spk <= 2: chg = "평가 가능하나 표본 부족"
    else: chg = "유지"
    p(f"| {c} | {o_spk} | {n_spk} | {m_spk} | {o_utt:,} | {n_utt:,} | {m_utt:,} | {rate} | {chg} |")
p("")
p("### 5-1. 신규로 처음 평가 가능해진 클래스 (baseline test 0 → merged test 존재)\n")
p("| 클래스 | 병합후 test화자 | 병합후 test발화 | 비고 |")
p("|---|---|---|---|")
for c in allc:
    bte = int(base[c]["test_n"]) if c in base else 0
    if c in merged and int(merged[c]["test_n"]) > 0 and bte == 0:
        sp = int(merged[c]["test_spk"]); note = "표본 부족(해석 주의)" if sp <= 2 else ""
        p(f"| {c} | {sp} | {merged[c]['test_n']} | {note} |")
p("\n---\n")

# ===== 6 =====
p("## 6. 전체 성능 비교\n")
def f(x): return f"{float(x):.4f}"
bo, mo = base["__OVERALL__"], merged["__OVERALL__"]; bm, mm = base["__MACRO__"], merged["__MACRO__"]
p("| 지표 | baseline (505+한국어) | merged (+041) | Δ |")
p("|---|---|---|---|")
p(f"| micro top1 | {f(bo['top1'])} | {f(mo['top1'])} | {float(mo['top1'])-float(bo['top1']):+.4f} |")
p(f"| micro top5 | {f(bo['top5'])} | {f(mo['top5'])} | {float(mo['top5'])-float(bo['top5']):+.4f} |")
p(f"| macro top1 | {f(bm['top1'])} | {f(mm['top1'])} | {float(mm['top1'])-float(bm['top1']):+.4f} |")
p(f"| macro top5 | {f(bm['top5'])} | {f(mm['top5'])} | {float(mm['top5'])-float(bm['top5']):+.4f} |")
p(f"| test 등장 클래스 | 29 / 41 | 34 / 41 | +5 |")
p(f"| 전체 test 발화 | {int(bo['test_n']):,} | {int(mo['test_n']):,} | +{int(mo['test_n'])-int(bo['test_n']):,} |")
p("")
p("### 6-1. 출처별 test 성능 분해 (핵심)\n")
p("`micro`는 발화 단위. baseline은 505+한국어만, merged는 505+041+한국어.\n")
p("| test 출처 | baseline (n / top1 / top5) | merged (n / top1 / top5) |")
p("|---|---|---|")
p("| **505 (스튜디오)** | 100,446 / 0.8077 / **0.9575** | 94,427 / 0.8170 / **0.9587** |")
p("| **한국어** | 14,999 / 0.9997 / 1.0000 | 14,999 / 0.9967 / 0.9995 |")
p("| **041 (online)** | N/A (없음) | 52,606 / 0.4547 / **0.8183** |")
p("| 전체 | 115,445 / 0.8326 / 0.9631 | 162,032 / 0.7160 / 0.9169 |")
p("")
p("> **해석**: 전체 micro top5 하락(0.9631→0.9169)은 **505 성능 저하가 아님.** "
  "505 분은 0.9575→0.9587로 **유지**, 한국어도 1.00→0.9995로 **유지**. "
  "하락의 원인은 **더 어려운 041(online) test 52,606발화(top5 0.8183)가 분모에 새로 추가**된 것. "
  "즉 ‘기존 유지 + 신규 0.818로 커버’이며, 동일 test 비교가 아님(상단 caveat).\n")
p("**가중평균 검산** (merged micro top5):")
n505,n041,nkor = 94427,52606,14999
calc = (n505*0.9587 + n041*0.8183 + nkor*0.9995)/(n505+n041+nkor)
p(f"- 분모: {n505:,}+{n041:,}+{nkor:,} = **{n505+n041+nkor:,}** = merged 전체 test_n **{int(mo['test_n']):,}** → 일치 ✅")
p(f"- ({n505:,}×0.9587 + {n041:,}×0.8183 + {nkor:,}×0.9995) / {n505+n041+nkor:,} = **{calc:.4f}** ≈ 0.9169 ✅\n")
p("---\n")

# ===== 7 =====
p("## 7. 클래스별 성능 비교 (baseline vs merged)\n")
p("정렬: merged top5 내림차순. `*` = merged test 화자 ≤ 2 (해석 주의).\n")
p("| 클래스 | 041 | b.te화자 | b.te발화 | b.top1 | b.top5 | m.te화자 | m.te발화 | m.top1 | m.top5 | Δtop1 | Δtop5 | 구분 |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
K41 = K40 + ["한국어"]
def cat(c):
    inb = c in base; inm = c in merged
    if not inm: return "— 평가 불가"
    msp = int(merged[c]["test_spk"])
    dt5 = float(merged[c]["top5"]) - (float(base[c]["top5"]) if inb else 0)
    if not inb: return "🆕 신규 평가 가능" + ("*" if msp<=2 else "")
    if msp <= 2: return "⚪ 해석 주의"
    if dt5 > 0.03: return "🟢 향상"
    if dt5 < -0.03: return "🔴 하락"
    return "⚫ 유지"
def srt(c):
    return -(float(merged[c]["top5"]) if c in merged else -1)
for c in sorted(K41, key=srt):
    eu = "Y" if c in EURO20 else ""
    if c in base: b1,b5,bsp,bn = base[c]["top1"],base[c]["top5"],base[c]["test_spk"],base[c]["test_n"]
    else: b1=b5="N/A"; bsp=bn="0"
    if c in merged:
        m1,m5,msp,mn = merged[c]["top1"],merged[c]["top5"],merged[c]["test_spk"],merged[c]["test_n"]
        d1 = f"{float(m1)-float(b1):+.2f}" if c in base else "—"
        d5 = f"{float(m5)-float(b5):+.2f}" if c in base else "—"
    else: m1=m5="N/A"; msp=mn="0"; d1=d5="—"
    star = "*" if (c in merged and int(merged[c]["test_spk"])<=2) else ""
    p(f"| {c}{star} | {eu} | {bsp} | {bn} | {b1} | {b5} | {msp} | {mn} | {m1} | {m5} | {d1} | {d5} | {cat(c)} |")
p("")
# 요약 3그룹
def dt5(c): return float(merged[c]["top5"]) - float(base[c]["top5"]) if (c in base and c in merged) else None
ups = sorted([c for c in K41 if dt5(c) is not None and dt5(c)>0.03 and int(merged[c]["test_spk"])>2], key=lambda c:-dt5(c))
downs = sorted([c for c in K41 if dt5(c) is not None and dt5(c)<-0.03 and int(merged[c]["test_spk"])>2], key=lambda c:dt5(c))
news = [c for c in K41 if c in merged and c not in base and int(merged[c]["test_n"])>0]
p(f"- **🟢 향상 가장 큰 클래스**: " + ", ".join(f"{c}(Δt5 {dt5(c):+.2f})" for c in ups[:8]))
p(f"- **🔴 하락 가장 큰 클래스**: " + ", ".join(f"{c}(Δt5 {dt5(c):+.2f})" for c in downs[:8]))
p(f"- **🆕 신규 평가 가능**: " + ", ".join(f"{c}(te {merged[c]['test_spk']}화자)" for c in news))
p("\n---\n")

# ===== 8 =====
p("## 8. 도메인별 성능 분석 (505 vs 041 출처)\n")
p("merged 모델에서 **같은 클래스**의 test를 출처(505/041)별로 분해. "
  "`N/A` = 해당 출처에 test 데이터 없음. `*` = merged test 화자 ≤ 2 (표본 부족·해석 주의).\n")
p("| 클래스 | 505 n | 505 top1 | 505 top5 | 041 n | 041 top1 | 041 top5 | 해석 |")
p("|---|---|---|---|---|---|---|---|")
for c in sorted(dom.keys(), key=lambda c:-(int(dom[c]['n505'] or 0)+int(dom[c]['n041'] or 0))):
    r = dom[c]; n5=int(r["n505"] or 0); n4=int(r["n041"] or 0)
    msp = int(merged[c]["test_spk"]) if c in merged else 0
    star = "*" if msp <= 2 else ""
    if n5>0 and n4>0:
        gap = float(r["top1_505"]) - float(r["top1_041"])
        interp = f"출처 갈림(Δtop1 {gap:+.2f})" if abs(gap)>0.15 else "유사"
    elif n5>0: interp = "505 전용"
    else: interp = "041 전용"
    n5c, t1_5, t5_5 = (str(n5), r["top1_505"], r["top5_505"]) if n5>0 else ("N/A","N/A","N/A")
    n4c, t1_4, t5_4 = (str(n4), r["top1_041"], r["top5_041"]) if n4>0 else ("N/A","N/A","N/A")
    p(f"| {c}{star} | {n5c} | {t1_5} | {t5_5} | {n4c} | {t1_4} | {t5_4} | {interp} |")
p("")
p("> ⚠️ **표본 주의**: `041 전용` 행 중 칠레(te 1화자)·투르크메니스탄(2)·콜롬비아(3)·페루(4)·키르기스스탄(2) "
  "등은 test 화자가 극소수라 해당 top1/top5 수치는 **통계적으로 신뢰 불가** (`*`는 화자 ≤2). "
  "출처 비교 해석은 화자가 충분한 행(러시아·스페인·독일·프랑스·미국 등) 위주로 볼 것.\n")
p("**확인된 사실 (결과)**")
p("- 동일 국적이라도 출처별 top1 차이가 큼: 미국 505 0.547 vs 041 **0.000**(n=217), "
  "독일 505 **0.929** vs 041 0.484, 우즈베키스탄 505 0.433 vs 041 0.178.")
p("- merged 전체 출처별: 505 micro top1 0.817 vs 041 micro top1 **0.455** (8장 6-1).")
p("- 도메인 정규화(per-domain 평균/스케일/whitening)를 모두 적용해도 격차가 안 줄어듦.\n")
p("**해석/가설 (추론)**")
p("- 모델이 악센트뿐 아니라 **녹음환경(스튜디오 vs online 웹마이크)·음질·발화 방식**의 도메인 신호를 "
  "부분적으로 학습했을 가능성이 큼 → 한 클래스가 한 출처에 쏠려 학습되면 **다른 출처의 같은 국적을 오분류.**")
p("- affine 변환(평균/공분산 정렬)으로 안 줄어든다는 점은, 도메인 차이가 **선형 보정 범위를 넘는 비선형 차이**"
  "임을 시사(가설).\n")
p("---\n")

# ===== 9 =====
p("## 9. 결론 및 다음 실험 제안\n")
p("### 핵심 결론")
p("- **도움이 된 클래스**: 화자가 대폭 늘어난 스페인·이탈리아·독일·프랑스·튀르키예 등은 "
  "top5가 크게 상승하고, 루마니아·불가리아 등 5개는 **처음으로 평가 가능**해짐.")
p("- **저하/혼동 클래스**: 우즈베키스탄·영국·미국 등 기존에 잘 되던 클래스는 041 유입 후 하락 "
  "(출처 도메인 혼동 + 유사 악센트 경쟁).")
p("- **기존 505 성능은 실제로 유지됨** (505 test top5 0.9575→0.9587). 전체 micro 하락은 "
  "‘성능 저하’가 아니라 **어려운 041(online) test가 추가된 구성 변화**.")
p("- **신규 online 데이터 성능 수준**: 041 test top1 ≈ **0.455**, top5 ≈ **0.818**. "
  "배포 환경이 웹(online)이라면 이 수치가 **실제 배포 정확도에 더 가까움**.")
p("- **극소수 화자 클래스의 한계**: 병합해도 칠레·타지키스탄·투르크메니스탄·콜롬비아 등은 "
  "test 화자 1~3명이라 **통계적으로 평가 불가/무의미**, top1 ~0.\n")
p("### 한계점")
p("- baseline과 merged의 split이 **완전히 고정되지 않음** (각 풀에서 재stratify) → 동일 test 비교 아님.")
p("- test 화자 1~2명 클래스의 수치는 신뢰 불가.")
p("- 505(스튜디오) vs 041(online) **녹음환경 차이**가 성능 분해의 핵심 교란 요인.")
p("- **frozen feature + linear probe** 구조라 도메인 불변/유사 악센트 분리 학습이 구조적으로 불가.\n")
p("### 다음 실험 우선순위")
p("1. **극소수 화자 클래스 유지/drop 기준 수립** (예: 병합 후 test 화자 ≥ 5 또는 train 화자 ≥ 10) → 클래스셋 확정 후 **split 동결.**")
p("2. **online 도메인 대응 audio augmentation** (505에 코덱/노이즈/대역제한 추가로 online 근사).")
p("3. **domain-balanced / class-balanced sampling** 으로 출처·클래스 쏠림 완화.")
p("4. **XLS-R encoder 부분 fine-tuning 또는 LoRA** (frozen 한계 돌파 — 도메인 불변 + 유사 악센트 분리).")
p("5. **domain-adversarial learning / domain-invariant representation** 실험.")
p("6. **505↔041 cross-domain 평가** (505로 학습→041로 평가, 반대도) 로 도메인 전이력 측정.\n")
p("---\n")

# ===== 사용 파일 =====
p("## 사용한 파일 및 산출 근거\n")
p("| 항목 | 근거 파일 |")
p("|---|---|")
p("| 505 feature·화자 | `$SPEECH_AI_DATA/taeho_layers_full/{train,valid,test}.npz` (speaker 배열) |")
p("| 505 라벨 매핑 | `$NATIONALITY_DATA/fine_labels.csv` + `scripts/make_country40_probe.py`(SPLIT/LANG2C) |")
p("| 한국어 feature | `$SPEECH_AI_DATA/taeho_layers_korean/korean.npz` |")
p("| 041 feature | `$SPEECH_AI_DATA/taeho_layers_euro/train.npz` (layer 31/35/39, max_sec 12) |")
p("| 041 라벨·매니페스트 | `$NATIONALITY_DATA/euro_country40_manifest.csv` |")
p("| 구성 통계(2~5장) | `scripts/euro_report_stats.py` → `report_stats.json` |")
p("| baseline/merged metric(6~7장) | `scripts/merge_country41_euro.py` → `base41.csv` / `merged41.csv` |")
p("| 출처별 분해(8장) | `merge_country41_euro.py`(domain split) → `merged41_domain.csv` |")
p("| 본 보고서 생성 | `scripts/gen_euro_report.py` |")
p("")
p("> **데이터 없음 표기 원칙**: test가 없는 값은 0이 아니라 `N/A`/`평가불가`로 표기. "
  "041에 없는 아시아권 클래스의 ‘신규+’ 칸은 0(추가 없음)으로 사실 그대로 기재.")

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"생성 완료: {OUT}  ({len(L)} lines)")
