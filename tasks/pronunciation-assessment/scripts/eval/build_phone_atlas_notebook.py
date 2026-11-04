"""Build and execute a Korean all-jamo notebook; make a self-contained HTML copy."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists(): ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]

import argparse
import base64
import html
import io
import json
import re

import nbformat
from nbclient import NotebookClient
from mdd.phone_atlas import Atlas, PHONES, GROUPS, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--kernel',default='qwen35')
    a=ap.parse_args();assets=a.assets.resolve();folder=assets.parent
    relative_assets=assets.relative_to(ROOT)
    atlas=Atlas(assets);r=atlas.report
    nb=nbformat.v4.new_notebook()
    nb.metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name=a.kernel),
                     language_info=dict(name='python',version='3.11'))
    cells=[]
    md=lambda text:cells.append(nbformat.v4.new_markdown_cell(text))
    code=lambda text:cells.append(nbformat.v4.new_code_cell(text))
    md('''# 한국어 전체 자모: 실제 음성·파형·M3 전사율

**먼저 ① 전체 표를 보고, ② 대립군을 듣고, ③ 원하는 글자 사례를 확인하세요.**

이 파일에는 실행한 표·그림·음성이 저장돼 있습니다. 다시 실행하려면 NumPy·SciPy·Matplotlib·pandas·IPython이 있는 Python 커널을 선택하세요. GPU나 모델 적재는 필요 없습니다.

| 항목 | 범위 |
|---|---|
| 모델 | 운영 식별자 M3 `w2vbert_synth_tense_v2/step_3000`의 **저장된** 평가 출력 |
| 음성 | AIHub 040~043 실제 한국어 학습자 녹음. 합성 음성 아님 |
| 주 평가 | 6,720녹음 중 입력 검사 통과 3,472녹음 · 대응 가능한 69,985자리 |
| 글자 범위 | 현대 자모 51개 표시: 모델 사전 46개 + 사전 밖 겹받침 표기 5개 |
| 확보 범위 | 전사율 38개 자모 · 음성/그림 예시 44개 자모. 없는 자료는 빈칸으로 유지 |
| 문항 조건 | M3 학습 문항의 별도 평가 녹음. **새 문장 일반화 점수가 아님** |

**기대 LAB**은 발음해야 하는 소리, **실제 TAG**는 원래 데이터에 전사된 소리, **M3 출력**은 저장된 `actual_jamo`입니다. 독립 청취로 TAG를 다시 검증한 것은 아닙니다.

예: 기대 `ㅆ` → TAG `ㅅ` → 모델 `ㅅ`이면 **전사는 성공**입니다. 이때 화면에서 경고했는지는 별도입니다. 경고가 없다고 발음이 맞았다는 뜻은 아닙니다.

목차: [1. 글자별 전사율](#전체표) · [2. 자주 헷갈리는 출력](#혼동) · [3. 소리 비교](#대립비교) · **[ㅅ·ㅆ 특징 분포](#ㅅㅆ특징)** · [4. 글자 선택](#선택) · [5. 모든 글자 사례](#글자사례) · [6. 집계 범위](#범위)
''')
    code('''from pathlib import Path
import sys, json, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown

search = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in search if (p / "pyproject.toml").exists()), None)
if ROOT is None:
    raise RuntimeError("speech-ai-project 폴더 안에서 이 노트북을 실행하세요.")
sys.path.insert(0, str(ROOT / "tasks/pronunciation-assessment/src"))
from mdd.phone_atlas import Atlas, PHONES, GROUPS, configure_font
ASSETS = ROOT / "tasks/pronunciation-assessment/notebooks/phone_atlas_assets"
atlas = Atlas(ASSETS)
configure_font()
metrics = atlas.table()
report = atlas.report
pd.set_option("display.max_rows", 60)
pd.set_option("display.max_colwidth", 100)
print("M3 저장 출력 재집계 · 새 모델 추론 0회 · 모델 학습 0회")
print(f"평가 {report['primary_items']}문항 · {report['primary_speakers']}화자 / 녹음 3,472개")
''')
    md('''<a id="전체표"></a>
## 1. 실제 소리를 글자별로 얼마나 그대로 전사했나?

**전사 일치율 = TAG가 해당 자모인 자리 중, 모델 출력도 같은 자리의 비율.**

정상과 오발음을 반드시 나눠 보세요. `정상`은 기대 LAB=실제 TAG, `오발음`은 기대 LAB≠실제 TAG입니다. 아래 행은 **실제 TAG 자모** 기준입니다. 따라서 실제 `ㅅ` 행의 오발음에는 `ㅆ→ㅅ` 외의 방향도 포함됩니다.

모델이 그 자리에 소리를 내지 않은 경우도 실패로 분모에 남깁니다. 실제 삭제 455자리는 발음된 글자가 아니므로 뒤에서 따로 보여줍니다. 표본 0은 0%가 아닙니다.
''')
    code('''def counted(k, n):
    return f"{int(k):,}/{int(n):,} ({100*k/n:.2f}%)" if n else "자료 없음"

overview = pd.DataFrame({
    "실제 TAG": metrics.phone,
    "전체 전사 일치": [counted(k,n) for k,n in zip(metrics.correct,metrics.n)],
    "정상 발음 전사": [counted(k,n) for k,n in zip(metrics.normal_correct,metrics.normal_n)],
    "오발음 그대로 전사": [counted(k,n) for k,n in zip(metrics.error_correct,metrics.error_n)],
    "주요 다른 출력": metrics.top_confusions.fillna("—"),
    "문항 수": metrics['items'], "화자 수": metrics.speakers,
    "표본 상태": metrics.support,
    "모델 사전": metrics.in_model_vocab.map({True:"있음",False:"단일 토큰 없음"}),
})
display(overview.style.hide(axis="index"))
overview.to_csv(ASSETS / "자모별_전사율.csv", index=False, encoding="utf-8-sig")
print("∅ = 해당 기준 자리에 모델 음소 연결 없음/삭제 출력. 실제 소리가 물리적으로 없다는 판정이 아닙니다.")
''')
    code('''shown = metrics[metrics.n > 0].copy()
fig, ax = plt.subplots(figsize=(12, 12), layout="constrained")
y = np.arange(len(shown))
ax.barh(y-.18, shown.normal_accuracy_pct, height=.35, label="정상 발음 전사율", color="#3883b8")
ax.barh(y+.18, shown.error_accuracy_pct, height=.35, label="오발음 그대로 전사율", color="#db7754")
ax.set_yticks(y, [f"{p}   정상 {n:,} / 오류 {e:,}" for p,n,e in zip(shown.phone,shown.normal_n,shown.error_n)])
ax.set_xlim(0,100); ax.invert_yaxis(); ax.grid(axis="x",alpha=.2)
ax.set_xlabel("모델 출력 = 실제 TAG (%)")
ax.set_title("같은 실제 소리라도, 정상일 때와 오발음일 때의 전사 성능은 다릅니다\\n표본 없는 비율은 막대 없음 · 적은 오류 표본의 수치에 주의")
ax.legend(loc="lower right")
display(fig); plt.close(fig)
''')
    md('''<a id="혼동"></a>
## 2. 실제 소리를 무엇으로 잘못 출력했나?

아래 혼동표의 행은 **실제 TAG**, 열은 **모델 출력**입니다. 기대 발음과 모델 출력을 비교하는 표가 아닙니다. 같은 소리를 맞힌 대각선이 높을수록 전사가 잘 된 것입니다.
''')
    code('''confusion = pd.read_csv(ASSETS / "confusion.csv")
active = metrics.loc[metrics.n > 0, "phone"].tolist()
outputs = [p for p in PHONES if p in set(confusion.predicted)] + ["∅"]
matrix = confusion.pivot(index="actual",columns="predicted",values="row_pct").reindex(index=active,columns=outputs).fillna(0)
fig, ax = plt.subplots(figsize=(14,12),layout="constrained")
im = ax.imshow(matrix.values,vmin=0,vmax=100,cmap="Blues",aspect="auto")
ax.set_xticks(range(len(outputs)), outputs); ax.set_yticks(range(len(active)),active)
ax.set_xlabel("모델 출력"); ax.set_ylabel("실제 TAG")
ax.set_title("실제 TAG별 모델 출력 비율 · 행 합계 100%")
fig.colorbar(im,ax=ax,label="비율 (%)")
display(fig); plt.close(fig)
''')
    code('''# 특정 혼동 방향은 기대 LAB → 실제 TAG로 집계합니다.
with gzip.open(ASSETS / "sites.jsonl.gz", "rt", encoding="utf-8") as f:
    sites = pd.DataFrame(json.loads(line) for line in f)
accepted = sites[sites.accepted].copy()
accepted["same"] = accepted.actual.fillna("∅") == accepted.predicted.fillna("∅")
accepted["exact_warning"] = accepted.same & accepted.warning
errors = accepted[accepted.expected != accepted.actual]
pairs = errors.assign(actual=errors.actual.fillna("∅")).groupby(["expected","actual"]).agg(
    n=("same","size"), correct=("same","sum"), exact_warning=("exact_warning","sum"))
pairs = pairs.reset_index()
pairs["전사 일치"] = [counted(k,n) for k,n in zip(pairs.correct,pairs.n)]
pairs["실제음까지 맞힌 경고"] = [counted(k,n) for k,n in zip(pairs.exact_warning,pairs.n)]

def pair_table(group="ㅅㅆㅈㅉㅊ"):
    z = pairs[pairs.expected.isin(list(group)) & pairs.actual.isin(list(group))]
    display(z[["expected","actual","전사 일치","실제음까지 맞힌 경고"]]
            .rename(columns={"expected":"기대 LAB","actual":"실제 TAG"}).style.hide(axis="index"))
pair_table()
''')
    md('''<a id="대립비교"></a>
## 3. 소리가 어떻게 다른지 나란히 듣기

**바로 다음 TAG 음과 위치를 맞춘 사례**를 먼저 보여줍니다. 가능한 경우 화자도 맞췄으며 각 그림 위에 일치 조건을 적었습니다. 다른 문항·화자의 음성을 섞었을 때 그 차이를 자음 고유의 차이라고 단정하면 안 됩니다.

왼쪽은 시간별 파형, 오른쪽은 주파수별 에너지입니다. 초록선은 기존 TAG 구간입니다. 같은 그림 안에서 시간·파형 세로축·스펙트로그램 색 범위를 공유합니다. 길이를 늘이거나 음량을 따로 정규화하지 않았습니다.

자음 뒤 모음의 시작도 보이도록 앞 100ms와 뒤 약 200ms를 함께 남겼습니다. 모음은 같은 방식으로 모음 전체와 주변 소리를 봅니다. **파형 한 장으로 그 글자의 고정된 소리 모양이나 오류 원인을 확정하지 않습니다.**
''')
    has_ss = (assets/'sibilant_context_v1/report.json').exists()
    if has_ss:
        md('''<a id="ㅅㅆ특징"></a>
### 3-1. ㅅ·ㅆ: 앞뒤 소리·위치를 맞추면 어떤 특징이 다른가?

**아래 점들의 높이가 얼마나 겹치는지 먼저 보세요. 파랑=TAG ㅅ, 주황=TAG ㅆ, ×=M3 전사 불일치입니다.**

앞 음소·뒤 모음·초성 위치·어절 처음/내부를 맞춘 집단입니다. 그중 화자까지 같은 쌍도 별도로 비교합니다. 표본 수가 충분한 문맥을 먼저 보여주며, 특징값이나 모델 성공 여부로 문맥·예시를 고르지 않았습니다.

| 측정 | 읽는 방법 |
|---|---|
| TAG 자음 구간 길이 | 원래 주석의 길이. 마찰 시간과 같다고 보지 않습니다. |
| 고주파 우세 연속 구간 | 저주파보다 고주파 에너지가 큰 구간의 자동 후보. 마찰·기식 경계를 확정한 값은 아닙니다. |
| 자음 중앙의 주파수 무게중심 | 고주파 에너지가 어느 주파수에 모이는지. 1~7.9kHz의 고정 범위에서 측정합니다. |
| 뒤 모음 크기 증가 | 모음 처음 0~30ms보다 60~90ms에서 얼마나 커졌는지. 전체 녹음 음량의 영향을 일부 줄이는 상대값입니다. |
| 뒤 모음 주기성 | 반복되는 파형이 얼마나 뚜렷한지의 자동 지표. 독립 검증한 성대 진동·F0는 아닙니다. |

측정 실패는 빈칸으로 남깁니다. **음향 특징의 분포가 다르다는 것과 발음 오류를 낮은 오탐으로 잡는다는 것은 별도입니다.** TAG로 구간·문맥을 고른 진단용 그림이며, 서비스 판별기 성능이나 원인 검증 결과가 아닙니다.

설계 참고: [Chang의 한국어 마찰음 연구](https://cbchang.com/wp-content/uploads/Chang2008_HSIKL12.pdf)는 마찰·기식과 뒤 모음의 여러 단서를 분리해 수동 측정했습니다. [Schertz & Kang의 지각 실험](https://benjamins.com/catalog/kl.16001.sch)도 복수의 시간·주파수 단서를 다룹니다. 이번 자동 대리값은 해당 수동 측정을 재현한 것이 아닙니다. 독립 청취, 확정 기식 시간, F0/F1·음질 측정은 하지 않았습니다.
''')
        code('''# SS_SETUP
from mdd.sibilant_atlas import SibilantAtlas, context_text
ss_atlas = SibilantAtlas(ASSETS / "sibilant_context_v1")
ss_report = ss_atlas.report
ss_context = ss_report["default_context"]
display(HTML("<h3>ㅅ·ㅆ: 비교 조건과 자료 수</h3>"))
print(f"TAG ㅅ·ㅆ 전체 {ss_report['source_sites']:,}자리 → 시간 대응 확인 {ss_report['timed_cases']:,}자리")
print(f"시간 대응 미확정 {ss_report['excluded']}자리: 전체 전사율 표에는 그대로 유지, 음향 비교에서는 별도 기록")
print("기본 비교:", context_text(ss_context))
display(ss_atlas.support_table().head(12).style.hide(axis="index"))
print("전체 문맥 목록: ss_atlas.support_table() / 모든 특징값: phone_atlas_assets/sibilant_context_v1/features.csv")
''')
        code('''# SS_DISTRIBUTION
display(HTML("<h3>① 같은 문맥의 여러 녹음: 특징 분포와 전사 실패</h3><p>검정선은 중앙값과 가운데 50% 범위입니다. 두 색이 겹치는 영역에도 ×가 있는지 보세요. 정상·오발음 표본 수는 그림 오른쪽에 있습니다.</p>"))
fig = ss_atlas.plot_distribution(ss_context)
display(fig); plt.close(fig)
display(ss_atlas.feature_table(ss_context).style.hide(axis="index"))
''')
        code('''# SS_ERRORS
display(HTML("<h3>② 같은 문맥에서 LAB≠TAG인 오발음만 보기</h3><p>위 전체 분포는 정상 사례가 많습니다. 아래는 실제 발음 오류가 있을 때도 TAG를 그대로 전사했는지 보는 부분집합입니다. 적은 표본의 점 몇 개를 일반화하지 마세요.</p>"))
if ss_atlas.subset(ss_context, normal=False):
    fig = ss_atlas.plot_distribution(ss_context, normal=False)
    display(fig); plt.close(fig)
    error_cases = ss_atlas.subset(ss_context, normal=False)
    error_table = pd.DataFrame([{
        "문장": r["prompt"], "기대 LAB": r["expected"], "실제 TAG": r["actual"],
        "M3 출력": r["predicted"] or "∅", "화면 경고": "있음" if r["warning"] else "없음",
        "고주파 우세(ms)": r["features"]["high_run_ms"],
        "모음 크기 증가(dB)": r["features"]["vowel_rise_db"]
    } for r in error_cases])
    display(error_table.round(2).style.hide(axis="index"))
    if len(error_cases) <= 10:
        display(HTML(ss_atlas.example_audio([r["case_id"] for r in error_cases])))
else:
    print("이 문맥에는 확인 가능한 오발음 표본이 없습니다.")
''')
        code('''# SS_PAIRS
display(HTML("<h3>③ 화자까지 같은 쌍</h3><p>앞뒤 소리·위치가 같은 동일 화자에서 ㅅ 한 개와 ㅆ 한 개를 고정 해시로 골랐습니다. 음량·문항·발화 속도까지 통제한 녹음 실험은 아닙니다.</p>"))
fig = ss_atlas.plot_pairs(ss_context)
display(fig); plt.close(fig)
''')
        code('''# SS_EXAMPLES
display(HTML("<h3>④ 실제 파형에서 측정한 위치 확인</h3><p>녹색은 TAG 자음, 파란 음영은 뒤 모음의 두 측정 창입니다. 주황·보라 막대는 고정 규칙으로 잡은 후보입니다. 주황 막대 끝부터 보라 막대 시작까지를 기식 시간으로 읽으면 안 됩니다.</p>"))
fig = ss_atlas.plot_examples(ss_report["example_ids"])
display(fig); plt.close(fig)
display(HTML(ss_atlas.example_audio(ss_report["example_ids"])))
''')
        code('''# SS_OTHER_CONTEXTS
display(HTML("<h3>⑤ 다른 문맥에서도 같은 모양인가?</h3><p>아래 두 문맥도 양쪽 TAG 표본 수로 선정했습니다. 서로 다른 문맥을 합쳐 하나의 분리도나 정확도로 계산하지 않았습니다.</p>"))
other_contexts = [r for r in ss_report["contexts"]
                  if min(r["s"], r["ss"]) >= 10 and r["context"] != ss_context
                  and r["context"][2] == "초성 추정" and r["context"][3] != "어절 미확정"
                  and "<불명/틈>" not in r["context"]][:2]
for context in other_contexts:
    fig = ss_atlas.plot_distribution(context["context"])
    display(fig); plt.close(fig)
# 다른 문맥도 직접 선택할 수 있습니다.
# fig = ss_atlas.plot_distribution(ss_report["contexts"][3]["context"])
''')
    for group in GROUPS:
        code('atlas.show_group('+repr(group)+')')
    md('''<a id="선택"></a>
## 4. 원하는 글자·사례 선택

위젯이 있으면 선택 메뉴를 쓰세요. 없어도 바로 아래 코드의 글자만 바꾸면 됩니다. 함께 저장한 **`한국어_자모별_음성_전사율.html`**은 커널 없이 선택 메뉴로 열어볼 수 있습니다.
''')
    code('atlas.widgets()')
    code('''# 원하는 글자와 조건으로 수정하세요. mode: 전체 / 정상 / 오발음
atlas.show_phone("ㅅ", n=2, mode="전체", full_audio=True)
# 예: atlas.show_phone("ㅋ", n=3, mode="오발음")
# 예: atlas.show_phone("ㅏ", n=2)
# 예: pair_table("ㄱㄲㅋ")
''')
    md('''<a id="글자사례"></a>
## 5. 모든 자모의 저장된 예시

각 글자의 대표 사례 1개를 저장했습니다. 더 보려면 위의 `show_phone`에서 `n`을 늘리세요. **예시 선정은 성능 계산과 별개**입니다. 정상/오발음 × 전사 성공/실패 유형에서 고정 해시로 뽑았습니다.

주 평가에 없는 ㅒ·ㅞ·일부 겹받침은 다른 테스트 녹음의 예시만 제공합니다. 그 사례의 M3 출력은 **미측정**이며 전사율 표에 합치지 않습니다. 겹받침 라벨을 독립된 단일 표면음으로 해석하지 마세요.

`ㅇ`도 원본 TAG의 표기를 따릅니다. 철자의 소리 없는 초성 `ㅇ`에 별도 소리가 있다고 가정해서 파형을 만든 것은 아닙니다.
''')
    for phone in PHONES:
        md('### '+phone)
        code('atlas.show_phone('+repr(phone)+', n=1, full_audio=True)')
    md('''<a id="범위"></a>
## 6. 무엇을 계산했고 무엇이 빠져 있나?

| 구분 | 해석 |
|---|---|
| 이 노트북의 일치율 | 입력 통과 + 기준음 자리 대응 가능 조건에서 실제 TAG와 저장 음소 출력의 일치 |
| 실제 삭제 | 발음된 자모와 분리. 아래 표에서 확인 |
| 입력 거절 3,248녹음 | 주 비율 밖. 원시 저장 출력 집계는 `per_phone_all_inputs.csv`에 별도 보존 |
| 삽입·대응 불가 | 원래 캐시에 전체 인식열/모든 gap이 없어 이 비율로 전체 전사 정확도나 PER를 주장하지 않음 |
| 라벨·시간 | 원본 TAG/LAB와 선정 사례를 대조했으나, 독립 청취 정답 또는 물리적 경계로 확정한 것은 아님 |
| 일반화 | 학습 문항의 선택 평가 표본. 새 문장/실사용 전체를 대표하는 수치 아님 |

전체 표는 고정된 한 편집 정렬 경로의 저장 출력을 집계합니다. 그림은 그중 TAG 시간 대응이 유일한 사례만 사용하므로, 그림의 수가 성능 분모가 아닙니다. 관측 수가 적은 자모의 0%·100%는 확정적인 모집단 성능이 아닙니다.
''')
    code('''d = report["deletion"]
display(pd.DataFrame([{
    "실제 TAG": "삭제 (∅)", "자리 수":d["n"],
    "삭제로 출력":counted(d["correct"],d["n"]),
    "정확한 삭제 경고":counted(d["exact_warnings"],d["n"])
}]).style.hide(axis="index"))
display(pd.DataFrame(report["membership"]).T)
print("주 평가에서 확인한 실제 자모:", report["evaluated_actual_phones"])
print("주 평가 표본 없음:", " ".join(report["missing_evaluation_phones"]))
print("모델 사전에 단일 토큰 없음:", " ".join(report["outside_model_vocab"]))
print("원음·원본 주석·입력 파일 hash 및 검사:", ASSETS / "verification.json")
''')
    nb.cells=cells
    cells[1].source=cells[1].source.replace(
        'ROOT / "tasks/pronunciation-assessment/notebooks/phone_atlas_assets"',
        'ROOT / '+repr(str(relative_assets)))
    for i,cell in enumerate(cells):
        if cell.cell_type=='code':compile(cell.source,'notebook_cell_%d' % i,'exec')
    path=folder/'한국어_자모별_음성_전사율.ipynb'
    nbformat.write(nb,path)
    print('Executing notebook:',path,flush=True)
    NotebookClient(nb,timeout=600,kernel_name=a.kernel,resources={'metadata':{'path':str(folder)}}).execute()
    nbformat.validate(nb);nbformat.write(nb,path)
    print('Executed cells:',sum(c.cell_type=='code' for c in nb.cells),flush=True)

    # Standalone HTML uses the executed outputs, including embedded WAV and PNG.
    # It does not publish files or expose a server. All 51 phone sections exist.
    def outputs(cell):
        result=[]
        for o in cell.get('outputs',[]):
            data=o.get('data',{})
            if 'image/png' in data:
                result.append('<img alt="음성 파형과 스펙트로그램" src="data:image/png;base64,'+data['image/png']+'">')
            elif 'text/html' in data:result.append(data['text/html'])
            elif o.get('output_type')=='stream':result.append('<pre>'+html.escape(o['text'])+'</pre>')
            elif 'text/plain' in data and not re.search(r'<.*(Figure|VBox)',data['text/plain']):
                result.append('<pre>'+html.escape(data['text/plain'])+'</pre>')
        return '\n'.join(result)
    overview_html=outputs(next(c for c in nb.cells if c.cell_type=='code' and 'overview = pd.DataFrame' in c.source))
    chart_html=outputs(next(c for c in nb.cells if c.cell_type=='code' and 'ax.barh' in c.source))
    phone_sections=[];group_sections=[];ss_sections=[]
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        if cell.source.startswith('atlas.show_phone(') and ', n=1, full_audio=True)' in cell.source:
            phone=re.search(r"atlas.show_phone\('([^']+)'",cell.source).group(1)
            phone_sections.append('<section class="phone" data-phone="'+phone+'">'+outputs(cell)+'</section>')
        if cell.source.startswith('atlas.show_group('):
            group=re.search(r"atlas.show_group\('([^']+)'",cell.source).group(1)
            group_sections.append('<details><summary>'+group+' 비교</summary>'+outputs(cell)+'</details>')
        if cell.source.startswith('# SS_'):
            body=outputs(cell)
            if not cell.source.startswith('# SS_DISTRIBUTION'):
                title={'SS_SETUP':'자료 수와 문맥 목록','SS_ERRORS':'오발음만 따로 보기','SS_PAIRS':'같은 화자 쌍',
                       'SS_EXAMPLES':'파형의 측정 위치와 음성','SS_OTHER_CONTEXTS':'다른 문맥 두 가지'}[cell.source.splitlines()[0][2:]]
                body='<details><summary>'+title+'</summary>'+body+'</details>'
            ss_sections.append(body)
    ss_html=''
    if ss_sections:
        ss_html='''<h2 id="ss-features">ㅅ·ㅆ: 조건을 맞춘 음향 특징</h2>
<p class="note"><b>파랑=TAG ㅅ · 주황=TAG ㅆ · ×=M3 전사 불일치.</b><br>
앞 음소·뒤 모음·초성 위치·어절 처음/내부가 같은 집단입니다. 화자까지 맞춘 쌍은 아래에서 펼치세요.<br>
TAG 길이는 원래 주석의 길이입니다. 고주파·주기성은 자동 대리값이며, 마찰·기식의 확정 경계나 F0가 아닙니다.
뒤 모음 크기 증가는 0~30ms 대비 60~90ms의 RMS 변화입니다. 측정 실패는 0으로 채우지 않았습니다.</p>
<p>목표는 특징의 겹침과 모델 실패 위치를 보는 것입니다. TAG 구간을 사용한 탐색이며 모델의 정보 부재·원인을 입증하거나 새 문장 성능을 평가한 결과가 아닙니다.
설계 참고: <a href="https://cbchang.com/wp-content/uploads/Chang2008_HSIKL12.pdf">Chang의 한국어 마찰음 연구</a>,
<a href="https://benjamins.com/catalog/kl.16001.sch">Schertz &amp; Kang의 지각 실험</a>. 해당 연구의 수동 측정을 재현한 것은 아닙니다.</p>'''+''.join(ss_sections)
    html_text='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>한국어 자모별 음성·M3 전사율</title><style>
body{font-family:system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 20px;line-height:1.6;color:#172638}
h1{font-size:26px}h2{margin-top:32px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px;width:100%}
th,td{padding:7px 9px;border-bottom:1px solid #d9e0e8;text-align:left}th{background:#eaf0f6;position:sticky;top:0}tr:hover{background:#f1f7fc}
.note{background:#eff5fb;border-left:4px solid #3883b8;padding:14px}select{font-size:22px;padding:8px;min-width:160px}
audio{width:340px;max-width:100%;vertical-align:middle}details{border:1px solid #d9e0e8;border-radius:8px;padding:12px;margin:10px 0}summary{font-weight:bold;cursor:pointer}
.phone{display:none}.phone.active{display:block}.table-wrap{overflow:auto;max-height:650px}pre{white-space:pre-wrap}
</style><h1>한국어 자모별 음성 · M3 전사율</h1>
<p class="note"><b>실제 TAG를 얼마나 그대로 출력했는지 확인하세요.</b><br>
M3 저장 평가: 6,720녹음 중 입력 통과 3,472개, 기준음 대응 69,985자리. 학습 문항의 별도 평가 녹음입니다.<br>
정상과 오발음을 분리합니다. 경고 없음은 정답 판정이 아닙니다. TAG는 독립 재청취 정답이 아닙니다.</p>
'''+('<p><a href="#ss-features"><b>새로 추가한 ㅅ·ㅆ 음향 특징 비교로 이동 ↓</b></a></p>' if ss_sections else '')+'''
<h2>1. 전체 자모 전사율</h2><div class="table-wrap">'''+overview_html+'''</div>
<details><summary>정상 / 오발음 전사율 그래프</summary>'''+chart_html+'''</details>
'''+ss_html+'''
<h2>2. 소리를 나란히 비교</h2><p>다음 TAG 음·위치를 맞춘 사례입니다. 화자 일치 여부는 각 그림 위에 표시됩니다.</p>'''+''.join(group_sections)+'''
<h2>3. 글자를 선택해서 듣기</h2><label for="phone">실제 TAG </label><select id="phone">'''+''.join('<option>'+p+'</option>' for p in PHONES)+'''</select>
<p>파형과 음성은 원래 음량·길이를 유지합니다. 초록선은 기존 TAG 시간입니다. 추가 음성의 미측정 출력은 전사율에 합치지 않습니다.</p>'''+''.join(phone_sections)+'''
<h2>집계 범위</h2><p>입력 거절 3,248녹음과 기준음에 대응하지 못한 자리는 주 비율 밖입니다. 삽입 gap을 포함한 전체 PER가 아닙니다.
실제 삭제 455자리는 자모 전사율과 별도이며, 삭제 출력 204건 / 정확 삭제 경고 95건입니다.
표본 없는 글자는 0%로 채우지 않았습니다. 더 많은 사례와 원시 표는 같은 이름의 ipynb 및 phone_atlas_assets 폴더에 있습니다.</p>
<script>const select=document.getElementById('phone');function show(){document.querySelectorAll('.phone').forEach(s=>s.classList.toggle('active',s.dataset.phone===select.value));document.querySelectorAll('audio').forEach(a=>a.pause());}select.value='ㅅ';select.addEventListener('change',show);show();</script></html>'''
    hp=path.with_suffix('.html');hp.write_text(html_text,encoding='utf-8')
    stats=dict(notebook=str(path),html=str(hp),executed_code_cells=sum(c.cell_type=='code' for c in nb.cells),
        error_outputs=sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),
        displayed_phone_sections=len(phone_sections),comparison_group_sections=len(group_sections),
        sibilant_feature_sections=len(ss_sections),
        notebook_bytes=path.stat().st_size,html_bytes=hp.stat().st_size,
        outputs_sha256={str(p):sha(p) for p in (path,hp,assets/'자모별_전사율.csv')},source_sha256=sha(__file__),
        renderer_sha256=sha(ROOT/'tasks/pronunciation-assessment/src/mdd/phone_atlas.py'))
    if has_ss:
        stats['sibilant_renderer_sha256']=sha(ROOT/'tasks/pronunciation-assessment/src/mdd/sibilant_atlas.py')
        stats['sibilant_report_sha256']=sha(assets/'sibilant_context_v1/report.json')
    assert stats['error_outputs']==0 and len(phone_sections)==51 and len(group_sections)==len(GROUPS)
    (assets/'notebook_execution.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
