"""Make a small, executed companion notebook and self-contained rank table."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse
import html
import json
import nbformat
from nbclient import NotebookClient
from mdd.phone_atlas import sha
from mdd.phone_probability_atlas import ProbabilityAtlas


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--folder',type=Path,required=True)
    folder=ap.parse_args().folder.resolve();atlas=ProbabilityAtlas(folder);r=atlas.report
    assert json.loads((folder/'verification.json').read_text())['status']=='passed'
    output=folder.parent.parent/'미전사_자모_확률순위.ipynb'
    total_ranked=sum(s['ranked'] for s in r['summary'])
    top2=sum(s['rank2'] for s in r['summary']);top3=sum(s['rank_le3'] for s in r['summary'])
    intro=f'''# 실제음을 놓친 자모: M3 확률 순위

**미전사 {r['target_sites']}자리 중 {total_ranked}자리의 후보 순위를 확인했습니다.**
실제 TAG가 2위인 자리는 **{top2}/{total_ranked}**, 3위 이내는 **{top3}/{total_ranked}**입니다.
삭제 출력 16자리는 출력 음소의 프레임이 없어 순위를 만들지 않았습니다.

대상은 기존 자모별 평가에서 **오발음 표본 20개 이상·오발음 전사율 40% 미만**인 16개 실제 자모입니다.
그중 **기대 LAB≠실제 TAG이고, 모델 출력≠실제 TAG**인 자리를 전부 비교했습니다.
정상 발음을 잘못 지적한 사례 또는 전사는 맞았으나 표시만 유보한 사례의 집계는 아닙니다.

**표 읽는 법:** 기대 LAB은 말해야 할 소리, 실제 TAG는 원래 데이터에 등록된 소리입니다.
1~3위는 해당 출력 음소가 가장 높은 확률을 보인 **한 프레임**에서의 자모 후보입니다.
실제 TAG를 보고 유리한 프레임을 고르지 않았습니다. 46자모 안의 순위이며, 확률은 blank 등을 포함한 원래 50개 출력 확률입니다.
따라서 표시된 세 확률을 더해도 100%가 아닙니다. **발음 정답률이나 보정된 정확도 확률이 아닙니다.**

M3 `w2vbert_synth_tense_v2/step_3000`으로 {r['recordings']}녹음을 다시 추론해
기존의 전체 대응 출력·인식 확률을 모두 재현했습니다. 새 학습은 0회입니다.
학습 문항의 별도 평가 녹음이며 새 문장 성적이 아닙니다. TAG 독립 청취는 하지 않았습니다.
'''
    nb=nbformat.v4.new_notebook()
    nb.metadata=dict(kernelspec=dict(display_name='Python (qwen35)',language='python',name='qwen35'))
    md=lambda s:nb.cells.append(nbformat.v4.new_markdown_cell(s))
    code=lambda s:nb.cells.append(nbformat.v4.new_code_cell(s))
    md(intro)
    code('''from pathlib import Path
import sys
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())
sys.path.insert(0, str(ROOT/'tasks/pronunciation-assessment/src'))
from mdd.phone_probability_atlas import ProbabilityAtlas
ranks = ProbabilityAtlas(ROOT/''' + repr(str(folder.relative_to(ROOT))) + ''')
display(ranks.summary_table().style.hide(axis='index'))
''')
    md('''## 자모별 실제 사례 — 하나씩 나란히 보기

자모별 사례는 UID·자리의 고정 hash 순서로 골랐습니다. 확률 차이가 큰 사례나 작은 사례만 고르지 않았습니다.
**경고 없음은 정상 확정이 아닙니다.** 이번 사례는 모두 TAG와 출력이 다른 전사 실패이며, 잘못된 다른 소리로 경고한 경우도 포함합니다.
''')
    code("display(ranks.case_table(examples=True).style.hide(axis='index'))")
    md('## 앞서 음향 특징을 비교한 ㅅ·ㅆ 오발음 3자리')
    code("display(ranks.case_table(known_ss=True).style.hide(axis='index'))")
    md('''## 자모를 바꿔서 전체 사례 보기

아래 `"ㅆ"`을 `"ㅅ"`, `"ㅈ"`, `"ㅉ"`, `"ㅊ"`, `"ㅋ"` 등으로 바꾸면 해당 실제 TAG의 미전사 사례를 볼 수 있습니다.
`ranks.case_table()`은 순위를 확인한 538자리 전부입니다.
삭제 출력을 포함한 554자리는 [전체 CSV](phone_atlas_assets/probability_ranks_v1/사례별_확률순위.csv)에 있습니다.

실제음이 2위라는 사실만으로 2위를 선택하면 좋아진다고 결론내릴 수 없습니다.
정상 발음에서도 같은 후보가 2위일 수 있으므로, 변경 효과는 정상 헛지적과 함께 따로 평가해야 합니다.
''')
    code('display(ranks.case_table("ㅆ").style.hide(axis="index"))')
    NotebookClient(nb,timeout=180,kernel_name='qwen35',resources={'metadata':{'path':str(ROOT)}}).execute()
    assert not [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    nbformat.write(nb,output)
    def table(df):return '<div class="scroll">'+df.to_html(index=False,escape=True)+'</div>'
    intro_html='''<p class="note"><b>미전사 554자리 중 538자리의 후보 순위를 확인했습니다.</b><br>
실제 TAG가 2위: %d/538 · 3위 이내: %d/538. 삭제 출력 16자리는 순위 없음.</p>
<p>실제 오발음 표본 20개 이상·전사율 40%% 미만인 16개 자모를 대상으로, TAG와 출력이 다른 사례 전부를 확인했습니다.
표의 순위는 <b>해당 출력 음소의 확률이 가장 높았던 한 프레임</b>에서의 46자모 순위입니다.
확률은 blank 등을 포함한 원래 50개 출력의 확률이며, 발음 정답 확률이 아닙니다.</p>
<p>M3 step_3000 · 420녹음 재추론 · 기존 전체 대응 출력 및 반올림 확률 420/420 일치 · 새 학습 0회.<br>
학습 문항의 별도 평가 녹음이며 새 문장 성적이 아닙니다. TAG 독립 청취는 하지 않았습니다.</p>'''%(top2,top3)
    sections=[]
    for s in r['summary']:
        p=s['phone']
        sections.append('<section class="phone" data-phone="'+p+'"><h3>실제 TAG '+p+'</h3>'+table(atlas.case_table(p))+'</section>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>미전사 자모 확률 순위</title><style>
body{font-family:system-ui,sans-serif;color:#172638;max-width:1500px;margin:24px auto;padding:0 20px;line-height:1.65}
table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border-bottom:1px solid #d4dfea;text-align:left}
th{background:#eef4fa;white-space:nowrap;position:sticky;top:0}tr:hover{background:#eff7ff}
.scroll{overflow:auto;max-height:650px}.note{background:#edf5ff;border-left:4px solid #3182ce;padding:14px}
select{font-size:22px;padding:8px}.phone{display:none}.phone.active{display:block}h2{margin-top:32px}
</style><h1>실제음을 놓친 자모: M3 확률 순위</h1>'''+intro_html+'''<h2>자모별 집계</h2>'''+table(atlas.summary_table())+'''
<h2>자모별 실제 사례 하나씩</h2><p>UID·자리의 고정 hash로 선정했습니다. 기대 LAB / 실제 TAG / 모델 출력 / 확률 후보를 나란히 보세요.
경고 없음은 정상 확정이 아닙니다.</p>'''+table(atlas.case_table(examples=True))+'''
<h2>앞서 비교한 ㅅ·ㅆ 오발음 3자리</h2>'''+table(atlas.case_table(known_ss=True))+'''
<h2>글자를 선택해 모든 미전사 사례 보기</h2><label for="phone">실제 TAG </label><select id="phone">'''+''.join('<option>'+s['phone']+'</option>' for s in r['summary'])+'''</select>'''+''.join(sections)+'''
<p>실제음이 2위에 있다고 2위를 선택하면 개선되는 것은 아닙니다. 정상 발음에도 같은 후보가 2위일 수 있어 정상 헛지적을 함께 평가해야 합니다.</p>
<p><a href="phone_atlas_assets/probability_ranks_v1/사례별_확률순위.csv">전체 554자리 CSV</a> · <a href="미전사_자모_확률순위.ipynb">실행된 노트북</a></p>
<script>const sel=document.getElementById('phone');function show(){document.querySelectorAll('.phone').forEach(s=>s.classList.toggle('active',s.dataset.phone===sel.value));}sel.addEventListener('change',show);show();</script></html>'''
    hp=output.with_suffix('.html');hp.write_text(page,encoding='utf-8')
    receipt=dict(notebook=str(output),html=str(hp),executed_cells=4,error_outputs=0,
        displayed_ranked_sites=sum(len(atlas.case_table(s['phone'])) for s in r['summary']),
        rank2=top2,rank_le3=top3,ranked=total_ranked,
        output_sha256={p.name:sha(p) for p in (output,hp)},source_sha256=sha(__file__),
        renderer_sha256=sha(ROOT/'tasks/pronunciation-assessment/src/mdd/phone_probability_atlas.py'))
    assert receipt['displayed_ranked_sites']==538
    (folder/'display_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
