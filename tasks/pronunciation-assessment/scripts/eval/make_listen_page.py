# -*- coding: utf-8 -*-
"""샘플 청취 페이지 — 오디오를 들으면서 표준발음 · 사람 전사 · 모델 출력을 나란히 본다.

입력: `<dir>/samples.json` (필드: stem prompt lang task sec split group canon tag final old *_ph d_final d_old)
      `<dir>/wav/<stem>.wav`
출력: `<dir>/index.html` (외부 의존 없음 — 같은 폴더에서 로컬 서버로 열면 된다)

    $PY .../make_listen_page.py --dir $MDD_OUTPUTS/experiments/final_ctc/listen
    cd <dir> && $PY -m http.server 8930        # 브라우저에서 localhost:8930

⚠️ AIHub 오디오가 들어 있으므로 외부에 공개하지 말 것(로컬·사내 확인용).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, html, json, os, time

CSS = """
:root{--paper:#F2F5F3;--surface:#fff;--ink:#0E1614;--muted:#5C6B66;--rule:#DCE4DF;--rule2:#EDF1EE;
 --canon:#0B6E62;--actual:#C2451E;--model:#2B4B92;--hit:#0B6E62;--accent:#0B6E62;
 --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--shadow:0 1px 2px rgba(16,32,28,.05),0 8px 24px rgba(16,32,28,.06)}
@media (prefers-color-scheme:dark){:root{--paper:#0D1412;--surface:#141C1A;--ink:#E9EFEB;--muted:#97A8A2;
 --rule:#27342F;--rule2:#1C2622;--canon:#5AD2C0;--actual:#FF9068;--model:#9FB8FF;--hit:#5AD2C0;--accent:#5AD2C0;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35)}}
*{box-sizing:border-box}
body{margin:0;padding:0 0 80px;background:var(--paper);color:var(--ink);
 font:17px/1.6 -apple-system,"Apple SD Gothic Neo","Pretendard","Noto Sans KR",Segoe UI,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px;display:flex;flex-direction:column;gap:20px}
header{padding:40px 0 4px}
h1{font-size:34px;line-height:1.2;margin:0 0 10px;letter-spacing:-.03em;font-weight:700}
.sub{color:var(--muted);font-size:15px}
header .sub{max-width:72ch}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:14px;font-size:14px;color:var(--muted)}
.legend b{font-weight:600}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:middle}

/* 도구막대 */
.toolbar{position:sticky;top:0;z-index:5;margin:0 -20px;padding:12px 20px;
 background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(10px);
 border-bottom:1px solid var(--rule)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:14px;max-width:1080px;margin:0 auto}
.controls .sub{font-size:13px;letter-spacing:.02em}
button{font:inherit;font-size:14px;padding:7px 14px;border-radius:99px;border:1px solid var(--rule);
 background:var(--surface);color:var(--ink);cursor:pointer;transition:.15s border-color,.15s color,.15s background}
button:hover{border-color:var(--muted)}
button[aria-pressed=true]{border-color:transparent;background:var(--accent);color:var(--surface);font-weight:600}

/* 카드 */
.card{background:var(--surface);border:1px solid var(--rule);border-radius:16px;padding:20px 22px;
 display:grid;grid-template-columns:288px 1fr;gap:26px;box-shadow:var(--shadow)}
.card .left{display:flex;flex-direction:column;gap:10px;min-width:0}
.prompt{font-size:20px;font-weight:650;line-height:1.35;word-break:keep-all;letter-spacing:-.01em}
.meta{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center}
audio{width:100%;height:36px}

/* 발음열 */
.rows{display:flex;flex-direction:column;gap:2px;min-width:0}
.row{display:grid;grid-template-columns:96px 1fr;gap:14px;align-items:baseline;
 padding:10px 0;border-top:1px solid var(--rule2)}
.rows .row:first-child{border-top:0;padding-top:0}
.tag{font-size:11px;letter-spacing:.09em;text-transform:uppercase;font-weight:700;padding-top:5px}
.t-canon{color:var(--canon)}.t-actual{color:var(--actual)}.t-model{color:var(--model)}.t-old{color:var(--muted)}
.seq{font-family:var(--mono);font-size:17px;word-break:break-all;line-height:1.7}
.seq .d{color:var(--actual);font-weight:700}
.seq .m{color:var(--hit);font-weight:700}
.seq.both{line-height:1.25}
.seq.both .tok{display:inline-block;text-align:center;min-width:26px;margin:0 3px 8px 0;vertical-align:top}
.seq.both .tok .j{display:block;font-size:21px;line-height:1.25}
.seq.both .tok .p{display:block;font-size:12px;color:var(--muted);letter-spacing:-.01em}
.seq.both .tok.d{background:color-mix(in srgb,var(--actual) 12%,transparent);border-radius:6px}
.seq.both .tok.d .j,.seq.both .tok.d .p{color:var(--actual);font-weight:700}

.badge{display:inline-block;font-size:12.5px;padding:3px 10px;border-radius:99px;border:1px solid var(--rule);color:var(--muted)}
.badge.err{border-color:var(--actual);color:var(--actual)}
.badge.ok{border-color:var(--hit);color:var(--hit)}
.hide{display:none}

/* 마이크 패널 */
.card.mic{border-color:color-mix(in srgb,var(--model) 45%,var(--rule));
 background:linear-gradient(180deg,color-mix(in srgb,var(--model) 6%,var(--surface)),var(--surface) 120px)}
.card.mic input,.card.mic select{font:inherit;font-size:16px;padding:10px 12px;border:1px solid var(--rule);
 border-radius:10px;background:var(--surface);color:var(--ink);width:100%}
.card.mic input:focus,.card.mic select:focus{outline:2px solid color-mix(in srgb,var(--model) 40%,transparent);outline-offset:1px}
.card.mic label.sub{font-size:12.5px;letter-spacing:.02em;margin-bottom:-6px}
.card.mic .controls{gap:10px;margin-top:4px}
#rec{background:var(--model);border-color:transparent;color:#fff;font-weight:600;font-size:15px;padding:9px 18px}
#rec[data-on=true]{background:var(--actual)}

/* 자리별 칩 */
.site{display:inline-flex;align-items:baseline;gap:3px;margin:0 7px 8px 0;padding:5px 11px;border-radius:9px;
 border:1px solid var(--rule);font-family:var(--mono);font-size:16px;background:var(--surface)}
.site .p{font-size:11.5px;color:var(--muted);font-weight:400}
.site.ok{color:var(--muted);border-color:var(--rule2)}
.site.diff{border-color:var(--actual);color:var(--actual);background:color-mix(in srgb,var(--actual) 8%,var(--surface))}
.site.del{border-color:var(--model);color:var(--model);background:color-mix(in srgb,var(--model) 8%,var(--surface))}

/* 피드백 */
.fb{display:flex;flex-direction:column;gap:10px;font-size:15px;font-family:inherit;line-height:1.6}
.fb .sum{color:var(--muted);font-size:14px}
.fb .item{border-left:3px solid var(--rule);padding:6px 0 6px 12px}
.fb .item.typ{border-left-color:var(--actual)}
.fb .item b{font-family:var(--mono);font-size:16px}
.fb .w{font-size:12.5px;color:var(--muted);margin-left:8px}
.fb .fix{margin-top:4px}

footer{color:var(--muted);font-size:13.5px;padding-top:6px}
@media (max-width:820px){
 h1{font-size:27px} body{font-size:16px}
 .card{grid-template-columns:1fr;gap:16px;padding:16px}
 .row{grid-template-columns:78px 1fr;gap:10px}
}
"""

JS = """
const st={mode:'both',filter:'all'};   // 기본은 자모+음소기호 병기
function render(){
  // data-mode 가 붙은 샘플 줄만 전환 대상이다 — 마이크 결과 줄(.seq[data-live])은 항상 보인다
  document.querySelectorAll('.seq[data-mode]').forEach(e=>{e.classList.toggle('hide', e.dataset.mode!==st.mode)});
  document.querySelectorAll('.card').forEach(c=>{
    const ok = st.filter==='all' || c.dataset.group===st.filter || c.dataset.split===st.filter;
    c.classList.toggle('hide', !ok);
  });
  document.querySelectorAll('[data-mode-btn]').forEach(b=>b.setAttribute('aria-pressed', b.dataset.modeBtn===st.mode));
  document.querySelectorAll('[data-filter-btn]').forEach(b=>b.setAttribute('aria-pressed', b.dataset.filterBtn===st.filter));
}
document.addEventListener('click',e=>{
  const m=e.target.closest('[data-mode-btn]'), f=e.target.closest('[data-filter-btn]');
  if(m){st.mode=m.dataset.modeBtn;render()} if(f){st.filter=f.dataset.filterBtn;render()}
});
render();
"""

MIC_JS = """
const $=id=>document.getElementById(id); let mr=null, chunks=[], stream=null, last=null;
function setStatus(t){ $('status').textContent=t; }
function seq(arr){ return (arr&&arr.length)? arr.join(st.mode==='phone'?' ':'') : '—'; }
function seqBoth(j,p,host){                 // 자모 위 · 음소기호 아래
  host.classList.add('both'); host.innerHTML='';
  if(!j||!j.length){ host.textContent='—'; return; }
  j.forEach((x,k)=>{ const e=document.createElement('span'); e.className='tok';
    e.innerHTML='<span class="j">'+esc(x)+'</span><span class="p">'+esc((p&&p[k])||'')+'</span>';
    host.appendChild(e); });
}
function modelRows(names){                 // 모델마다 [전사 · 자리별] 두 줄
  const host=$('m-models'); host.innerHTML='';
  names.forEach((n,i)=>{
    const cls = i===0? 't-model':'t-old';
    host.insertAdjacentHTML('beforeend',
      `<div class="row"><div class="tag ${cls}">${esc(n)}</div><div class="seq" data-live id="m-pred-${i}">—</div></div>`+
      `<div class="row"><div class="tag">${esc(n)} 자리별</div><div class="seq" data-live id="m-sites-${i}">녹음하면 자리마다 판정한다</div></div>`);
  });
}
function renderSites(sites, host, canonLen){
  host.innerHTML='';
  if(!sites || !sites.length){ host.textContent = canonLen? '—' : '읽을 문장을 적으면 자리별로 비교한다'; return; }
  sites.forEach(s=>{
    const el=document.createElement('span');
    const ph = st.mode==='phone';
    const c = (ph && s.canon_ph)? s.canon_ph : s.canon;
    const p = s.pred===null? '∅' : ((ph && s.pred_ph)? s.pred_ph : s.pred);
    el.className='site '+(s.verdict==='일치'?'ok':(s.verdict==='삭제'?'del':'diff'));
    if(st.mode==='both'){
      const t=(a,b)=>esc(a)+(b?'<span class="p"> '+esc(b)+'</span>':'');
      el.innerHTML = s.verdict==='일치' ? t(s.canon,s.canon_ph)
        : t(s.canon,s.canon_ph)+'→'+(s.pred===null? '∅' : t(s.pred,s.pred_ph));
    } else el.textContent = s.verdict==='일치' ? c : c+'→'+p;
    host.appendChild(el);
  });
}
function renderMic(){
  if(!last) return;
  const el=$('m-canon');
  if(st.mode==='both') seqBoth(last.canon_jamo, last.canon_phone, el);
  else { el.classList.remove('both'); el.textContent = seq(st.mode==='phone'? last.canon_phone : last.canon_jamo); }
  const ms = last.models || [{name:'모델', pred_jamo:last.pred_jamo, pred_phone:last.pred_phone, sites:last.sites}];
  if($('m-models').children.length !== ms.length*2) modelRows(ms.map(m=>m.name));
  ms.forEach((m,i)=>{
    const pe=$('m-pred-'+i);
    if(st.mode==='both') seqBoth(m.pred_jamo, m.pred_phone, pe);
    else { pe.classList.remove('both'); pe.textContent = seq(st.mode==='phone'? m.pred_phone : m.pred_jamo); }
    renderSites(m.sites, $('m-sites-'+i), last.canon_jamo.length);
  });
  renderFeedback();
}
const _render=render; render=function(){ _render(); renderMic(); };
$('rec').addEventListener('click', async ()=>{
  if(mr && mr.state==='recording'){ mr.stop(); return; }
  try{
    stream = stream || await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, noiseSuppression:true}});
  }catch(e){ setStatus('마이크를 열지 못했다: '+e.message+' (http://127.0.0.1 로 접속했는지 확인)'); return; }
  chunks=[]; mr=new MediaRecorder(stream);
  mr.ondataavailable=e=>{ if(e.data.size) chunks.push(e.data); };
  mr.onstop=async ()=>{
    $('rec').textContent='● 녹음 시작'; $('rec').dataset.on='false';
    const blob=new Blob(chunks,{type:mr.mimeType||'audio/webm'});
    $('play').src=URL.createObjectURL(blob); $('play').classList.remove('hide');
    setStatus('추론 중…');
    const fd=new FormData(); fd.append('audio', blob, 'rec.webm'); fd.append('sentence', $('sent').value); fd.append('country', $('country').value);
    try{
      const r=await fetch('transcribe',{method:'POST',body:fd}); const d=await r.json();
      if(!r.ok){ setStatus('오류: '+(d.error||r.status)); return; }
      last=d; renderMic();
      const per=(d.models||[]).map(m=>`${m.name} ${m.ms}ms`).join(' · ');
      setStatus(`${d.sec}초 · 음량 ${d.rms}${per? ' · '+per : ''}`);
    }catch(e){ setStatus('요청 실패: '+e.message); }
  };
  mr.start(); $('rec').textContent='■ 정지'; $('rec').dataset.on='true'; setStatus('녹음 중…');
});
setStatus('녹음 버튼을 누르면 마이크 권한을 묻는다');
"""


def mark_both(seq, seq_ph, ref):
    """자모와 음소기호를 위아래로 병기. 둘은 1:1 이라 위치로 짝짓는다."""
    import edlib
    enc, dec = {}, {}
    def e(xs):
        s = ""
        for x in xs:
            if x not in enc:
                c = chr(0xE000 + len(enc)); enc[x] = c; dec[c] = x
            s += enc[x]
        return s
    a = edlib.getNiceAlignment(edlib.align(e(seq), e(ref), task="path"), e(seq), e(ref))
    out, k = [], 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            continue
        ph = seq_ph[k] if k < len(seq_ph) else ""
        cls = "tok" if x == y else "tok d"
        out.append(f'<span class="{cls}"><span class="j">{html.escape(dec[x])}</span>'
                   f'<span class="p">{html.escape(ph)}</span></span>')
        k += 1
    return "".join(out)


def mark(seq, ref):
    """ref 와 다른 토큰을 표시 (같은 길이가 아니면 정렬해서 비교)."""
    import edlib
    enc, dec = {}, {}
    def e(xs):
        s = ""
        for x in xs:
            if x not in enc:
                c = chr(0xE000 + len(enc)); enc[x] = c; dec[c] = x
            s += enc[x]
        return s
    a = edlib.getNiceAlignment(edlib.align(e(seq), e(ref), task="path"), e(seq), e(ref))
    out = []
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            continue
        tok = html.escape(dec[x])
        out.append(tok if x == y else f'<span class="d">{tok}</span>')
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--title", default="모델이 들린 대로 적는가 — 샘플 청취")
    ap.add_argument("--model-label", dest="label", default="FINAL 모델", help="모델 줄 이름")
    a = ap.parse_args()
    rows = json.load(open(os.path.join(a.dir, "samples.json"), encoding="utf-8"))
    cards = []
    for r in rows:
        preds = r.get("preds") or [{"name": a.label, "jamo": r["final"], "ph": r["final_ph"], "d": r["d_final"]},
                                   {"name": "기존 모델", "jamo": r["old"], "ph": r["old_ph"], "d": r["d_old"]}]

        def rowsets(seq, seq_ph, ref, ref_ph):
            j = f'<div class="seq hide" data-mode="jamo">{mark(seq, ref)}</div>'
            p = f'<div class="seq hide" data-mode="phone">{mark(seq_ph, ref_ph)}</div>'
            b = f'<div class="seq both" data-mode="both">{mark_both(seq, seq_ph, ref)}</div>'
            return j + p + b

        def seqs(key, ref_key):
            return rowsets(r[key], r[key + "_ph"], r[ref_key], r[ref_key + "_ph"])

        # 모델 줄 — 사람 전사와 비교한다(= 모델 오류)
        model_rows = "".join(
            f'<div class="row"><div class="tag {"t-model" if k == 0 else "t-old"}">{html.escape(m["name"])}</div>'
            f'{rowsets(m["jamo"], m["ph"], r["tag"], r["tag_ph"])}</div>' for k, m in enumerate(preds))
        badges = "".join(f'<span class="badge {"ok" if m["d"] == 0 else "err"}">{html.escape(m["name"])} 차이 {m["d"]}자모</span>'
                         for m in preds)
        cards.append(f"""
<div class="card" data-group="{html.escape(r['group'])}" data-split="{html.escape(r['split'])}">
  <div class="left">
    <div class="prompt">{html.escape(r['prompt'] or '')}</div>
    <div class="meta"><span>{html.escape(r['lang'] or '')}</span><span>{r['task']}</span><span>{r['sec']}초</span>
      <span>{html.escape(r['split'])}</span><span>{html.escape(r['group'])}</span></div>
    <audio controls preload="none" src="wav/{html.escape(r['stem'])}.wav"></audio>
    <div class="meta">{badges}</div>
    <div class="meta"><span>{html.escape(r['stem'])}</span></div>
  </div>
  <div class="rows">
    <div class="row"><div class="tag t-canon">표준발음</div>{seqs('canon','tag')}</div>
    <div class="row"><div class="tag t-actual">사람 전사</div>{seqs('tag','canon')}</div>
    {model_rows}
  </div>
</div>""")
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.title)}</title><style>{CSS}</style></head><body>
<div class="wrap"><header>
<h1>{html.escape(a.title)}</h1>
<p class="sub">기준과 <strong>다른 소리만 색으로</strong> 표시한다.
<b>사람 전사</b> 줄의 색은 화자가 실제로 낸 오발음, <b>모델</b> 줄의 색은 모델이 잘못 들은 자리다.</p>
<div class="legend">
  <span><span class="sw" style="background:var(--canon)"></span><b>표준발음</b> 문장대로 읽었을 때의 소리</span>
  <span><span class="sw" style="background:var(--actual)"></span><b>사람 전사</b> 전문가가 들은 실제 발음</span>
  <span><span class="sw" style="background:var(--model)"></span><b>모델</b> 우리 모델이 받아쓴 것</span>
</div>
</header></div>
<div class="toolbar"><div class="controls">
  <span class="sub">표기</span>
  <button data-mode-btn="jamo">자모</button><button data-mode-btn="phone">음소기호</button><button data-mode-btn="both">둘 다</button>
  <span class="sub" style="margin-left:10px">보기</span>
  <button data-filter-btn="all">전체</button><button data-filter-btn="오발음">오발음 발화</button>
  <button data-filter-btn="정상">정상 발음</button>
  <button data-filter-btn="test_unseen">안 본 문항</button><button data-filter-btn="test_seen">본 문항</button>
</div></div>
<div class="wrap">
<div class="card mic" id="mic">
  <div class="left">
    <div class="prompt">직접 말해 보기</div>
    <div class="meta"><span>읽을 문장을 적고 녹음하면 {html.escape(a.label)} 이 들린 대로 받아쓴다</span></div>
    <label class="sub" for="sent">읽을 문장</label>
    <input id="sent" value="안녕하세요" autocomplete="off">
    <label class="sub" for="country">화자 국적 (피드백용)</label>
    <select id="country"><option value="">선택 안 함</option></select>
    <div class="controls">
      <button id="rec" type="button">● 녹음 시작</button>
      <span id="status" class="sub">마이크 권한이 필요하다</span>
    </div>
    <audio id="play" controls preload="none" class="hide"></audio>
  </div>
  <div class="rows" id="micout">
    <div class="row"><div class="tag t-canon">표준발음</div><div class="seq" data-live id="m-canon">—</div></div>
    <div id="m-models"></div>
    <div class="row"><div class="tag">피드백</div><div class="seq fb" data-live id="m-fb">—</div></div>
  </div>
</div>
{''.join(cards)}
<footer>샘플 {len(rows)}개 · 생성 {time.strftime('%Y-%m-%d %H:%M')} · 모델 {html.escape(a.label)} ·
예측은 greedy CTC(LM·사전 없음) · AIHub 데이터 — 외부 공개 금지</footer>
</div><script>{JS}{MIC_JS}</script></body></html>"""
    out = os.path.join(a.dir, "index.html")
    open(out, "w", encoding="utf-8").write(doc)
    print(f"[done] {out} · 샘플 {len(rows)}개")


if __name__ == "__main__":
    main()
