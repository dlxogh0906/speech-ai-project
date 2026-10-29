# -*- coding: utf-8 -*-
"""비교 결과 → 터미널 요약 · 혼자 열리는 HTML 보고서(음성 내장, 외부 파일·네트워크 없음).

보고서 구성은 계획서 5.2 를 따른다: 글자 칩(어긋난 곳 번호) → 지적 카드(무엇이/어떻게/보조 숫자 +
그 구간 원본·내 녹음 듣기) → 음높이 곡선(녹음을 원본 시간축에 음절 단위로 맞춰 겹침) → 음절 길이 비.
"""
import base64
import json
import math

import numpy as np

from .audio import wav_bytes
from .feedback import retake

CIRCLED = "❶❷❸❹❺❻❼❽❾"
TYPE_KO = {"pitch_level": "음높이", "pitch_dir": "끝 억양", "duration": "길이", "pause": "쉼", "energy": "세기",
           "tempo": "빠르기", "voicing": "목소리"}
PAD = 0.3            # 구간 듣기: 앞뒤로 더 들려줄 폭(초)


def terminal(A, B, C):
    S = A["syllables"]
    mark = {}
    for k, iss in enumerate(C["selected"]):
        if iss.get("i") is not None:
            for j in range(iss["i"], iss.get("i_end", iss["i"]) + 1):
                mark[j] = CIRCLED[k] if j == iss["i"] else "~"
    unc = {row["i"] for row in C["rows"] if row["uncertain"]}
    words = []
    for wi, w in enumerate(A["words"]):
        chips = "".join("[%s%s]" % (s["text"], mark.get(s["i"], "·" if s["i"] in unc else ""))
                        for s in S if s["word"] == wi)
        words.append(chips)
    q = C["quality"]
    L = ["대사: %s" % " ".join(A["words"]),
         "  " + " ".join(words),
         "  정렬 표기 %s · 경계 일치 %s · 대사 받아 적기 JER %s · 빠르기 %.2f배%s" % (
             A["token_source"].split(":")[0], "—" if q["bound_agree"] is None else "%.0f%%" % (100 * q["bound_agree"]),
             "—" if q.get("line_jer") is None else "%.2f" % q["line_jer"],
             C["tempo"], "" if not unc else " · 확신 없어 안 잰 음절(·) %d개" % len(unc))]
    if not q["ok"]:
        say = q.get("say") or retake(q, A["words"], [s["text"] for s in S], [s["word"] for s in S])
        L.append("  ⚠ %s → %s" % (say["what"], say["how"]))
    if q.get("note"):
        L.append("  ※ %s%s" % (q["note"], " (녹음 %.2f~%.2f초)" % tuple(B["crop"]) if B.get("crop") else ""))
    elif not C["selected"]:
        L.append("  원본과 거의 같아요." + (" 더 다듬으려면: %s → %s" % (C["minor"][0]["say"]["what"], C["minor"][0]["say"]["how"])
                                        if C.get("minor") else " 어긋난 곳을 찾지 못했어요."))
    for k, iss in enumerate(C["selected"]):
        say = iss["say"]
        L.append("  %s %s → %s   (%s)" % (CIRCLED[k], say["what"], say["how"], say["detail"]))
    if C["match"] is not None:
        L.append("  (내부 참고) 대사 일치도 %d" % C["match"])
    return "\n".join(L)


def _seg(S, iss, dur):
    if iss.get("i") is None:
        return [0.0, dur]
    a = S[iss["i"]]["start"]
    b = S[iss.get("i_end", iss["i"])]["end"]
    if iss["type"] == "pause" and iss["i"] > 0:
        a = S[iss["i"] - 1]["start"]
    return [round(max(0.0, a - PAD), 3), round(min(dur, b + PAD), 3)]


def _curve(res, mapper=None):
    tr = res.get("tracks")
    if not tr:
        return []
    s0, s1 = res["syllables"][0]["start"], res["syllables"][-1]["end"]
    out = []
    for t, v in zip(tr["t_pitch"], tr["st"]):
        if v is None or t < s0 or t > s1:
            continue
        out.append([round(float(mapper(t) if mapper else t), 3), round(float(v), 2)])
    return out


def payload(A, B, C, wav_ref, wav_take, title=None):
    """wav_take 는 원래 녹음 그대로 준다 — 앞뒤를 잘라 분석했으면(B["crop"]) 그 구간만 넣는다."""
    SA, SB = A["syllables"], B["syllables"]
    if B.get("crop"):
        wav_take = wav_take[int(round(B["crop"][0] * 16000)):int(round(B["crop"][1] * 16000))]
    kb = [s["start"] for s in SB] + [SB[-1]["end"]]
    ka = [s["start"] for s in SA] + [SA[-1]["end"]]
    mapper = lambda t: float(np.interp(t, kb, ka))         # 녹음 시각 → 원본 시간축(음절 단위 꺾은선)
    sel, span = {}, {}
    for k, iss in enumerate(C["selected"]):
        if iss.get("i") is not None:
            sel[iss["i"]] = k
            for j in range(iss["i"], iss.get("i_end", iss["i"]) + 1):
                span[j] = k
    syl = []
    for r, t, row in zip(SA, SB, C["rows"]):
        dq = row["checks"].get("duration", {}).get("diff")
        syl.append(dict(text=r["text"], word=r["word"], score=row["score"], uncertain=row["uncertain"],
                        sel=sel.get(r["i"]), span=span.get(r["i"]), flags=[k for k, v in row["checks"].items() if v["flag"]],
                        ref=dict(start=r["start"], end=r["end"], level=r["level"], delta=r["delta"], pause=r["pause_before"]),
                        take=dict(start=t["start"], end=t["end"], level=t["level"], delta=t["delta"], pause=t["pause_before"]),
                        dur_ratio=dq))
    issues = []
    for k, iss in enumerate(C["selected"]):
        issues.append(dict(n=k, i=iss.get("i"), i_end=iss.get("i_end", iss.get("i")), type=iss["type"],
                           type_ko=TYPE_KO[iss["type"]], **iss["say"],
                           seg_ref=_seg(SA, iss, A["duration"]), seg_take=_seg(SB, iss, B["duration"])))
    minor = [dict(type=x["type"], type_ko=TYPE_KO[x["type"]], **x["say"]) for x in C.get("minor", []) if "say" in x]
    b64 = lambda w: "data:audio/wav;base64," + base64.b64encode(wav_bytes(w)).decode("ascii")
    q = dict(C["quality"])
    if not q["ok"] and "say" not in q:
        q["say"] = retake(q, A["words"], [s["text"] for s in SA], [s["word"] for s in SA])
    blocked = q.get("reason") in ("line", "silent", "whisper", "missing")   # 다른 대사·잡음·빠진 어절이면 녹음 곡선은 원본에 맞춘 게 아니다
    return dict(title=title or "성대모사 교정", line=" ".join(A["words"]), words=A["words"], syllables=syl, issues=issues, minor=minor,
                quality=q, tempo=C["tempo"], match=C["match"], token_source=A["token_source"],
                bounds_ref=ka, curves=dict(ref=_curve(A), take=[] if blocked else _curve(B, mapper)),
                audio=dict(ref=b64(wav_ref), take=b64(wav_take)), dur_tol=math.exp(0.336472))


def html(A, B, C, wav_ref, wav_take, title=None):
    data = json.dumps(payload(A, B, C, wav_ref, wav_take, title), ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>성대모사 교정 결과</title>
<style>
:root{--bg:#f7f6f2;--fg:#1c1c1e;--muted:#6c6c72;--card:#ffffff;--line:#e4e1d8;--ref:#2b63d9;--take:#e05a1f;
--ok:#2f9a5a;--okbg:#e6f4ec;--warn:#b98a0c;--warnbg:#fbf2d9;--bad:#cf3f3f;--badbg:#fbe5e3;--unc:#9a9aa2;--uncbg:#efeff1;--band:#f0eee8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;--fg:#ececef;--muted:#9c9ca6;--card:#1c1c20;
--line:#2d2d34;--ref:#77a1ff;--take:#ff8b55;--ok:#5fd08f;--okbg:#17301f;--warn:#f0c14b;--warnbg:#332a12;--bad:#ff7a73;
--badbg:#3a1b1a;--unc:#80808a;--uncbg:#26262b;--band:#232328}}
:root[data-theme="dark"]{--bg:#121214;--fg:#ececef;--muted:#9c9ca6;--card:#1c1c20;--line:#2d2d34;--ref:#77a1ff;--take:#ff8b55;
--ok:#5fd08f;--okbg:#17301f;--warn:#f0c14b;--warnbg:#332a12;--bad:#ff7a73;--badbg:#3a1b1a;--unc:#80808a;--uncbg:#26262b;--band:#232328}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:15px;font-weight:600;color:var(--muted);margin:0 0 4px}
.line{font-size:26px;font-weight:700;margin:0 0 18px;word-break:keep-all}
h2{font-size:14px;font-weight:600;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-top:14px}
.chips{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:flex-end}
.word{display:flex;gap:5px}
.chip{position:relative;min-width:42px;height:50px;padding:0 6px;border-radius:10px;display:flex;align-items:center;
justify-content:center;font-size:23px;font-weight:650;border:2px solid transparent;cursor:pointer;user-select:none}
.chip.g{background:var(--okbg);color:var(--ok)}.chip.y{background:var(--warnbg);color:var(--warn)}
.chip.r{background:var(--badbg);color:var(--bad)}.chip.u{background:var(--uncbg);color:var(--unc)}
.chip.cur{border-color:currentColor}
.badge{position:absolute;top:-9px;right:-9px;min-width:21px;height:21px;border-radius:11px;background:var(--bad);color:#fff;
font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;padding:0 5px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12.5px;margin-top:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.issue{border-left:4px solid var(--bad)}
.issue .k{font-size:12.5px;color:var(--muted)}
.issue .what{font-size:19px;font-weight:700;margin-top:2px;word-break:keep-all}
.issue .how{margin-top:4px;word-break:keep-all}
.issue .detail{color:var(--muted);font-size:13px;margin-top:6px}
.issue.minor .what{font-size:16px}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:9px;
padding:7px 12px;cursor:pointer}
button:hover{border-color:var(--muted)}
button .sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.banner{background:var(--warnbg);color:var(--fg);border:1px solid var(--warn);border-radius:12px;padding:12px 14px;margin-top:14px}
.good{background:var(--okbg);border:1px solid var(--ok);border-radius:12px;padding:12px 14px;margin-top:14px}
svg{width:100%;height:auto;display:block}
svg text{fill:var(--muted);font-size:11px}
svg .lbl{fill:var(--fg);font-size:14px;font-weight:600}
.meta{color:var(--muted);font-size:12.5px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:center}
summary{cursor:pointer;color:var(--muted);font-weight:600}
.tw{overflow-x:auto}
label{color:var(--muted);font-size:13px}
</style>
</head>
<body>
<main>
<h1 id="title"></h1>
<div class="line" id="line"></div>
<div id="state"></div>
<section class="card"><div class="chips" id="chips"></div>
<div class="legend"><span><i class="dot" style="background:var(--ok)"></i>원본과 비슷</span>
<span><i class="dot" style="background:var(--warn)"></i>조금 다름</span><span><i class="dot" style="background:var(--bad)"></i>어긋남</span>
<span><i class="dot" style="background:var(--unc)"></i>확신 없어 안 봄</span></div></section>
<div id="issues"></div>
<section class="card"><h2>음높이 — 원본 <span style="color:var(--ref)">━</span> · 내 녹음 <span style="color:var(--take)">━</span>
(각자 평소 음높이 기준, 내 녹음은 음절별로 원본 시간에 맞춤)</h2><svg id="pitch"></svg></section>
<section class="card"><h2>음절 길이 — 원본 대비 (전체 빠르기를 맞춘 뒤)</h2><svg id="dur"></svg></section>
<section class="card"><h2>전체 듣기</h2>
<div class="btns"><button data-play="ref"><i class="sw" style="background:var(--ref)"></i>원본 전체</button>
<button data-play="take"><i class="sw" style="background:var(--take)"></i>내 녹음 전체</button>
<label><input type="checkbox" id="slow"> 0.75배속</label></div>
<div class="meta" id="meta"></div></section>
<details class="card"><summary>음절별 수치</summary><div class="tw"><table id="tbl"></table></div></details>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = s => document.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent, text) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; if (parent) parent.appendChild(e); return e; };
const CIR = '❶❷❸❹❺';
// 그래프는 실제 폭(px)에 맞춰 그린다 — 고정 좌표계를 줄이면 폰에서 글자가 6px 로 작아진다
const widthOf = svg => Math.max(300, Math.round((svg.getBoundingClientRect ? svg.getBoundingClientRect().width : 0) || 860));
$('#title').textContent = D.title; $('#line').textContent = D.line;
// 재생
const players = {ref: new Audio(D.audio.ref), take: new Audio(D.audio.take)};
let stopAt = null, raf = null;
function play(which, seg) {
  for (const k in players) players[k].pause();
  const a = players[which]; const rate = $('#slow').checked ? 0.75 : 1.0;
  a.playbackRate = rate; a.preservesPitch = true;
  a.currentTime = seg ? seg[0] : 0; stopAt = seg ? seg[1] : null; a.play();
  cancelAnimationFrame(raf);
  const tick = () => { if (stopAt != null && a.currentTime >= stopAt) { a.pause(); return; } if (!a.paused) raf = requestAnimationFrame(tick); };
  raf = requestAnimationFrame(tick);
}
document.querySelectorAll('[data-play]').forEach(b => b.onclick = () => play(b.dataset.play, null));
// 상태
const st = $('#state');
if (!D.quality.ok) { const s = D.quality.say || {what: '녹음을 원본과 확실히 맞추지 못했어요', how: '조용한 곳에서 다시 녹음해 주세요'};
  const b = document.createElement('div'); b.className = 'banner'; b.textContent = s.what + ' — ' + s.how + '.'; st.appendChild(b); }
else if (!D.issues.length) { const g = document.createElement('div'); g.className = 'good';
  g.textContent = '원본과 거의 같아요.' + (D.minor && D.minor.length ? ' 더 다듬으려면: ' + D.minor[0].what + ' → ' + D.minor[0].how : ' 어긋난 곳을 찾지 못했어요.');
  st.appendChild(g); }
if (D.quality.note) { const n = document.createElement('div'); n.className = 'meta'; n.textContent = '※ ' + D.quality.note; st.appendChild(n); }
// 칩
const chips = $('#chips'); let cur = D.issues.length && D.issues[0].i != null ? [D.issues[0].i, D.issues[0].i_end] : null;
const isCur = i => cur != null && i >= cur[0] && i <= cur[1];
function cls(s) { if (s.uncertain) return 'u'; if (s.flags.length) return 'r'; return s.score >= 75 ? 'g' : 'y'; }
function drawChips() {
  chips.innerHTML = '';
  D.words.forEach((w, wi) => { const g = document.createElement('div'); g.className = 'word';
    D.syllables.forEach((s, i) => { if (s.word !== wi) return; const c = document.createElement('div');
      c.className = 'chip ' + cls(s) + (isCur(i) ? ' cur' : ''); c.textContent = s.text;
      c.title = s.uncertain ? '경계를 확실히 못 잡아 안 봤어요' : (s.flags.length ? '어긋남: ' + s.flags.join(', ') : '');
      if (s.sel != null) { const b = document.createElement('span'); b.className = 'badge'; b.textContent = s.sel + 1; c.appendChild(b); }
      c.onclick = () => { cur = [i, i]; drawChips(); drawPitch(); }; g.appendChild(c); });
    chips.appendChild(g); });
}
// 지적 카드
const box = $('#issues');
D.issues.forEach((it, k) => { const c = document.createElement('section'); c.className = 'card issue' + (k ? ' minor' : '');
  c.innerHTML = '<div class="k">' + (k ? '다음 ' : '') + CIR[k] + ' ' + it.type_ko + ' · ' + it.where + '</div>' +
    '<div class="what"></div><div class="how"></div><div class="detail"></div>' +
    '<div class="btns"><button data-k="ref"><i class="sw" style="background:var(--ref)"></i>원본 이 부분</button>' +
    '<button data-k="take"><i class="sw" style="background:var(--take)"></i>내 녹음 이 부분</button></div>';
  c.querySelector('.what').textContent = it.what; c.querySelector('.how').textContent = it.how; c.querySelector('.detail').textContent = it.detail;
  c.querySelector('[data-k=ref]').onclick = () => play('ref', it.seg_ref);
  c.querySelector('[data-k=take]').onclick = () => play('take', it.seg_take);
  c.onmouseenter = () => { if (it.i != null) { cur = [it.i, it.i_end]; drawChips(); drawPitch(); } };
  box.appendChild(c); });
// 음높이 곡선
function drawPitch() {
  const svg = $('#pitch'); svg.innerHTML = ''; const W = widthOf(svg), H = 260, L = 34, R = 8, T = 10, B = 40;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const b = D.bounds_ref, x0 = b[0], x1 = b[b.length - 1];
  const all = D.curves.ref.concat(D.curves.take).map(p => p[1]);
  let lo = Math.min(-6, ...all), hi = Math.max(6, ...all); lo = Math.max(lo, -16); hi = Math.min(hi, 16);
  const X = t => L + (t - x0) / Math.max(1e-6, x1 - x0) * (W - L - R), Y = v => T + (hi - v) / (hi - lo) * (H - T - B);
  D.syllables.forEach((s, i) => { const a = X(b[i]), z = X(b[i + 1]);
    const fill = isCur(i) ? 'var(--badbg)' : (i % 2 ? 'var(--band)' : 'none');
    if (fill !== 'none') el('rect', {x: a, y: T, width: Math.max(0, z - a), height: H - T - B, fill: fill}, svg);
    el('text', {x: (a + z) / 2, y: H - B + 20, 'text-anchor': 'middle', class: 'lbl', style: s.flags.length && !s.uncertain ? 'fill:var(--bad)' : ''}, svg, s.text); });
  for (let v = Math.ceil(lo / 3) * 3; v <= hi; v += 3) { el('line', {x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: 'var(--line)'}, svg);
    el('text', {x: L - 6, y: Y(v) + 4, 'text-anchor': 'end'}, svg, (v > 0 ? '+' : '') + v); }
  el('text', {x: 2, y: T + 8}, svg, '반음');
  const line = (pts, color) => { let d = '', prev = null;
    pts.forEach(p => { if (p[1] < lo || p[1] > hi) { prev = null; return; } d += (prev == null || p[0] - prev > 0.035 ? 'M' : 'L') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1); prev = p[0]; });
    el('path', {d: d, fill: 'none', stroke: color, 'stroke-width': 2.4, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg); };
  line(D.curves.ref, 'var(--ref)'); line(D.curves.take, 'var(--take)');
}
// 길이 비
function drawDur() {
  const svg = $('#dur'); svg.innerHTML = ''; const W = widthOf(svg), H = 170, L = 34, R = 8, T = 10, B = 34;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const n = D.syllables.length, bw = (W - L - R) / n, lim = Math.log(2.5), Y = v => T + (lim - v) / (2 * lim) * (H - T - B);
  const tol = Math.log(D.dur_tol);
  [0, tol, -tol].forEach((v, k) => el('line', {x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: k ? 'var(--warn)' : 'var(--muted)', 'stroke-dasharray': k ? '4 4' : ''}, svg));
  [['2배', Math.log(2)], ['1배', 0], ['½배', -Math.log(2)]].forEach(([t, v]) => el('text', {x: L - 6, y: Y(v) + 4, 'text-anchor': 'end'}, svg, t));
  D.syllables.forEach((s, i) => { const cx = L + bw * i + bw / 2;
    if (s.dur_ratio != null && !s.uncertain) { const v = Math.max(-lim, Math.min(lim, Math.log(s.dur_ratio)));
      const bad = Math.abs(v) > tol; el('rect', {x: cx - bw * 0.3, y: Math.min(Y(0), Y(v)), width: bw * 0.6, height: Math.abs(Y(v) - Y(0)),
        rx: 3, fill: bad ? 'var(--bad)' : 'var(--ok)', opacity: bad ? 1 : 0.55}, svg); }
    el('text', {x: cx, y: H - B + 20, 'text-anchor': 'middle', class: 'lbl', style: s.uncertain ? 'fill:var(--unc)' : ''}, svg, s.text); });
}
// 표
(function () { const t = $('#tbl'); const f = v => v == null ? '—' : (Math.round(v * 10) / 10).toFixed(1);
  let h = '<tr><th>음절</th><th>원본 시작</th><th>길이</th><th>내 시작</th><th>길이</th><th>길이 비</th><th>원본 높이</th><th>내 높이</th><th>원본 끝억양</th><th>내 끝억양</th><th>원본 앞쉼</th><th>내 앞쉼</th></tr>';
  D.syllables.forEach(s => { h += '<tr><td>' + s.text + (s.uncertain ? '·' : '') + '</td><td>' + s.ref.start.toFixed(2) + '</td><td>' + (s.ref.end - s.ref.start).toFixed(2) +
    '</td><td>' + s.take.start.toFixed(2) + '</td><td>' + (s.take.end - s.take.start).toFixed(2) + '</td><td>' + (s.dur_ratio == null ? '—' : s.dur_ratio.toFixed(2)) +
    '</td><td>' + f(s.ref.level) + '</td><td>' + f(s.take.level) + '</td><td>' + f(s.ref.delta) + '</td><td>' + f(s.take.delta) +
    '</td><td>' + s.ref.pause.toFixed(2) + '</td><td>' + s.take.pause.toFixed(2) + '</td></tr>'; });
  t.innerHTML = h; })();
$('#meta').textContent = '정렬 표기 ' + D.token_source.split(':')[0] + ' · 경계 일치 ' + (D.quality.bound_agree == null ? '—' : Math.round(D.quality.bound_agree * 100) + '%') +
  ' · 전체 빠르기 원본의 ' + D.tempo.toFixed(2) + '배' + (D.match != null ? ' · (내부 참고) 일치도 ' + D.match : '');
drawChips(); drawPitch(); drawDur();
if (typeof window !== 'undefined' && window.addEventListener) { let rt = null; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { drawPitch(); drawDur(); }, 150); }); }
</script>
</body>
</html>
"""
