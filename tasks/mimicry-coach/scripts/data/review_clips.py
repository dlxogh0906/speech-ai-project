# -*- coding: utf-8 -*-
"""원작 대사 클립 사람 검수(계획서 7.1)용 HTML 한 장 — 클립마다 음절 구간을 눌러 듣고 판정을 골라 JSON 으로 내려받는다.

자동으로 자른 클립(prepare_clips.py)은 대사가 잘렸는지·다른 말이 섞였는지·음절 경계가 맞는지를 사람이 들어 봐야 한다.
이 페이지는 클립마다 원본(목소리만)·배경음 포함 듣기, 음절 버튼(그 음절 구간만 재생), 앞뒤 여유(첫 음절 앞·끝 음절 뒤 0.3초)
듣기, 판정(잘림 없음 · 앞이 잘림 · 끝이 잘림 · 다른 말 섞임 · 경계 틀림)과 메모를 두고, "결과 내려받기"로 JSON 을 준다.
원작 음성이 들어가므로 저장소 밖(클립 폴더)에만 만든다 — 공유하지 말 것(내부 시험용).

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/data/review_clips.py --clips /data/speech-ai/out/mimic/original_clips_v1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, base64, json, os

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>원작 클립 검수</title>
<style>
:root{--bg:#f7f6f2;--fg:#1c1c1e;--muted:#6c6c72;--card:#fff;--line:#e4e1d8;--acc:#2b63d9;--sel:#eaf0fd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;--fg:#ececef;--muted:#9c9ca6;--card:#1c1c20;
--line:#2d2d34;--acc:#77a1ff;--sel:#1d2840}}
:root[data-theme="dark"]{--bg:#121214;--fg:#ececef;--muted:#9c9ca6;--card:#1c1c20;--line:#2d2d34;--acc:#77a1ff;--sel:#1d2840}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 16px;word-break:keep-all}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-top:14px}
.line{font-size:20px;font-weight:700;word-break:keep-all}
.meta{color:var(--muted);font-size:12.5px;margin-top:2px;word-break:keep-all}
.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:10px}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:9px;padding:6px 11px;cursor:pointer}
button:hover{border-color:var(--muted)}
button.syl{font-size:18px;font-weight:650;min-width:44px}
button.on{border-color:var(--acc);background:var(--sel)}
label{margin-right:12px;white-space:nowrap}
input[type=text]{font:inherit;width:100%;padding:7px 10px;border:1px solid var(--line);border-radius:9px;background:var(--bg);color:var(--fg)}
.go{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:700;padding:10px 18px}
</style>
</head>
<body>
<main>
<h1>원작 클립 검수</h1>
<p class="sub">클립마다 음절 버튼을 눌러 그 구간만 들어 보고, 앞뒤 여유도 들어 본 뒤 판정을 고르세요. 다 했으면 맨 아래 "결과 내려받기".
원작 음성이 들어 있어요 — 내부 시험용이라 밖으로 공유하지 마세요.</p>
<div id="list"></div>
<div class="row"><button class="go" id="save">결과 내려받기</button><span class="meta" id="saved"></span></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const V = [['ok', '잘림 없음'], ['cut_start', '앞이 잘림'], ['cut_end', '끝이 잘림'], ['extra', '다른 말 섞임'], ['bounds', '경계 틀림']];
const audio = new Audio(); let stopAt = null;
audio.ontimeupdate = () => { if (stopAt != null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; } };
function play(src, a, b) { audio.pause(); if (audio.src !== src) audio.src = src; audio.currentTime = Math.max(0, a); stopAt = b; audio.play(); }
const list = document.getElementById('list');
D.clips.forEach(c => {
  const s = document.createElement('section'); s.className = 'card';
  const t = document.createElement('div'); t.className = 'line'; t.textContent = c.id + ' · ' + c.line; s.appendChild(t);
  const m = document.createElement('div'); m.className = 'meta'; m.textContent = c.meta; s.appendChild(m);
  const r1 = document.createElement('div'); r1.className = 'row';
  const all = document.createElement('button'); all.textContent = '▶ 전체(목소리만)'; all.onclick = () => play(c.ref, 0, null); r1.appendChild(all);
  if (c.mix) { const mx = document.createElement('button'); mx.textContent = '▶ 배경음 포함'; mx.onclick = () => play(c.mix, 0, null); r1.appendChild(mx); }
  const pre = document.createElement('button'); pre.textContent = '▶ 첫 음절 앞 여유'; pre.onclick = () => play(c.ref, 0, c.syl[0].start + 0.15); r1.appendChild(pre);
  const post = document.createElement('button'); post.textContent = '▶ 끝 음절 뒤 여유'; post.onclick = () => play(c.ref, c.syl[c.syl.length - 1].end - 0.15, null); r1.appendChild(post);
  s.appendChild(r1);
  const r2 = document.createElement('div'); r2.className = 'row';
  c.syl.forEach(x => { const b = document.createElement('button'); b.className = 'syl'; b.textContent = x.text;
    b.title = x.start.toFixed(2) + '~' + x.end.toFixed(2) + '초'; b.onclick = () => play(c.ref, x.start, x.end); r2.appendChild(b); });
  s.appendChild(r2);
  const r3 = document.createElement('div'); r3.className = 'row';
  V.forEach(([v, label]) => { const l = document.createElement('label'); const i = document.createElement('input'); i.type = 'checkbox';
    i.name = c.id; i.value = v; l.appendChild(i); l.appendChild(document.createTextNode(' ' + label)); r3.appendChild(l); });
  s.appendChild(r3);
  const note = document.createElement('input'); note.type = 'text'; note.placeholder = '메모(예: 끝 \'야\'가 반쯤 잘림)'; note.id = 'note-' + c.id;
  const r4 = document.createElement('div'); r4.className = 'row'; r4.appendChild(note); s.appendChild(r4);
  list.appendChild(s);
});
document.getElementById('save').onclick = () => {
  const out = D.clips.map(c => ({id: c.id, line: c.line,
    verdict: Array.from(document.querySelectorAll('input[name="' + c.id + '"]:checked')).map(x => x.value),
    note: document.getElementById('note-' + c.id).value}));
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify({reviewed: new Date().toISOString(), clips: out}, null, 1)],
    {type: 'application/json'})); a.download = 'clip_review.json'; a.click();
  document.getElementById('saved').textContent = '내려받았어요 — clips.json 옆에 두고 알려 주세요.';
};
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="원작 발췌 clips.json 폴더")
    ap.add_argument("--out", default="", help="HTML 경로(기본: 클립 폴더/review.html)")
    a = ap.parse_args()
    from mimicry.audio import load_audio, wav_bytes
    from mimicry.pipeline import Coach
    coach = Coach()
    meta = json.load(open(os.path.join(a.clips, "clips.json"), encoding="utf-8"))
    b64 = lambda x: "data:audio/wav;base64," + base64.b64encode(wav_bytes(x)).decode("ascii")
    clips = []
    for c in meta["clips"]:
        if not (c.get("qc") or {}).get("ok", True):
            continue
        x = load_audio(os.path.join(a.clips, c["files"]["ref"]))
        A, _ = coach.analyze(x, c["line"])
        src = c.get("source") or {}
        clips.append(dict(id=c["id"], line=c["line"],
                          meta="%s · %s · %s %.1f~%.1f초 · 받아 적기 JER %.2f" % (
                              c.get("title", ""), src.get("channel", ""), src.get("video_id", ""), src.get("t0", 0), src.get("t1", 0),
                              (c.get("qc") or {}).get("jer", float("nan"))),
                          ref=b64(x), mix=b64(load_audio(os.path.join(a.clips, c["files"]["mix"]))) if "mix" in c["files"] else None,
                          syl=[dict(text=s["text"], start=round(s["start"], 3), end=round(s["end"], 3)) for s in A["syllables"]]))
    out = a.out or os.path.join(a.clips, "review.html")
    data = json.dumps(dict(clips=clips), ensure_ascii=False).replace("</", "<\\/")
    open(out, "w", encoding="utf-8").write(PAGE.replace("__DATA__", data))
    print("검수 페이지: %s (클립 %d개)" % (out, len(clips)))


if __name__ == "__main__":
    main()
