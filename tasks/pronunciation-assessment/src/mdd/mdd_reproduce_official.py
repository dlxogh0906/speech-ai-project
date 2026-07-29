# -*- coding: utf-8 -*-
"""공식 0.96 재현: 공식 모델 × TAG 전사자, 공식 파이프라인(edlib 3자 정렬 + weighted F1) 그대로.
  canonical = LAB phonemic.phones(표준)
  manual    = TAG phones 티어(전사자=실현형)  ← 정답
  predicted = 공식 인식기 디코드
공식 calculate_f1.py 방식(버그 포함) + clean 방식 둘 다 출력.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT

import sys, os, glob, json, re, argparse
import numpy as np
import edlib
from sklearn.metrics import f1_score, confusion_matrix
from mdd.mdd_eval_official import load_official, load_wav, decode

# 유럽어권(eu 체크포인트 매칭) langcode
EU = {"PL","PT","DE","IT","FR","ES","RU","NL","CS","RO","BG","HU","EL","SV","FI",
      "DA","NO","UK","HR","SR","SK","SL","LT","LV","ET","TR","IS","GA","MT","SQ","MK","BE","CA","EU"}

def get_canon(lab):
    d = json.load(open(lab, encoding="utf-8"))
    ph = d.get("RecordingMetadata", {}).get("phonemic", {})
    if not ph or "phones" not in ph:
        return None
    return [x["phone"] for x in ph["phones"]]

def get_manual(tg):
    try: txt = open(tg, encoding="utf-16").read()
    except:
        try: txt = open(tg, encoding="utf-8").read()
        except: return None
    m = re.search(r'name = "phones".*?(?=name = "|\Z)', txt, re.S)
    if not m: return None
    return [s for s in re.findall(r'text = "([^"]*)"', m.group(0)) if s.strip() and s not in (".", "-")]

# ── 공식 align_result.py align() 그대로 ──
def align(t1, t2, t3):
    t1 = t1.replace('spn', 's').replace('<unk>', 'u').replace(' ', '')
    t2 = t2.replace(' ', ''); t3 = t3.replace(' ', '')
    r = edlib.align(t1, t2, task="path"); a = edlib.getNiceAlignment(r, t1, t2)
    t1_t2, t2_t1 = a["query_aligned"], a["target_aligned"]
    r = edlib.align(t1, t3, task="path"); a = edlib.getNiceAlignment(r, t1, t3)
    t1_t3, t3_t1 = a["query_aligned"], a["target_aligned"]
    r = edlib.align(t1_t2, t1_t3, task="path"); a = edlib.getNiceAlignment(r, t1_t2, t1_t3)
    t1_t2_d = a["query_aligned"]
    r = edlib.align(t1_t2_d, t2, task="path"); a = edlib.getNiceAlignment(r, t1_t2_d, t2)
    q, tgt2 = a["query_aligned"], a["target_aligned"]
    r = edlib.align(t1_t2_d, t3, task="path"); a = edlib.getNiceAlignment(r, t1_t2_d, t3)
    tgt3 = a["target_aligned"]
    return q, tgt2, tgt3   # canonical, manual, predicted (aligned)

# ── 공식 calculate_f1.py 길이보정 ──
def fixup(can, aud, asr):
    if not (len(can) == len(aud) == len(asr)):
        if len(can) == len(aud):
            n = len(can) - len(asr)
            if n > 0: asr = (re.sub('-', '--', asr, n) if '-' in asr else asr + '-'*n)
            else: asr = (re.sub('--', '-', asr, abs(n)) if '--' in asr else re.sub('-', '', asr, abs(n)))
        elif len(can) == len(asr):
            n = len(can) - len(aud)
            if n > 0: aud = (re.sub('-', '--', aud, n) if '-' in aud else aud + '-'*n)
            else: aud = (re.sub('--', '-', aud, abs(n)) if '--' in aud else re.sub('-', '', aud, abs(n)))
        elif len(aud) == len(asr):
            n = len(aud) - len(can)
            if n > 0: can = (re.sub('-', '--', can, n) if '-' in can else can + '-'*n)
            else: can = (re.sub('--', '-', can, abs(n)) if '--' in can else re.sub('-', '', can, abs(n)))
    return can, aud, asr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_official_ckpt/checkpoint_mdd_eu.pt")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    print("[1] triple 수집(유럽어)...", flush=True)
    tags = {os.path.basename(p).split(".")[0]: p for p in glob.glob(f"{DATA_ROOT}/aihub/**/*pronunciation_tag*/*.TextGrid", recursive=True)}
    labs = {os.path.basename(p).split(".")[0]: p for p in glob.glob(f"{DATA_ROOT}/aihub/**/*pronunciation_lab*/*.json", recursive=True)}
    def wp(st): return f"{DATA_ROOT}/mdd_gold_wav16k/{st[:2]}/{st}.wav"
    stems = sorted(st for st in tags if st in labs and st.split("-")[3] in EU and os.path.exists(wp(st)))
    stems = stems[:a.n * 2]  # 여유(파싱실패 대비)

    data = []
    for st in stems:
        C = get_canon(labs[st]); M = get_manual(tags[st])
        if not C or not M: continue
        data.append({"stem": st, "wav": wp(st), "canon": C, "manual": M})
        if len(data) >= a.n: break
    print(f"[유효 triple] {len(data)}", flush=True)

    print("[2] 공식 모델 로드 + 디코드...", flush=True)
    model, dic, norm = load_official(a.ckpt)
    B = 8
    for i in range(0, len(data), B):
        batch = data[i:i+B]
        wavs = [load_wav(d["wav"]) for d in batch]
        outs = decode(model, dic, wavs, norm)
        for d, o in zip(batch, outs): d["pred"] = o
        if i % 80 == 0: print(f"  decode {i}/{len(data)}", flush=True)

    print("[3] 정렬 + F1...", flush=True)
    Xg, Yg = [], []      # 공식(버그 포함)
    Tc, Pc = [], []      # clean(위치정렬)
    for d in data:
        can = "".join(d["canon"]); man = "".join(d["manual"]); pred = "".join(d.get("pred", [])) or "-"
        n1, n2, n3 = align(can, man, pred)
        n1, n2, n3 = fixup(n1, n2, n3)
        L = min(len(n1), len(n2), len(n3))
        # clean: 위치별 정렬 라벨
        for i in range(L):
            Tc.append('1' if n1[i] != n2[i] else '0')
            Pc.append('1' if n1[i] != n3[i] else '0')
        # official: 버그 그대로(매치 local, 미스매치 global)
        x, y = [], []
        for i in range(L):
            if n1[i] == n2[i]: x.append('0')
            else: Xg.append('1')
            if n1[i] == n3[i]: y.append('0')
            else: Yg.append('1')
        Xg += x; Yg += y

    f1_off = f1_score(Xg, Yg, average='weighted')
    f1_cln = f1_score(Tc, Pc, average='weighted')
    print("\n" + "=" * 55)
    print(f"공식 모델 × TAG 전사자 · {len(data)}발화 (유럽어, eu ckpt)")
    print(f"  ▶ 공식 방식 F1 (calculate_f1.py 그대로, 버그포함): {f1_off:.4f}")
    print(f"  ▶ clean 위치정렬 F1                              : {f1_cln:.4f}")
    print(f"  (공식 발표: 0.96)")
    print("=" * 55)

if __name__ == "__main__":
    main()
