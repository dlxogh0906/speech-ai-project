# -*- coding: utf-8 -*-
"""MDD 제품 파이프라인 데모: 오디오 → 우리 모델 → 표준음 정렬 → 진단 리포트.
실제 TAG test 발화에서 '모델이 실제 오류를 잡은' 깔끔한 예시를 골라 JSON으로 덤프.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, glob, json, re
from mdd import mdd_eval
mdd_eval.CKPT=f"{DATA_ROOT}/mdd_ckpt_tag041/checkpoint_best.pt"
from mdd.mdd_eval import Decoder, load_wav, batched
from mdd.mdd_eval_pos import _nw

M=f"{MDD_OUTPUTS}/manifest_tag_041"
LAB_DIR=f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted/TL_pronunciation_lab"

VOWELS=set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
TENSE={'ㄱ':'ㄲ','ㄷ':'ㄸ','ㅂ':'ㅃ','ㅅ':'ㅆ','ㅈ':'ㅉ'}
ASP={'ㄱ':'ㅋ','ㄷ':'ㅌ','ㅂ':'ㅍ','ㅈ':'ㅊ'}
NASAL={'ㄱ':'ㅇ','ㄷ':'ㄴ','ㅂ':'ㅁ','ㅋ':'ㅇ','ㅌ':'ㄴ','ㅍ':'ㅁ'}
def ctype(t,a,kind):
    if kind=="del": return "음운탈락"
    if kind=="ins": return "음운첨가"
    if (t in VOWELS) or (a in VOWELS): return "모음오류"
    if TENSE.get(t)==a: return "경음화"
    if ASP.get(t)==a: return "격음화"
    if NASAL.get(t)==a: return "비음화"
    if {t,a}=={'ㄴ','ㄹ'}: return "유음화"
    return "자음오류"

def prompt_of(st):
    p=os.path.join(LAB_DIR, st+".json")
    if not os.path.exists(p): return ""
    try:
        d=json.load(open(p,encoding="utf-8"))
        return d.get("RecordingMetadata",{}).get("prompt","") or ""
    except: return ""

def main():
    rel=[l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
    canon=[l.split() for l in open(f"{M}/test.canon").read().splitlines()]
    manual=[l.split() for l in open(f"{M}/test.ltr").read().splitlines()]
    K=3000
    data=[{"st":os.path.basename(rel[i][0])[:-4],"wp":"/"+rel[i][0],"fr":int(rel[i][1]),
           "C":canon[i],"MAN":manual[i]} for i in range(min(K,len(rel)))]
    print(f"[디코드] {len(data)}발화...",flush=True)
    dec=Decoder()
    items=[(d["fr"],d) for d in data]
    for bi,idxs in enumerate(batched(items)):
        outs=dec.decode([load_wav(items[i][1]["wp"]) for i in idxs])
        for j,i in enumerate(idxs): items[i][1]["P"]=outs[j]

    picks=[]
    for d in data:
        C,MAN,P=d["C"],d["MAN"],d.get("P",[])
        if not C or not P: continue
        # 정답(전사자) 기준 진짜 오류
        gold={op["tpos"]:op for op in _nw(C,MAN) if op["kind"] in("sub","del")}
        # 모델 예측 오류
        pred_ops=[op for op in _nw(C,P) if op["kind"] in("sub","del")]
        pred={op["tpos"]:op for op in pred_ops}
        if not(1<=len(gold)<=2): continue          # 오류 1~2개 깔끔한 것
        # 모델이 그 오류를 잡았나 (위치+실현 일치)
        hit=sum(1 for pos in gold if pos in pred and pred[pos].get("actual")==gold[pos].get("actual"))
        if hit<len(gold): continue                 # 완전히 맞춘 것만(데모 신뢰성)
        pr=prompt_of(d["st"])
        if not pr or len(C)>14: continue
        errs=[]
        for pos in sorted(gold):
            op=gold[pos]
            errs.append({"pos":pos,"target":op["target"],
                         "actual":op.get("actual") or "∅","kind":op["kind"],
                         "type":ctype(op["target"],op.get("actual") or "",op["kind"])})
        picks.append({"prompt":pr,"canon":C,"model":P,"errors":errs,"stem":d["st"]})
        if len(picks)>=8: break

    out="/tmp/claude-7778/-home-tta/86db1df8-e52d-4e72-89e0-2554abdda961/scratchpad/demo_examples.json"
    json.dump(picks,open(out,"w"),ensure_ascii=False,indent=2)
    print(f"[완료] {len(picks)}개 예시 → {out}",flush=True)
    for p in picks:
        print(f"  · {p['prompt']}: "+", ".join(f"{e['target']}→{e['actual']}({e['type']})" for e in p['errors']),flush=True)

if __name__=="__main__":
    main()
