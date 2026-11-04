"""범용 음성 인식 기준선: Whisper 받아쓰기 → KoG2P 발음 자모열 → `eval_actual_ctc.py score` 입력.

Whisper(기본 large-v3, language=ko)가 글자로 받아 적은 문장을 KoG2P 로 표준 발음 자모열로 바꿔,
CTC 모델과 같은 `<out>/whisper_<split>.preds.jsonl` 형식으로 쓴다. 받아쓰기는 `.text.jsonl` 에 이어 쓰므로 중단 후 다시 돌리면 이어서 한다.
숫자·영문은 KoG2P 가 읽지 못해 빠진다(2026-11-03 test_unseen 에서 4.4%, 빼고 재도 결론 같음 — results/24).

Whisper 는 transformers ≥ 4.36 이 필요해 저장소 기본 $PY(3.8) 환경이 아니라 별도 환경으로 돌렸다:
    <whisper 환경 python> baseline_whisper_kog2p.py --out <출력 폴더>
    $PY eval_actual_ctc.py score --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 --split test_unseen \
        --ckpt <출력 폴더>/whisper_large_v3_plus_kog2p.ckpt --name whisper_v3_kog2p --out <출력 폴더>/whisper_test_unseen
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
from shared.paths import MDD_OUTPUTS, G2P_TOOLS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)

import argparse
import json
import os
import re
import wave


def read_wav(p):
    import numpy as np
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, default=MDD_OUTPUTS / 'manifest_foreign_clean_v2')
    ap.add_argument('--split', default='test_unseen')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='openai/whisper-large-v3')
    ap.add_argument('--cache-dir', default=None, help='HF 캐시 폴더 (없으면 기본)')
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # KoG2P 는 import 시점에 argv 를 파싱한다 — 잠시 비우고 import (src/mdd/pipeline.py 와 같은 패턴)
    sys.path.insert(0, str(G2P_TOOLS / 'KoG2P'))
    argv, sys.argv = sys.argv, sys.argv[:1]
    from g2p import readRules, graph2prono, ver_info
    sys.argv = argv
    from mdd.jamo_to_kog2p import phones_to_jamo, INV

    lines = open(a.manifest / (a.split + '.tsv'), encoding='utf-8').read().splitlines()
    root, rows = lines[0], [l.split('\t') for l in lines[1:]]
    raw = a.out / ('whisper_%s.text.jsonl' % a.split)
    done = {}
    if raw.exists():
        for l in open(raw, encoding='utf-8'):
            o = json.loads(l); done[o['i']] = o['text']
    todo = [i for i in range(len(rows)) if i not in done]
    if todo:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if dev == 'cuda' else torch.float32
        proc = WhisperProcessor.from_pretrained(a.model, cache_dir=a.cache_dir)
        model = WhisperForConditionalGeneration.from_pretrained(a.model, cache_dir=a.cache_dir, torch_dtype=dtype).to(dev).eval()
        with open(raw, 'a', encoding='utf-8') as f:
            for k in range(0, len(todo), a.batch):
                ids = todo[k:k + a.batch]
                feats = proc([read_wav(os.path.join(root, rows[i][0])) for i in ids], sampling_rate=16000,
                             return_tensors='pt').input_features.to(dev, dtype)
                with torch.no_grad():
                    gen = model.generate(feats, language='ko', task='transcribe', max_new_tokens=64)
                for i, t in zip(ids, proc.batch_decode(gen, skip_special_tokens=True)):
                    f.write(json.dumps({'i': i, 'text': t.strip()}, ensure_ascii=False) + '\n'); done[i] = t.strip()
                if k % (a.batch * 40) == 0: print('[whisper] %d/%d' % (k + len(ids), len(todo)), flush=True)

    ri, ro = readRules(ver_info[0], str(G2P_TOOLS / 'KoG2P' / 'rulebook.txt'))

    def g2p(text):
        t = re.sub(r'\s+', ' ', re.sub(r'[^가-힣\s]', ' ', text or '')).strip()
        if not t: return []
        out = []
        for j in phones_to_jamo([p for p in graph2prono(t, ri, ro).split() if p in INV]):
            out += {'ㅙ': ['ㅗ', 'ㅐ'], 'ㅞ': ['ㅜ', 'ㅔ']}.get(j, [j])
        return out

    preds = a.out / ('whisper_%s.preds.jsonl' % a.split)
    with open(preds, 'w', encoding='utf-8') as f:
        for i in range(len(rows)): f.write(json.dumps({'i': i, 'pred': g2p(done[i])}, ensure_ascii=False) + '\n')
    dummy = a.out / 'whisper_large_v3_plus_kog2p.ckpt'   # 채점기 캐시 메타가 요구하는 가짜 체크포인트 (음향 체크포인트 아님)
    dummy.write_text('Whisper (%s, language=ko) + KoG2P pronunciation; no acoustic checkpoint\n' % a.model)
    meta = {'ckpt': str(dummy.resolve()), 'ckpt_size': dummy.stat().st_size, 'n': len(rows),
            'first': os.path.join(root, rows[0][0]), 'last': os.path.join(root, rows[-1][0]), 'decoder': 'greedy_ctc_no_lm'}
    json.dump(meta, open(str(preds) + '.meta.json', 'w'))
    print('[whisper] preds written', preds, flush=True)


if __name__ == '__main__':
    main()
