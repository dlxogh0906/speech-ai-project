"""Unscored raw-TAG prompt inventory outside M2/H0 item lists, metadata only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import collections
import gzip
import hashlib
import json
import wave
from diagnose_h0_p4 import write_json
from train_h0_backbone_control import file_sha


def manifest_items(path):
    items=set()
    with Path(path).open() as f:
        next(f)
        for line in f:
            parts=Path(line.split('\t')[0]).stem.split('-')
            if len(parts)==7:items.add(parts[5])
    return items


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);ap.add_argument('--m2-manifest',required=True)
    a=ap.parse_args();root=Path(a.root);p=json.loads((root/'protocol.json').read_text())
    manifest=Path(p['manifest']);stats=json.loads((manifest/'stats.json').read_text());cache=Path(stats['args']['cache'])
    m2=Path(a.m2_manifest);seen=manifest_items(m2/'train.tsv')
    known=manifest_items(manifest/'train.tsv')
    for split in ('valid','valid_unseen','test','test_unseen'):known |= manifest_items(manifest/(split+'.tsv'))
    counts=collections.defaultdict(collections.Counter);speakers=collections.defaultdict(set);examples=collections.defaultdict(list);prompts={};n=0
    with gzip.open(cache,'rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);n+=1;item=r.get('item')
            if not item or item in seen or item in known or r.get('fail'):continue
            counts[item]['records']+=1;speakers[item].add(r['spk']);prompts[item]=r.get('prompt')
            if r.get('dropped'):counts[item]['with_dropped_annotation']+=1
            if not r.get('canon') or not r.get('ltr'):counts[item]['empty_sequences']+=1
            split=int(hashlib.md5(r['spk'].encode()).hexdigest()[:8],16)%10
            if split==0:
                counts[item]['historical_test_speaker_records']+=1
                digest=hashlib.sha256(r['stem'].encode()).hexdigest()
                examples[item].append((digest,dict(uid=r['stem'],speaker=r['spk'],item=item,source_dataset=r['ds'],
                    source_part=r['part'],canonical_tokens=len(r.get('canon',[])),actual_tokens=len(r.get('ltr',[])),
                    dropped=r.get('dropped'),tag_end=max([t[1] for t in r.get('tag_phones',[])] or [0]),
                    audio_path=str(Path(stats['args']['wav16k'])/r['stem'][:2]/(r['stem']+'.wav')))))
                examples[item]=sorted(examples[item],key=lambda x:x[0])[:8]
    out=[]
    for item,c in sorted(counts.items()):
        samples=[]
        for _,e in examples[item]:
            path=Path(e['audio_path']);e['audio_exists']=path.exists();e['model_scored']=False;e['audio_listened']=False
            if path.exists():
                with wave.open(str(path),'rb') as w:e['audio_seconds']=w.getnframes()/w.getframerate()
            samples.append(e)
        out.append(dict(item=item,prompt=prompts[item],speakers=len(speakers[item]),**dict(c),unscored_examples=samples))
    write_json(root/'raw_source_candidate_inventory.json',dict(status='metadata_inventory_not_validated_new_dataset',
        parsed_records=n,m2_train_items=len(seen),h0_all_items=len(known),outside_items=out,
        new_audio_collected=0,new_model_inferences=0,source_sha256=file_sha(__file__),
        bindings={str(f):file_sha(f) for f in (cache,m2/'train.tsv',manifest/'stats.json')},
        limitations=['outside item IDs alone do not prove new text, no substring overlap, or independent speakers',
            'these are existing annotations; original long-form prompts may be excluded by duration/quality constraints',
            'sample WAV headers only; no listening, segment extraction, or label validation',
            'not a sealed final test: historical research exposure, text overlap, speaker aliases, native data and segmentation must be audited before use']))


if __name__=='__main__':main()
