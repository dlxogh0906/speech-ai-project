"""Independently check saved row selections and count training phone support."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,collections,csv,hashlib,json,re
import numpy as np
from train_h0_backbone_control import file_sha
from audit_h0_p5_readiness import write
from prepare_h0_p5_balance_design import tsv


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--design',type=Path,required=True)
    ap.add_argument('--backbone-protocol',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.backbone_protocol.read_text());m=Path(p['stage2']['manifest']);native=Path(p['stage1']['manifest'])
    protocol=json.loads((a.design/'protocol.json').read_text());report=json.loads((a.design/'report.json').read_text())
    for name,h in report['bindings'].items():assert file_sha(name)==h
    with (a.design/'prompt_ancestry.tsv').open() as f:items={r['item']:r for r in csv.DictReader(f,delimiter='\t')}
    def norm(s):return re.sub('[^가-힣]','',s)
    patterns=[];indices=[];native_drops=[];train_sets=[]
    for fold in protocol['folds']:
        f=fold['fold'];indices.append(dict(np.load(a.design/('fold_%d'%f)/'row_indices.npz')))
        held=set(fold['inner_selection_items']+fold['outer_items']+protocol['old_diagnostics']);terms=set()
        for k in held:
            text=norm(items[k]['text']);terms.add(text)
            terms.update(text[j:j+12] for j in range(max(0,len(text)-11)))
        patterns.append(re.compile('|'.join(re.escape(t) for t in sorted(terms) if t)))
        train_sets.append(set(fold['train_items']));native_drops.append(set())
    alphabet=[line.split()[0] for line in (m/'dict.ltr.txt').read_text().splitlines() if len(line.split()[0])==1 and 'ㄱ'<=line.split()[0]<='ㅣ']
    token_counts=collections.defaultdict(collections.Counter);baselines=collections.defaultdict(collections.Counter)
    for split in ('train','dev'):
        excluded=[[] for _ in range(5)]
        with (native/(split+'.ltr')).open() as fl:
            for j,line in enumerate((native/(split+'.meta.jsonl')).open()):
                r=json.loads(line);text=norm(r.get('text',r['norm_text']));phones=fl.readline().split()
                baselines['stage1_'+split].update(phones)
                for f,regex in enumerate(patterns):
                    if regex.search(text):
                        excluded[f].append(j)
                        if split=='train':native_drops[f].add(Path(r['rel']).stem)
                    else:token_counts[f,'stage1_'+split].update(phones)
            assert not fl.readline()
        key='stage1_excluded' if split=='train' else 'stage1_dev_excluded'
        for f in range(5):np.testing.assert_array_equal(indices[f][key],excluded[f])
    sources=(m/'train.source').read_text().splitlines();expected=[[] for _ in range(5)];speakers=set();source_counts=[collections.Counter() for _ in range(5)]
    with (m/'train.tsv').open() as ft,(m/'train.ltr').open() as fl:
        next(ft)
        for j,line in enumerate(ft):
            uid=Path(line.split('\t')[0]).stem;src=sources[j];phones=fl.readline().split();baselines['stage2_'+src].update(phones)
            item=None if src=='native' else uid.split('-')[5]
            if item is not None:speakers.add(uid.split('-')[0])
            for f in range(5):
                if (item in train_sets[f]) or (item is None and uid not in native_drops[f]):
                    expected[f].append(j);source_counts[f][src]+=1;token_counts[f,'stage2_'+src].update(phones)
        assert not fl.readline()
    for f in range(5):np.testing.assert_array_equal(expected[f],indices[f]['stage2_train'])
    valid=[json.loads(line) for line in (m/'valid.meta.jsonl').open()];test=[json.loads(line) for line in (m/'test.meta.jsonl').open()]
    rolechecks=[]
    for fold in protocol['folds']:
        f=fold['fold'];sets=[]
        for name,records,allowed in [('cal_valid',valid,set(fold['train_items'])),('inner_valid',valid,set(fold['inner_selection_items'])),('outer_test',test,set(fold['outer_items']))]:
            selected=[records[j] for j in indices[f][name]];spks={r['spk'] for r in selected};sets.append(spks)
            assert not speakers&spks;assert {r['item'] for r in selected}<=allowed
            def role(s):return int(hashlib.sha256(('h0_backbone_v1:'+s).encode()).hexdigest()[:8],16)%2
            if name=='cal_valid':assert all(role(s)==0 for s in spks)
            if name=='inner_valid':assert all(role(s)==1 for s in spks)
            expected_rows=[j for j,r in enumerate(records) if r['item'] in allowed and (name=='outer_test' or role(r['spk'])==(name=='inner_valid'))]
            np.testing.assert_array_equal(indices[f][name],expected_rows)
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        rolechecks.append(dict(fold=f,speaker_overlap=0,source_counts=dict(source_counts[f])))
    table=[]
    for (f,stage),counts in sorted(token_counts.items()):
        for phone in alphabet:
            base=baselines[stage][phone];after=counts[phone]
            table.append(dict(fold=f,stage=stage,phone=phone,manifest_token_count=after,original_manifest_token_count=base,retained_fraction=after/base if base else None))
    tsv(a.design/'training_phone_support.tsv',table)
    write(a.design/'verification.json',dict(status='passed',native_exclusions='independent regular-expression search, exact index match',
        stage2_and_evaluation_indices='independently reconstructed, exact match',role_checks=rolechecks,
        token_scope='manifest token support, not optimizer exposure, not label quality or performance',
        model_optimizer_updates=0,independent_final_evaluation=False))
    print('holdout row, text, speaker-role and training support verification passed',flush=True)

if __name__=='__main__':main()
