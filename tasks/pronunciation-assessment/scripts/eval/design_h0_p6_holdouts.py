"""Materializable multi-prompt development folds and supervised ancestry audit.

No audio inference, new labels, training, or claim of a sealed final dataset.
Uses existing split metadata; holdout assignment never reads model predictions.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse,collections,gzip,hashlib,json,re,time
import numpy as np
from train_h0_backbone_control import file_sha
from audit_h0_backbone_protocol import calibration_role
from audit_h0_p5_readiness import write
from prepare_h0_p5_balance_design import tsv


def norm(s):
    return re.sub('[^가-힣]','',s)


def families(texts):
    parent={k:k for k in texts}
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    keys=sorted(texts);edges=[]
    grams={k:{texts[k][j:j+12] for j in range(max(0,len(texts[k])-11))} for k in keys}
    for i,k in enumerate(keys):
        for j in keys[i+1:]:
            if texts[k] and texts[j] and (texts[k] in texts[j] or texts[j] in texts[k] or grams[k]&grams[j]):
                parent[root(j)]=root(k);edges.append([k,j])
    grouped=collections.defaultdict(list)
    for k in keys:grouped[root(k)].append(k)
    return sorted(grouped.values(),key=lambda x:tuple(x)),edges


def trie_add(trie,text,item):
    node=trie
    for c in text:node=node.setdefault(c,{})
    node.setdefault('',set()).add(item)


def trie_matches(trie,text):
    matches=set()
    for start in range(len(text)):
        node=trie
        for c in text[start:]:
            if c not in node:break
            node=node[c];matches.update(node.get('',()))
    return matches


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--backbone-protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    p=json.loads(a.backbone_protocol.read_text());m=Path(p['stage2']['manifest']);native=Path(p['stage1']['manifest'])
    bindings={str(a.backbone_protocol):file_sha(a.backbone_protocol)};metadata={};items={};spksets={}
    for split in ('valid','valid_unseen','test','test_unseen'):
        path=m/(split+'.meta.jsonl');bindings[str(path)]=file_sha(path)
        records=[json.loads(line) for line in path.open()];metadata[split]=records;spksets[split]={x['spk'] for x in records}
        for r in records:
            text=norm(r['prompt'])
            if r['item'] in items:assert items[r['item']]['norm']==text
            items[r['item']]=dict(prompt=r['prompt'],norm=text,task=r['task'])
    old=set(r['item'] for s in ('valid_unseen','test_unseen') for r in metadata[s])
    groups,edges=families({k:r['norm'] for k,r in items.items()})
    # Entire connected components touching historical unseen prompts stay diagnostic.
    diagnostic=set(k for g in groups if set(g)&old for k in g)
    candidates=[g for g in groups if not set(g)&diagnostic and all(items[k]['task'] in ('RS','RW') for k in g)]
    # Deterministic balancing of counts of scripts/words, never of model scores.
    seed='h0-p6-multiholdout-20261029';folds=[[] for _ in range(5)];counts=np.zeros((5,2),dtype=int)
    target=np.array([sum(items[k]['task']==task for g in candidates for k in g)/5 for task in ('RS','RW')])
    candidates.sort(key=lambda g:(-len(g),hashlib.sha256((seed+','.join(g)).encode()).hexdigest()))
    for g in candidates:
        v=np.array([sum(items[k]['task']==task for k in g) for task in ('RS','RW')])
        f=min(range(5),key=lambda j:(float(np.sum((counts[j]+v)**2/np.maximum(target,1))),j))
        folds[f].extend(g);counts[f]+=v
    owner={k:f for f,g in enumerate(folds) for k in g};plan=[]
    for f in range(5):
        outer=set(folds[f]);inner=set(folds[(f+1)%5]);train=set(owner)-outer-inner
        assert not (outer&inner or outer&train or inner&train)
        for k in outer|inner:
            assert not any(items[k]['norm'] in items[j]['norm'] or items[j]['norm'] in items[k]['norm'] for j in train)
        plan.append(dict(fold=f,outer_items=sorted(outer),inner_selection_items=sorted(inner),train_items=sorted(train)))
    write(a.out/'protocol.json',dict(created=time.time(),source_sha256=file_sha(__file__),bindings=bindings,
        fold_seed=seed,assignment='five components folds; full-string containment or shared 12-Hangul substring; balance RS/RW counts only',
        roles='old train speakers for Stage2; old valid calibration-role speakers and TRAIN prompts for CAL; old valid selection-role speakers and inner prompts for method selection; old test speakers and outer prompts for cross-fold development estimation',
        ancestry='restart each fold from public SSL, rebuild Stage1 and Stage2 after excluding outer and inner text matches; old H0/M2/M3 cannot be strict-unseen warm starts',
        text_exclusion='full heldout string contained in training text, or any shared 12-Hangul substring for longer prompts; conservative string screen, not semantic equivalence',
        old_diagnostics=sorted(diagnostic),folds=plan,
        metrics=['fixed-full-known denominator exact diagnosis and FPR bounds','observed conditional FPR/recall','normal/error Coverage separately','substitution/deletion/insertion exact match','all 46 phones, positions, L1, script and word groups; zero support remains undefined'],
        selection='CAL fits threshold once per fixed method/fold; never retune on outer/RP; no posthoc choice of best fold or suri result',
        statistical_unit='report all 5 fold and all prompt results; paired speaker and prompt resampling conditional on fitted models; folds have overlapping training so do not treat five scores as independent runs',
        original_label_rules='preserve unknown and ambiguous annotation; readiness of future manifests must verify coverage and label parsing before training',
        limits=['existing analyzed data, not new final independent confirmation','public SSL exposure unknown','native/foreign cross-corpus identity and approximate PCM aliases not yet fully audited','metadata selection plan does not execute Stage1/Stage2 or determine model hyperparameters'],
        training_allowed=False,model_optimizer_updates=0,model_inference_count=0))
    # Exact row selections over preserved manifests; archive counts and hashes.
    sources=(m/'train.source').read_text().splitlines();train_rows=[];train_speakers=set();replay=set()
    path=m/'train.tsv';bindings[str(path)]=file_sha(path);bindings[str(m/'train.source')]=file_sha(m/'train.source')
    with path.open() as f:
        next(f)
        for i,line in enumerate(f):
            uid=Path(line.split('\t')[0]).stem
            if sources[i]=='native':replay.add(uid);train_rows.append((None,None,uid))
            else:
                z=uid.split('-');train_rows.append((z[5],z[0],uid));train_speakers.add(z[0])
    for s,spks in spksets.items():assert not train_speakers&spks
    assert not (spksets['valid']|spksets['valid_unseen'])&(spksets['test']|spksets['test_unseen'])
    # Search all native TRAIN texts, without using native audio or any model output.
    trie={};ngram={}
    for k,r in items.items():
        trie_add(trie,r['norm'],k)
        for j in range(max(0,len(r['norm'])-11)):ngram.setdefault(r['norm'][j:j+12],set()).add(k)
    overlap=collections.defaultdict(collections.Counter);native_exclusions=[[] for _ in range(5)];native_drop_sets=[set() for _ in range(5)]
    forbidden_sets=[set(f['outer_items']+f['inner_selection_items'])|diagnostic for f in plan]
    native_path=native/'train.meta.jsonl';bindings[str(native_path)]=file_sha(native_path)
    native_tsv=(native/'train.tsv').read_text().splitlines()[1:]
    for suffix in ('tsv','ltr','meta.jsonl'):
        for split in ('train','dev'):
            path=native/(split+'.'+suffix);bindings[str(path)]=file_sha(path)
    with gzip.open(a.out/'native_matches.jsonl.gz','wt') as fw:
        for i,line in enumerate(native_path.open()):
            r=json.loads(line);text=norm(r.get('text',r['norm_text']));uid=Path(r['rel']).stem
            assert uid==Path(native_tsv[i].split('\t')[0]).stem
            full=trie_matches(trie,text);shared=set()
            for j in range(max(0,len(text)-11)):shared.update(ngram.get(text[j:j+12],()))
            matches=full|shared
            for k in matches:
                overlap[k]['native_full']+=int(k in full);overlap[k]['native_12gram']+=int(k in shared)
                overlap[k]['replay_full']+=int(uid in replay and k in full)
            if matches:fw.write(json.dumps(dict(row=i,uid=uid,full=sorted(full),shared_12gram=sorted(shared)),ensure_ascii=False)+'\n')
            for fold in plan:
                f=fold['fold']
                if matches&forbidden_sets[f]:native_exclusions[f].append(i);native_drop_sets[f].add(uid)
            if (i+1)%100000==0:print('native metadata rows',i+1,flush=True)
    native_count=i+1;assert native_count==len(native_tsv)
    dev_exclusions=[[] for _ in range(5)];dev_tsv=(native/'dev.tsv').read_text().splitlines()[1:]
    for j,line in enumerate((native/'dev.meta.jsonl').open()):
        r=json.loads(line);text=norm(r.get('text',r['norm_text']));assert Path(r['rel']).stem==Path(dev_tsv[j].split('\t')[0]).stem
        matches=trie_matches(trie,text)
        for q in range(max(0,len(text)-11)):matches.update(ngram.get(text[q:q+12],()))
        for f in range(5):
            if matches&forbidden_sets[f]:dev_exclusions[f].append(j)
    assert j+1==len(dev_tsv)
    reports=[];phone_support=[]
    alphabet=[line.split()[0] for line in (m/'dict.ltr.txt').read_text().splitlines() if len(line.split()[0])==1 and 'ㄱ'<=line.split()[0]<='ㅣ']
    for fold in plan:
        f=fold['fold'];folder=a.out/('fold_%d'%f);folder.mkdir();tr=set(fold['train_items']);inn=set(fold['inner_selection_items']);outer=set(fold['outer_items'])
        native_drop=native_drop_sets[f]
        indices={'stage2_train':np.array([j for j,(item,_,uid) in enumerate(train_rows) if (item in tr) or (item is None and uid not in native_drop)],dtype=np.int64),
            'stage1_excluded':np.array(native_exclusions[f],dtype=np.int64),
            'stage1_dev_excluded':np.array(dev_exclusions[f],dtype=np.int64),
            'cal_valid':np.array([j for j,r in enumerate(metadata['valid']) if r['item'] in tr and calibration_role(r['spk'])=='calibration'],dtype=np.int64),
            'inner_valid':np.array([j for j,r in enumerate(metadata['valid']) if r['item'] in inn and calibration_role(r['spk'])=='selection'],dtype=np.int64),
            'outer_test':np.array([j for j,r in enumerate(metadata['test']) if r['item'] in outer],dtype=np.int64)}
        np.savez_compressed(folder/'row_indices.npz',**indices)
        stat=dict(fold=f,train_items=len(tr),inner_items=len(inn),outer_items=len(outer),
            train_by_task=dict(collections.Counter(items[k]['task'] for k in tr)),
            outer_by_task=dict(collections.Counter(items[k]['task'] for k in outer)),
            stage1_total=native_count,stage1_excluded=len(native_exclusions[f]),stage1_retained=native_count-len(native_exclusions[f]),
            row_counts={k:len(v) for k,v in indices.items()},row_indices_sha256=file_sha(folder/'row_indices.npz'),
            historical_stage2_outer_item_membership=len(outer),historical_stage1_outer_full_text_items_membership=sum(overlap[k]['native_full']>0 for k in outer),
            stage1_dev_total=len(dev_tsv),stage1_dev_excluded=len(dev_exclusions[f]))
        for role,split in [('cal_valid','valid'),('inner_valid','valid'),('outer_test','test')]:
            rs=[metadata[split][j] for j in indices[role]];stat[role]=dict(speakers=len({r['spk'] for r in rs}),items=len({r['item'] for r in rs}),L1=len({r['stem'].split('-')[3] for r in rs}))
            # Support only, no derived performance or relabeling.
            counter=collections.Counter();unknown=0;deletion=0
            for r in rs:
                for _,_,label in r['tag_phones']:
                    token=label.strip()
                    if token=='-':deletion+=1
                    elif token and all(c in alphabet for c in token):counter.update(token)
                    elif token:unknown+=1
            stat[role].update(explicit_deletion_segments=deletion,unknown_segments=unknown)
            for phone in alphabet:phone_support.append(dict(fold=f,role=role,phone=phone,actual_TAG_token_count=counter[phone]))
        reports.append(stat)
    tsv(a.out/'phone_support.tsv',phone_support)
    tsv(a.out/'prompt_ancestry.tsv',[dict(item=k,task=r['task'],text=r['prompt'],role='historical_diagnostic' if k in diagnostic else 'fold_%s'%owner.get(k),
        historical_stage2_member=k not in old,**{name:overlap[k][name] for name in ('native_full','native_12gram','replay_full')}) for k,r in sorted(items.items())])
    write(a.out/'report.json',dict(folds=reports,candidate_items=len(owner),family_components=len(candidates),containment_edges=edges,
        all_items=len(items),diagnostic_items=sorted(diagnostic),speaker_intersections=0,bindings=bindings,
        materialization='indices bind existing manifests; native exclusions apply before Stage1/replay sampling',
        missing_before_training=['PCM/alias checks beyond UID','source-label and partial coverage validation','compute budget, fixed method hypotheses and hyperparameters'],
        independent_final_evaluation=False,training_allowed=False,model_optimizer_updates=0))
    write(a.out/'finished.json',dict(status='design_and_metadata_audit_complete_no_training',report_sha256=file_sha(a.out/'report.json')))
    print(json.dumps(reports,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
