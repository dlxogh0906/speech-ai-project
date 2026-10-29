"""Context-controlled TRAIN source audit; no acoustic or human labels invented."""
import collections
import functools
import hashlib
import numpy as np
from mdd.sibilant_partial_annotation import partial_mapping


def context(label,feature):
    return (label['item'],feature['q_runtime'],feature['canonical'],feature['expected_V'],
            label['position'],label.get('language'),label['uid'].split('-')[1])


def speaker_fold(speaker):
    return int(hashlib.sha256(('source-audit-speaker-v1:'+speaker).encode()).hexdigest(),16)%2


def matched_source_pairs(y,labels,features,offset):
    pools = collections.defaultdict(lambda:collections.defaultdict(list))
    used = []
    for i,lab in sorted(labels.items()):
        f = features[i]
        if (y[i] not in (0,1) or lab['label_kind'] not in ('correct','pair_error')
                or lab['expected']!=f['canonical'] or lab['following']!=f['expected_V']):
            continue
        key = context(lab,f);fold = speaker_fold(lab['speaker'])
        pools[key][(fold,int(i>=offset),int(y[i]))].append(i);used.append(i)
    pairs = [[],[]];quadruples = [[],[]]
    def priority(i):
        s = 'source-shift-pair-v1:'+labels[i]['uid']+':'+str(features[i]['q_runtime'])
        return hashlib.sha256(s.encode()).hexdigest(),i
    for key,cells in sorted(pools.items(),key=lambda z:repr(z[0])):
        for fold in (0,1):
            for cls in (0,1):
                old,new = [sorted(cells.get((fold,s,cls),[]),key=priority) for s in (0,1)]
                pairs[fold].extend(zip(old[:min(len(old),len(new))],new[:min(len(old),len(new))]))
            cells4 = [sorted(cells.get((fold,s,c),[]),key=priority) for s in (0,1) for c in (0,1)]
            n = min(map(len,cells4))
            quadruples[fold].extend(zip(*(z[:n] for z in cells4)))
    for fold,group in enumerate(pairs):
        for a,b in group:
            if not a<offset<=b or context(labels[a],features[a])!=context(labels[b],features[b]) or y[a]!=y[b]:
                raise ValueError('source pair not context/actual matched')
            if any(speaker_fold(labels[i]['speaker'])!=fold for i in (a,b)):
                raise ValueError('speaker fold leaked')
    speakers = [{labels[i]['speaker'] for pair in group for i in pair} for group in pairs]
    if speakers[0]&speakers[1]:raise ValueError('speaker split overlap')
    return [np.asarray(z,np.int64).reshape(-1,2) for z in pairs],quadruples,used


def auc(y,score):
    """Exact Mann-Whitney AUC, including tied and unobserved(.5) scores."""
    y,score = np.asarray(y),np.asarray(score)
    if len(y)!=len(score) or not np.isin(y,[0,1]).all() or not np.isfinite(score).all():
        raise ValueError('binary finite fixed population required')
    positives,negatives = int((y==1).sum()),int((y==0).sum())
    if not positives or not negatives:return None
    order = np.argsort(score,kind='mergesort');ranks = np.empty(len(y),float)
    start = 0
    while start<len(y):
        stop = start+1
        while stop<len(y) and score[order[stop]]==score[order[start]]:stop+=1
        ranks[order[start:stop]]=(start+1+stop)/2.;start=stop
    return float((ranks[y==1].sum()-positives*(positives+1)/2.)/(positives*negatives))


def context_lookup(indices,labels,features,fields):
    """Out-of-speaker-fold context-only error propensity, including unseen cells."""
    tables = [collections.defaultdict(collections.Counter) for _ in (0,1)]
    for i in indices:
        z = labels[i];fold = speaker_fold(z['speaker']);key = context(z,features[i])[:fields]
        tables[fold][key][int(z['label_kind']=='pair_error')]+=1
    y,pred,known,tokens = [],[],[],[]
    for i in indices:
        z = labels[i];fold = speaker_fold(z['speaker']);key = context(z,features[i])[:fields]
        cell = tables[1-fold].get(key,{});n = sum(cell.values())
        # Unseen cells stay in the denominator with chance prediction.
        pred.append((cell.get(1,0)+1.)/(n+2.));known.append(bool(n))
        y.append(int(z['label_kind']=='pair_error'));tokens.append(features[i]['canonical'])
    result = {'observations':len(indices),'observed_contexts':sum(known),
        'unseen_contexts_chance_kept':len(indices)-sum(known),'AUC':auc(y,pred),'by_expected':{}}
    for token in ('ㅅ','ㅆ'):
        mask = np.asarray(tokens)==token
        result['by_expected'][token] = {'n':int(mask.sum()),'AUC':auc(np.asarray(y)[mask],np.asarray(pred)[mask])}
    return result


@functools.lru_cache(maxsize=4096)
def _mapping(canonical,partial):
    return partial_mapping(canonical,partial)[0]


def prototype_normal_targets(row,canonical):
    """Source-stage selection only; runtime mapping/support must be rechecked."""
    options = _mapping(tuple(canonical),tuple(row['expected_partial']));by_partial = {}
    for site in row['sites']:
        if (site.get('source_time_witness') and site['label_kind']=='correct'
                and site.get('actual')==site['expected']):
            by_partial[site['partial_q']] = site
    result = []
    for q,token in enumerate(canonical[:-1]):
        if token not in ('ㅅ','ㅆ') or len(options[q])!=1:continue
        partial = next(iter(options[q]));site = by_partial.get(partial)
        if site is None or token!=site['expected'] or canonical[q+1]!=site['following']:continue
        if options[q+1]!={partial+1}:continue
        lab = dict(site,uid=row['uid'],item=row['item'],language=row.get('language'))
        feature = {'q_runtime':q,'canonical':token,'expected_V':canonical[q+1]}
        result.append((context(lab,feature),site['partial_q'],q))
    return result


def ridge_source_probe(x,pairs,seed=20261026,permutations=200):
    """Two speaker-fold transfers; fixedridge10, paired-label-swap null controls."""
    import torch
    torch.set_num_threads(2)
    results = []
    for test in (0,1):
        train = 1-test;ix = pairs[train].ravel();jx = pairs[test].ravel()
        if len(pairs[train])<100 or len(pairs[test])<100:
            results.append({'test_fold':test,'status':'insufficient_pairs'});continue
        a,b = np.asarray(x[ix],np.float64),np.asarray(x[jx],np.float64)
        mean,scale = a.mean(0),a.std(0);scale = np.where(scale>1e-6,scale,1.)
        a,b = (a-mean)/scale,(b-mean)/scale
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite observations')
        ta,tb = torch.from_numpy(a),torch.from_numpy(b)
        identity = torch.eye(a.shape[1],dtype=torch.float64)
        # Transform fitted exclusively on the opposite speaker fold.
        operator = torch.linalg.solve(ta.T@ta+10.*identity,ta.T)
        target = np.tile([-1.,1.],len(pairs[train]))
        rng = np.random.RandomState(seed+test)
        swaps = rng.choice([-1.,1.],(len(pairs[train]),permutations))
        null_y = np.stack([-swaps,swaps],axis=1).reshape(len(ix),permutations)
        targets = torch.from_numpy(np.column_stack([target,null_y]))
        scores = (tb@operator@targets).numpy()
        labels = np.tile([0,1],len(pairs[test]))
        correct = np.where(scores>0,labels[:,None]==1,
                           np.where(scores<0,labels[:,None]==0,.5)).astype(float).mean(0)
        paired = scores.reshape(len(pairs[test]),2,-1)
        margin = (paired[:,1]-paired[:,0]).mean(0)
        results.append({'test_fold':test,'status':'complete','train_pairs':len(pairs[train]),
            'test_pairs':len(pairs[test]),'balanced_accuracy':float(correct[0]),
            'AUC':auc(labels,scores[:,0]),'null_balanced_accuracy_mean':float(correct[1:].mean()),
            'null_balanced_accuracy_95percentile':float(np.quantile(correct[1:],.95)),
            'mean_new_minus_old_score_margin':float(margin[0]),
            'null_margin_mean':float(margin[1:].mean()),
            'null_margin_95percentile':float(np.quantile(margin[1:],.95)),
            'ridge_alpha':10.,'null_permutations':permutations})
    return results
