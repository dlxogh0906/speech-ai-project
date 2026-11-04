"""Bounded TRAIN/CAL/test comparison of fixed-M3 candidate re-ranking."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT: raise RuntimeError('저장소 루트(pyproject.toml)를 찾지 못했습니다')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/data')]
from shared.paths import DATA_ROOT, MDD_OUTPUTS  # 경로는 .env 에서 (CLAUDE.md 경로 규칙)
import argparse, collections, csv, datetime, gzip, hashlib, json, shutil, time, wave
import numpy as np
from mdd.candidate_rerank import FAMILIES, features, budget_threshold, outputs
from mdd.h0_partial_reference import UNKNOWN, DELETED, partial_tokens, classify_truth, input_windows
from mdd.h0_partial_scoring import mapping_indices, value_options, annotation_mask

BASE = MDD_OUTPUTS
CKPT = DATA_ROOT/'checkpoints/w2vbert_synth_tense_v2/step_3000'
OUT = BASE/'experiments/m3_candidate_rerank_2026-11-01_v1'
RP = BASE/'experiments/h0_rp_stress_eval_2026-10-29_v2/annotations.jsonl.gz'
READINESS = BASE/'experiments/h0_p5_readiness_2026-10-28_v1/reserved_unscored_candidates.jsonl'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def emit(f, obj): f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')


def read_rows(p):
    with gzip.open(p, 'rt') as f: return list(map(json.loads, f))


def choose_tsv(folder, split, count, kinds=None):
    candidates = {}; all_speakers = set(); all_items = set()
    with (folder/(split+'.tsv')).open() as f:
        root = Path(next(f).strip())
        for n, line in enumerate(f):
            rel, frames = line.rstrip().split('\t'); path = root/rel; uid = path.stem
            parts = uid.split('-')
            if len(parts) != 7: continue  # native replay has no AIHub TAG
            all_speakers.add(parts[0]); all_items.add(parts[5])
            if kinds is not None and kinds[n] != 'real': continue
            if not parts[5].startswith(('RS', 'RW')): continue
            candidates[uid] = dict(uid=uid, path=str(path), frames=int(frames),
                                   speaker=parts[0], item=parts[5])
    ids = sorted(candidates, key=lambda u: hashlib.sha256(('rerank20261101:'+u).encode()).hexdigest())[:count]
    assert len(ids) == count
    return {u:candidates[u] for u in ids}, all_speakers, all_items


def label_record(m, path_record, vocabulary):
    ref, ro, ri = partial_tokens(m['lab_phones'], vocabulary, reference=True)
    act, ao, ai = partial_tokens(m['tag_phones'], vocabulary)
    idx = mapping_indices(ref, act); sites = []
    for q, (e, indices) in enumerate(zip(ref, idx)):
        opts = value_options(indices, act); status, actual = classify_truth(e, opts)
        sites.append(dict(index=q, expected=e, identified_actual=actual, status=status,
                          owner=ro[q]))
    r = dict(path_record, prompt=m['prompt'], reference=ref, actual=act,
             actual_owners=ao, sites=sites, ignored_reference=ri, ignored_actual=ai)
    r['annotation_mask'] = annotation_mask(r, idx)
    return r


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT/'tasks/pronunciation-assessment/src/mdd/candidate_rerank.py',
               ROOT/'tasks/pronunciation-assessment/src/mdd/h0_partial_reference.py',
               ROOT/'tasks/pronunciation-assessment/src/mdd/h0_partial_scoring.py',
               ROOT/'tasks/pronunciation-assessment/src/mdd/phone_probability_atlas.py']
    protocol = dict(created=datetime.datetime.now().astimezone().isoformat(), checkpoint=str(CKPT),
        conditions=['A_M3_peak', 'B_probability', 'C_segment'], families=FAMILIES,
        selection='UID SHA256 hash order, seed rerank20261101, no model-output or label-dependent selection',
        sizes=dict(train=3000, cal=2000, seen=1000, rp=1118),
        train='Unique real AIHub RS/RW recordings in M3 train manifest; retain all selected recordings',
        cal='Hash-selected M2 valid speakers; seen prompts; threshold only',
        seen='Hash-selected M2 test; held-out speakers, seen prompts; no tuning',
        rp='All reserved RP002/3/4; prompts outside M2/M3 supervised train; prior research test speakers; three-script stress, not independent final validation',
        input='Raw 16k PCM, no VAD or truncation; 12-second cores + 400ms halos; keep frame centers 280+320*i in core before CTC collapse; same preprocessing for all roles',
        anchor='Unique canonical-to-greedy occurrence across all minimum partial edit paths; emitted family phone peak; no TAG frame/time input',
        scope='Expected S/SS or J/JJ/CH; one emitted family candidate. Other-phone or deletion outputs/ambiguous mapping retained in fixed denominators as unavailable. This is a substitution reranker, not full API/deletion/insertion replacement.',
        features_B='family probabilities, log-probability gaps to top family candidate, blank/max probability, expected and greedy one-hots',
        features_C='B + consonant +/-60ms and immediately following canonical vowel peak +/-40ms posterior means/max/log gaps, emitted-run duration and vowel distance; raw high/low spectral ratio, centroid, zero crossings, autocorrelation proxies for both windows. These are not physical segment boundaries/VOT/F0.',
        head='Per family StandardScaler fit on TRAIN; multinomial logistic regression C=1, balanced class weights, lbfgs max_iter=2000 tol=1e-5; actual TAG family phone or OTHER. OTHER vetoes warning and keeps original emitted phone. No hyperparameter search.',
        score='A: emitted peak logP(pred)-logP(expected). B/C: selected class logP(pred)-logP(expected); require actual family winner different from expected and not OTHER. One final phone; no top-k oracle.',
        calibration='Each condition and family independently: maximize accepted positive-score proposals within floor(.01*observed CAL normal count), exclude boundary ties via nextafter. Never tune on seen/RP.',
        common_observation='Require same available family peak, fixed annotation boundary mask, no feature-window overlap with LAB/TAG unknown times +/-40ms, no feature window or LAB interval within40ms of input seam; no sample deletion',
        primary='C vs A and B on all RP. Proceed only if BOTH family FPR<=1% and total fixed-denominator exact diagnosis exceeds A, without lost coverage. Otherwise fail, no rescue search.',
        reporting='Raw single-phone accuracy, warning FPR, exact diagnosis, third-phone errors, total/observed denominators and bounds, per-family/phone/script, speaker bootstrap1000; no claim of unseen sentence population certainty',
        encoder_updates=0, independent_listening=False, production_changes=False,
        source_hashes={str(p):sha(p) for p in sources},
        checkpoint_hashes={p.name:sha(p) for p in CKPT.iterdir() if p.is_file()},
        input_hashes={str(p):sha(p) for p in [RP, READINESS]})
    write(out/'protocol.json', protocol)
    (out/'source').mkdir()
    for p in sources: shutil.copy2(p, out/'source'/p.name)
    vocab = json.loads((CKPT/'jamo_vocab.json').read_text())['vocab']
    phones = {c for c in vocab if len(c)==1 and '\u3131' <= c <= '\u3163'}
    full = BASE/'manifest_full_tag_replay10'; real = BASE/'manifest_synth_tense_v2'
    kinds = (real/'train.kind').read_text().splitlines()
    train, _, _ = choose_tsv(real, 'train', 3000, kinds)
    cal, _, _ = choose_tsv(full, 'valid', 2000)
    seen, _, _ = choose_tsv(full, 'test', 1000)
    _, train_speakers, train_items = choose_tsv(full, 'train', 1)
    groups = dict(train=train, cal=cal, seen=seen); counts = {}; recordings = {}
    for role in ('cal', 'seen'):
        selected = groups[role]; split = 'valid' if role=='cal' else 'test'; rows = []
        with (full/(split+'.meta.jsonl')).open() as f:
            for line in f:
                m = json.loads(line)
                if m['stem'] in selected: rows.append(label_record(m, selected[m['stem']], phones))
        recordings[role] = sorted(rows, key=lambda r:r['uid'])
        assert len(rows)==len(selected)
    from build_foreign_clean_v2 import parse_one
    dirs = sorted((DATA_ROOT/'aihub').glob('0[4][0-3].*/01-1.정식개방데이터/*/02.라벨링데이터/*_pronunciation_lab'))
    rows = []
    for r in train.values():
        found = [d/(r['uid']+'.json') for d in dirs if (d/(r['uid']+'.json')).exists()]
        assert len(found)==1, (r['uid'], found)
        lab = found[0]; part = lab.parent.name[:2]; tag = lab.parent.parent/(part+'_pronunciation_tag')/(r['uid']+'.TextGrid')
        m = parse_one((r['uid'], 'source', part, str(tag), str(lab)))
        assert 'fail' not in m, m
        z = label_record(m, r, phones); z['source_hashes']={str(p):sha(p) for p in (lab,tag)}; rows.append(z)
    recordings['train'] = sorted(rows, key=lambda r:r['uid'])
    rp_paths = {r['uid']:r for r in map(json.loads, READINESS.open())}
    recordings['rp'] = read_rows(RP)
    for r in recordings['rp']:
        p = rp_paths[r['uid']]; r.update(path=p['path'], frames=p['frames'], pcm_sha256=p['pcm_sha256'])
        assert r['item'] not in train_items and r['speaker'] not in train_speakers
        # Original text is read without changing frozen annotation sequences.
        lab = next(p for p in r['source_hashes'] if p.endswith('.json'))
        r['prompt'] = json.loads(Path(lab).read_text())['RecordingMetadata']['prompt']
    assert len(recordings['rp'])==1118
    ss = {role:{r['speaker'] for r in rows} for role, rows in recordings.items()}
    assert not (ss['train'] & ss['cal'] or ss['train'] & ss['seen'] or ss['train'] & ss['rp'] or ss['cal'] & ss['seen'] or ss['cal'] & ss['rp'])
    for role, rows in recordings.items():
        counts[role] = dict(recordings=len(rows), speakers=len(ss[role]), prompts=len({r['item'] for r in rows}),
            target_status=dict(collections.Counter(s['status'] for r in rows for s in r['sites'] if any(s['expected'] in f for f in FAMILIES.values()))))
        with gzip.open(out/(role+'_inputs.jsonl.gz'), 'wt') as f:
            for r in rows: emit(f,r)
    bindings={str(p):sha(p) for p in [full/'train.tsv', full/'valid.tsv', full/'test.tsv',
        full/'valid.meta.jsonl',full/'test.meta.jsonl',real/'train.tsv',real/'train.kind']}
    write(out/'prepared.json',dict(counts=counts, protocol_sha256=sha(out/'protocol.json'), input_hashes=bindings,
        selected_hashes={role:sha(out/(role+'_inputs.jsonl.gz')) for role in recordings},
        train_cal_test_speaker_overlap=0, seen_rp_speaker_overlap=len(ss['seen'] & ss['rp']),
        RP_M2_M3_train_prompt_overlap=0, RP_M2_train_speaker_overlap=0))
    print(json.dumps(counts, ensure_ascii=False), flush=True)


def audio_read(path):
    with wave.open(str(path),'rb') as w:
        assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
        pcm=w.readframes(w.getnframes())
    return np.frombuffer(pcm,'<i2').astype(np.float32)/32768.,hashlib.sha256(pcm).hexdigest()


def stitch(dec, audio):
    from mdd.gop_feats_v2 import forward_lprobs
    pieces=[]; centers=[]; windows=[]
    for w in input_windows(len(audio)):
        b,lens=forward_lprobs(dec,[audio[w['read_start']:w['read_end']]])
        lp=b[0,:int(lens[0])].numpy(); ts=w['read_start']+280+320*np.arange(len(lp))
        keep=(ts>=w['core_start'])&(ts<w['core_end'])
        pieces.append(lp[keep]);centers.extend(ts[keep].tolist());windows.append(w)
    assert all(b-a==320 for a,b in zip(centers,centers[1:]))
    return np.concatenate(pieces),np.array(centers),windows


def masked(r, q, z, audio_length):
    if r['annotation_mask'][q]: return r['annotation_mask'][q]
    if not z['available']: return z['reason']
    lo,hi=z['support_samples']
    unknown=[s['owner'] for s in r['sites'] if s['expected']==UNKNOWN]
    unknown += [o for p,o in zip(r['actual'],r['actual_owners']) if p==UNKNOWN]
    for o in unknown:
        if hi >= o['start']*16000-640 and lo <= o['end']*16000+640: return 'unknown_time_overlap'
    o=r['sites'][q]['owner']
    for seam in range(192000,audio_length,192000):
        if lo-640 <= seam <= hi+640 or o['start']*16000-640 <= seam <= o['end']*16000+640: return 'core_seam'
    return None


def extract(out, role, dec, vocab, smoke=False):
    rows=read_rows(out/(role+'_inputs.jsonl.gz')); started=time.time()
    if smoke: rows=rows[:4]
    target=out/(('smoke_' if smoke else '')+role+'_features.jsonl.gz')
    assert not target.exists()
    counts=collections.Counter();ledger=[]
    with gzip.open(target,'wt') as f:
        for n,r in enumerate(rows,1):
            audio,digest=audio_read(r['path']);assert len(audio)==r['frames']
            if 'pcm_sha256' in r: assert digest==r['pcm_sha256']
            lp,centers,windows=stitch(dec,audio)
            zs,hyp=features(audio,lp,centers,r['reference'],vocab,dec.blank)
            if smoke:
                # Exact repeat and short-input equivalence; altered actual labels cannot reach features().
                from mdd.gop_feats_v2 import forward_lprobs
                if len(windows)==1:
                    direct,lens=forward_lprobs(dec,[audio]);np.testing.assert_array_equal(lp,direct[0,:int(lens[0])].numpy())
                again,_=features(audio,lp,centers,r['reference'],vocab,dec.blank)
                assert zs==again
                np.savez_compressed(out/('smoke_'+r['uid']+'.npz'),lp=lp,centers=centers)
            for q,z in zs.items():
                s=r['sites'][q];reason=masked(r,q,z,len(audio))
                z.update(uid=r['uid'],speaker=r['speaker'],item=r['item'],prompt=r.get('prompt',''),
                    actual=s['identified_actual'],truth_status=s['status'],observation_reason=reason,
                    observed=reason is None, audio_path=r['path'])
                emit(f,z);counts[s['status']]+=1;counts['observed']+=int(reason is None)
            ledger.append(dict(uid=r['uid'],pcm_sha256=digest,audio_samples=len(audio),frames=len(lp),
                cores=len(windows),greedy=''.join(hyp)))
            if n%100==0:
                f.flush();status=dict(stage='extract_'+role,recordings=n,total=len(rows),seconds=round(time.time()-started,1))
                write(out/'live_status.json',status);print(json.dumps(status),flush=True)
    with gzip.open(out/(('smoke_' if smoke else '')+role+'_ledger.jsonl.gz'),'wt') as f:
        for r in ledger:emit(f,r)
    write(out/(('smoke_' if smoke else '')+role+'_finished.json'),dict(recordings=len(rows),counts=dict(counts),
        elapsed_seconds=time.time()-started,features_sha256=sha(target),encoder_updates=0))


def proposals(out, rows, condition, family, model=None):
    observed=[r for r in rows if r['observed'] and r['family']==family]
    if condition=='A_M3_peak':
        values=[(r['original'],r['original_score'] if r['original']!=r['expected'] else -np.inf,False) for r in observed]
    else: values=outputs(model,observed,'base' if condition=='B_probability' else 'segment') if observed else []
    return observed,values


def fit(out):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    train=read_rows(out/'train_features.jsonl.gz');cal=read_rows(out/'cal_features.jsonl.gz')
    models={};receipt={};thresholds={}
    for family,phones in FAMILIES.items():
        tr=[r for r in train if r['family']==family and r['observed']]
        y=np.array([r['actual'] if r['actual'] in phones else 'OTHER' for r in tr])
        assert set(phones).issubset(set(y))
        receipt[family]={'labels':dict(collections.Counter(y)), 'n':len(tr)}
        for cond in ('A_M3_peak','B_probability','C_segment'):
            model=None
            if cond!='A_M3_peak':
                key='base' if cond=='B_probability' else 'segment'
                x=np.array([r[key] for r in tr],dtype=float)
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',
                    solver='lbfgs',multi_class='multinomial',max_iter=2000,tol=1e-5,random_state=20261101))
                model.fit(x,y);assert model[-1].n_iter_.max()<2000
                path=out/(cond+'_'+family+'.joblib');joblib.dump(model,path)
                np.testing.assert_array_equal(model.predict_proba(x),joblib.load(path).predict_proba(x))
                models[(cond,family)]=model
                receipt[family][cond]=dict(dimensions=x.shape[1],iterations=model[-1].n_iter_.tolist(),sha256=sha(path))
            rr,pp=proposals(out,cal,cond,family,model)
            scores=[s for r,(_,s,_) in zip(rr,pp) if r['actual']==r['expected']]
            threshold,budget=budget_threshold(scores)
            assert np.isfinite(threshold)
            count=int((np.array(scores)>=threshold).sum());assert count<=budget
            thresholds[cond+'_'+family]=dict(threshold=threshold,normal=len(scores),budget=budget,false_alarms=count)
    write(out/'fit.json',receipt);write(out/'thresholds.json',thresholds)
    write(out/'frozen_head.json',dict(created=datetime.datetime.now().astimezone().isoformat(),
        thresholds_sha256=sha(out/'thresholds.json'),fit_sha256=sha(out/'fit.json'),
        train_sha256=sha(out/'train_features.jsonl.gz'),cal_sha256=sha(out/'cal_features.jsonl.gz'),
        test_used_for_fitting=False))
    print(json.dumps(dict(training=receipt,calibration=thresholds),ensure_ascii=False),flush=True)


def evaluate(out,role):
    import joblib
    rows=read_rows(out/(role+'_features.jsonl.gz')); thresholds=json.loads((out/'thresholds.json').read_text())
    frozen=json.loads((out/'frozen_head.json').read_text());assert frozen['thresholds_sha256']==sha(out/'thresholds.json')
    predictions=[];summaries=[]
    for cond in ('A_M3_peak','B_probability','C_segment'):
        results={}
        for family in FAMILIES:
            model=None if cond=='A_M3_peak' else joblib.load(out/(cond+'_'+family+'.joblib'))
            rr,pp=proposals(out,rows,cond,family,model);threshold=thresholds[cond+'_'+family]['threshold']
            for r,(p,s,veto) in zip(rr,pp): results[(r['uid'],r['q'])]=(p,s,veto,s>=threshold)
        groups=collections.defaultdict(collections.Counter)
        for r in rows:
            z=dict(role=role,condition=cond,uid=r['uid'],q=r['q'],speaker=r['speaker'],item=r['item'],
                family=r['family'],expected=r['expected'],actual=r['actual'],truth_status=r['truth_status'],
                observed=r['observed'],reason=r['observation_reason'],prediction=None,score=None,warning=False,veto=False)
            if r['observed']:
                pred,score,veto,warning=results[(r['uid'],r['q'])]
                z.update(prediction=pred,score=None if not np.isfinite(score) else score,warning=bool(warning),veto=bool(veto))
            predictions.append(z)
            keys=['all','family:'+r['family'],'phone:'+r['expected'],'item:'+r['item']]
            if r['truth_status'] not in ('normal','substitution','deletion'):
                for k in keys:groups[k]['unidentified']+=1
                continue
            error=r['actual']!=r['expected'];pair=error and r['actual'] in FAMILIES[r['family']]
            for key in keys:
                c=groups[key];kind='error' if error else 'normal';c[kind]+=1;c['pair_errors']+=int(pair)
                if not r['observed']:c['unobserved_'+kind]+=1;continue
                c['observed_'+kind]+=1;c['observed_pair_errors']+=int(pair)
                correct=z['prediction']==r['actual'];warn=z['warning']
                c['raw_correct_'+kind]+=int(correct);c['raw_pair_correct']+=int(pair and correct)
                c['fp']+=int(warn and not error);c['exact']+=int(warn and error and correct)
                c['pair_exact']+=int(warn and pair and correct);c['wrong_actual']+=int(warn and error and not correct)
                c['veto']+=int(z['veto'])
        for group,c in groups.items():
            def frac(a,b):return c[a]/c[b] if c[b] else None
            summaries.append(dict(role=role,condition=cond,group=group,**dict(c),
                FPR=frac('fp','observed_normal'),exact_recall=frac('exact','error'),pair_recall=frac('pair_exact','pair_errors'),
                normal_coverage=frac('observed_normal','normal'),error_coverage=frac('observed_error','error'),
                fpr_lower=frac('fp','normal'),fpr_upper=(c['fp']+c['unobserved_normal'])/c['normal'] if c['normal'] else None))
    with gzip.open(out/(role+'_predictions.jsonl.gz'),'wt') as f:
        for r in predictions:emit(f,r)
    write(out/(role+'_report.json'),summaries)
    print(json.dumps([r for r in summaries if r['group'] in ('all','family:sibilants','family:affricates')],ensure_ascii=False),flush=True)
    return summaries


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('mode',choices=['prepare','run','summarize']);a=ap.parse_args();out=a.out
    if a.mode=='prepare':prepare(out);return
    if a.mode=='summarize':
        evaluate(out,'seen');evaluate(out,'rp');return
    p=json.loads((out/'protocol.json').read_text())
    for path,digest in p['source_hashes'].items():assert sha(path)==digest,path
    assert shutil.disk_usage(out).free>150*2**20
    import torch
    torch.set_num_threads(2)
    from mdd.gop_feats_v2 import HFDecoder
    dec=HFDecoder(str(CKPT));vocab=[dec.d[i] for i in range(len(dec.d))]
    extract(out,'train',dec,vocab,smoke=True)
    extract(out,'train',dec,vocab);extract(out,'cal',dec,vocab)
    fit(out)
    for role in ('seen','rp'):
        extract(out,role,dec,vocab);evaluate(out,role)
    write(out/'completed.json',dict(status='complete',encoder_updates=0,
        protocol_sha256=sha(out/'protocol.json'),frozen_head_sha256=sha(out/'frozen_head.json'),
        finished=datetime.datetime.now().astimezone().isoformat()))
    write(out/'live_status.json',dict(status='complete',encoder_updates=0))


if __name__=='__main__':main()
