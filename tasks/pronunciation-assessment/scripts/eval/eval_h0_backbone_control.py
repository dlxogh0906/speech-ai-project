"""Validation-only H0 trajectory and low-FPR diagnostic from a new checkpoint.

Canonical forced-alignment GOP margin gates a greedy mismatch. Actual TAG is
used only after inference for scoring. This is a common offline policy, not the
production API/VAD. FPR thresholds are fit on calibration speakers and evaluated
on different selection speakers, without retuning on the latter.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/train')]
import argparse
import itertools
import json
import math
import numpy as np
import torch
from train_h0_backbone_control import Backbone, file_sha
from train_w2vbert_ctc import read_split, read_wav, load_vocab
from audit_h0_backbone_protocol import calibration_role
from eval_actual_ctc import project_on_canon


def infer_utterance_sites(model, audio, expected, actual, vocab, meta, overlap, split, row, sequence_out=None):
    """Audio/reference-only inference; attach actual TAG only after prediction."""
    from mdd.gop_feats_v2 import viterbi_align
    inv = {v:k for k,v in vocab.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=model.device.type == 'cuda'):
        logits, lengths = model([audio])
    lp = logits[0,:int(lengths[0])].float().log_softmax(-1).cpu().numpy()
    if not np.isfinite(lp).all():
        raise ValueError('nonfinite evaluation logits')
    prediction = [inv[t] for t,_ in itertools.groupby(lp.argmax(-1)) if t in inv]
    if sequence_out is not None:
        sequence_out['predicted'] = prediction
    expected_ids = [vocab[t] for t in expected]
    segments = viterbi_align(lp, expected_ids, blank=0)
    scores = []
    for q, target in enumerate(expected_ids):
        if segments is None or not segments[q]:
            scores.append(None)
        else:
            l = lp[segments[q]]
            other = l.copy(); other[:, target] = -np.inf
            scores.append(float((other.max(-1)-l[:,target]).mean()))
    projected = project_on_canon(expected, prediction)
    truth = project_on_canon(expected, actual)
    if not len(expected) == len(projected) == len(truth) == len(scores):
        raise ValueError('canonical projection lost sites')
    return [dict(split=split, row=row, uid=meta['stem'], speaker=meta['spk'], item=meta['item'],
        task=meta['task'], language=meta['language'], role=calibration_role(meta['spk']),
        strict_native_unseen=bool(overlap and not overlap['native_contains']),
        q=q, expected=c, predicted=p, actual=t, score=s)
        for q,(c,p,t,s) in enumerate(zip(expected,projected,truth,scores))]


def cluster_intervals(sites, threshold, cluster_key, repeats=2000, seed=42):
    """Resample complete speakers or prompts, keeping the CAL threshold fixed."""
    groups = {}
    for s in sites:
        groups.setdefault(s[cluster_key], []).append(s)
    if len(groups) < 2:
        return dict(clusters=len(groups), status='insufficient_clusters')
    totals = []
    for key in sorted(groups):
        m = metrics(groups[key], threshold)
        totals.append([m['normal_false_alarms'], m['normal'], m['exact'], m['errors'],
                       m['scored_sites'], m['total_sites']])
    totals = np.asarray(totals, dtype=np.float64)
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(totals), size=(repeats, len(totals)))
    sampled = totals[draws].sum(axis=1)
    result = dict(clusters=len(groups), repeats=repeats, status='conditional_on_fixed_CAL_threshold')
    for label, numerator, denominator in [('normal_fpr',0,1), ('exact_recall',2,3), ('score_coverage',4,5)]:
        valid = sampled[:,denominator] > 0
        values = sampled[valid,numerator] / sampled[valid,denominator]
        result[label] = dict(ci95=np.quantile(values,[.025,.975]).tolist() if len(values) else None,
                             valid_replicates=int(valid.sum()))
    return result


def threshold_for_normal_budget(sites, fpr=0.01):
    normals = [s for s in sites if s['actual'] == s['expected']]
    if not normals:
        raise ValueError('normal calibration sites required')
    k = math.floor(len(normals) * fpr)
    scores = sorted([s['score'] for s in normals if s['score'] is not None
                     and s['predicted'] != s['expected']], reverse=True)
    # Strictly exclude the (k+1)th normal score, including all ties.
    return float(np.nextafter(scores[k], np.inf)) if k < len(scores) else -np.inf


def metrics(sites, threshold=None):
    normal = error = fp = exact = wrong = covered = unavailable_errors = 0
    for s in sites:
        is_error = s['actual'] != s['expected']
        normal += not is_error; error += is_error
        available = s['score'] is not None
        covered += available; unavailable_errors += is_error and not available
        warned = s['predicted'] != s['expected'] and (threshold is None or (
            available and s['score'] >= threshold))
        fp += warned and not is_error
        exact += warned and is_error and s['predicted'] == s['actual']
        wrong += warned and is_error and s['predicted'] != s['actual']
    return dict(normal=normal, errors=error, normal_false_alarms=fp, exact=exact, wrong_actual=wrong,
        normal_fpr=fp/normal if normal else None, exact_recall=exact/error if error else None,
        scored_sites=covered, total_sites=len(sites), unscorable_errors=unavailable_errors)


def compare_trajectory(baseline, candidates):
    """Interpolate adjacent checkpoints only, never threshold-shift/extrapolate."""
    baseline = sorted(baseline, key=lambda r: r['step'])
    result = []
    for candidate in candidates:
        if any(r['protocol_sha256'] != candidate['protocol_sha256'] for r in baseline):
            raise ValueError('cannot compare different protocols')
        x = candidate['results']['valid']['all']['greedy']['normal_fpr']
        y = candidate['results']['valid_unseen']['all']['greedy']['normal_fpr']
        brackets = []
        for left,right in zip(baseline,baseline[1:]):
            lx = left['results']['valid']['all']['greedy']['normal_fpr']
            rx = right['results']['valid']['all']['greedy']['normal_fpr']
            ly = left['results']['valid_unseen']['all']['greedy']['normal_fpr']
            ry = right['results']['valid_unseen']['all']['greedy']['normal_fpr']
            if min(lx,rx) <= x <= max(lx,rx):
                by = min(ly,ry) if lx == rx else ly+(ry-ly)*(x-lx)/(rx-lx)
                brackets.append(dict(steps=[left['step'],right['step']], baseline_unseen=by,
                    candidate_minus_baseline=y-by))
        result.append(dict(backbone=candidate['backbone'],step=candidate['step'],
            seen_fpr=x,unseen_fpr=y,brackets=brackets,
            status='descriptive_interpolation_no_significance_claim' if brackets else 'outside_common_range'))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(Path(a.protocol).read_text())
    audit = json.loads(Path(protocol['input_audit']).read_text())
    ck = torch.load(a.checkpoint, map_location='cpu')
    if ck['protocol_sha256'] != file_sha(a.protocol) or ck['stage'] != 'stage2':
        raise ValueError('matched Stage2 checkpoint required')
    kind, step = ck['backbone'], ck['step']
    vocab = ck['vocab']
    manifest = Path(protocol['stage2']['manifest'])
    if vocab != load_vocab(str(manifest/'dict.ltr.txt')):
        raise ValueError('dictionary mismatch')
    model = Backbone(kind, protocol['backbones'][kind]['path'], max(vocab.values())+1,
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(ck['model'], strict=True); model.eval()
    del ck
    by_split = {}
    ledger = (out/'sites.jsonl').open('w', encoding='utf-8')
    with torch.inference_mode():
        for split in ('valid', 'valid_unseen'):
            for ext in ('tsv','ltr','canon','meta.jsonl'):
                p = str(manifest / (split+'.'+ext))
                if file_sha(p) != audit['sha256'][p]:
                    raise ValueError('evaluation manifest changed')
            rows, actual = read_split(str(manifest), split)
            canon = [s.split() for s in (manifest/(split+'.canon')).read_text().splitlines()]
            meta = [json.loads(s) for s in (manifest/(split+'.meta.jsonl')).read_text().splitlines()]
            assert len(rows) == len(canon) == len(meta)
            sites = []
            for i, (path, frames) in enumerate(rows):
                x = read_wav(path, 20.)
                if len(x) != frames:
                    raise ValueError('evaluation audio truncated')
                m = meta[i]
                inferred = infer_utterance_sites(model,x,canon[i],actual[i],vocab,m,
                    audit['native_text_overlap'].get(m['item']),split,i)
                for site in inferred:
                    ledger.write(json.dumps(site, ensure_ascii=False)+'\n'); sites.append(site)
                if i % 1000 == 0:
                    ledger.flush(); print(json.dumps(dict(split=split, utterances=i+1)), flush=True)
            by_split[split] = sites
    ledger.close()
    # One threshold per model/checkpoint, using seen calibration normals only.
    calibration = [s for s in by_split['valid'] if s['role']=='calibration']
    threshold = threshold_for_normal_budget(calibration)
    results, uncertainty, composition = {}, {}, {}
    for split, sites in by_split.items():
        selected = [s for s in sites if s['role']=='selection']
        groups = dict(all=selected, sentences=[s for s in selected if s['task']=='RS'],
            strict_native_unseen=[s for s in selected if s['strict_native_unseen']],
            expected_s=[s for s in selected if s['expected']=='ㅅ'],
            expected_ss=[s for s in selected if s['expected']=='ㅆ'],
            pair_s_to_ss=[s for s in selected if s['expected']=='ㅅ' and s['actual'] in ('ㅅ','ㅆ')],
            pair_ss_to_s=[s for s in selected if s['expected']=='ㅆ' and s['actual'] in ('ㅅ','ㅆ')])
        results[split] = {k:dict(greedy=metrics(v), calibrated=metrics(v,threshold)) for k,v in groups.items()}
        for group in results[split].values():
            fpr = group['calibrated']['normal_fpr']
            group['calibrated']['measured_fpr_within_1pct'] = fpr <= .01 if fpr is not None else None
        uncertainty[split] = {k:{axis:cluster_intervals(v,threshold,axis)
            for axis in ('speaker','item')} for k,v in groups.items()}
        composition[split] = dict(utterances=len({s['uid'] for s in selected}),
            speakers=len({s['speaker'] for s in selected}), items=len({s['item'] for s in selected}),
            languages={lang:dict(utterances=len({s['uid'] for s in selected if s['language']==lang}),
                speakers=len({s['speaker'] for s in selected if s['language']==lang}))
                for lang in sorted({s['language'] for s in selected})})
    report = dict(backbone=kind, step=step, checkpoint_sha256=file_sha(a.checkpoint),
        protocol_sha256=file_sha(a.protocol), calibration=metrics(calibration,threshold),
        threshold=None if threshold == -np.inf else threshold, threshold_none_means='all scored mismatches',
        results=results, uncertainty=uncertainty, composition=composition,
        limitations=['offline whole-audio policy, no production VAD/quality rejection',
            'insertion gaps are excluded from canonical-site metrics',
            'selection FPR may exceed 1%; this is a failure, not a reason to retune',
            'bootstrap CIs condition on the fixed CAL threshold; they exclude calibration and training-seed uncertainty',
            'speaker and prompt bootstrap are separate sensitivity analyses, not a joint crossed bootstrap',
            'no final-test evaluation or statistical superiority claim in this report'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
