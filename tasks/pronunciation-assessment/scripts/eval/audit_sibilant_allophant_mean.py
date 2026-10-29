"""Fixed phonetic-supervised mean representation versus native SSL mean."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import functools
import json
import time
import numpy as np
import torch
import eval_sibilant_directional as service
from mdd.sibilant_allophant import load_attributes
from mdd.sibilant_hubert_adaptation import tensor_digest
from mdd.sibilant_trajectory_abx import unit_rows, angular_matrix, class_balanced_abx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p, out = json.loads(Path(args.protocol).read_text()), Path(args.out)
    for name,digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen mean attribute input changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('original TRAIN-only representation diagnosis required')
    preflight = json.loads(Path(p['preflight_report']).read_text())
    if preflight['functional_inputs'] != 16 or not preflight['exact_reload_first_input']:
        raise ValueError('functional preflight required before representations')
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.protocol).read_bytes())
    tick = time.monotonic()
    torch.set_num_threads(2)
    features = service.read_jsonl(p['features'])
    cells = json.loads(Path(p['cells']).read_text())
    selected = sorted({i for c in cells for i in c['indices']})
    if len(selected) != 5059 or len(cells) != 241:
        raise ValueError('frozen diagnostic population changed')
    wanted = {features[i]['uid'] for i in selected}
    rows = {}
    for line in (cache/'data.responses.jsonl').open():
        row = json.loads(line)['row']
        if row['uid'] in wanted:
            rows[row['uid']] = {k:row[k] for k in ['uid','wav','frames']}
    if set(rows) != wanted:
        raise ValueError('missing original source rows')
    @functools.lru_cache(maxsize=128)
    def audio(uid):
        return service.load_audio(rows[uid])[:320000]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, metadata = load_attributes(torch.load(p['checkpoint'],map_location='cpu',weights_only=True),
        json.loads(Path(p['backbone_config']).read_text()),json.loads(Path(p['preprocessor']).read_text()),device)
    before = tensor_digest(model.state_dict())
    means = np.empty((len(selected),metadata['posterior_width']),np.float32)
    lengths = np.empty(len(selected),np.int64)
    failures = []
    for row_index,i in enumerate(selected):
        try:
            source = audio(features[i]['uid'])
            with torch.no_grad():
                logits,_ = model.observed(source,features[i])
                posterior = torch.cat([logits[z['name']].softmax(-1) for z in metadata['attributes']],-1)
                value = posterior.mean(0)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('nonfinite attributes')
            means[row_index],lengths[row_index] = value.cpu().numpy(),len(posterior)
        except (ValueError,RuntimeError) as exc:
            failures.append({'feature_index':i,'reason':str(exc)})
        if row_index%256 == 0 or row_index+1 == len(selected):
            print(json.dumps({'observations_done':row_index+1,'failures':len(failures),
                              'seconds':time.monotonic()-tick}),flush=True)
    if failures:
        (out/'measurement_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
        raise ValueError('failed observations retained; no silently reduced population')
    np.savez(out/'mean_observations.npz',indices=np.asarray(selected),mean=means,lengths=lengths)
    unit = unit_rows(means)
    index = {i:j for j,i in enumerate(selected)}
    y = np.load(p['targets'])['y']
    original = np.load(p['original_distances'])
    original_cells = json.loads(Path(p['original_cell_scores']).read_text())
    records, scores, all_distances = [], [], []
    offset = 0
    for k,cell in enumerate(cells):
        left,right = cell['halves']
        n = len(left)*len(right)
        np.testing.assert_array_equal(original['pairs'][offset:offset+n],[(k,i,j) for i in left for j in right])
        matrix = angular_matrix(unit[[index[i] for i in left]],unit[[index[j] for j in right]])
        value,halves = class_balanced_abx(matrix,y[left],y[right])
        old_matrix = original['distances'][offset:offset+n,2].reshape(len(left),len(right))
        base,_ = class_balanced_abx(old_matrix,y[left],y[right])
        np.testing.assert_allclose(base,original_cells[k]['accuracy']['mean'],atol=1e-12)
        scores.append([value,base])
        records.append({'cell':cell['cell'],'attribute_mean_accuracy':value,
            'native_mean_accuracy':base,'attribute_halfclass_scores':halves})
        all_distances.extend(matrix.reshape(-1));offset += n
    scores = np.asarray(scores)
    vowels = np.asarray([c['cell'][3] for c in cells])
    groups = {}
    for name in ['all','ㅣ','ㅡ','ㅜ']:
        mask = np.ones(len(cells),bool) if name=='all' else vowels==name
        groups[name] = {'cells':int(mask.sum()),'accuracy_attribute_native_mean':scores[mask].mean(0).tolist()}
    delta = scores[:,0]-scores[:,1]
    sample = np.random.RandomState(p['seed']).randint(0,len(cells),(2000,len(cells)))
    reference = np.percentile(delta[sample].mean(1),[2.5,50,97.5]).tolist()
    passed = bool(delta.mean() >= .02 and reference[0] > 0 and groups['ㅣ']['cells'] >= 10
        and groups['ㅣ']['accuracy_attribute_native_mean'][0] >= groups['ㅣ']['accuracy_attribute_native_mean'][1])
    if before != tensor_digest(model.state_dict()):
        raise ValueError('frozen public model changed')
    np.savez(out/'distances.npz',pairs=original['pairs'],attribute_mean=np.asarray(all_distances))
    (out/'cells.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    report = {'scope':p['scope'],'diagnostic_rows':len(selected),'pair_distances':offset,'measurement_failures':0,
        'metadata':metadata,'groups':groups,'attribute_minus_native_mean':float(delta.mean()),
        'paired_context_bootstrap_p025_p50_p975':reference,'proceed_attribute_readout_candidate':passed,
        'source_flow':json.loads(Path(p['source_report']).read_text())['source_flow'],
        'frozen_public_model_unchanged':True,'no_newclassifier_or_thresholdtraining':True,
        'seconds':time.monotonic()-tick,'deployment':False,'goal_complete':False}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'finished.json').write_text(json.dumps({'report_sha256':service.sha(out/'report.json')}))
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
