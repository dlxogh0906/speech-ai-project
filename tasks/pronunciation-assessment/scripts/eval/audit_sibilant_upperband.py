"""Fixed cross-speaker upper-band spectral-shape feasibility diagnostic on TRAIN."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse
import collections
import functools
import json
import time
import numpy as np
import soundfile as sf
import eval_sibilant_directional as service
from mdd.sibilant_trajectory_abx import angular_matrix,class_balanced_abx,unit_rows
from mdd.sibilant_upperband import mean_pair,missing_balanced_abx,summarize,SCHEMA,METHODS,BINS


def write(path,value):
    Path(path).write_text(json.dumps(service.jsonable(value),ensure_ascii=False,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    p,out = json.loads(Path(a.protocol).read_text()),Path(a.out)
    for name,digest in p['sha256'].items():
        if service.sha(name) != digest:
            raise ValueError('frozen upperband source changed: '+name)
    cache = Path(p['train_cache'])
    if json.loads((cache/'report.json').read_text())['partition'] != 'train':
        raise ValueError('original TRAIN-only representation diagnosis required')
    features = service.read_jsonl(p['features'])
    labels = service.read_jsonl(cache/'data.labels.jsonl')
    indexed = {z['feature_index']:z for z in labels if z['feature_index'] is not None}
    cells = json.loads(Path(p['cells']).read_text())
    selected = sorted({i for c in cells for i in c['indices']})
    if len(cells) != 241 or len(selected) != 5059:
        raise ValueError('fixed diagnostic population differs')
    wanted = {features[i]['uid'] for i in selected}
    rows = {}
    for line in (cache/'data.responses.jsonl').open():
        row = json.loads(line)['row']
        if row['uid'] in wanted:
            rows[row['uid']] = {k:row[k] for k in ('uid','wav','frames')}
    if set(rows) != wanted:
        raise ValueError('original source rows missing')
    @functools.lru_cache(maxsize=128)
    def audio(uid):
        info = sf.info(rows[uid]['wav'])
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError('actual original16k mono data required, no synthetic bandwidth restoration')
        return service.load_audio(rows[uid])[:320000]
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_bytes(Path(a.protocol).read_bytes())
    tick = time.monotonic()
    means = np.full((len(selected),2,BINS),np.nan,np.float64)
    failures,observations,flow = [],[],collections.Counter()
    for j,i in enumerate(selected):
        means[j],measurement = mean_pair(audio(features[i]['uid']),features[i])
        observations.append(dict(feature_index=i,**measurement))
        flow.update({k:measurement[k] for k in ('frames','valid_frames','failed_frames')})
        if not np.isfinite(means[j]).all() or (np.linalg.norm(means[j],axis=-1) <= 1e-12).any():
            means[j] = np.nan
            failures.append({'feature_index':i,'reason':'no finite nonzero norm mean spectrum; chance credit retained'})
        if (j+1)%512 == 0 or j+1 == len(selected):
            progress = {'observations_done':j+1,'total':len(selected),'failed_queries':len(failures),
                        'seconds':time.monotonic()-tick}
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.savez(out/'mean_observations.npz',indices=np.asarray(selected),means=means)
    write(out/'observation_flow.json',observations)
    write(out/'measurement_failures.json',failures)
    unit = np.full_like(means,np.nan)
    valid = np.isfinite(means).all((1,2))
    for method in range(2):
        if valid.any():
            unit[valid,method] = unit_rows(means[valid,method])
    index = {i:j for j,i in enumerate(selected)}
    original = np.load(p['original_distances'])
    original_cells = json.loads(Path(p['original_cell_scores']).read_text())
    y = np.load(p['targets'])['y']
    records,scores,distances = [],[],[]
    offset = 0
    for k,cell in enumerate(cells):
        left,right = cell['halves']
        if (set(left+right) != set(cell['indices'])
                or len({indexed[i]['speaker'] for i in left+right}) != len(left+right)):
            raise ValueError('original disjoint speaker halves changed')
        n = len(left)*len(right)
        np.testing.assert_array_equal(original['pairs'][offset:offset+n],[(k,i,j) for i in left for j in right])
        values,halves,matrices = [],{},[]
        for method in range(2):
            matrix = angular_matrix(unit[[index[i] for i in left],method],unit[[index[i] for i in right],method])
            value,half = missing_balanced_abx(matrix,y[left],y[right])
            values.append(value);halves[METHODS[method]] = half;matrices.append(matrix.reshape(-1))
        native = original['distances'][offset:offset+n,2].reshape(len(left),len(right))
        baseline,base_half = class_balanced_abx(native,y[left],y[right])
        np.testing.assert_allclose(baseline,original_cells[k]['accuracy']['mean'],atol=1e-12)
        values.append(baseline);halves['native_mean'] = base_half
        scores.append(values);distances.extend(np.stack(matrices,-1));offset += n
        records.append({'cell':cell['cell'],'accuracy':dict(zip(METHODS,values)),'half_class_accuracy':halves})
    if offset != 41280 or len(original['pairs']) != offset:
        raise ValueError('all original crossspeaker pairs must be retained')
    np.savez(out/'distances.npz',pairs=original['pairs'],raw_mean_distances=np.asarray(distances))
    result = summarize(scores,[c['cell'][3] for c in cells],p['seed'])
    write(out/'cells.json',records)
    report = {'scope':p['scope'],'schema':SCHEMA,'diagnostic_rows':len(selected),'pair_distances':offset,
        'original_PCM_sha_verified_each_ROI':True,'actual_source_16k_mono':True,
        'measurement_flow':dict(flow),'measurement_failed_queries':len(failures),'failed_query_reference_credit':.5,
        'result':result,'source_flow':json.loads(Path(p['source_report']).read_text())['source_flow'],
        'new_classifier_training':False,'CAL_primary_reserved_scoring':False,
        'seconds':time.monotonic()-tick,'deployment':False,'goal_complete':False}
    write(out/'report.json',report)
    write(out/'finished.json',{'report_sha256':service.sha(out/'report.json')})
    print(json.dumps({k:v for k,v in report.items() if k != 'source_flow'},ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
