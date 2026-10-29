"""Label-defined vowel contrasts, including paired speakers; no fitted classifier."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from diagnose_h0_p4 import MODES,read_jsonl,write_json
from fit_h0_representation_probes import table
from train_h0_backbone_control import file_sha


def prepare(root):
    source=root/'audio_labels/review_manifest_NOT_LISTENED.json';ss=json.loads(source.read_text())['sites']
    us=[s for s in ss if 'normal_suri_475' in s['cohorts']]
    err=[s for s in ss if s['item']=='RW054' and s['actual']=='ㅗ' and s['expected']=='ㅜ']
    os=[s for s in ss if s['item']=='RW052' and s['actual']=='ㅗ' and s['expected']=='ㅗ']
    shared=sorted(set(s['speaker'] for s in us)&set(s['speaker'] for s in os));pairs=[]
    for speaker in shared:
        u=sorted([s for s in us if s['speaker']==speaker],key=lambda s:s['uid'])[0]
        o=sorted([s for s in os if s['speaker']==speaker],key=lambda s:s['uid'])[0]
        pairs.append(dict(speaker=speaker,u_id=u['id'],o_id=o['id'],u_uid=u['uid'],o_uid=o['uid']))
    write_json(root/'audio_labels/pair_design.json',dict(source_sha256=file_sha(source),source_code_sha256=file_sha(__file__),
        same_prompt=dict(u_ids=[s['id'] for s in us],o_ids=[s['id'] for s in err]),same_speaker=pairs,
        chosen_without_predictions=True,limitations=['same prompt contrast has few errors',
            'same-speaker Suri/Sori contrasts unseen versus seen prompts; not a pure phonetic or causal experiment',
            'no independent listening or label correction']))


def compare(root):
    p=json.loads((root/'protocol.json').read_text());design=json.loads((root/'audio_labels/pair_design.json').read_text())
    classes=json.loads((Path(p['p1'])/'probes/fit_protocol.json').read_text())['classes'];u,o=classes.index('ㅜ'),classes.index('ㅗ')
    rows=[];paired=[]
    cohorts=[('same_prompt_suri',design['same_prompt']['u_ids'],design['same_prompt']['o_ids']),
             ('same_speaker_suri_sori',[s['u_id'] for s in design['same_speaker']],[s['o_id'] for s in design['same_speaker']])]
    scope=root/'audio_labels/pair_scope_audit.json'
    if scope.exists():
        strict=json.loads(scope.read_text())['whole_TAG_normal_pairs']
        cohorts.append(('same_speaker_both_whole_TAG_normal',[s['u_id'] for s in strict],[s['o_id'] for s in strict]))
    for arm in p['arms']:
        ids=np.load(root/arm/'site_ids.npy');index={int(g):i for i,g in enumerate(ids)};available=np.load(root/arm/'available.npy')
        for k,mode in enumerate(MODES):
            logits=np.load(root/arm/(mode+'.npy'));pred=logits.argmax(-1);score=logits[:,o]-logits[:,u]
            for name,uids,oids in cohorts:
                ui=np.array([index[i] for i in uids],int);oi=np.array([index[i] for i in oids],int)
                ua=ui[available[ui,k]];oa=oi[available[oi,k]]
                binary=np.r_[np.zeros(len(ua)),np.ones(len(oa))];values=np.r_[score[ua],score[oa]]
                rows.append(dict(arm=arm,mode=mode,cohort=name,normal_u=len(ui),actual_o=len(oi),u_available=len(ua),o_available=len(oa),
                    u_correct=int((pred[ua]==u).sum()),o_correct=int((pred[oa]==o).sum()),o_to_u=int((pred[oa]==u).sum()),
                    auc=float(roc_auc_score(binary,values)) if len(ua) and len(oa) else None,
                    u_score_quantiles=np.quantile(score[ua],[.1,.5,.9]).tolist() if len(ua) else None,
                    o_score_quantiles=np.quantile(score[oa],[.1,.5,.9]).tolist() if len(oa) else None))
                if name.startswith('same_speaker'):
                    ok=available[ui,k]&available[oi,k];delta=score[oi[ok]]-score[ui[ok]]
                    paired.append(dict(arm=arm,mode=mode,cohort=name,total_pairs=len(ui),available_pairs=int(ok.sum()),
                        o_score_greater_than_u=int((delta>0).sum()),score_ties=int((delta==0).sum()),
                        both_phone_correct=int(((pred[ui[ok]]==u)&(pred[oi[ok]]==o)).sum()),
                        median_o_minus_u_score=float(np.median(delta)) if len(delta) else None))
    table(root/'audio_labels/vowel_contrasts.tsv',rows);table(root/'audio_labels/speaker_pairs.tsv',paired)
    write_json(root/'audio_labels/contrast_report.json',dict(contrasts=rows,paired=paired,limitations=design['limitations'],
        pair_design_sha256=file_sha(root/'audio_labels/pair_design.json'),source_sha256=file_sha(__file__)))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);ap.add_argument('--stage',choices=['prepare','compare'],required=True)
    a=ap.parse_args();(prepare if a.stage=='prepare' else compare)(Path(a.root))


if __name__=='__main__':main()
