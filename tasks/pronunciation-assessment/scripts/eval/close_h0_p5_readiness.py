"""Independently reaggregate the RP audit and close the pre-training data gate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import argparse
import collections
import csv
import json
import numpy as np
from audit_h0_p5_readiness import write,sha,emit


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit',required=True)
    ap.add_argument('--balance',required=True);a=ap.parse_args();root=Path(a.audit);balance=Path(a.balance)
    if (root/'closure.json').exists():raise ValueError('closure exists; preserve prior result')
    report=json.loads((root/'report.json').read_text());records=[json.loads(l) for l in (root/'recordings.jsonl').open()]
    byuid={r['uid']:r for r in records};reserved=[r for r in records if r['historical_role']=='test']
    verification=collections.Counter();last={};aggregate=collections.defaultdict(collections.Counter)
    for line in (root/'windows.jsonl').open():
        c=json.loads(line);key=(c['uid'],c['mode']);end=last.get(key,0)
        if c['start_sample']!=end or c['end_sample']<=end:raise ValueError('window continuity')
        last[key]=c['end_sample'];verification['windows']+=1
        if c['eligible_for_primary_evaluation'] or c['model_scored']:raise ValueError('unexpected evaluated/selected window')
        for role in ('all',c['historical_role']):
            k=(c['item'],role,c['mode']);aggregate[k]['chunks']+=1
            aggregate[k]['samples']+=c['end_sample']-c['start_sample']
            aggregate[k]['complete']+=int(c['complete_annotation_window'])
            aggregate[k].update(c.get('conditional_annotation_counts',{}))
            aggregate[k].update({'normal:'+x:n for x,n in c.get('conditional_normal_phone_counts',{}).items()})
            aggregate[k].update({'error:'+x:n for x,n in c.get('conditional_error_phone_counts',{}).items()})
    for (uid,mode),end in last.items():
        if end!=byuid[uid]['frames']:raise ValueError('uncovered audio tail')
    for r in report['windows']:
        c=aggregate[r['item'],r['role'],r['mode']]
        for k in ('chunks','samples'):
            if c[k]!=r[k]:raise ValueError('aggregate mismatch')
        if c['complete']!=r['complete_annotation_chunks']:raise ValueError('complete mismatch')
        for k,v in r['conditional_counts'].items():
            if c[k]!=v:raise ValueError('conditional denominator mismatch')
    verification['recording_mode_pairs']=len(last)
    verification['original_recordings']=len(records)
    # Inspect the preserved LAB source, without changing any reference or TAG.
    unknown_words=collections.Counter();unknown_sources=[]
    for r in reserved:
        if not r['lab_check']['kinds'].get('unknown'):continue
        path=next(Path(s) for s in r['source_hashes'] if s.endswith('.json'))
        if sha(path)!=r['source_hashes'][str(path)]:raise ValueError('LAB source changed')
        ph=json.loads(path.read_text())['RecordingMetadata']['phonemic']
        for t in ph['phones']:
            if t['phone']!='spn':continue
            words=[w['word'] for w in ph['words'] if min(w['end'],t['end'])>max(w['start'],t['start'])+.001]
            unknown_words[r['item'],'|'.join(words)]+=1
            unknown_sources.append(dict(uid=r['uid'],item=r['item'],start=t['start'],end=t['end'],
                reference_phone='spn',overlapping_original_LAB_words=words,source_sha256=r['source_hashes'][str(path)]))
    write(root/'reference_unknown_words.json',dict(counts=[dict(item=k[0],words=k[1],intervals=v)
        for k,v in sorted(unknown_words.items())],sources=unknown_sources,
        interpretation='time overlap with original LAB word annotations, not an independent pronunciation judgment'))
    aliases=[json.loads(l) for l in (root/'reference_speaker_aliases.jsonl').open()]
    users=collections.defaultdict(set)
    for r in records+aliases:
        if r.get('user_id_sha256'):users[r['user_id_sha256']].add(r['speaker'])
    alias_collisions=sum(len(s)>1 for s in users.values())
    bindings=json.loads((root/'manifest_bindings.json').read_text())
    old_test_speakers=set()
    for key in ('H0_test','H0_test_unseen'):
        with Path(bindings[key]['path']).open() as f:
            next(f)
            for line in f:
                parts=Path(line.split('\t')[0]).stem.split('-')
                if len(parts)==7:old_test_speakers.add(parts[0])
    with (root/'reserved_unscored_candidates.jsonl').open('x') as f:
        for r in sorted(reserved,key=lambda r:r['uid']):
            emit(f,dict(uid=r['uid'],speaker=r['speaker'],item=r['item'],path=r['path'],frames=r['frames'],
                pcm_sha256=r['pcm_sha256'],source_hashes=r['source_hashes'],
                historical_test_speaker=r['speaker'] in old_test_speakers,
                actual_unknown=bool(r['tag_check']['kinds'].get('unknown')),
                reference_unknown=bool(r['lab_check']['kinds'].get('unknown')),
                model_scored=False,primary_evaluation_eligible=False))
    support=[]
    for phone in report['vocabulary']:
        per_item={item:aggregate[item,'test','energy'] for item in ('RP002','RP003','RP004')}
        support.append(dict(phone=phone,conditional_normal=sum(c['normal:'+phone] for c in per_item.values()),
            conditional_errors=sum(c['error:'+phone] for c in per_item.values()),
            normal_script_families=sum(c['normal:'+phone]>0 for c in per_item.values()),
            error_script_families=sum(c['error:'+phone]>0 for c in per_item.values()),
            scope='complete-annotation energy windows only; not primary evaluation'))
    with (root/'phone_support.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(support[0]),delimiter='\t');w.writeheader();w.writerows(support)
    # Recheck stored weights and independent table sums, rather than trust report flags.
    breport=json.loads((balance/'report.json').read_text())
    groups=list(csv.DictReader((balance/'contribution_groups.tsv').open(),delimiter='\t'))
    prompt=list(csv.DictReader((balance/'prompt_caps.tsv').open(),delimiter='\t'))
    source_rows={(r['arm'],r['key']):r for r in groups if r['dimension']=='source'}
    for arm in ('C0','C1','C2'):
        weights=np.load(balance/(arm+'_weights.npy'))
        if not np.isfinite(weights).all() or not (weights>0).all():raise ValueError('invalid weight')
        if arm=='C0' and not (weights==1).all():raise ValueError('C0 changed')
        for (a0,source),r in source_rows.items():
            if a0!=arm:continue
            baseline=source_rows['C0',source]
            for field in ('row_draws','audio_samples'):
                if float(r[field])!=float(baseline[field]):raise ValueError('physical exposure changed')
            if abs(float(r['ctc_coefficient'])-float(baseline['ctc_coefficient']))>1e-6:
                raise ValueError('source loss mass mismatch')
    for r in prompt:
        if float(r['C1_mass'])>float(r['C1_cap'])+1e-7:raise ValueError('prompt cap violated')
    for source in {r['source'] for r in prompt}:
        for arm in ('C0','C1','C2'):
            total=sum(float(r[arm+'_mass']) for r in prompt if r['source']==source)
            if abs(total-float(source_rows[arm,source]['ctc_coefficient']))>1e-6:raise ValueError('prompt/source mismatch')
    cstats=[]
    for c in breport['context_balance']:
        cstats.append(dict(arm=c['arm'],weighted_s_u_l=c['s_u_l_weighted_tokens'],
            weighted_s_o_l=c['s_o_l_weighted_tokens'],o_to_u_ratio=c['s_o_l_weighted_tokens']/c['s_u_l_weighted_tokens']))
    write(balance/'verification.json',dict(status='stored_weights_and_independent_table_sums_verified',
        checks=['C0 all ones; all weights positive finite', 'physical draws/audio unchanged',
            'foreign clean/error and native CTC mass preserved', 'C1 per-source item cap and total verified'],
        context_diagnostic=cstats,optimizer_updates=0,source_sha256=sha(__file__)))
    collection=dict(status='requirements_only_no_new_collection',
        primary_need='new prompts and speakers, with auditable actual-phone annotations; RP remains secondary stress data',
        new_speakers='disjoint IDs/UserID and known aliases from TRAIN, CAL, model-selection and historically analyzed evaluation speakers',
        prompt_independence='new script families, not splits of these three; multiple independent prompts per important phone/context',
        lengths='include the short-utterance service range and predeclared long-form strata separately; audio-only boundaries before label scoring',
        sampling='include normal and naturally occurring substitution/deletion/insertion; report L1, proficiency, device and prompt distributions',
        vocabulary='audit all 46 model symbols, but distinguish phonetic surface symbols from annotation conventions; do not solicit impossible literal compound-coda pronunciations',
        annotation='retain raw TAG/LAB, unknowns, intervals, annotator provenance; separate normative allowed variants from actual-phone judgments',
        incomplete_labels='predeclare local unknown masking and boundary-crossing accounting with sensitivity bounds; no silent complete-case filtering',
        frozen_roles='lock prompt/speaker split and cohort before any model scores; CAL separate; release confirmation only after candidate and thresholds freeze',
        analysis='exact diagnosis plus normal FPR, per phone/context/prompt, deletion and insertion gaps, unknown coverage, speaker/prompt uncertainty',
        no_fixed_sample_size_claim='sample size depends on product recall target, error prevalence and prompt/speaker clustering; current 1% point limit is not a population upper-bound guarantee',
        current_support=support,people_contacted=False,independent_listening_performed=False)
    write(root/'new_collection_requirements.json',collection)
    closure=dict(status='audit_complete_P5_training_gate_not_met',
        candidates_reserved_not_evaluation_frozen=True,reserved_recordings=len(reserved),
        reserved_speakers=len({r['speaker'] for r in reserved}),script_families=3,
        reused_historical_test_speaker_recordings=sum(r['speaker'] in old_test_speakers for r in reserved),
        reused_historical_test_speakers=len({r['speaker'] for r in reserved if r['speaker'] in old_test_speakers}),
        unknown_tag_or_lab_recordings=sum(bool(r['tag_check']['kinds'].get('unknown') or r['lab_check']['kinds'].get('unknown')) for r in reserved),
        alias_collisions_found=alias_collisions,original_labels_reparsed=22634,
        can_be_used_now_as_complete_independent_primary_evaluation=False,
        potential_use='additional unseen-script stress test after partial-reference/unknown/boundary scoring policy is validated; existing speakers and 3 script families must remain explicit',
        P5_optimizer_updates=0,new_model_inferences=0,gate_reasons=[
            '1117 of 1118 reserved recordings share speakers with historically analyzed H0 test',
            'reference unknowns concentrate in specific words; complete-window filtering changes context composition',
            'current short-evaluation path cannot score all long audio or partial references without a declared tested policy',
            '3 scripts do not supply broad independent prompt confirmation'],
        TRAIN_design_completed_not_trained=str(balance),design_findings=[
            'universal C1 cap is not equivalent to removing the Sori context concentration; Sori contribution increases',
            'C2 utterance weights only weakly reduce the motivating context ratio; not independent token-local control',
            'sampling remains unchanged; only intended loss contributions were simulated'],
        next_step='obtain suitable new sentence/speaker annotations or explicitly complete a partial-label stress protocol before P5 training',
        source_sha256=sha(__file__),bindings={str(f):sha(f) for f in (
            root/'protocol.json',root/'recordings.jsonl',root/'windows.jsonl',root/'reserved_unscored_candidates.jsonl',
            root/'memberships.json',root/'text_overlap.json',balance/'design.json',balance/'report.json')})
    write(root/'verification.json',dict(status='independent_reaggregation_passed',checks=dict(verification),
        full_audio_covered_by_both_window_rules=True,no_window_selected_or_scored=True))
    write(root/'closure.json',closure)
    write(root/'live_status.json',dict(status=closure['status'],model_inferences=0,optimizer_updates=0))


if __name__=='__main__':main()
