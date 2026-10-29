"""Write a small live status file for an already-running H0 experiment.

Never starts/stops training, changes data, or promotes a model. Exits after the
launcher terminates. The experiment's own run.exit remains authoritative.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time


def last_update(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(0,2)
        f.seek(max(0,f.tell()-8192))
        lines=f.read().splitlines()
    for line in reversed(lines):
        try:
            row=json.loads(line)
            if 'step' in row:
                return row
        except (ValueError,UnicodeDecodeError):
            pass
    return None


def snapshot(output,session):
    stages={}
    for backbone in ('w2vbert','xlsr300m'):
        for stage,steps in [('stage1',10000),('stage2',4000)]:
            name=backbone+'_'+stage
            finished=output/name/'finished.json'
            stages[name]=dict(target_updates=steps,finished=finished.exists(),
                             latest_update=last_update(output/name/'updates.jsonl'))
        for step in (2000,4000):
            name=backbone+'_eval_'+str(step)
            stages[name]=dict(finished=(output/name/'report.json').exists())
    exit_file=output/'run.exit'
    alive=subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL).returncode==0
    exit_code=int(exit_file.read_text().strip()) if exit_file.exists() else None
    status='running' if alive else 'launcher_absent_without_exit_record'
    exposure={}
    if exit_code is not None:
        status='pipeline_finished' if exit_code==0 else 'pipeline_failed'
    if status=='pipeline_finished':
        if not all(s['finished'] for s in stages.values()) or not (output/'trajectory.json').exists():
            status='incomplete_outputs_despite_zero_exit'
        else:
            for stage in ('stage1','stage2'):
                records=[json.loads((output/(b+'_'+stage)/'finished.json').read_text())
                         for b in ('w2vbert','xlsr300m')]
                keys=('row_draws','audio_samples','batch_order_sha256')
                exposure[stage]={k:records[0]['final_exposure'][k]==records[1]['final_exposure'][k]
                                 for k in keys}
            if not all(all(v.values()) for v in exposure.values()):
                status='paired_exposure_mismatch'
    return dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,
        tmux_session=session,launcher_alive=alive,run_exit=exit_code,stages=stages,
        paired_exposure_checks=exposure,free_disk_bytes=shutil.disk_usage(str(output)).free,
        service_changed=False,performance_improvement_established=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True)
    ap.add_argument('--session',required=True)
    ap.add_argument('--once',action='store_true')
    a=ap.parse_args()
    output=Path(a.output).resolve()
    if not output.is_dir():
        raise ValueError('existing experiment directory required')
    while True:
        status=snapshot(output,a.session)
        temp=output/'live_status.json.tmp'
        temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        temp.replace(output/'live_status.json')
        if a.once or status['status']!='running':
            print(json.dumps(status,ensure_ascii=False),flush=True)
            break
        time.sleep(30)


if __name__=='__main__':
    main()
