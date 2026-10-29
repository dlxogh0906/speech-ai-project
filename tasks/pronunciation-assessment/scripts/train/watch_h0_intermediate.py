"""Observe P3 progress only; never alter training, checkpoints or deployment."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
from watch_h0_backbone_control import last_update


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True); ap.add_argument('--session',required=True)
    ap.add_argument('--once',action='store_true')
    a=ap.parse_args(); out=Path(a.output)
    while True:
        alive=subprocess.run(['tmux','has-session','-t',a.session],stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode==0
        code=int((out/'run.exit').read_text()) if (out/'run.exit').exists() else None
        status=('pipeline_finished' if code==0 else 'pipeline_failed') if code is not None else (
                'running' if alive else 'launcher_absent_without_exit_record')
        stages={}; done=0
        for arm in ('B0','B1','B2','B3'):
            u=last_update(out/arm/'updates.jsonl')
            done+=u['continuation_updates'] if u else 0
            stages[arm]=dict(training_finished=(out/arm/'finished.json').exists(),latest_update=u,
                evaluations={str(step):(out/('%s_eval_%d'%(arm,step))/'finished.json').exists()
                             for step in (3000,4000)})
        if code==0 and not (out/'completion.json').exists(): status='incomplete_outputs_despite_zero_exit'
        report=dict(updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
            status=status,tmux_session=a.session,launcher_alive=alive,run_exit=code,
            optimizer_updates_completed=done,optimizer_updates_target=8000,stages=stages,
            free_disk_bytes=shutil.disk_usage(out).free,service_model_changed=False,
            results_available=(out/'completion.json').exists())
        temp=out/'live_status.json.tmp'; temp.write_text(json.dumps(report,indent=2)+'\n'); temp.replace(out/'live_status.json')
        if a.once or status!='running':
            print(json.dumps(report)); break
        time.sleep(30)


if __name__=='__main__': main()
