"""Run the frozen strict fold pilot and retain every stage's exit and artifacts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent==ROOT:raise RuntimeError('repository root not found')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src'),str(ROOT/'tasks/pronunciation-assessment/scripts/eval')]
import argparse
import json
import subprocess
import time
from audit_h0_p5_readiness import write,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--solution',type=Path,required=True);a=ap.parse_args()
    assert json.loads((a.root/'runtime_check.json').read_text())['status']=='passed'
    assert json.loads((a.root/'speaker_alias_verification.json').read_text())['status']=='passed'
    logs=a.root/'pipeline';logs.mkdir(exist_ok=False)
    args=['--root',str(a.root)]
    train=ROOT/'tasks/pronunciation-assessment/scripts/train/train_h0_joint_pilot.py'
    evaluate=ROOT/'tasks/pronunciation-assessment/scripts/eval/eval_h0_joint_pilot.py'
    commands=[(p,[sys.executable,str(train)]+args+['--solution',str(a.solution),'--phase',p])
              for p in ('stage1','common','C0','weighted')]
    commands += [('evaluate_'+arm,[sys.executable,str(evaluate)]+args+['--arm',arm]) for arm in ('C0','weighted','compare')]
    write(logs/'launch.json',dict(created=time.time(),commands=commands,protocol_sha256=sha(a.root/'protocol.json'),
        runtime_sha256=sha(a.root/'runtime_check.json'),source_sha256=sha(__file__),
        speaker_alias_sha256=sha(a.root/'speaker_alias_verification.json'),conditional_training_authorized=True,
        no_deployment=True,independent_final=False))
    for phase,cmd in commands:
        start=time.time()
        with (logs/(phase+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                update=None;path=a.root/'training'/phase/'updates.jsonl'
                if path.exists():
                    # Each update is small; read only the tail even after 10k steps.
                    with path.open('rb') as f:
                        f.seek(max(0,path.stat().st_size-4096));lines=f.read().decode().splitlines()
                    for line in reversed(lines):
                        try:update=json.loads(line);break
                        except ValueError:pass
                write(a.root/'live_status.json',dict(status='running',phase=phase,pid=process.pid,
                    phase_elapsed_seconds=time.time()-start,last_training_update=update,
                    performance_complete=False,all_five_folds_complete=False))
                time.sleep(30)
        (logs/(phase+'.exit')).write_text(str(process.returncode)+'\n')
        if process.returncode:
            write(a.root/'live_status.json',dict(status='stopped_on_execution_failure',phase=phase,
                exit_code=process.returncode,all_five_folds_complete=False))
            (logs/'run.exit').write_text(str(process.returncode)+'\n')
            raise SystemExit(process.returncode)
    comparison=json.loads((a.root/'comparison/report.json').read_text())
    continued=comparison['pilot_gate']['continue_folds']
    write(a.root/'live_status.json',dict(status='pilot_complete',pilot_gate=continued,
        next_action='prepare remaining fixed folds with the same gates and storage checks' if continued else
                    'stop this weighting family under the fixed pilot compute rule',
        all_five_folds_complete=False,independent_final=False))
    write(logs/'completion.json',dict(status='pilot_complete',comparison_sha256=sha(a.root/'comparison/report.json'),
        pilot_gate=continued,full_five_fold_success=False,production_changes=False))
    (logs/'run.exit').write_text('0\n')


if __name__=='__main__':main()
