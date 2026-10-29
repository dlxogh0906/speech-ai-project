"""Stage newly approved dialect data, finish person splits, and compare encoders.

An optional existing preparation process is allowed to finish before this runner
opens its SQLite index. Downloads use separate dataset directories in that time.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tasks/dialect-classification/scripts/data'))
from prepare_speakerid_dialect import plan_jobs
from download_aihub_dialect import download, write_json


def process_identity(pid):
    try:
        fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--stage-datasets', nargs='+', type=int, default=[71517])
    parser.add_argument('--datasets', nargs='+', type=int, default=[71558, 71517])
    parser.add_argument('--catalog', default='/data/speech-ai/tmp/aihub-dialect-catalog')
    parser.add_argument('--corpus', default='/data/speech-ai/corpus/dialect_speakerid_20261011')
    parser.add_argument('--data-out', default='/data/speech-ai/out/dialect/speakerid_20261011')
    parser.add_argument('--out', default='/data/speech-ai/out/dialect/compare_speakerid_20261011')
    parser.add_argument('--reserve-gib', type=float, default=40)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing_identity = process_identity(args.wait_for_pid) if args.wait_for_pid else None
    state = {'pid': os.getpid(), 'datasets': args.datasets,
             'wait_for_pid': args.wait_for_pid, 'wait_for_start_ticks': existing_identity,
             'corpus': args.corpus, 'data_out': args.data_out, 'comparison_out': args.out}

    def status(stage, **extra):
        state.update(status='running', stage=stage,
                     updated_at=datetime.datetime.now().astimezone().isoformat(), **extra)
        write_json(out / 'pipeline_status.json', state)
        print('[pipeline] ' + json.dumps(state, ensure_ascii=False), flush=True)

    def run(command, logfile):
        with (out / logfile).open('a') as output:
            subprocess.run(command, cwd=str(REPO), stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    try:
        jobs = plan_jobs(args.stage_datasets, Path(args.catalog))
        write_json(out / 'staging_plan.json', {'datasets': args.stage_datasets, 'jobs': jobs})
        total = sum(int(job[k]['fileSize']) for job in jobs for k in ['labels', 'audio'])
        status('stage_approved_downloads', staged_archives=0,
               staging_archives_total=2 * len(jobs), staging_bytes=total)
        completed = 0
        for job in jobs:
            for kind in ['labels', 'audio']:
                folder = Path(args.corpus) / 'archives' / str(job['dataset']) / str(job['audio']['fileSn'])
                if kind == 'audio' and (folder / 'processed.json').exists() and (folder / 'rows.csv').exists():
                    completed += 1
                    continue
                status('stage_approved_downloads', dataset=job['dataset'],
                       filekey=int(job[kind]['fileSn']), archive_kind=kind,
                       staged_archives=completed)
                download(job['dataset'], job[kind], Path(args.corpus) / 'archives', args.reserve_gib)
                completed += 1
                status('stage_approved_downloads', staged_archives=completed)
        status('wait_for_existing_preparation')
        while existing_identity and process_identity(args.wait_for_pid) == existing_identity:
            time.sleep(30)
        status('prepare_all_five_regions')
        run([sys.executable, '-u', str(REPO / 'tasks/dialect-classification/scripts/data/prepare_speakerid_dialect.py'),
             '--datasets'] + [str(d) for d in args.datasets] +
            ['--catalog', args.catalog, '--root', args.corpus, '--out', args.data_out,
             '--reserve-gib', str(args.reserve_gib)], 'data_preparation.log')
        data_status = json.loads((Path(args.data_out) / 'status.json').read_text())
        if not data_status.get('five_region_classifier_ready'):
            raise RuntimeError('All five regions must be ready before model comparison')
        summary = json.loads((Path(args.data_out) / 'manifests/summary.json').read_text())
        if not summary.get('person_disjoint_verified') or any(summary['person_overlap'].values()):
            raise RuntimeError('Provider person ID separation failed')
        status('compare_four_encoders', manifest_sha256=summary['manifest_sha256'],
               people=summary['people'], counts=summary['counts'])
        run(['bash', str(REPO / 'tasks/dialect-classification/scripts/train/run_dialect_comparison.sh'),
             args.out, str(Path(args.data_out) / 'manifests')], 'comparison_pipeline.log')
        comparison = json.loads((out / 'comparison.json').read_text())
        if not comparison['complete'] or not comparison['person_disjoint_evaluation_complete']:
            raise RuntimeError('Model comparison did not complete all four person-split reports')
        status('complete', selected_model_by_validation=comparison['selected_model_by_validation'])
        state['status'] = 'complete'
        write_json(out / 'pipeline_status.json', state)
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                     updated_at=datetime.datetime.now().astimezone().isoformat())
        write_json(out / 'pipeline_status.json', state)
        raise


if __name__ == '__main__':
    main()
