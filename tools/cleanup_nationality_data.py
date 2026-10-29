"""Remove authorized nationality training data while preserving pronunciation use.

Plan and execution are separate. Model weights, labels, reports and captures are
outside the deletion rules. Pronunciation manifests and live process references
protect shared audio paths; hardlinked pronunciation copies are never removed.
"""
import argparse
import array
import collections
import datetime
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

CORPUS = Path('/data/speech-ai/corpus/aihub')
NATIONALITY = Path('/data/speech-ai/out/nationality')
MDD = Path('/data/speech-ai/out/mdd')
AUDIO = {'.wav', '.flac', '.mp3', '.pcm', '.ogg', '.m4a'}


def roots():
    return [p for code in ['040', '041', '042', '043', '131'] for p in CORPUS.glob(code + '.*') if p.is_dir() and not p.is_symlink()]


def in_scope(path, source_roots):
    p = str(path)
    return any(p.startswith(str(root) + '/') for root in source_roots) or p.startswith(str(NATIONALITY) + '/')


def protected_paths(source_roots):
    protected = set()
    command = ['rg', '-l', '-0', '-F']
    for root in source_roots:
        command.extend(['-e', str(root).lstrip('/')])
    command.extend([str(MDD), '--glob', '*.tsv', '--glob', '*.csv', '--glob', '*.json', '--glob', '*.jsonl'])
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in [0, 1]:
        raise RuntimeError('Could not audit pronunciation references')
    references = [Path(p) for p in result.stdout.decode().split('\0') if p]

    def add(value, base='/'):
        if not isinstance(value, str) or not value:
            return
        if value.startswith('/data/') or value.startswith('data/'):
            candidate = os.path.normpath('/' + value.lstrip('/'))
        elif base != '/':
            candidate = os.path.normpath(os.path.join(base, value))
        else:
            return
        if in_scope(candidate, source_roots):
            protected.add(candidate)

    def visit(value):
        if isinstance(value, str):
            add(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in references:
        if path.suffix == '.tsv':
            with path.open() as handle:
                base = handle.readline().strip()
                for line in handle:
                    add(line.split('\t', 1)[0].rstrip('\n'), base)
        elif path.suffix == '.json':
            visit(json.loads(path.read_text()))
        elif path.suffix == '.jsonl':
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        visit(json.loads(line))
        else:
            import csv
            with path.open(newline='') as handle:
                for row in csv.reader(handle):
                    for value in row:
                        add(value)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    add(os.readlink(str(fd)))
                except OSError:
                    pass
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6:
                    add(fields[5])
            # Only read the sample-directory field; never record process secrets.
            for entry in (proc / 'environ').read_bytes().split(b'\0'):
                if entry.startswith(b'SAMPLE_WAV_DIR='):
                    sample_root = Path(entry.split(b'=', 1)[1].decode())
                    for sample in sorted(sample_root.glob('*.wav'))[:10]:
                        add(str(sample))
        except (OSError, UnicodeError):
            pass
    return protected, references


def eligible(path, source_roots):
    if any(str(path).startswith(str(root) + '/') for root in source_roots):
        return path.suffix.lower() in AUDIO
    if str(path).startswith(str(NATIONALITY) + '/'):
        relative = path.relative_to(NATIONALITY)
        if path.suffix == '.npy':
            return 'prefix_cache' in relative.parts or path.name.endswith('features.npy')
        return relative.parts[0].startswith('youtube_') and path.suffix.lower() in AUDIO
    return False


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/data/speech-ai/out/cleanup/nationality_20261011')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    source_roots = roots()
    protected, references = protected_paths(source_roots)
    write_json(destination / 'protected_references.json', {'files': [str(p) for p in references], 'protected_path_count': len(protected)})
    plan = destination / 'delete_plan.jsonl'
    if not args.execute:
        stats = collections.defaultdict(collections.Counter)
        with plan.open('w') as sink:
            for root in source_roots + [NATIONALITY]:
                for folder, dirs, names in os.walk(root, followlinks=False):
                    dirs[:] = [d for d in dirs if not Path(folder, d).is_symlink()]
                    for name in names:
                        path = Path(folder, name)
                        if not eligible(path, source_roots):
                            continue
                        info = path.lstat()
                        if not stat.S_ISREG(info.st_mode):
                            continue
                        category = root.name
                        if str(path) in protected:
                            stats[category]['protected_files'] += 1
                            stats[category]['protected_bytes'] += info.st_size
                            continue
                        item = {'path': str(path), 'dev': info.st_dev, 'ino': info.st_ino, 'size': info.st_size,
                                'mtime_ns': info.st_mtime_ns, 'nlink': info.st_nlink, 'allocated_bytes': info.st_blocks * 512}
                        sink.write(json.dumps(item, ensure_ascii=False) + '\n')
                        stats[category]['delete_files'] += 1
                        stats[category]['delete_logical_bytes'] += info.st_size
                        stats[category]['estimated_free_bytes'] += info.st_blocks * 512 if info.st_nlink == 1 else 0
                print('[plan] ' + root.name + ' ' + json.dumps(dict(stats[root.name])), flush=True)
        report = {'created_at': datetime.datetime.now().astimezone().isoformat(), 'categories': dict(stats),
                  'source_roots': [str(p) for p in source_roots], 'protected_paths': len(protected),
                  'preserved': ['pronunciation-referenced audio', 'mdd_gold_wav16k hardlink copies', 'all labels', 'all model weights', 'all reports and captures', '037 English', 'KsponSpeech', 'Zeroth', 'dialect data']}
        write_json(destination / 'plan_summary.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    if not plan.is_file():
        raise RuntimeError('Create and inspect the deletion plan first')
    before = shutil.disk_usage('/data').free
    result = collections.Counter()
    # Successful indices refer back to the exact predeletion inode/size manifest.
    with plan.open() as source, (destination / 'deleted_indexes.uint64').open('ab') as journal:
        pending = array.array('Q')
        for index, line in enumerate(source):
            item = json.loads(line)
            path = Path(item['path'])
            if str(path) in protected:
                result['protected_at_execution'] += 1
                continue
            if not eligible(path, source_roots) or str(path.resolve()) != str(path):
                raise RuntimeError('Deletion scope or symlink changed: ' + str(path))
            try:
                info = path.lstat()
            except FileNotFoundError:
                result['already_missing'] += 1
                continue
            if not stat.S_ISREG(info.st_mode) or any(getattr(info, 'st_' + key) != item[key] for key in ['dev', 'ino', 'size', 'mtime_ns']):
                result['changed_since_plan'] += 1
                continue
            path.unlink()
            pending.append(index)
            result['deleted_files'] += 1
            result['deleted_logical_bytes'] += info.st_size
            if len(pending) >= 1000:
                pending.tofile(journal)
                journal.flush()
                pending = array.array('Q')
            if result['deleted_files'] % 25000 == 0:
                print('[delete] ' + json.dumps(dict(result)), flush=True)
        if pending:
            pending.tofile(journal)
            journal.flush()
    report = {'finished_at': datetime.datetime.now().astimezone().isoformat(), **dict(result),
              'disk_free_before': before, 'disk_free_after': shutil.disk_usage('/data').free,
              'disk_free_change': shutil.disk_usage('/data').free - before,
              'plan': str(plan), 'protected_paths_at_execution': len(protected)}
    write_json(destination / 'deletion_report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
