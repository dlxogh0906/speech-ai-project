"""Export five distinct source people per model class for direct listening.

Export the exact test windows as WAV, with no enhancement or volume adjustment.
The provisional standard label and residence-based regional labels are explicit.
Five North Korean course excerpts are references, not an extra trained class.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf

NAMES = ['강원', '경상', '전라', '제주', '충청', '표준어후보_미확정']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifests', type=Path, default=Path('/data/speech-ai/out/dialect/standard_six_class_20261013/manifests'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selection = args.out / 'selected_source_rows.json'
    if selection.is_file():
        chosen = json.loads(selection.read_text())
    else:
        with (args.manifests / 'test.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        chosen = []
        for label in range(6):
            candidates = [r for r in rows if int(r['label_id']) == label]
            candidates.sort(key=lambda r: hashlib.sha256(('listen-review-20261013:' + r['row_id']).encode()).hexdigest())
            people, taken = set(), []
            for r in candidates:
                if r['person_id'] in people: continue
                people.add(r['person_id']); taken.append(r)
                if len(taken) == 5: break
            if len(taken) != 5: raise RuntimeError('Insufficient distinct source people')
            chosen.extend(taken)
        selection.write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + '\n')
    standard = {Path(r['audio_path']).stem: r for r in chosen if int(r['label_id']) == 5}
    archives = Path('/data/speech-ai/corpus/dialect_standard_speaker_20261013/archives/537')
    metadata = {}
    for key, name in [(55147, 'TL_common_01.zip'), (55159, 'VL_common_01.zip')]:
        path = archives / str(key) / name
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if member.endswith('.json') and Path(member).stem in standard:
                    metadata[Path(member).stem] = {'label': json.loads(z.read(member)), 'annotation_archive': str(path), 'annotation_member': member}
    if set(metadata) != set(standard): raise RuntimeError('Selected standard labels missing')
    original_audio = {}
    for key, name in [(55154, 'TS_common_01.zip'), (55163, 'VS_common_01.zip')]:
        path = archives / str(key) / name
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if member.endswith('.wav') and Path(member).stem in standard:
                    original_audio[Path(member).stem] = (str(path), member)
    if set(original_audio) != set(standard): raise RuntimeError('Selected original WAV sources missing')
    entries, counters = [], Counter()

    def export(relative, source, start, duration):
        dest = args.out / relative
        dest.parent.mkdir(exist_ok=True)
        with sf.SoundFile(source) as f:
            if f.samplerate != 16000 or f.channels != 1: raise RuntimeError('Expected 16kHz mono')
            f.seek(round(start * 16000)); wave = f.read(round(duration * 16000), dtype='float32')
        if not np.isfinite(wave).all() or len(wave) != round(duration * 16000):
            raise RuntimeError('Audio interval does not match requested source window')
        sf.write(dest, wave, 16000, subtype='PCM_16')
        return len(wave) / 16000

    for r in chosen:
        label = int(r['label_id']); counters[label] += 1
        folder = '%02d_%s' % (label + 1, NAMES[label])
        relative = folder + '/%02d.wav' % counters[label]
        duration = export(relative, r['audio_path'], float(r['start_sec']), float(r['duration_sec']))
        entry = {'sample_file': relative, 'class_label': NAMES[label], 'role': 'model_test_window',
                 'source_dataset': r['source_dataset'], 'source_path': r['audio_path'],
                 'source_start_sec': float(r['start_sec']), 'duration_sec': duration,
                 'person_id': r['person_id'], 'source_person_number': r.get('source_person_id', ''),
                 'dialect_code': '', 'region_code': '', 'gender': '', 'age': '',
                 'transcript': r.get('transcript', ''), 'source_url': '',
                 'label_basis': 'speaker.residenceProvince; regional accent is not independently verified',
                 'original_audio_archive': '', 'original_audio_member': '',
                 'annotation_archive': '', 'annotation_member': ''}
        if label == 5:
            record = metadata[Path(r['audio_path']).stem]; j = record['label']
            entry.update(dialect_code=j['Speaker']['Dialect'], region_code=j['Speaker']['Region'],
                         gender=j['Speaker']['Gender'], age=j['Speaker']['Age'],
                         source_person_number=j['Basic']['NumberOfSpeaker'],
                         annotation_archive=record['annotation_archive'], annotation_member=record['annotation_member'],
                         label_basis='537 Speaker.Dialect=01 candidate; code meaning and Seoul residence are NOT verified')
            entry['original_audio_archive'], entry['original_audio_member'] = original_audio[Path(r['audio_path']).stem]
        entries.append(entry)
    north_root = Path('/data/speech-ai/out/dialect/north_korean_review_20261012')
    north = json.loads((north_root / 'media_inventory.json').read_text())
    clips = {Path(r['audio_path']).stem: r for r in north['clips']}
    for index, (clip, start) in enumerate([('pre1', 5), ('pre1', 30), ('pre2', 3), ('pre2', 28), ('pre3', 2)], 1):
        item = clips[clip]; relative = '07_북한말_참고_미학습/%02d.wav' % index
        duration = export(relative, item['audio_path'], start, 12.)
        entry = {k: '' for k in entries[0]}
        entry.update(sample_file=relative, class_label='북한말_참고_미학습', role='reference_only_not_model_class',
                     source_dataset='DLIFLC_course', source_path=item['audio_path'],
                     source_start_sec=start, duration_sec=duration,
                     source_url=item['source_url'], label_basis='Course source clip; speaker identity/dialect not independently verified. Five excerpts from three source clips.')
        entries.append(entry)
    with (args.out / 'sample_metadata.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(entries[0])); w.writeheader(); w.writerows(entries)
    (args.out / 'sample_metadata.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    with (args.out / 'listening_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['sample_file', 'source_class', 'heard_dialect', 'sounds_standard', 'multiple_speakers', 'notes'])
        w.writeheader(); w.writerows({'sample_file': r['sample_file'], 'source_class': r['class_label']} for r in entries)
    notice = '표준어 후보는 실제 사람이 녹음한 537번 자료의 Dialect=01 부분입니다. 이 코드의 뜻은 설명서 확인 전이며, 서울·경기 화자만 고른 자료도 아닙니다. 기존 5지역은 거주지역 라벨을 사용했습니다. 북한말 5개는 원본 3개를 나눈 참고 구간이고 현재 모델 클래스가 아닙니다.'
    sections = []
    for name in NAMES + ['북한말_참고_미학습']:
        values = [e for e in entries if e['class_label'] == name]
        players = ''.join('<article><p><a href="%s">%s</a> · %.2f초</p><audio controls preload="none" src="%s"></audio><p>%s</p>%s</article>' %
                          (html.escape(e['sample_file']), html.escape(Path(e['sample_file']).name), e['duration_sec'], html.escape(e['sample_file']), html.escape(e['transcript']),
                           '<p>원본 방언 코드: %s · 거주지역 코드: %s</p>' % (html.escape(e['dialect_code']), html.escape(e['region_code'])) if e['dialect_code'] else '') for e in values)
        sections.append('<section><h2>' + html.escape(name) + '</h2>' + players + '</section>')
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>방언 샘플 청취</title><style>body{font-family:sans-serif;max-width:1000px;margin:32px auto;padding:0 16px;line-height:1.7}article{border:1px solid #ddd;padding:12px;margin:12px 0}audio{width:100%%}code{overflow-wrap:anywhere}.notice{background:#fff4d8;padding:18px}</style><h1>클래스별 청취 샘플</h1><p class="notice">%s</p><p>6개 모델 클래스는 클래스마다 서로 다른 화자 ID 5개를 선택한 테스트 구간입니다. 모델 판정값과 관계없이 결정한 샘플이며 음량 조정이나 잡음 제거를 하지 않았습니다.</p><p><a href="sample_metadata.csv">원본 경로·라벨·전사 CSV</a> · <a href="listening_review.csv">청취 기록 양식</a></p>%s</html>' % (html.escape(notice), ''.join(sections))
    (args.out / 'index.html').write_text(page)
    (args.out / '안내.txt').write_text(notice + '\n\n' +
        '기존 5지역 음성: /data/speech-ai/corpus/dialect_speakerid_20261011/audio/71517 및 /71558\n' +
        '표준어 후보 음성: /data/speech-ai/corpus/dialect_standard_speaker_20261013/audio\n' +
        '클래스별 분할 목록: ' + str(args.manifests) + '\n' +
        'index.html을 열면 각 샘플을 재생할 수 있습니다. 원본 파일과 구간 위치는 sample_metadata.csv에 있습니다.\n')
    archive = args.out / '클래스별_청취샘플_35개.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in args.out.rglob('*'):
            if path.is_file() and path != archive: z.write(path, str(path.relative_to(args.out)))
    print(json.dumps({'output': str(args.out), 'wav_count': len(entries), 'model_samples': 30,
                      'north_reference_samples': 5, 'class_counts': dict(Counter(e['class_label'] for e in entries)),
                      'distinct_source_people_by_model_class': {name: len({e['person_id'] for e in entries if e['class_label'] == name}) for name in NAMES},
                      'standard_raw_region_codes': [e['region_code'] for e in entries if e['class_label'] == NAMES[5]],
                      'archive_bytes': archive.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
