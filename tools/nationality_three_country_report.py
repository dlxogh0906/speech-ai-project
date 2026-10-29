"""Audit the final zeroth loader and write three-country training data statistics.

Uses retained training manifests and previously reconstructed label statistics.
Does not load a model, download data, or touch ongoing training jobs.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
NAT = Path('/data/speech-ai/out/nationality')
BASE = NAT / 'webmic_kd'
META = NAT / 'data_distribution_20261012'
LABELS = ['China', 'Uzbekistan', 'Philippines']
KO = dict(zip(LABELS, ['중국', '우즈베키스탄', '필리핀']))
SPLITS = {'train': '학습', 'valid': '검증', 'test': '테스트'}
EOJEOL_BINS = ['1-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51+']


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quantile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


def distribution(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'min': min(values),
            'p10': quantile(values, .1), 'p50': quantile(values, .5),
            'p90': quantile(values, .9), 'max': max(values)}


def n(value):
    return format(int(value), ',')


def f(value, places=2):
    return format(float(value), '.' + str(places) + 'f')


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def link(path, title=None):
    return '[%s](%s)' % (title or path.name, str(path))


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((BASE / 'run_zeroth/history.json').read_text())['cfg']
    model_info = json.loads((BASE / 'run_zeroth/deploy/model_info.json').read_text())
    manifest_root = Path(cfg['data_dir'])
    provenance = json.loads((META / 'provenance.json').read_text())
    existing = {(r['label_en'], r['split']): r for r in read(META / 'model_country_split_distribution.csv')}
    lengths = {(r['label_en'], r['split'], r['length_type']): r for r in read(META / 'sentence_length_country_summary.csv')}
    mother = [r for r in read(META / 'model_country_mother_tongue_distribution.csv')
              if r['label_en'] in LABELS and r['split'] == 'train']
    inventory = [r for r in read(META / 'historical_corpus_distribution.csv') if r['model_label'] in LABELS]
    selected, totals, verification = {}, {}, {}
    input_files = [BASE / 'run_zeroth/history.json', BASE / 'run_zeroth/deploy/model_info.json',
                   META / 'provenance.json', META / 'model_country_split_distribution.csv',
                   META / 'sentence_length_country_summary.csv', META / 'model_country_mother_tongue_distribution.csv',
                   META / 'historical_corpus_distribution.csv']
    for split in SPLITS:
        manifest = manifest_root / (split + '.csv')
        selected[split] = {label: [] for label in LABELS}
        digest, duration_digest, count = hashlib.sha256(), hashlib.sha256(), 0
        with manifest.open() as stream:
            for row in csv.DictReader(stream):
                record = [row['audio_path'], row['label_en']]
                digest.update((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                duration_digest.update((json.dumps(record + [float(row['duration_sec'])], ensure_ascii=False) + '\n').encode())
                count += 1
                if row['label_en'] in LABELS:
                    selected[split][row['label_en']].append(row)
        expected = provenance['input_files'][str(manifest)]
        assert count == expected['rows'] and digest.hexdigest() == expected['ordered_audio_path_label_sha256']
        totals[split] = count
        verification[split] = {'rows': count, 'ordered_audio_path_label_sha256': digest.hexdigest(),
                                'matches_retained_metadata_provenance': True}
        input_files.append(manifest)
        if split != 'test':
            loader = BASE / 'run_zeroth' / ('_' + split + '_loader.csv')
            h, dh, loader_count = hashlib.sha256(), hashlib.sha256(), 0
            with loader.open() as stream:
                for row in csv.DictReader(stream):
                    record = [row['audio_path'], row['label_name']]
                    h.update((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                    dh.update((json.dumps(record + [float(row['duration_sec'])], ensure_ascii=False) + '\n').encode())
                    loader_count += 1
            assert loader_count == count and h.hexdigest() == digest.hexdigest()
            assert dh.hexdigest() == duration_digest.hexdigest()
            verification[split]['actual_loader_rows_order_labels_and_durations_match'] = True
            input_files.append(loader)
        print('Verified %s: %s rows' % (split, n(count)), flush=True)

    stats, overlaps, sources = {}, {}, []
    for label in LABELS:
        rows = selected['train'][label]
        train = existing[label, 'train']
        speakers = Counter(r['speaker'] for r in rows)
        durations = [float(r['duration_sec']) for r in rows]
        assert len(rows) == int(train['data_rows'])
        assert len(speakers) == int(train['speaker_keys'])
        assert len({r['audio_path'] for r in rows}) == len(rows) == int(train['unique_source_recordings'])
        assert int(train['augmentation_rows']) == 0
        assert abs(sum(durations) / 3600 - float(train['hours_annotated'])) < .001
        count_by_task = Counter(r['task'] for r in rows)
        spk_counts = sorted(speakers.values(), reverse=True)
        item = {'rows': len(rows), 'speakers': len(speakers), 'unique_recordings': len(rows),
                'share_of_all_training_pct': 100 * len(rows) / totals['train'],
                'duration_seconds': distribution(durations), 'hours': sum(durations) / 3600,
                'estimated_front_crop_hours': sum(min(v, cfg['max_sec']) for v in durations) / 3600,
                'longer_than_train_cap': sum(v > cfg['max_sec'] for v in durations),
                'speaker_recording_counts': distribution(list(speakers.values())),
                'largest_10_speakers_recording_share_pct': 100 * sum(spk_counts[:10]) / len(rows),
                'task_codes': dict(count_by_task), 'task_types': {k: int(train[k + '_records']) for k in ['word', 'reading', 'question_answer', 'free', 'other']}}
        assert sum(item['task_types'].values()) == len(rows)
        item['duration_bins'] = {}
        for name, low, high in [('0–2초 미만', 0, 2), ('2–4초 미만', 2, 4), ('4–8초 미만', 4, 8),
                                ('8–12초 미만', 8, 12), ('12–20초 미만', 12, 20), ('20초 이상', 20, float('inf'))]:
            cnt = sum(low <= v < high for v in durations)
            item['duration_bins'][name] = {'count': cnt, 'pct': 100 * cnt / len(rows)}
        assert sum(x['count'] for x in item['duration_bins'].values()) == len(rows)
        item['text_lengths'] = {kind: lengths[label, 'train', kind] for kind in
                               ['reading_prompt_recording', 'reading_prompt_unique', 'spoken_transcript_recording']}
        for kind, values in item['text_lengths'].items():
            total = int(values['distribution_items'])
            assert sum(int(values['bin_' + key + '_count']) for key in EOJEOL_BINS) == total
            assert int(values['recordings_with_text']) + int(values['recordings_missing_text']) == int(values['eligible_recordings'])
            if kind.startswith('reading'):
                assert int(values['eligible_recordings']) == item['task_types']['reading']
            else:
                assert int(values['eligible_recordings']) == len(rows) - item['task_types']['word']
                assert int(values['recordings_with_text']) == int(train['transcript_available_records'])
        for split in SPLITS:
            sr = selected[split][label]
            assert len(sr) == int(existing[label, split]['data_rows'])
            assert len({r['speaker'] for r in sr}) == int(existing[label, split]['speaker_keys'])
        overlaps[label] = {}
        for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
            value = len({r['speaker'] for r in selected[a][label]} & {r['speaker'] for r in selected[b][label]})
            assert value == 0
            overlaps[label][a + '_' + b] = value
        for domain in sorted({r['domain'] for r in rows}):
            values = [r for r in rows if r['domain'] == domain]
            sources.append({'label': label, 'domain': domain, 'rows': len(values),
                            'speakers': len({r['speaker'] for r in values}), 'pct': 100 * len(values) / len(rows)})
        assert sum(int(r['data_rows']) for r in mother if r['label_en'] == label) == len(rows)
        item['historical_index_recordings'] = sum(int(r['recordings_historical']) for r in inventory if r['model_label'] == label)
        stats[label] = item

    combined = sum(v['rows'] for v in stats.values())
    for item in stats.values():
        item['share_within_three_countries_pct'] = item['rows'] / combined * 100
    audit = {'generated_at': datetime.now().astimezone().isoformat(), 'model': model_info, 'config': cfg,
             'scope': 'Final run_zeroth training loader only; excludes cumulative predecessor/teacher/gate training.',
             'all_country_split_rows': totals, 'statistics': stats, 'sources': sources, 'mother_tongues': mother,
             'verification': verification, 'speaker_key_overlap': overlaps,
             'input_sha256': {str(p): sha(p) for p in input_files},
             'audio_lengths_source': 'Retained manifest duration_sec; original audio not re-decoded.',
             'text_lengths_source': 'Retained full-recording annotations summarized on 2026-10-12; original transcript, prompt and missingness kept separate.',
             'all_checks_passed': True}
    (args.out / 'statistics.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    md = ['# 중국·우즈베키스탄·필리핀 국적분류 학습 데이터 정리',
          '작성: ' + audit['generated_at'][:10] + ' · 대상: **41클래스 `run_zeroth` 주 모델의 마지막 재학습 입력**',
          '중국 **40,000개·1,139화자키**, 우즈베키스탄 **33,399개·164화자키**, 필리핀 **24,188개·2,119화자키**가 최종 학습 목록에 포함됐습니다. 세 국가 합계는 **' + n(combined) + '개**, 전체 학습 ' + n(totals['train']) + '개의 **' + f(combined / totals['train'] * 100) + '%**입니다.',
          '## 1. 집계 범위와 단위',
          '- 기준은 배포 정보의 `run_zeroth/ep0.pt (EMA)`와 연결된 `lab131full_zeroth`입니다. 마지막 단계는 `run4/ep0.pt`에서 이어 학습했으므로, 이 문서의 숫자는 이전 초기 모델·교사 모델까지 합친 누적 고유 데이터 수가 아닙니다. 별도 게이트, YouTube·LoRA 추가 실험도 합산하지 않았습니다. 서비스 모델 식별 근거는 10/14 검증 기록과 보존된 배포 정보입니다.\n'
          '- **샘플 1개 = 학습 CSV 1행의 음성 녹음 파일 1개**입니다. 세 국가에는 중복 경로와 별도 증강 행이 없어 학습 행 수와 고유 녹음 수가 같습니다.\n'
          '- **화자 수 = 데이터셋 구분자가 포함된 고유 화자키 수**입니다. 로더의 `speaker_id`는 행 번호이므로 사람 수로 세지 않았습니다. 서로 다른 데이터셋에 같은 사람이 다른 ID로 등장하는 경우까지 확인한 실제 인원 수는 아닙니다.\n'
          '- **문장 길이 = 전체 녹음 라벨의 공백 기준 어절 수**입니다. 낭독 제시문과 실제 발화 전사를 따로 계산하며, 문법적 문장 수·형태소 수·LLM 토큰 수가 아닙니다. 한 녹음에 여러 문장이 들어갈 수 있습니다. 전사·제시문은 이 문서의 통계 작성에 사용했고, 국적분류 모델의 직접 입력은 음성입니다.\n'
          '- 학습 목록의 경로·라벨·순서·길이를 실제 학습 로더와 다시 대조했습니다. 원본 음성 일부는 10/11 정리 이후 삭제된 상태이므로 음성 길이는 보존된 매니페스트 값을 사용했습니다. 과거 학습량과 현재 보유 음성량은 다릅니다. 원문 누락을 0어절로 채우지 않았습니다.',
          '## 2. 학습량과 화자 분포',
          table(['국가', '학습 녹음', '화자키', '전체 41클래스 학습 비중', '세 국가 내 비중', '원본 녹음 시간'],
                [[KO[label], n(stats[label]['rows']), n(stats[label]['speakers']), f(stats[label]['share_of_all_training_pct']) + '%',
                  f(stats[label]['share_within_three_countries_pct']) + '%', f(stats[label]['hours']) + '시간'] for label in LABELS]),
          '### 화자당 녹음 수',
          table(['국가', '평균', '최소', '중앙값(P50)', 'P90', '최대', '상위 10화자 녹음 비중'],
                [[KO[label], f(stats[label]['speaker_recording_counts']['mean'])] +
                 [n(stats[label]['speaker_recording_counts'][key]) for key in ['min', 'p50', 'p90', 'max']] +
                 [f(stats[label]['largest_10_speakers_recording_share_pct']) + '%'] for label in LABELS]),
          '우즈베키스탄은 필리핀보다 녹음 수가 많지만 화자키는 훨씬 적습니다. 따라서 녹음 수만으로 학습 데이터의 다양성을 판단하면 안 됩니다. P10·P50·P90은 누적 순위(nearest-rank) 기준입니다.',
          '### 학습·검증·테스트 분할',
          table(['국가', '학습 녹음 / 화자키', '검증 녹음 / 화자키', '테스트 녹음 / 화자키'],
                [[KO[label]] + [n(existing[label, split]['data_rows']) + ' / ' + n(existing[label, split]['speaker_keys']) for split in SPLITS] for label in LABELS]),
          '세 국가 각각 학습↔검증, 학습↔테스트, 검증↔테스트 간 **화자키 교집합 0개**를 재확인했습니다. 음성 중복·다른 ID의 동일 인물까지 이번 문서에서 새로 감사한 것은 아닙니다. 검증·테스트 자료는 위 학습량에 포함하지 않습니다.',
          '## 3. 발화 과제와 데이터 출처',
          '### 발화 과제 — 개수와 해당 국가 학습 데이터 내 비율',
          table(['과제', '중국', '우즈베키스탄', '필리핀'],
                [[title] + [n(stats[label]['task_types'][kind]) + ' (' + f(stats[label]['task_types'][kind] / stats[label]['rows'] * 100) + '%)' for label in LABELS]
                 for kind, title in [('word', '단어 읽기'), ('reading', '문장·문단 낭독'), ('question_answer', '질문 응답'), ('free', '자유발화'), ('other', '기타')]]),
          '문장형 녹음(낭독+질문 응답)은 중국 **39,327개**, 우즈베키스탄 **19,936개**, 필리핀 **16,217개**입니다. 자유발화는 별도로 계산했습니다. 위 숫자는 파일 수이므로 문법적 문장 개수로 해석하지 않습니다.',
          table(['국가', '출처/도메인', '학습 녹음', '화자키', '국가 내 비율'],
                [[KO[row['label']], row['domain'], n(row['rows']), n(row['speakers']), f(row['pct']) + '%'] for row in sources]),
          '`web040`=교육용 영어 모국어 사용자의 한국어 음성, `web041`=유럽어, `web042`=중·일어, `web043`=아시아어(중·일어 제외). `p505`는 내부 도메인명이며 실제 보존 코퍼스 폴더는 `131.인공지능_학습을_위한_외국인_한국어_발화_음성_데이터`입니다.',
          '## 4. 음성 길이 — 초 단위',
          '아래는 단어 과제를 포함한 **원본 녹음 전체 길이**입니다.',
          table(['국가', '평균', '최소', 'P10', '중앙값(P50)', 'P90', '최대'],
                [[KO[label]] + [f(stats[label]['duration_seconds'][key]) for key in ['mean', 'min', 'p10', 'p50', 'p90', 'max']] for label in LABELS]),
          table(['원본 음성 길이', '중국', '우즈베키스탄', '필리핀'],
                [[name] + [n(stats[label]['duration_bins'][name]['count']) + ' (' + f(stats[label]['duration_bins'][name]['pct']) + '%)' for label in LABELS]
                 for name in stats[LABELS[0]]['duration_bins']]),
          '### 모델이 받는 학습 입력 길이',
          '해당 최종 학습은 **녹음 앞 최대 ' + f(cfg['max_sec'], 0) + '초**를 사용했습니다(`max_sec=' + str(cfg['max_sec']) + '`, 증강 확률 `aug=' + str(cfg['aug']) + '`). 검증 입력 제한은 12초입니다. 따라서 원본 녹음의 전체 시간과 모델 입력 시간은 다릅니다.',
          table(['국가', '원본이 8초 초과인 녹음', '국가 내 비율', '앞 8초 제한 후 입력 시간 합계(추정)'],
                [[KO[label], n(stats[label]['longer_than_train_cap']), f(stats[label]['longer_than_train_cap'] / stats[label]['rows'] * 100) + '%',
                  f(stats[label]['estimated_front_crop_hours']) + '시간'] for label in LABELS]),
          '입력 시간 합계는 매니페스트 길이에 `min(길이, 8초)`를 적용한 값입니다. 당시 디코딩 성공 여부·스킵·패딩·실제 옵티마이저 업데이트별 노출 시간을 재구성한 값은 아닙니다. 8초 구간에 실제 몇 어절이 들어가는지는 시간 정렬 자료가 없어 계산하지 않았습니다.',
          '## 5. 문장·발화 길이 — 어절 단위',
          '### 낭독 제시문: 녹음 횟수 기준',
          '같은 문장을 여러 사람이 읽으면 그 녹음 수만큼 집계합니다. 단어 읽기와 질문 제시문은 제외했습니다.',
          table(['국가', '낭독 녹음 / 제시문 확인', '평균 어절', 'P10', '중앙값', 'P90', '최소–최대', '고유 제시문'],
                [[KO[label], n(lengths[label, 'train', 'reading_prompt_recording']['eligible_recordings']) + ' / ' + n(lengths[label, 'train', 'reading_prompt_recording']['recordings_with_text'])] +
                 [f(lengths[label, 'train', 'reading_prompt_recording']['mean_eojeol'])] +
                 [n(lengths[label, 'train', 'reading_prompt_recording'][key + '_eojeol']) for key in ['p10', 'p50', 'p90']] +
                 [lengths[label, 'train', 'reading_prompt_recording']['min_eojeol'] + '–' + lengths[label, 'train', 'reading_prompt_recording']['max_eojeol'],
                  n(lengths[label, 'train', 'reading_prompt_recording']['unique_texts'])] for label in LABELS])]
    for kind, heading in [('reading_prompt_recording', '낭독 제시문 길이 분포'), ('spoken_transcript_recording', '실제 발화 전사 길이 분포')]:
        if kind == 'spoken_transcript_recording':
            md.extend(['### 실제 발화 전사: 텍스트 확인률과 길이',
                       '단어 과제를 제외한 낭독·응답·자유발화를 대상으로 합니다. 실제 전사가 있는 녹음만 길이 통계에 넣었습니다. 질문 문장을 답변 전사로 대체하지 않았습니다.',
                       table(['국가', '대상 녹음', '전사 확인', '전사 미확인', '확인률', '평균 어절', '중앙값', 'P90', '최소–최대'],
                             [[KO[label]] + [n(lengths[label, 'train', kind][key]) for key in ['eligible_recordings', 'recordings_with_text', 'recordings_missing_text']] +
                              [f(lengths[label, 'train', kind]['text_coverage_pct']) + '%', f(lengths[label, 'train', kind]['mean_eojeol']),
                               n(lengths[label, 'train', kind]['p50_eojeol']), n(lengths[label, 'train', kind]['p90_eojeol']),
                               lengths[label, 'train', kind]['min_eojeol'] + '–' + lengths[label, 'train', kind]['max_eojeol']] for label in LABELS])])
        md.extend(['#### ' + heading,
                   table(['어절 구간', '중국', '우즈베키스탄', '필리핀'],
                         [[key] + [n(lengths[label, 'train', kind]['bin_' + key + '_count']) + ' (' + f(lengths[label, 'train', kind]['bin_' + key + '_pct']) + '%)' for label in LABELS] for key in EOJEOL_BINS]),
                   '각 비율의 분모는 해당 종류의 텍스트가 확인된 녹음 수입니다.'])
    md.extend(['### 같은 제시문을 한 번만 셌을 때',
               table(['국가', '고유 낭독 제시문', '평균 어절', '중앙값', 'P90'],
                     [[KO[label], n(lengths[label, 'train', 'reading_prompt_unique']['unique_texts']),
                       f(lengths[label, 'train', 'reading_prompt_unique']['mean_eojeol']),
                       n(lengths[label, 'train', 'reading_prompt_unique']['p50_eojeol']),
                       n(lengths[label, 'train', 'reading_prompt_unique']['p90_eojeol'])] for label in LABELS]),
               '고유 제시문 평균은 세 국가 모두 약 14어절이지만, 녹음 횟수로 가중한 평균은 다릅니다. 특히 필리핀에서는 짧은 제시문 녹음이 상대적으로 많이 포함돼 있습니다. 고유 제시문 수는 문법적 문장 경계로 나눈 개수가 아니라 공백 정규화 후 서로 다른 전체 문자열 수입니다.',
               '## 6. 모델 국가 라벨 안의 모국어 구성',
               '국가 라벨 하나에 여러 모국어가 섞여 있습니다. 아래 언어명은 원본 메타데이터 표기를 유지했습니다. 040~043은 `nationality`, 131은 기존 모국어·거주국 매핑 규칙을 사용하므로 모델 라벨을 실제 국적과 동일시하지 않습니다.',
               table(['모델 국가', '출처', '원본 모국어 표기', '학습 녹음', '화자키', '국가 내 비율'],
                     [[KO[row['label_en']], row['dataset'], row['mother_tongue_raw'], n(row['data_rows']), n(row['speaker_keys']),
                       f(int(row['data_rows']) / stats[row['label_en']]['rows'] * 100) + '%']
                      for label in LABELS for row in sorted([r for r in mother if r['label_en'] == label], key=lambda r: -int(r['data_rows']))]),
               '우즈베키스탄에는 러시아어 표기 **7,079개(' + f(7079 / stats['Uzbekistan']['rows'] * 100) + '%)**, 필리핀에는 영어 표기 **19,978개(' + f(19978 / stats['Philippines']['rows'] * 100) + '%)**가 포함됩니다. 이는 한국어 음성을 말한 화자의 모국어 메타데이터이며, 학습 음성의 언어가 러시아어·영어라는 뜻은 아닙니다.',
               '## 7. 보존된 원본 후보 목록과 최종 학습량',
               table(['국가', '삭제 전 원본 인덱스 녹음', '최종 학습 녹음', '원본 인덱스 대비 비율'],
                     [[KO[label], n(stats[label]['historical_index_recordings']), n(stats[label]['rows']),
                       f(stats[label]['rows'] / stats[label]['historical_index_recordings'] * 100) + '%'] for label in LABELS]),
               '원본 인덱스는 이 프로젝트에서 확보해 목록화한 040~043·131 데이터의 역사적 기록이며 **AI Hub 전체 제공량**을 뜻하지 않습니다. 검증·테스트 후보와 학습 선별에서 빠진 자료도 포함하므로 위 비율을 단순 무작위 샘플링률로 해석하지 않습니다. 현재 남은 음성 파일 비율도 아닙니다.',
               '## 8. 해석과 확인 결과',
               '- **녹음 수와 화자 다양성의 순위가 다릅니다.** 중국의 녹음 수가 가장 많지만 화자키는 필리핀이 가장 많습니다. 우즈베키스탄은 화자당 녹음이 집중돼 있습니다.\n'
               '- **발화 과제 비율이 다릅니다.** 중국은 낭독 비율이 높고, 우즈베키스탄·필리핀은 단어 과제 비율이 높습니다. 문장 길이 비교에서는 이 차이를 분리해야 합니다.\n'
               '- **원본 길이와 실제 입력 길이를 구분해야 합니다.** 학습 입력은 음성 앞 최대 8초이며, 원본 전사 전체의 어절 수는 그 구간의 어절 수와 일치한다고 볼 수 없습니다.\n'
               '- **국가 라벨은 순수한 단일 모국어 집단이 아닙니다.** 특히 필리핀의 영어, 우즈베키스탄의 러시아어 비중을 함께 봐야 합니다. 이 분포만으로 오분류의 인과 원인을 확정하지는 않습니다.\n'
               '- 실제 로더의 학습·검증 목록과 매니페스트 **행 수·입력 순서·경로·라벨·길이 일치**, 기존 통계의 클래스별 행 수·화자 수·과제 합계·문장 길이 구간 합계 일치를 확인했습니다. 세 국가의 분할 간 화자키 중복은 0개입니다.',
               '## 9. 근거 파일과 재현',
               '\n'.join('- ' + link(p, title) for p, title in [
                   (manifest_root / 'train.csv', '최종 학습 매니페스트'),
                   (BASE / 'run_zeroth/_train_loader.csv', '실제 학습 로더 목록'),
                   (BASE / 'run_zeroth/history.json', '실제 학습 설정 및 이력'),
                   (BASE / 'run_zeroth/deploy/model_info.json', '배포 체크포인트 정보'),
                   (META / 'model_country_split_distribution.csv', '전체 국가별 분할 통계'),
                   (META / 'sentence_length_country_summary.csv', '전사·제시문 길이 통계'),
                   (META / 'model_country_mother_tongue_distribution.csv', '모국어 메타데이터 통계'),
                   (NAT / 'uzbek_china_validation_20261014/health_before.json', '10/14 서비스 모델 확인 기록'),
                   (args.out / 'statistics.json', '이번 보고서의 수치·입력 해시·검증 결과'),
                   (Path(__file__).resolve(), '집계 및 Markdown 생성 스크립트')]),
               '```bash\npython tools/nationality_three_country_report.py\n```'])
    filename = '중국_우즈베키스탄_필리핀_학습데이터_20261015.md'
    body = '\n\n'.join(md) + '\n'
    path = args.out / filename
    path.write_text(body, encoding='utf-8')
    repo_copy = REPO / 'tasks/nationality-classification/results' / filename
    repo_copy.write_text(body, encoding='utf-8')
    assert sha(path) == sha(repo_copy)
    print(json.dumps({'report': str(repo_copy), 'artifact_copy': str(path), 'all_checks_passed': True,
                      'countries': {label: {key: stats[label][key] for key in ['rows', 'speakers', 'hours', 'duration_seconds', 'speaker_recording_counts', 'historical_index_recordings']} for label in LABELS}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=NAT / 'china_uzbekistan_philippines_training_20261015')
    main(parser.parse_args())
