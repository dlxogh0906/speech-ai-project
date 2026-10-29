"""Compare identical saved recordings on SpeechSuper and the local sentence API.

Credentials: SPEECHSUPER_APP_KEY / SPEECHSUPER_SECRET_KEY (environment only).
Uses sequential requests. Saves redacted responses and paired scores, including errors.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
import wave


def post(url, params, audio, token=None):
    boundary = uuid.uuid4().hex
    body = ('--' + boundary + '\r\nContent-Disposition: form-data; name="text"\r\n\r\n'
            + json.dumps(params) + '\r\n--' + boundary
            + '\r\nContent-Disposition: form-data; name="audio"; filename="sample.wav"'
            + '\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += audio + ('\r\n--' + boundary + '--\r\n').encode()
    headers = {'Content-Type': 'multipart/form-data; boundary=' + boundary, 'Request-Index': '0'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    started = time.monotonic()
    try:
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=45)
        except urllib.error.HTTPError as e:
            response = e
        with response:
            status, raw = response.status, response.read()
        data = json.loads(raw)
        for key in ('params', 'applicationId'):
            data.pop(key, None)
    except (OSError, ValueError) as e:
        status, data = None, {'error': type(e).__name__}
    return {'http_status': status, 'seconds': round(time.monotonic() - started, 3), 'body': data}


def external_params(text, record_id):
    app, secret = os.environ['SPEECHSUPER_APP_KEY'], os.environ['SPEECHSUPER_SECRET_KEY']
    ts, user = str(int(time.time())), 'comparison'
    sig = lambda value: hashlib.sha1(value.encode()).hexdigest()
    return {
        'connect': {'cmd': 'connect', 'param': {
            'sdk': {'version': 16777472, 'source': 9, 'protocol': 2},
            'app': {'applicationId': app, 'timestamp': ts, 'sig': sig(app + ts + secret)}}},
        'start': {'cmd': 'start', 'param': {
            'app': {'applicationId': app, 'timestamp': ts, 'userId': user, 'sig': sig(app + ts + user + secret)},
            'audio': {'audioType': 'wav', 'channel': 1, 'sampleBytes': 2, 'sampleRate': 16000},
            'request': {'coreType': 'sent.eval.kr', 'refText': text, 'tokenId': record_id}}}}


def select_samples(directory, count):
    candidates, seen = [], set()
    for meta in sorted(directory.glob('*.json')):
        d = json.loads(meta.read_text())
        wav = meta.with_suffix('.wav')
        if not wav.exists() or not d.get('sentence') or not d.get('scores'):
            continue
        with wave.open(str(wav)) as w:
            duration = w.getnframes() / w.getframerate()
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2) or not 1 <= duration <= 20:
                continue
        digest = hashlib.sha256(wav.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append({'id': meta.stem, 'audio': str(wav), 'sha256': digest,
                           'text': d['sentence'], 'seconds': duration,
                           'selection_score': d['scores'].get('pron', 0)})
    candidates.sort(key=lambda x: (x['selection_score'], x['id']))
    n = min(count, len(candidates))
    return [candidates[round(i * (len(candidates) - 1) / max(1, n - 1))] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recordings', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--count', type=int, default=30)
    ap.add_argument('--local-url', default='http://127.0.0.1:8932/sent.eval.kr')
    a = ap.parse_args()
    if a.count < 1:
        ap.error('--count must be positive')
    # Fail before sending any audio when credentials are absent.
    external_params('', 'check')
    a.out.mkdir(parents=True, exist_ok=True)
    samples = select_samples(a.recordings, a.count)
    (a.out / 'manifest.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    rows = []
    for i, sample in enumerate(samples):
        audio = Path(sample['audio']).read_bytes()
        local_params = {'start': {'param': {'request': {'refText': sample['text'], 'tokenId': sample['id']}}}}
        local = post(a.local_url, local_params, audio, os.environ.get('MDD_API_TOKEN'))
        remote = post('https://api.speechsuper.com/sent.eval.kr', external_params(sample['text'], sample['id']), audio)
        (a.out / (sample['id'] + '.json')).write_text(json.dumps(
            {'sample': sample, 'local': local, 'speechsuper': remote}, ensure_ascii=False, indent=2))
        row = {'id': sample['id'], 'text': sample['text'], 'seconds': sample['seconds']}
        for name, response in [('local', local), ('speechsuper', remote)]:
            row[name + '_status'] = response['http_status']
            result = response['body'].get('result') or {}
            row[name + '_error'] = response['body'].get('error') or response['body'].get('errorId')
            for key in ('overall', 'pronunciation', 'fluency', 'rhythm'):
                row[name + '_' + key] = result.get(key)
        rows.append(row)
        with (a.out / 'paired_scores.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print('{}/{} {} local={} speechsuper={}'.format(
            i + 1, len(samples), sample['id'], row['local_overall'], row['speechsuper_overall']), flush=True)
        if remote['http_status'] in (401, 403, 429) or not remote['body'].get('result'):
            print('Stopping after external API error; inspect saved response.', flush=True)
            break


if __name__ == '__main__':
    main()
