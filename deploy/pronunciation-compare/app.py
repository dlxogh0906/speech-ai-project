"""Read-only comparison viewer. Serves cached results; never calls assessment APIs."""
import argparse
import hmac
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8940)
    args = ap.parse_args()
    token = os.environ['COMPARE_VIEW_TOKEN']
    manifest = json.loads((args.results / 'manifest.json').read_text())
    records = []
    audio = {}
    for sample in manifest:
        record = json.loads((args.results / (sample['id'] + '.json')).read_text())
        audio[sample['id']] = Path(sample['audio'])
        records.append({'id': sample['id'], 'text': sample['text'], 'seconds': sample['seconds'],
                        'local': record['local']['body'], 'speechsuper': record['speechsuper']['body']})
    template = Path(__file__).with_name('index.html').read_text()
    payload = json.dumps(records, ensure_ascii=False).replace('<', '\\u003c')
    html = template.replace('__RECORDS__', payload).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            supplied = parse_qs(url.query).get('token', [''])[0]
            if not hmac.compare_digest(supplied, token):
                self.send_error(403)
                return
            name = url.path.rstrip('/').rsplit('/', 1)[-1]
            if name.endswith('.wav') and name[:-4] in audio:
                data = audio[name[:-4]].read_bytes()
                mime = 'audio/wav'
            elif url.path.endswith('/') or name in ('', 'pron-compare'):
                data, mime = html, 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Robots-Tag', 'noindex, nofollow')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass  # Do not log the access token in request URLs.

    print('Cached comparison viewer listening on 127.0.0.1:{}'.format(args.port), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
