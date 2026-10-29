"""Hold the selected dialect ensemble in memory and score decoded waveforms (serving, 2026-10-26).

The models and the combination rule are the ones the sealed test scored (scripts/eval/score_sealed_test.py):
EAT Large and XLS-R 2B, each rebuilt as generic encoder + saved encoder_delta + head + mu/sd, combined by the rule
selected on validation (comparison_report.json 'selected'). Every window goes through the evaluation preprocessing
itself — train_dialect_comparison.PreparedClips reads it from an in-memory WAV and Collator pads it — one window at a time.
scripts/eval/check_serving_parity.py compares this path with the saved sealed-test logits.

Windows must be 16 kHz mono float32, 1-12 s (the loader rejects < 1 s; training windows were 3-12 s).
"""
import io
import json
from pathlib import Path
import sys
import threading
import time

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
LABELS = ['Gangwon', 'Gyeongsang', 'Jeolla', 'Jeju', 'Chungcheong', 'Standard']
KO = dict(Gangwon='강원', Gyeongsang='경상', Jeolla='전라', Jeju='제주', Chungcheong='충청', Standard='표준어')
SR = 16000
MIN_SEC, MAX_SEC = 1.0, 12.0
KINDS = ('eat_large', 'xlsr2b')


class DialectEnsemble:
    def __init__(self, root, device=None):
        import torch
        for sub in ('train', 'eval'):
            if str(SCRIPTS / sub) not in sys.path:
                sys.path.insert(0, str(SCRIPTS / sub))
        import train_dialect_comparison as engine
        from score_sealed_test import candidates, softmax
        self.engine, self.candidates, self.softmax = engine, candidates, softmax
        self.root = Path(root)
        report = json.loads((self.root / 'comparison_report.json').read_text())
        self.rule = report['selected']
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.models = {kind: self._load(kind) for kind in KINDS}

    def _load(self, kind):
        """Same reconstruction as score_sealed_test.run()."""
        import torch
        from torch import nn
        encoder = self.engine.Encoder(kind, self.device)
        if kind == 'xlsr2b':
            from fairseq.modules.checkpoint_activations import unwrap_checkpoint
            unwrap_checkpoint(encoder.model)
        ck = torch.load(self.root / kind / 'best.pt', map_location='cpu')
        params = dict(encoder.model.named_parameters())
        missing = [k for k in ck['encoder_delta'] if k not in params]
        assert not missing, missing[:3]
        with torch.no_grad():
            for name, value in ck['encoder_delta'].items():
                params[name].copy_(value.to(self.device))
        head = nn.Linear(len(ck['mu']), len(LABELS)).to(self.device)
        head.load_state_dict(ck['head'])
        encoder.model.eval(); head.eval()
        return dict(encoder=encoder, head=head, selected=[encoder.names.index(n) for n in ck['feature_names']],
                    mu=torch.as_tensor(ck['mu'], device=self.device), sd=torch.as_tensor(ck['sd'], device=self.device))

    def _logits(self, kind, wav):
        import soundfile as sf
        import torch
        buffer = io.BytesIO()
        sf.write(buffer, wav, SR, format='WAV', subtype='FLOAT')
        row = dict(audio_path=io.BytesIO(buffer.getvalue()), start_sec='0', duration_sec='%.6f' % (len(wav) / SR))
        _, values, lengths, errors = self.engine.Collator(kind)([self.engine.PreparedClips([row], kind)[0]])
        if errors:
            raise ValueError(errors[0][1])
        m = self.models[kind]
        with torch.no_grad():
            feature = m['encoder'](values, lengths)[:, m['selected'], :].reshape(1, -1).half().float()
            return m['head']((feature - m['mu']) / m['sd']).cpu().numpy()

    def score(self, windows):
        """windows: list of float32 arrays → per-window probabilities of each model and of the selected rule."""
        import numpy as np
        for w in windows:
            if not MIN_SEC <= len(w) / SR <= MAX_SEC + 1e-3:
                raise ValueError('window must be %.0f-%.0f s, got %.2f s' % (MIN_SEC, MAX_SEC, len(w) / SR))
        t0 = time.perf_counter()
        with self.lock:
            logits = {kind: np.concatenate([self._logits(kind, w) for w in windows]) for kind in KINDS}
        probs = {kind: self.softmax(logits[kind]) for kind in KINDS}
        probs['selected'] = self.candidates(probs['eat_large'], probs['xlsr2b'])[self.rule]
        probs['encode_ms'] = (time.perf_counter() - t0) * 1000
        return probs
