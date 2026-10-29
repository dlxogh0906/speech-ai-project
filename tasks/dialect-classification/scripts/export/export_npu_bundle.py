"""Check the pure NPU encoders against the sealed-test scoring on GPU, then write the NPU bundle's model folder (2026-10-28).

Rebuilds both encoders exactly as the sealed test scored them (dialect_classifier.ensemble.DialectEnsemble), copies the
weights into the pure modules (dialect_classifier.npu_models) rounded to fp16 as they will be shipped, and scores --n
sealed-test windows (fixed seed, the same draw as scripts/eval/check_serving_parity.py) through the pure path with
12 s static padding. Compares with the saved sealed-test logits and with the current serving path one window at a time.
Prints aggregate agreement only.

    PYTHONPATH=/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38 \
        $PY tasks/dialect-classification/scripts/export/export_npu_bundle.py \
        --root /data/speech-ai/out/dialect/content_grouped_full_20261015 [--out <bundle>/models]

--out writes meta.json · <kind>/pure_fp16.pt · <kind>/head.npz and parity.npz (the --n windows as int16 PCM with their
label ids, the saved sealed-test probabilities and the GPU pure-path pooled features — no paths or speaker fields), about 5 GB.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import csv
import datetime
import json
from pathlib import Path


def agreement(got, ref):
    return dict(argmax_agreement=float((got.argmax(1) == ref.argmax(1)).mean()),
                max_abs_prob_diff=float(abs(got - ref).max()))


def main(args):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from dialect_classifier.ensemble import DialectEnsemble, KINDS, LABELS, SR
    from dialect_classifier import npu_models as nm
    from dialect_classifier.npu_ensemble import combine, softmax

    ens = DialectEnsemble(args.root)
    dev = ens.device
    with (args.root / 'manifests' / 'test.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    saved = {k: np.load(args.root / args.sealed / ('%s_test_logits.npz' % k)) for k in KINDS}
    for k in KINDS:
        assert list(saved[k]['row_id']) == [r['row_id'] for r in rows]
    pick = np.random.default_rng(args.seed).choice(len(rows), args.n, replace=False)
    ref = {k: softmax(saved[k]['logits'][pick]) for k in KINDS}
    ref['selected'] = combine(ens.rule, ref['eat_large'], ref['xlsr2b'])
    assert np.allclose(ref['selected'], ens.candidates(ref['eat_large'], ref['xlsr2b'])[ens.rule])

    banks, _ = torchaudio.compliance.kaldi.get_mel_banks(128, 512, float(SR), 20.0, 0.0, 100.0, -500.0, 1.0)
    mel_banks = torch.nn.functional.pad(banks, (0, 1)).float().numpy()          # kaldi.fbank 와 같게 오른쪽 0 한 칸

    pure, states, heads = {}, {}, {}
    for kind in KINDS:
        m = ens.models[kind]
        names = [m['encoder'].names[i] for i in m['selected']]
        assert names == ['layer%d_mean' % v for v in nm.SPECS[kind]['out_layers']], names
        original = m['encoder'].model.model if kind == 'eat_large' else m['encoder'].model
        states[kind] = {k: v.half() for k, v in nm.pure_state(kind, original).items()}
        model = nm.build(kind)
        model.load_state_dict({k: v.float() for k, v in states[kind].items()}, strict=True)
        pure[kind] = model.to(dev).eval()
        heads[kind] = dict(weight=m['head'].weight.detach().float().cpu().numpy(), bias=m['head'].bias.detach().float().cpu().numpy(),
                           mu=m['mu'].float().cpu().numpy(), sd=m['sd'].float().cpu().numpy())
        print('[pure] %s tensors=%d params=%d' % (kind, len(states[kind]), sum(v.numel() for v in states[kind].values())), flush=True)
    head = {k: nm.Head(heads[k]) for k in KINDS}

    got = {mode: {k: [] for k in KINDS} for mode in ('fp32', 'fp16')}
    serving = {k: [] for k in KINDS}
    pooled16 = {k: [] for k in KINDS}
    fbank_diff, pcm, labels, exact = 0.0, [], [], True
    for i in pick:
        r = rows[i]
        with sf.SoundFile(r['audio_path']) as source:
            source.seek(round(float(r['start_sec']) * SR))
            wav = source.read(round(float(r['duration_sec']) * SR), dtype='float32')
        q = np.round(wav * 32768).astype(np.int16)
        exact = exact and bool(np.array_equal(q.astype(np.float32) / 32768, wav))
        pcm.append(q)
        labels.append(int(r['label_id']))
        for k in KINDS:
            serving[k].append(ens._logits(k, wav)[0])
            a, b = nm.inputs(k, wav, mel_banks)
            if k == 'eat_large':
                w = torch.from_numpy(wav.copy())
                fb = torchaudio.compliance.kaldi.fbank((w - w.mean()).unsqueeze(0), htk_compat=True, sample_frequency=16000,
                                                       use_energy=False, window_type='hanning', num_mel_bins=128,
                                                       dither=0.0, frame_shift=10)
                mine = a[0, 0, :len(fb)] * nm.EAT_SCALE - nm.EAT_SHIFT
                fbank_diff = max(fbank_diff, float(np.abs(mine - fb.numpy()).max()))
            ta, tb = torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev)
            with torch.no_grad():
                p32 = pure[k](ta, tb).float().cpu().numpy()
                with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                    p16 = pure[k](ta, tb).float().cpu().numpy()
            got['fp32'][k].append(head[k](p32)[0])
            got['fp16'][k].append(head[k](p16)[0])
            pooled16[k].append(p16[0])
    serving = {k: softmax(np.array(v)) for k, v in serving.items()}
    serving['selected'] = combine(ens.rule, serving['eat_large'], serving['xlsr2b'])
    result = dict(rule=ens.rule, windows=int(args.n), fbank_max_abs_diff=fbank_diff,
                  serving_vs_saved={k: agreement(serving[k], ref[k]) for k in ref})
    for mode in got:
        p = {k: softmax(np.array(v)) for k, v in got[mode].items()}
        p['selected'] = combine(ens.rule, p['eat_large'], p['xlsr2b'])
        result['pure_%s_vs_saved' % mode] = {k: agreement(p[k], ref[k]) for k in ref}
        result['pure_%s_vs_serving' % mode] = {k: agreement(p[k], serving[k]) for k in ref}
        result['pure_%s_top1_on_labels' % mode] = float((p['selected'].argmax(1) == np.array(labels)).mean())
    result['saved_top1_on_labels'] = float((ref['selected'].argmax(1) == np.array(labels)).mean())
    print(json.dumps(result, indent=2), flush=True)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for kind in KINDS:
            files[kind] = dict(state='pure_fp16.pt', rbln='%s_static12s.rbln' % kind, spec=nm.SPECS[kind])
            if args.parity_only:
                continue
            (out / kind).mkdir(parents=True, exist_ok=True)
            torch.save(states[kind], str(out / kind / 'pure_fp16.pt'))
            extra = dict(mel_banks=mel_banks) if kind == 'eat_large' else {}
            np.savez(str(out / kind / 'head.npz'), **heads[kind], **extra)
        lengths = np.array([len(x) for x in pcm])
        np.savez_compressed(str(out / args.parity_name), pcm=np.concatenate(pcm), lengths=lengths, labels=np.array(labels),
                            pcm_exact=exact, **{'saved_%s' % k: v for k, v in ref.items()},
                            **{'pooled_gpu_fp16_%s' % k: np.array(v) for k, v in pooled16.items()})
        meta = dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'), labels=LABELS, rule=ens.rule,
                    source='%s (sealed test %s)' % (args.root.name, args.sealed), files=files,
                    inputs=dict(xlsr2b=['source [1,%d]' % nm.STATIC_SAMPLES, 'frame_valid [1,%d]' % nm.XLSR_FRAMES],
                                eat_large=['fbank [1,1,%d,128]' % nm.EAT_FRAMES, 'patch_valid [1,%d]' % nm.EAT_PATCHES]),
                    gpu_check=result)
        if not args.parity_only:
            (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[out] %s' % out, flush=True)
    print('EXPORT_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sealed', default='sealed_test_20261026')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', help='번들의 models 폴더 — 주면 가중치 · 헤드 · 대조 세트를 쓴다(약 5 GB)')
    parser.add_argument('--parity-only', action='store_true', help='--out 에 대조 세트만 쓴다(가중치 · 헤드 · meta.json 은 그대로 둔다)')
    parser.add_argument('--parity-name', default='parity.npz')
    main(parser.parse_args())
