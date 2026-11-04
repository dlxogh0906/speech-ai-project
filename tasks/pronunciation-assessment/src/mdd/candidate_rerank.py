"""Fixed-M3 pair reranking. Acoustic inputs never contain TAG or TAG times."""
import numpy as np
from .phone_probability_atlas import greedy_runs
from .h0_partial_reference import UNKNOWN
from .h0_partial_scoring import mapping_indices, unique_index

FAMILIES = {'sibilants': ['ㅅ', 'ㅆ'], 'affricates': ['ㅈ', 'ㅉ', 'ㅊ']}
VOWELS = set('ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ')


def acoustic_summary(audio, center, radius):
    """Broad spectral/periodicity proxies, not measured VOT or true boundaries."""
    lo = max(0, int(center-radius)); hi = min(len(audio), int(center+radius))
    x = np.asarray(audio[lo:hi], dtype=np.float64)
    if len(x) < 160: return [0.] * 6
    frames = np.lib.stride_tricks.sliding_window_view(x, 160)[::80]
    power = abs(np.fft.rfft(frames*np.hanning(160), n=256))**2
    hz = np.fft.rfftfreq(256, 1/16000.)
    low = power[:, (hz >= 100) & (hz < 3000)].sum(1)
    high = power[:, (hz >= 3000) & (hz < 7900)].sum(1)
    centroid = (power*hz).sum(1)/(power.sum(1)+1e-15)/8000.
    ratios = 10*np.log10((high+1e-12)/(low+1e-12))
    zcr = (frames[:, 1:]*frames[:, :-1] < 0).mean()
    w = x-x.mean(); ac = np.correlate(w, w, 'full')[len(w)-1:]
    periodicity = float(ac[32:min(201, len(ac))].max()/(ac[0]+1e-12))
    return [float(ratios.mean()), float(ratios.std()), float(centroid.mean()),
            float(centroid.std()), float(zcr), periodicity]


def features(audio, lp, centers, reference, vocab, blank=0):
    """Return output-selected anchors/features before examining any actual label.

    Candidate families are complete (2 or 3 phones). A unique emitted in-family
    token is required; deleted/third-phone/ambiguous outputs stay outside this
    bounded re-ranking input. All evaluation denominators retain those sites.
    """
    runs = greedy_runs(lp, vocab, blank); hyp = [r['token'] for r in runs]
    idx = mapping_indices(reference, hyp); result = {}
    for q, expected in enumerate(reference):
        family = next((f for f, phones in FAMILIES.items() if expected in phones), None)
        if family is None: continue
        phones = FAMILIES[family]; j = unique_index(idx[q])
        row = dict(q=q, family=family, expected=expected, available=False)
        if j is None:
            row['reason'] = 'no_unique_emitted_token'; result[q] = row; continue
        run = runs[j]; row['original'] = run['token']
        if run['token'] not in phones:
            row['reason'] = 'original_output_outside_family'; result[q] = row; continue
        peak = run['peak']; ids = [vocab.index(p) for p in phones]
        logp = lp[peak, ids].astype(float); probs = np.exp(logp)
        base = list(probs) + list(logp-logp.max())
        base += [float(np.exp(lp[peak, blank])), float(np.exp(lp[peak]).max())]
        base += [float(expected == p) for p in phones]
        base += [float(run['token'] == p) for p in phones]
        cf = np.arange(max(0, peak-3), min(len(lp), peak+4))
        vj = unique_index(idx[q+1]) if q+1 < len(reference) and reference[q+1] in VOWELS else None
        vp = runs[vj]['peak'] if vj is not None and vj > j else None
        vf = np.arange(max(0, vp-2), min(len(lp), vp+3)) if vp is not None else np.array([], int)
        extra = [float(vp is not None), 0. if vp is None else min(1., (vp-peak)*.02),
                 (run['end']-run['start'])*.02]
        for frames in (cf, vf):
            if len(frames):
                block = lp[frames][:, ids].astype(float)
                extra += list(np.exp(block).mean(0)) + list(np.exp(block).max(0))
                extra += list((block-block.max(1, keepdims=True)).mean(0))
            else: extra += [0.]*(3*len(ids))
        extra += acoustic_summary(audio, centers[peak], 960)
        extra += acoustic_summary(audio, centers[vp], 640) if vp is not None else [0.]*6
        support = list(cf) + list(vf)
        row.update(available=True, peak=peak, center=int(centers[peak]),
                   vowel_center=None if vp is None else int(centers[vp]),
                   support_samples=[int(centers[min(support)]-160), int(centers[max(support)]+160)],
                   base=base, segment=base+extra,
                   original_score=float(logp[phones.index(run['token'])]-logp[phones.index(expected)]),
                   raw_family_probabilities=probs.tolist())
        result[q] = row
    return result, hyp


def budget_threshold(scores, budget_fraction=.01):
    """Strictly exclude a boundary tie; -inf means the policy makes no proposal."""
    scores = np.asarray(scores, dtype=float)
    assert not np.isnan(scores).any()
    budget = int(np.floor(len(scores)*budget_fraction))
    finite = np.sort(scores[np.isfinite(scores)])[::-1]
    if len(finite) <= budget: return 0., budget
    return float(np.nextafter(finite[budget], np.inf)), budget


def outputs(model, rows, key):
    """OTHER is an internal veto, never an emitted phoneme or an accepted error."""
    x = np.array([r[key] for r in rows], dtype=float)
    lp = model.predict_log_proba(x); classes = model.classes_.tolist()
    result = []
    for r, p in zip(rows, lp):
        chosen = classes[int(p.argmax())]
        pred = r['original'] if chosen == 'OTHER' else chosen
        score = -np.inf
        if chosen != 'OTHER' and pred != r['expected']:
            score = float(p[classes.index(chosen)]-p[classes.index(r['expected'])])
        result.append((pred, score, chosen == 'OTHER'))
    return result
