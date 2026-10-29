"""Fixed observed upper-band shape comparison, without phonetic cutoff rules.

25ms Hann frames, global5ms grid, 16kHz original runtime ROI. FFT1024 zero
padding supplies frequency samples, not independent15.625Hz resolution.
No TAG boundary, actual-phone identity, F0 or expected consonant is an input.
"""
import hashlib
import numpy as np

SCHEMA = 'original16k_PCM_ROI_RMS_logpower25ms_hop5ms_full7900_upper4000_flat_v1'
WINDOW,HOP,NFFT = 400,80,1024
FREQUENCIES = np.fft.rfftfreq(NFFT,1./16000)
FREQUENCIES = FREQUENCIES[FREQUENCIES < 7900]
UPPER = FREQUENCIES >= 4000
BINS = len(FREQUENCIES)
METHODS = ('fine','upper_flat','native_mean')


def power_frames(source,feature):
    source = np.asarray(source)
    left,right = int(feature['left']),int(feature['right'])
    if source.ndim != 1 or not 0 <= left < right <= len(source):
        raise ValueError('original mono PCM and contained runtime ROI required')
    if hashlib.sha256(source[left:right].tobytes()).hexdigest() != feature['clip_sha256']:
        raise ValueError('original runtime PCM changed')
    first = (left+HOP-1)//HOP
    starts = np.arange(first*HOP,right-WINDOW+1,HOP,dtype=np.int64)
    power = np.full((len(starts),BINS),np.nan,np.float64)
    clip = source[left:right].astype(np.float64)
    if not np.isfinite(clip).all() or np.ptp(clip) == 0:
        return power,starts
    rms = np.sqrt(np.mean((clip-clip.mean())**2))
    window = np.hanning(WINDOW)
    for j,start in enumerate(starts):
        frame = source[start:start+WINDOW].astype(np.float64)
        if not np.isfinite(frame).all() or np.ptp(frame) == 0:
            continue
        frame = (frame-frame.mean())/rms
        power[j] = abs(np.fft.rfft(frame*window,NFFT))[:BINS]**2/np.sum(window**2)
    return power,starts


def flatten_upper(power):
    """Keep low bins exact and preserve every valid frame's total upper power."""
    value = np.asarray(power,np.float64)
    if value.ndim != 2 or value.shape[-1] != BINS:
        raise ValueError('fixed full observed spectrum required')
    result = value.copy()
    result[:,UPPER] = value[:,UPPER].mean(-1,keepdims=True)
    return result


def mean_pair(source,feature):
    power,starts = power_frames(source,feature)
    valid = np.isfinite(power).all(-1)
    result = np.full((2,BINS),np.nan,np.float64)
    energy_difference = 0.
    if valid.any():
        observed = power[valid]
        control = flatten_upper(observed)
        # Do not change the low-band input or trade spectral shape for level.
        np.testing.assert_array_equal(control[:,~UPPER],observed[:,~UPPER])
        a,b = control[:,UPPER].sum(-1),observed[:,UPPER].sum(-1)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        energy_difference = float(np.max(abs(a-b)/np.maximum(abs(b),1e-20)))
        result[0] = (10*np.log10(np.maximum(observed,1e-12))).mean(0)
        result[1] = (10*np.log10(np.maximum(control,1e-12))).mean(0)
    return result,{'frames':len(starts),'valid_frames':int(valid.sum()),
        'failed_frames':int((~valid).sum()),'first_start':int(starts[0]) if len(starts) else None,
        'last_end':int(starts[-1]+WINDOW) if len(starts) else None,
        'upper_power_max_relative_error':energy_difference}


def missing_balanced_abx(distances,actual_left,actual_right):
    """All fixed queries/references retained: any missing member gets chance."""
    distances = np.asarray(distances,np.float64)
    left,right = np.asarray(actual_left),np.asarray(actual_right)
    if (distances.shape != (len(left),len(right)) or np.isinf(distances).any()
            or any(set(z.tolist()) != {0,1} for z in (left,right))):
        raise ValueError('fixed two-class speaker halves and finite/NaN costs required')
    halves = []
    for matrix,query,reference in [(distances,left,right),(distances.T,right,left)]:
        scores = []
        for i,cls in enumerate(query):
            difference = matrix[i,reference == cls][:,None]-matrix[i,reference != cls][None]
            value = np.where(np.isfinite(difference),(difference < 0)+.5*(difference == 0),.5)
            scores.append(float(value.mean()))
        scores = np.asarray(scores)
        halves.append([float(scores[query == cls].mean()) for cls in (0,1)])
    return float(np.mean(halves)),halves


def summarize(scores,vowels,seed):
    scores,vowels = np.asarray(scores,np.float64),np.asarray(vowels)
    if scores.shape != (241,3) or len(vowels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('fixed241contexts and3pairedmean representations required')
    groups = {}
    for name in ('all','ㅣ','ㅡ','ㅜ'):
        mask = np.ones(len(scores),bool) if name == 'all' else vowels == name
        groups[name] = {'cells':int(mask.sum()),'accuracy':dict(zip(METHODS,scores[mask].mean(0).tolist()))}
    delta = scores[:,:1]-scores[:,1:]
    sample = np.random.RandomState(seed).randint(0,len(scores),(2000,len(scores)))
    reference = np.percentile(delta[sample].mean(1),[2.5,50,97.5],axis=0)
    means = scores.mean(0)
    i = groups['ㅣ']
    passed = bool(means[0] >= max(means[1:])+.02 and (reference[0] > 0).all()
        and i['cells'] == 51 and i['accuracy']['fine'] >= max(i['accuracy'][k] for k in METHODS[1:]))
    return {'groups':groups,'fine_minus_control_native':delta.mean(0).tolist(),
        'paired_context_bootstrap_p025_p50_p975':reference.tolist(),
        'bootstrap_scope':'descriptive TRAIN context resampling; speakers/recordings can span contexts, not independent confidence',
        'proceed_readout_training':passed}
