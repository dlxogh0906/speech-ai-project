"""Fixed Korean label attributes for a controlled auxiliary-CTC experiment.

These are properties of retained actual-phone labels, not measured VOT/F0,
independently adjudicated articulation, or context-sensitive surface allophones.
Vowels, h, and composite dictionary labels are explicitly distinguished from
the plain/tense/aspirated contrast; no annotation is repaired here.
"""
import torch

VOWELS = set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
COMPOSITES = set('ㄳㄵㄺㄻㄼㅄ')
GROUPS = {
    'manner': [('vowel', VOWELS), ('stop', set('ㄱㄲㅋㄷㄸㅌㅂㅃㅍ')),
               ('fricative', set('ㅅㅆㅎ')), ('affricate', set('ㅈㅉㅊ')),
               ('nasal', set('ㅁㄴㅇ')), ('liquid', set('ㄹ')),
               ('composite_label', COMPOSITES)],
    'series': [('vowel', VOWELS), ('plain', set('ㄱㄷㅂㅅㅈ')),
               ('tense', set('ㄲㄸㅃㅆㅉ')), ('aspirated', set('ㅋㅌㅍㅊ')),
               ('sonorant', set('ㅁㄴㅇㄹ')), ('h_unclassified', set('ㅎ')),
               ('composite_label', COMPOSITES)],
    'place': [('vowel', VOWELS), ('labial', set('ㅂㅃㅍㅁ')),
              ('alveolar', set('ㄷㄸㅌㅅㅆㄴㄹ')), ('palatal', set('ㅈㅉㅊ')),
              ('velar', set('ㄱㄲㅋㅇ')), ('glottal', set('ㅎ')),
              ('composite_label', COMPOSITES)],
}


def schema(vocab):
    result = {}
    for axis, groups in GROUPS.items():
        labels = ['<blank>'] + [name for name, _ in groups]
        mapping = {}
        for token in vocab:
            matches = [i + 1 for i, (_, phones) in enumerate(groups) if token in phones]
            if len(matches) != 1:
                raise ValueError('unmapped or multiply mapped phone: ' + token)
            mapping[token] = matches[0]
        result[axis] = dict(classes=labels, phone_to_id=mapping)
    return result


def feature_targets(phone_targets, vocab, spec):
    inv = {v: k for k, v in vocab.items()}
    # One feature label per actual phone. Adjacent equal classes are retained;
    # the auxiliary CTC must separate them with a blank, not drop a phone.
    return {axis: [[info['phone_to_id'][inv[t]] for t in y] for y in phone_targets]
            for axis, info in spec.items()}


def minimum_frames(target):
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def checked_ctc(logits, lengths, targets):
    sizes = torch.tensor([len(y) for y in targets], device=logits.device)
    needed = torch.tensor([minimum_frames(y) for y in targets], device=logits.device)
    if bool((sizes == 0).any()) or bool((needed > lengths).any()):
        raise ValueError('empty or unalignable auxiliary CTC target; no row may be dropped')
    flat = torch.tensor([t for y in targets for t in y], device=logits.device, dtype=torch.long)
    with torch.backends.cudnn.flags(enabled=False):
        losses = torch.nn.functional.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1),
            flat, lengths, sizes, blank=0, reduction='none', zero_infinity=False)
    if not torch.isfinite(losses).all():
        raise ValueError('nonfinite auxiliary CTC')
    return (losses / sizes).mean()


class FeatureHeads(torch.nn.Module):
    def __init__(self, width, spec):
        super().__init__()
        self.heads = torch.nn.ModuleDict({axis: torch.nn.Linear(width, len(info['classes']))
                                       for axis, info in spec.items()})
        for head in self.heads.values():
            torch.nn.init.normal_(head.weight, std=.02)
            torch.nn.init.zeros_(head.bias)

    def forward(self, features):
        return {axis: head(features) for axis, head in self.heads.items()}


def auxiliary_loss(outputs, lengths, targets):
    losses = {axis: checked_ctc(logits, lengths, targets[axis])
              for axis, logits in outputs.items()}
    return torch.stack(list(losses.values())).mean(), losses
