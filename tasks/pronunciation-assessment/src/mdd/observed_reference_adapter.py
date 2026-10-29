"""Inactive request-local reference rescoring; no second acoustic-model forward.

Instantiate explicitly against a loaded app, then call adapter.analyze(). The
original app.analyze remains untouched. close() restores the forward callback.
This adapter is restricted to apps without the optional segment detector; its
interaction with acceptance-aware detector policies needs separate validation.
"""
import contextvars
import copy
import numpy as np
from mdd.canonical_observed_time import observed_time_reference
from mdd.canonical_time import TIME_PHRASE


def _rescore(app, prior, wav, lp, canonical, country):
    from mdd.gop_feats_v2 import attach_pred_probs
    state, decoder = app.S, app.S["dec"]
    ids = [decoder.d.index(t) for t in canonical]
    if not ids or decoder.d.unk() in ids:
        raise ValueError("attested reference is outside the original vocabulary")
    features, extra = state["feats"](decoder, lp, ids, canonical, len(wav) / app.SR)
    actual = [t for t in extra["pred"] if len(t) == 1 and 0x3131 <= ord(t[0]) <= 0x3163]
    sites = app.align(canonical, actual)
    attach_pred_probs(decoder, lp, sites, actual)
    cph, pph = state["j2p"](canonical), state["j2p"](actual)
    for site in sites:
        site["canon_ph"] = cph[site["i"]] if len(cph) == len(canonical) else None
        site["pred_ph"] = pph[site["j"]] if site["j"] is not None and len(pph) == len(actual) else None
    feedback = state["l1fb"].annotate(canonical, sites, country or None) if state.get("l1fb") else None
    margins = extra.get("gops_m") or []
    scores = state["scorer"].utt_scores(features) if features else None
    raw = {"canon": list(canonical), "sites": sites, "margins": margins, "scores": scores,
           "vad": copy.deepcopy(prior["vad"]), "feats": features}
    primary = {"canon_jamo": list(canonical), "canon_phone": cph, "pred_jamo": actual,
               "pred_phone": pph, "sites": sites, "feedback": feedback, "scores": scores, "vad": raw["vad"]}
    session = {"scores": scores, "phone": [(state["ctx"](canonical, i, state["vowels"]), margins[i]) for i in range(len(margins))]}
    return primary, session, raw


class ObservedReferenceAdapter:
    def __init__(self, app, rescorer=None):
        if not app.S.get("cb"):
            raise ValueError("a canonical builder is required")
        if app.S.get("sz") is not None:
            raise ValueError("optional segment detector needs acceptance-aware integration first")
        if getattr(app.S["fwd"], "_observed_reference_recorder", False):
            raise ValueError("reference recorder already installed")
        self.app, self.original_forward = app, app.S["fwd"]
        self.original_analyze = app.analyze
        self.rescorer = rescorer or _rescore
        self.context = contextvars.ContextVar("observed_reference_frame_cache", default=None)

        def record_forward(decoder, wavs):
            lp, lengths = self.original_forward(decoder, wavs)
            cache = self.context.get()
            if cache is not None:
                if len(wavs) != 1:
                    raise ValueError("single-utterance analysis required")
                cache.update(wav=wavs[0].copy(), lp=lp.clone(), lengths=copy.deepcopy(lengths))
                cache["forward_calls"] += 1
            return lp, lengths

        record_forward._observed_reference_recorder = True
        self.record_forward = record_forward
        app.S["fwd"] = record_forward

    def close(self):
        if self.app.S["fwd"] is not self.record_forward:
            raise ValueError("forward callback changed after adapter installation")
        self.app.S["fwd"] = self.original_forward

    def analyze(self, wav, text, country):
        if not TIME_PHRASE.search(text):
            return self.original_analyze(wav, text, country)
        return self.analyze_with_trace(wav, text, country)[0]

    def analyze_with_trace(self, wav, text, country, apply_reference=True):
        cache = {"forward_calls": 0}
        token = self.context.set(cache)
        try:
            original = self.original_analyze(wav, text, country)
            before = original[2]
            trace = {"reason": "disabled", "accepted_positions": [], "reference_changed": False,
                     "forward_calls": cache["forward_calls"], "raw_before": before}
            if cache["forward_calls"] == 1:
                # Private diagnostic/detector integration data, never an API payload.
                trace["frame_log_probs"] = cache["lp"][0, :cache["lengths"][0]].numpy()
            if not apply_reference:
                return original, trace
            proposed, accepted, reason = observed_time_reference(self.app.S["cb"], text, before["canon"], before["sites"])
            trace.update(reason=reason, accepted_positions=accepted)
            if proposed == before["canon"]:
                return original, trace
            if cache["forward_calls"] != 1:
                raise ValueError("original acoustic posterior was not captured exactly once")
            lp = cache["lp"][0, :cache["lengths"][0]].numpy()
            result = self.rescorer(self.app, before, cache["wav"], lp, proposed, country)
            if result[0]["pred_jamo"] != original[0]["pred_jamo"]:
                raise AssertionError("reference rescoring changed acoustic recognition")
            old_sites, new_sites = before["sites"], result[2]["sites"]
            if (len(old_sites) != len(new_sites) or any(a.get("pred") != b.get("pred")
                    for i, (a, b) in enumerate(zip(old_sites, new_sites)) if i not in accepted)
                    or any(new_sites[q].get("pred") != "ㅆ" for q in accepted)):
                trace["reason"] = "deferred_other_phone_alignment_change"
                return original, trace
            formatter = self.app.S.get("ss")
            if formatter is not None:
                def phones(raw):
                    body = formatter.build(text, raw["canon"], raw["sites"], raw["margins"],
                                           utt_scores=raw["scores"], record_id=None,
                                           vowels=self.app.S["vowels"], feats=raw.get("feats"))
                    return [p for word in body["result"]["words"] for p in word["phonemes"]]
                old_phones, new_phones = phones(before), phones(result[2])
                keys = ("actual_jamo", "errorType", "pronunciation")
                if len(old_phones) != len(new_phones) or any(
                        any(a.get(k) != b.get(k) for k in keys)
                        for i, (a, b) in enumerate(zip(old_phones, new_phones)) if i not in accepted):
                    trace["reason"] = "deferred_other_phone_response_change"
                    return original, trace
            capture_context = getattr(self.app, "capture_context", None)
            capture = capture_context.get() if capture_context is not None else None
            if capture is not None:
                capture["pronunciation"] = dict(result[2], pred_jamo=result[0]["pred_jamo"])
            trace.update(reference_changed=True, forward_calls=cache["forward_calls"],
                         cached_posterior_bytes=int(cache["lp"].numel() * cache["lp"].element_size()))
            return result, trace
        finally:
            self.context.reset(token)
