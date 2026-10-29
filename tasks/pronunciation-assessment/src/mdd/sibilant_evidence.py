"""Join offline acoustic observations without promoting them to pronunciation truth.

Source identity, numerical availability, labeled-span containment and phonetic
interpretation are separate claims. This module never produces a score, fixes
an octave, infers a deleted vowel, or certifies a glottal source.
Branch-specific local support is separate from the larger common support used
for raw/filtered comparisons. Full-clip path dependence is retained explicitly.
"""
import math
import re

from mdd.sibilant_observations import interval_relation


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _span(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_finite(x) for x in value):
        raise ValueError("finite two-endpoint support required")
    if not 0 <= value[0] < value[1]:
        raise ValueError("positive nonempty support required")
    return [float(x) for x in value]


def _same_span(left, right):
    return all(abs(a - b) <= 1e-9 for a, b in zip(_span(left), _span(right)))


def support_evidence(span, target_span, phones, sample_rate=None):
    """Compare supports with existing labels, never with asserted acoustic gold."""
    span, target_span = _span(span), _span(target_span)
    relation = interval_relation(*span, *target_span, "existing_TAG_not_independent_acoustic_gold")
    within = relation["within_reference_interval"]
    basis = "continuous_seconds_with_1e-9_numerical_tolerance"
    if sample_rate is not None:
        if not _finite(sample_rate) or sample_rate <= 0:
            raise ValueError("positive sample rate required")
        a, b = [round(t * sample_rate) for t in span]
        lo, hi = [round(t * sample_rate) for t in target_span]
        within = lo <= a < b <= hi
        basis = "same_nearest_sample_quantization_as_requested_anchor"
    overlaps = [{"label": label, "interval_s": [s, e],
                 "overlap_s": min(e, span[1]) - max(s, span[0]),
                 "is_target_TAG_interval": _same_span([s, e], target_span)}
                for s, e, label in phones if min(e, span[1]) > max(s, span[0])]
    flags = [] if within else ["support_extends_outside_existing_TAG_vowel"]
    if not within and any(not r["is_target_TAG_interval"] for r in overlaps):
        flags.append("support_overlaps_other_existing_TAG_interval")
    return {"support_original_s": span, "reference_relation": relation,
            "within_existing_TAG_vowel": within, "containment_basis": basis,
            "overlapping_existing_TAG_intervals": overlaps, "review_flags": flags,
            "acoustic_phone_containment_verified": False}


def _candidate_support(frames, branch, support_key, interval, target, offset):
    a, b = interval
    supported = [(i, f) for i, f in enumerate(frames)
                 if f[support_key][0] >= a and f[support_key][1] <= b]
    candidates = [(i, f) for i, f in supported if f[branch]["candidate_hz"] is not None]
    if any(not _finite(f[branch]["candidate_hz"]) or f[branch]["candidate_hz"] <= 0 for _, f in candidates):
        raise ValueError("nonfinite or nonpositive selected periodicity candidate")
    inside = [i for i, f in candidates if interval_relation(
        *[t + offset for t in f[support_key]], *target, "existing_TAG")["within_reference_interval"]]
    return {"local_support_key_in_source_trace": support_key,
            "locally_supported_frame_indices": [i for i, _ in supported],
            "candidate_frame_indices": [i for i, _ in candidates],
            "candidate_frame_indices_contained_in_existing_TAG_vowel": inside,
            "state": "no_locally_supported_frame" if not supported else "candidate_returned" if candidates else "no_candidate_returned",
            "path_choice_disagreement_frames": sum(f[branch].get("path_comparison", {}).get("selected_candidate_changed", False) for _, f in supported),
            "path_choice_octave_review_frames": sum(f[branch].get("path_comparison", {}).get("near_octave_review_flag", False) for _, f in supported),
            "glottal_source_verified": False}


def _periodicity_views(observation, item, phones):
    offset = observation["offset_s"]
    result = {}
    for name, view in observation["intervals"].items():
        record = {"observation_status": view["status"], "interval_provenance": view["interval_provenance"],
                  "review_flags": [], "settings": [], "validated_glottal_frequency_hz": None,
                  "any_common_support_candidate_contained_in_existing_TAG_vowel": None,
                  "any_branch_specific_candidate_contained_in_existing_TAG_vowel": None,
                  "support_modes": {"common": "shared CC plus FIR support for raw/low-band comparisons",
                                    "branch_specific": "raw CC support for original PCM; CC plus FIR for low band"},
                  "global_dependencies_outside_local_support": observation["periodicity"].get("full_clip_dependence", []),
                  "pronunciation_decision": None}
        result[name] = record
        if "original_interval_s" in view:
            record["requested_support"] = support_evidence(view["original_interval_s"], item["tag_vowel"], phones)
            record["review_flags"].extend(record["requested_support"]["review_flags"])
        if view["status"] != "observed_candidates_only":
            record["review_flags"].append(view["status"])
            continue
        if view.get("glottal_source_validated") is not False or view.get("pronunciation_decision") is not None:
            raise ValueError("unsupported phonetic validation claim in candidate observations")
        if not _same_span([v + offset for v in view["interval_s"]], view["original_interval_s"]):
            raise ValueError("relative and original interval timestamps disagree")
        if any(c["near_octave_review_flag"] for c in view["cross_floor_comparisons"]):
            record["review_flags"].append("cross_floor_octave_disagreement_on_common_support")
        if len(view["settings"]) != len(observation["periodicity"]["settings"]):
            raise ValueError("periodicity setting count differs from trace")
        totals = {mode: [0, 0, 0] for mode in ("common", "branch_specific")}
        for setting, full in zip(view["settings"], observation["periodicity"]["settings"]):
            if setting["floor_hz"] != full["floor_hz"]:
                raise ValueError("periodicity settings do not match")
            modes = {mode: {} for mode in totals}
            for branch in ("raw", "low_band"):
                for mode in totals:
                    support_key = "raw_cc_support_s" if mode == "branch_specific" and branch == "raw" else "paired_support_s"
                    info = _candidate_support(full["frames"], branch, support_key, view["interval_s"], item["tag_vowel"], offset)
                    modes[mode][branch] = info
                    for k, field in enumerate(("locally_supported_frame_indices", "candidate_frame_indices", "candidate_frame_indices_contained_in_existing_TAG_vowel")):
                        totals[mode][k] += len(info[field])
                    if info["path_choice_disagreement_frames"]:
                        record["review_flags"].append("pitch_path_selection_sensitive_on_" + mode + "_support")
                    if info["path_choice_octave_review_frames"]:
                        record["review_flags"].append("pitch_path_octave_disagreement_on_" + mode + "_support")
                common = modes["common"][branch]
                summary = setting["branches"][branch]
                if len(common["locally_supported_frame_indices"]) != setting["supported_frames"]:
                    raise ValueError("summarized support count differs from trace")
                if len(common["candidate_frame_indices"]) != summary["returned_frames"]:
                    raise ValueError("summarized candidate count differs from trace")
                if common["state"] != summary["state"]:
                    raise ValueError("summarized candidate state differs from trace")
                for field in ("path_choice_disagreement_frames", "path_choice_octave_review_frames"):
                    if common[field] != summary[field]:
                        raise ValueError("summarized candidate path state differs from trace")
            if setting["branch_return_disagreements"]:
                record["review_flags"].append("raw_low_band_return_disagreement_on_common_support")
            record["settings"].append({"floor_hz": setting["floor_hz"], "common_support": modes["common"],
                                       "branch_specific_support": modes["branch_specific"]})
        for mode, (supported, returned, contained) in totals.items():
            if not supported:
                record["review_flags"].append("no_locally_supported_periodicity_frame_on_" + mode + "_support")
            elif not returned:
                record["review_flags"].append("no_periodicity_candidate_returned_on_" + mode + "_support")
            if returned:
                field = "any_" + mode + ("_support" if mode == "common" else "") + "_candidate_contained_in_existing_TAG_vowel"
                record[field] = bool(contained)
                if not contained:
                    record["review_flags"].append("no_" + mode + "_support_candidate_fully_contained_in_existing_TAG_vowel")
        record["review_flags"] = sorted(set(record["review_flags"]))
        record["interpretation"] = "Candidate availability and labeled containment do not establish glottal voicing or vowel deletion."
    return result


def _intensity_views(intensity, item, phones, sr):
    result = {}
    for anchor_name in ("TAG_anchor", "provisional_voicing_anchor"):
        anchor = intensity[anchor_name]
        variants = []
        for variant in anchor["variants"]:
            metrics = {}
            for name, source in variant["metrics"].items():
                target = {"observation_status": source["status"], "value_db_per_s": source.get("ols_db_per_s"),
                          "endpoint_value_db_per_s": source.get("endpoint_db_per_s"), "review_flags": []}
                metrics[name] = target
                if "ols_db_per_s" not in source:
                    target["review_flags"].append(source["status"])
                    continue
                if not all(_finite(source[k]) for k in ("ols_db_per_s", "endpoint_db_per_s")):
                    raise ValueError("nonfinite intensity slope")
                if source.get("containment_basis") != "same_nearest_sample_quantization_as_requested_anchor":
                    raise ValueError("intensity support schema requires current sample-based containment")
                supports = [_span(s) for s in source["frame_support_original_s"]]
                if not supports or not _same_span([min(s[0] for s in supports), max(s[1] for s in supports)], source["support_s"]):
                    raise ValueError("intensity support does not equal actual frame union")
                support = support_evidence(source["support_s"], item["tag_vowel"], phones, sr)
                if support["within_existing_TAG_vowel"] != source["within_labeled_vowel"]:
                    raise ValueError("intensity containment flag differs from actual support")
                target.update(support=support, overlapping_frame_count=len(supports),
                              frames_are_independent=False, carrier_phase_validated=False)
                target["review_flags"] = support["review_flags"]
            contrast = variant["cv_contrast"]
            cv = {"observation_status": contrast["status"], "value_db": contrast.get("vowel_minus_noise_db"),
                  "review_flags": [], "reference_is_verified_pure_frication": False}
            if cv["value_db"] is not None:
                if not _finite(cv["value_db"]):
                    raise ValueError("nonfinite relative intensity")
                support = support_evidence(contrast["vowel_support_s"], item["tag_vowel"], phones, sr)
                if support["within_existing_TAG_vowel"] != contrast["within_labeled_vowel"]:
                    raise ValueError("relative intensity containment differs from support")
                cv.update(vowel_support=support, noise_support_original_s=_span(contrast["noise_support_s"]),
                          review_flags=support["review_flags"])
            else:
                cv["review_flags"].append(contrast["status"])
            variants.append({"onset_offset_ms": variant["offset_ms"], "metrics": metrics, "cv_contrast": cv})
        result[anchor_name] = {"observation_status": anchor["status"], "variants": variants,
                              "sensitivity_grid_is_confidence_interval": False,
                              "actual_acoustic_onset_verified": False}
    return result


def assemble_evidence(item, observation, intensity, tag_phones):
    """Return a research review record. Callers separately verify actual file hashes."""
    if observation.get("schema_version") != 3:
        raise ValueError("expected observation schema 3 with explicit vowel context")
    if not item["id"] == observation["id"] == intensity["id"]:
        raise ValueError("target identity mismatch")
    checksum = item["source_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", checksum) or checksum != observation["source_sha256"] or checksum != intensity["source_sha256"]:
        raise ValueError("source fingerprint mismatch")
    if observation["TAG_sha256"] != item["tag_sha256"] or not _same_span(intensity["TAG_vowel_span"], item["tag_vowel"]):
        raise ValueError("reference TAG identity mismatch")
    if observation["pronunciation_decision"] is not None or observation["new_human_gold"] is not False:
        raise ValueError("candidate export cannot carry an unverified decision or new gold")
    context = observation["vowel_context"]
    if context["expected_following_vowel"] != item["following_vowel"] or context["existing_TAG_following_vowel"] != item["actual_vowel"]:
        raise ValueError("vowel context differs from frozen packet")
    if intensity["expected_vowel"] != item["following_vowel"] or intensity["TAG_vowel"] != item["actual_vowel"]:
        raise ValueError("intensity vowel context differs from frozen packet")
    if context["labels_agree"] is not (item["following_vowel"] == item["actual_vowel"]):
        raise ValueError("vowel context agreement flag differs from labels")
    phones = [[*_span([s, e]), label] for s, e, label in tag_phones]
    if sum(_same_span([s, e], item["tag_vowel"]) and label == item["tag_vowel_label"] for s, e, label in phones) != 1:
        raise ValueError("target vowel not unique in source TAG")
    periodicity = _periodicity_views(observation, item, phones)
    intensity_views = _intensity_views(intensity, item, phones, observation["sample_rate"])
    context_flags = ["vowel_context_label_disagreement"] if context["labels_agree"] is False else []
    return {"schema_version": 2, "id": item["id"], "source_sha256": checksum, "TAG_sha256": item["tag_sha256"],
            "scope": "offline_research_review_not_blinded_listener_material_or_pronunciation_scoring",
            "source_identity_consistent": True, "actual_file_integrity_verified_here": False,
            "vowel_context": context, "context_review_flags": context_flags,
            "target_association": observation["target_association"],
            "acoustic_target_identity_verified": False,
            "noise": {"waveform": observation["noise"]["waveform"],
                      "frames": len(observation["noise"]["frames"]),
                      "spectral_top_hz": observation["noise"]["spectral_top_hz"],
                      "effective_recording_bandwidth_verified": False,
                      "frication_or_aspiration_classes_assigned": False},
            "periodicity_views": periodicity, "intensity_views": intensity_views,
            "interpretation_requirements": ["independent_actual_pronunciation_labels", "verified_target_acoustic_events",
                "vowel_context_and_channel_conditions", "no_borrowing_harmonics_from_a_neighboring_phone"],
            "pronunciation_decision": None, "pronunciation_error_inferred_from_missingness": False,
            "new_human_gold": False}
