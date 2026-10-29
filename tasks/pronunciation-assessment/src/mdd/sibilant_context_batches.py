"""Training-only normal/error balance within the same phonemic item position.

No expected phone, TAG actual or pairing information is added to model inputs.
The loss stays independent four-class cross entropy; only sampling changes.
"""
import collections
import numpy as np


def make_plan(y, labels, features, seed, steps, mode):
    if mode not in ("all_actual", "eligible_actual", "paired_context"):
        raise ValueError("unknown sampling mode")
    y = np.asarray(y)
    original = {i: np.flatnonzero(y == i) for i in range(4)}
    if not all(len(v) for v in original.values()):
        raise ValueError("all four actual classes required")
    by_index = collections.defaultdict(list)
    for label in labels:
        if label.get("feature_index") is not None:
            by_index[label["feature_index"]].append(label)
    pools = collections.defaultdict(lambda: {0: [], 1: []})
    for i, matches in by_index.items():
        if len(matches) != 1 or y[i] not in (0, 1):
            continue
        row = matches[0]
        if (row["alignment_ambiguous"] or row["label_kind"] not in ("correct", "pair_error")
                or features[i]["canonical"] != row["expected"]):
            continue
        actual = "ㅅ" if y[i] == 0 else "ㅆ"
        if row["actual"] != actual or (row["label_kind"] == "correct") != (row["expected"] == actual):
            raise ValueError("inconsistent human actual/normal labels")
        key = (row["item"], row["q"], row["expected"], row["following"])
        pools[key][int(y[i])].append(i)
    contexts = {key: {k: np.array(sorted(v), np.int64) for k, v in values.items()}
                for key, values in pools.items() if values[0] and values[1]}
    keys = {expected: sorted(key for key in contexts if key[2] == expected) for expected in ("ㅅ", "ㅆ")}
    eligible = {i: np.array(sorted({v for c in contexts.values() for v in c[i]}), np.int64) for i in (0, 1)}
    if mode != "all_actual" and (not all(keys.values()) or not all(len(v) for v in eligible.values())):
        raise ValueError("both expected directions need paired contexts")
    rng = np.random.RandomState(seed)
    groups = dict(original)
    if mode == "eligible_actual":
        groups.update(eligible)
    plan, selected_contexts = [], []
    for _ in range(steps):
        if mode != "paired_context":
            plan.append(np.concatenate([rng.choice(groups[i], 4, replace=True) for i in range(4)]))
            continue
        draw, selected = {0: [], 1: []}, []
        for expected in ("ㅅ", "ㅆ"):
            for _ in range(2):
                key = keys[expected][int(rng.randint(len(keys[expected])))]
                selected.append(key)
                for actual in (0, 1):
                    draw[actual].append(int(rng.choice(contexts[key][actual])))
        plan.append(np.concatenate([draw[0], draw[1], rng.choice(original[2], 4, replace=True),
                                    rng.choice(original[3], 4, replace=True)]))
        selected_contexts.append(selected)
    report = {"mode": mode, "all_actual_class_counts": {str(k): len(v) for k, v in original.items()},
              "paired_contexts": len(contexts), "paired_contexts_by_expected": {k: len(v) for k, v in keys.items()},
              "eligible_actual_class_counts": {str(k): len(v) for k, v in eligible.items()},
              "context_rule": "same item, original phonemic position, expected phone and following vowel; one human-normal and one opposite-actual example, no pairwise loss"}
    return np.asarray(plan, np.int64), selected_contexts, report
