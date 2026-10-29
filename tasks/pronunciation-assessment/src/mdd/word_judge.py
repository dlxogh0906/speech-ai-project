# -*- coding: utf-8 -*-
"""단어 하나 판정 — 발음 챌린지용(PERFECT / GOOD / MISS).

발음평가 API(`deploy/pronunciation-demo/app.py`)와 같은 모델·표준발음·전사 정렬 규칙을 쓰되, 게임 판정에 필요 없는
문장 점수(GBM)·유창성·L1 피드백은 돌리지 않는다(results/20: GBM 이 처리 시간의 60%).
**API 쪽 `canonical`·`align` 과 규칙이 같아야 한다** — 저쪽을 고치면 여기도 맞출 것.

    J = WordJudge(hf_ckpt, devices=("cuda", "cpu"))
    out = J.judge(wave_16k_float32, "소리", device="cuda")      # {"result": "PERFECT", "edit": 0, ...}
    J = WordJudge(hf_ckpt, devices=("npu",), rbln="w2vbert_ctc_f192_448_1024.rbln", rbln_device=6)   # NPU 서버

판정 = 모델이 들은 자모열과 표준발음 자모열의 편집거리. 0 → PERFECT · 1 → GOOD · 2 이상 → MISS.
results/20 §4 실측(학습 단어 · 맞게 읽은 발화): MISS 2.2%, 1개라도 다르면 MISS 로 치면 12.0%.
처음 보는 단어에서는 각각 22.0% · 69.0% 로 오른다.
"""
import itertools
import os
import re
import sys
import threading
import time

SR = 16000
# 단어 발화의 유성 구간 중앙값이 0.63초, 한 글자 단어(귀·키·입)는 0.39초다(results/20).
# 문장용 기준(1.0초)이면 단어의 95% 가 거절된다. 0.2초도 한 글자 단어 7.4% 를 거절해서 0.12초로 둔다(0.2%).
# 브라우저 말끝 감지도 0.12초 이상 소리가 나야 녹음을 시작하므로 그 기준과 맞춘다.
MIN_VOICED_SEC = 0.12
GRADES = {0: "PERFECT", 1: "GOOD"}            # 편집거리 → 등급. 나머지는 MISS


def training_prompts(manifest):
    """서비스 모델(step_38000)이 학습에서 본 문항 → (단어 읽기 RW 목록, 전체 문항 목록).

    매니페스트 메타(prompt·task)에서 읽는다. hold-out 분할의 unseen 문항도 이 모델에겐 본 문항이다.
    문항 목록(AIHub)은 저장소에 두지 않는다 — 매니페스트가 없는 서버로는 JSON 으로 옮긴다(`save_prompts_json`).
    """
    import json
    rw, other = set(), set()
    for f in ("valid", "valid_unseen", "test_unseen"):
        p = os.path.join(manifest, f + ".meta.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            d = json.loads(l)
            (rw if d.get("task") == "RW" else other).add(d["prompt"])
    return sorted(rw), sorted(rw | other)


def save_prompts_json(manifest, path):
    import json
    rw, allp = training_prompts(manifest)
    json.dump({"rw": rw, "all": allp}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return len(rw), len(allp)


def load_prompts_json(path):
    import json
    d = json.load(open(path, encoding="utf-8"))
    return sorted(d["rw"]), sorted(d["all"])


def load_kog2p():
    from shared.paths import G2P_TOOLS, ensure_kog2p
    ensure_kog2p()
    argv, sys.argv = sys.argv, sys.argv[:1]             # g2p.py 는 import 시점에 argv 를 파싱한다
    try:
        from g2p import readRules, graph2prono, ver_info
    finally:
        sys.argv = argv
    return graph2prono, readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))


class KoCanonical:
    """문장 → 표준발음 자모열(KoG2P). 모델 없이 쓴다 — 발음 API 의 canonical 과 같은 규칙.

    KoG2P 는 공백이 정확히 2칸이면 음절을 깨뜨려서 1칸으로 맞춘다. ㅙ·ㅞ 는 TAG/LAB 관례대로 나눈다.
    """

    def __init__(self):
        from mdd.jamo_to_kog2p import INV, phones_to_jamo
        self.g2p, self.rules = load_kog2p()
        self.p2j, self.inv = phones_to_jamo, INV
        self._cache, self._lock = {}, threading.Lock()

    def __call__(self, text):
        text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text or "")).strip()
        if not text:
            return []
        with self._lock:
            if text not in self._cache:
                ri, ro = self.rules
                out = []
                for j in self.p2j([p for p in self.g2p(text, ri, ro).split() if p in self.inv]):
                    out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
                self._cache[text] = out
            return list(self._cache[text])


def align(canon, hyp):
    """표준발음 자리마다 모델이 무엇을 냈는지(`deploy/pronunciation-demo/app.py` 의 align 과 같은 규칙)."""
    import edlib
    c, h = "".join(canon), "".join(hyp)
    if not c:
        return []
    if not h:
        return [{"i": i, "canon": x, "pred": None, "verdict": "삭제"} for i, x in enumerate(canon)]
    a = edlib.getNiceAlignment(edlib.align(c, h, task="path"), c, h)
    out, i = [], 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            continue
        out.append({"i": i, "canon": x, "pred": None if y == "-" else y,
                    "verdict": "일치" if x == y else ("삭제" if y == "-" else "다름")})
        i += 1
    return out


class WordJudge:
    def __init__(self, hf_ckpt, devices=("cuda",), rbln=None, rbln_device=0):
        """devices: "cuda" · "cpu" · "npu" 중 올릴 것. "npu" 는 rbln(컴파일된 .rbln)이 있어야 하고
        hf_ckpt 에서는 jamo_vocab.json·preprocessor_config.json 만 읽는다(`mdd.npu_w2vbert.RBLNDecoder`)."""
        import torch
        from mdd.gop_feats_v2 import HFDecoder
        from mdd.jamo_to_kog2p import CODA, ONSET, VOWEL
        self.canonical = KoCanonical()
        self.jamo = set(VOWEL) | set(ONSET) | set(CODA)
        self.decs = {}
        for d in devices:
            if d == "npu":
                if not rbln:
                    raise ValueError("npu 는 rbln(.rbln 경로)이 필요하다")
                from mdd.npu_w2vbert import RBLNDecoder
                self.decs["npu"] = RBLNDecoder(hf_ckpt, rbln, device=rbln_device)   # .device 는 "npu:N"
                continue
            if d == "cuda" and not torch.cuda.is_available():
                continue
            old = os.environ.get("MDD_FORCE_CPU")
            os.environ["MDD_FORCE_CPU"] = "1" if d == "cpu" else "0"     # HFDecoder 는 생성 시점에 이 값을 본다
            try:
                dec = HFDecoder(hf_ckpt)
            finally:
                if old is None:
                    os.environ.pop("MDD_FORCE_CPU", None)
                else:
                    os.environ["MDD_FORCE_CPU"] = old
            self.decs[dec.device] = dec
        if not self.decs:
            raise RuntimeError("올릴 장치가 없다: %r" % (devices,))
        self.locks = {k: threading.Lock() for k in self.decs}           # 장치마다 한 번에 하나씩
        self.default = next(k for k in ("npu", "cuda", "cpu") if k in self.decs)

    def warmup(self, n=3):
        """첫 호출(CUDA 초기화·커널 선택)이 느려서 미리 몇 번 돌린다."""
        import numpy as np
        x = (np.random.RandomState(0).randn(SR) * 0.05).astype(np.float32)
        for d in self.decs:
            for _ in range(n):
                self._forward(d, x)

    def check_word(self, word):
        """판정할 수 있는 단어인가(표준발음이 모델 사전 안에 있는가)."""
        canon = self.canonical(word)
        d = self.decs[self.default].d
        return bool(canon) and d.unk() not in [d.index(t) for t in canon]

    def _forward(self, device, wav):
        from mdd.gop_feats_v2 import forward_lprobs
        dec = self.decs[device]
        with self.locks[device]:
            lp, Ls = forward_lprobs(dec, [wav])
        return dec, lp[0, :Ls[0]].numpy()

    def judge(self, wav, word, device=None):
        """wav: 16 kHz float32 · word: 읽어야 할 단어. 말소리가 너무 적으면 result=NO_SPEECH(채점 안 함)."""
        import edlib
        from mdd.vad import FRAME_SEC, trim, voiced_mask
        t0 = time.perf_counter()
        device = device if device in self.decs else self.default
        if not self.check_word(word):
            raise ValueError("표준발음으로 바꿀 수 없는 단어: %r" % word)
        canon = self.canonical(word)
        m, _ = voiced_mask(wav, SR)
        voiced = float(m.sum()) * FRAME_SEC
        base = {"word": word, "canon": canon, "device": device, "voiced_sec": round(voiced, 2)}
        if voiced < MIN_VOICED_SEC:
            return dict(base, result="NO_SPEECH", ms={"total": round((time.perf_counter() - t0) * 1e3, 1)})
        wav, _ = trim(wav, SR)
        t1 = time.perf_counter()
        dec, l = self._forward(device, wav)
        t2 = time.perf_counter()
        g = l.argmax(-1)
        pred = [dec.d[int(v)] for v, _ in itertools.groupby(g) if v != dec.blank]
        pred = [p for p in pred if p in self.jamo and len(p) == 1]
        edit = edlib.align("".join(pred), "".join(canon))["editDistance"]
        t3 = time.perf_counter()
        return dict(base, result=GRADES.get(edit, "MISS"), edit=edit, pred=pred, sites=align(canon, pred),
                    ms={"prep": round((t1 - t0) * 1e3, 1), "forward": round((t2 - t1) * 1e3, 1),
                        "post": round((t3 - t2) * 1e3, 1), "total": round((t3 - t0) * 1e3, 1)})
