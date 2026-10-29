# -*- coding: utf-8 -*-
"""영어 문장 → 단어별 표준발음(ARPAbet 39 + 강세 숫자).

CMUdict 를 먼저 찾고(여러 발음이 있으면 전부 후보로), 없는 단어만 g2p_en 신경망으로 만든다.
동형이의어(read·live·record …)는 g2p_en 처럼 품사로 고른다.

화면에 보여줄 단어는 **원문 철자 그대로** 둔다 — "3" 은 그대로 "3" 이고 소리는 three 로,
"well-known" 은 한 단어이고 소리는 두 단어를 잇는다. 문장부호는 버린다.

g2p_en · nltk 는 고정된 파이썬 환경에 없다. 덧씌우기 폴더를 PYTHONPATH 에, nltk 데이터
(cmudict · averaged_perceptron_tagger)를 NLTK_DATA 에 둔다(`deploy/pronunciation-en/README.md`).
"""

import re
import unicodedata

# 숫자는 쉼표·소수점·통화·% 를 붙인 채 한 토큰으로(1,000 · $5 · 3.14 · 50%), 단어는 유니코드 글자(café)와
# 아포스트로피·하이픈으로 이어진 것을 한 토큰으로. 한글 등 영어가 아닌 토큰은 _parts 에서 소리가 없어 빠진다.
TOKEN = re.compile(u"[$£€]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?%?(?![^\\W_])|[^\\W_]+(?:['\\-][^\\W_]+)*")
APOS = {u"‘": "'", u"’": "'", u"ʼ": "'", u"＇": "'", u"`": "'"}


def strip_stress(p):
    return re.sub(r"\d", "", p)


class EnglishG2P:
    def __init__(self, max_variants=4):
        try:
            from g2p_en import G2p
            from g2p_en.expand import normalize_numbers
            from nltk import pos_tag
        except ImportError as e:                          # 덧씌우기 폴더가 PYTHONPATH 에 없다
            raise ImportError("g2p_en/nltk 를 찾지 못했다 — PYTHONPATH 에 영어 덧씌우기 폴더를, "
                              "NLTK_DATA 에 nltk 데이터를 둘 것 (%s)" % e)
        self.g = G2p()
        self.norm_num = normalize_numbers
        self.pos_tag = pos_tag
        self.max_variants = max_variants

    def _parts(self, token):
        """표시 단어 1개 → 소리 낼 소문자 단어들(숫자 풀기 · 하이픈 나누기 · 악센트 떼기)."""
        t = unicodedata.normalize("NFD", token)
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        t = t.replace("%", " percent")
        # 소수점 뒤는 한 자리씩 읽는다(3.14 → three point one four)
        t = re.sub(r"(\d)\.(\d+)", lambda m: m.group(1) + " point " + " ".join(m.group(2)), t)
        t = self.norm_num(t).lower()
        return [p for p in re.split(r"[\s\-]+", re.sub(r"[^a-z'\s\-]", " ", t)) if re.search("[a-z]", p)]

    def _pron(self, w, pos):
        """단어 1개 → 발음 후보들(첫 번째가 기본)."""
        g = self.g
        if w in g.homograph2features:
            p1, p2, pos1 = g.homograph2features[w]
            return [list(p1) if pos.startswith(pos1) else list(p2)]
        if w in g.cmu:
            out = []
            for p in g.cmu[w]:
                if p not in out:
                    out.append(list(p))
            return out[: self.max_variants]
        bare = w.strip("'")
        if bare != w and bare in g.cmu:
            return [list(g.cmu[bare][0])]
        return [list(g.predict(w))]

    def words(self, text):
        """refText → [{"word": 원문 철자, "variants": [[ARPAbet+강세 …], …]}]. 소리 없는 토큰은 뺀다."""
        text = unicodedata.normalize("NFC", "".join(APOS.get(c, c) for c in (text or "")))
        toks = TOKEN.findall(text)
        parts = [self._parts(t) for t in toks]
        flat = [p for ps in parts for p in ps]
        tags = [pos for _, pos in self.pos_tag(flat)] if flat else []
        out, k = [], 0
        for tok, ps in zip(toks, parts):
            if not ps:
                continue
            per = []
            for p in ps:
                per.append(self._pron(p, tags[k]))
                k += 1
            # 여러 부분(하이픈·숫자)이면 각 부분 기본 발음을 잇는다. 한 부분이면 후보를 그대로 둔다.
            variants = per[0] if len(per) == 1 else [[ph for v in per for ph in v[0]]]
            out.append({"word": tok, "variants": variants})
        return out
