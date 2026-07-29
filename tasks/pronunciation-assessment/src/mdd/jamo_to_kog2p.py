# -*- coding: utf-8 -*-
"""자모(46) ↔ KoG2P 음소(71 인벤토리) 결정적 변환.

규칙: 자음 토큰은 '다음 토큰이 모음이면 초성, 아니면 종성'. 예외 둘:
- ㅇ: 초성 ㅇ은 무음이라 TAG 전사에 안 적히므로, 등장하는 ㅇ은 위치 무관 전부 종성 ng.
  (모음 앞 ㅇ = 병원[ㅂㅕㅇㅝㄴ]·시장에[ㅅㅣㅈㅏㅇㅔ]처럼 연음 안 되는 앞 음절 종성 — 실데이터 10.7만 건 확인)
- 겹받침(ㄳㄵㄻ...)은 항상 종성 (모음 앞 등장은 전체 2.18M 라인 중 3건뿐).
심볼 테이블은 KoG2P(_g2p_tools/KoG2P/g2p.py)의 ONS/NUC/COD와 동일, 종성 ㅇ만 표면형 ng.

python jamo_to_kog2p.py <manifest_dir>  → 전 split 스캔: 예외 케이스 카운트 + 왕복(자모→음소→자모) 무손실 검증
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS, ensure_kog2p

import sys

VOWEL = {"ㅏ":"aa","ㅐ":"qq","ㅑ":"ya","ㅒ":"yq","ㅓ":"vv","ㅔ":"ee","ㅕ":"yv","ㅖ":"ye",
         "ㅗ":"oo","ㅘ":"wa","ㅙ":"wq","ㅚ":"wo","ㅛ":"yo","ㅜ":"uu","ㅝ":"wv","ㅞ":"we",
         "ㅟ":"wi","ㅠ":"yu","ㅡ":"xx","ㅢ":"xi","ㅣ":"ii"}
ONSET = {"ㄱ":"k0","ㄲ":"kk","ㄴ":"nn","ㄷ":"t0","ㄸ":"tt","ㄹ":"rr","ㅁ":"mm","ㅂ":"p0",
         "ㅃ":"pp","ㅅ":"s0","ㅆ":"ss","ㅈ":"c0","ㅉ":"cc","ㅊ":"ch","ㅋ":"kh","ㅌ":"th",
         "ㅍ":"ph","ㅎ":"h0"}
CODA  = {"ㄱ":"kf","ㄲ":"kk","ㄳ":"ks","ㄴ":"nf","ㄵ":"nc","ㄶ":"nh","ㄷ":"tf","ㄹ":"ll",
         "ㄺ":"lk","ㄻ":"lm","ㄼ":"lb","ㄽ":"ls","ㄾ":"lt","ㄿ":"lp","ㅀ":"lh","ㅁ":"mf",
         "ㅂ":"pf","ㅄ":"ps","ㅅ":"s0","ㅆ":"ss","ㅇ":"ng","ㅈ":"c0","ㅊ":"ch","ㅋ":"kh",
         "ㅌ":"th","ㅍ":"ph","ㅎ":"h0"}
# 역변환: 초성/종성이 같은 음소 심볼을 공유하는 경우(kk,s0,ss,c0,ch,kh,th,ph,h0)는
# 자모 글자도 동일하므로, phone→jamo 는 위치 규칙 없이 단일 dict 로 충분 (충돌 전부 동일 자모 확인됨)
INV = {}
for k, v in list(VOWEL.items()) + list(ONSET.items()) + list(CODA.items()):
    assert INV.get(v, k) == k, f"inverse collision: {v} -> {INV[v]} vs {k}"
    INV[v] = k
ALL_PHONES = sorted(set(ONSET.values()) | set(CODA.values()) | set(VOWEL.values()))
COMPOUND = {"ㄳ","ㄵ","ㄶ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅄ"}


def jamo_to_phones(toks):
    """자모 토큰열 → KoG2P 음소열 (1:1, 길이 보존)."""
    out = []
    n = len(toks)
    for i, t in enumerate(toks):
        if t in VOWEL:
            out.append(VOWEL[t])
        elif t == "ㅇ" or t in COMPOUND:          # 위치 무관 종성
            out.append(CODA[t])
        else:
            nxt_vowel = (i + 1 < n) and (toks[i + 1] in VOWEL)
            if nxt_vowel and t in ONSET:
                out.append(ONSET[t])
            elif t in CODA:
                out.append(CODA[t])
            elif t in ONSET:                     # ㅃㅉㄸ: 종성형 없음 → 초성 심볼 폴백
                out.append(ONSET[t])             # (더듬기 전사 등 극소수, 2.18M 중 109건)
            else:
                raise ValueError(f"unmappable {t!r} at {i}: {' '.join(toks)}")
    return out


def phones_to_jamo(phs):
    """역변환 — 심볼별 단일 매핑으로 무손실."""
    return [INV[p] for p in phs]


def scan(manifest_dir):
    import glob, collections
    stats = collections.Counter(); bad = []
    used = collections.Counter()
    for f in sorted(glob.glob(f"{manifest_dir}/*.ltr") + glob.glob(f"{manifest_dir}/*.canon")):
        for ln, line in enumerate(open(f, encoding="utf-8")):
            toks = line.split()
            if not toks:
                stats["empty"] += 1; continue
            stats["lines"] += 1
            # 예외 패턴 카운트
            for i, t in enumerate(toks):
                if t == "ㅇ" and i + 1 < len(toks) and toks[i + 1] in VOWEL:
                    stats["onset_ㅇ"] += 1
                if t in ("ㄳ","ㄵ","ㄶ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅄ") and i + 1 < len(toks) and toks[i + 1] in VOWEL:
                    stats["compound_before_vowel"] += 1
            try:
                phs = jamo_to_phones(toks)
                used.update(phs)
                assert len(phs) == len(toks), "length changed"
                assert phones_to_jamo(phs) == toks, "roundtrip mismatch"
                stats["ok"] += 1
            except Exception as e:
                stats["fail"] += 1
                if len(bad) < 5:
                    bad.append((f.split("/")[-1], ln, str(e)[:120]))
    print(f"scan {manifest_dir}")
    for k, v in stats.most_common():
        print(f"  {k:24}{v:>10,}")
    print(f"  distinct phones used  {len(used):>10,}  (inventory {len(ALL_PHONES)})")
    print("  used:", " ".join(sorted(used)))
    for b in bad:
        print("  FAIL:", b)
    return stats


if __name__ == "__main__":
    scan(sys.argv[1] if len(sys.argv) > 1 else f"{MDD_OUTPUTS}/manifest_tag_combined")
