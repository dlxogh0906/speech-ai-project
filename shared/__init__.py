"""저장소 공통 유틸 — 두 task가 실제로 공유하는 것만 둔다.

오디오 전처리는 여기 없다. 두 task의 WAV 로더는 요구사항이 달라서(§docs/DECISIONS.md)
각자 자기 패키지 안에 있다. 억지로 합치지 말 것.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

