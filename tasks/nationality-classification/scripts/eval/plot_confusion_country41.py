"""Heatmap of the country41 row-normalized confusion (present classes only)."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import NAT_OUTPUTS

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = f"{NAT_OUTPUTS}/fonts/NotoSansKR.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

D = Path(f"{NAT_OUTPUTS}/0629_country41_12s/confusion")
counts = list(csv.reader(open(D / "confusion_counts.csv", encoding="utf-8")))
header = counts[0][1:]
C = np.array([[int(x) for x in r[1:]] for r in counts[1:]], float)

present = [i for i in range(len(header)) if C[i].sum() > 0]          # true classes seen in test
order = sorted(present, key=lambda i: -C[i].sum())                    # by support desc
labels = [header[i] for i in order]
M = C[np.ix_(order, order)]
M = M / M.sum(1, keepdims=True).clip(min=1)                          # row-normalized (recall)

n = len(order)
fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("예측 (pred)"); ax.set_ylabel("정답 (true)")
ax.set_title("country41 혼동행렬 (행 정규화 = 클래스별 recall)  ·  test top-1 0.833", fontsize=11)
for i in range(n):
    for j in range(n):
        v = M[i, j]
        if v >= 0.08:
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="비율")
fig.tight_layout()
outp = D / "confusion_heatmap.png"
fig.savefig(outp, dpi=130, bbox_inches="tight")
print("wrote", outp, "| classes:", n)
