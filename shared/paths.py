# -*- coding: utf-8 -*-
"""경로 단일 소스.

이 파일에는 **서버 경로가 하드코딩돼 있지 않다.** 코퍼스·모델·산출물 위치는 배포 환경마다
다르고 저장소에 남길 정보도 아니므로, 환경변수 또는 저장소 루트의 `.env` 에서 읽는다.

설정
    cp .env.example .env      # 그리고 값을 자기 환경에 맞게 채운다
    # 또는
    export SPEECH_AI_DATA=/mnt/corpus

필수 값이 없으면 그 상수를 **처음 쓸 때** 무엇을 설정해야 하는지 알려주며 실패한다.
import 시점에는 실패하지 않으므로, 일부 경로만 필요한 스크립트는 그대로 돌아간다.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAT_TASK = REPO_ROOT / "tasks" / "nationality-classification"
MDD_TASK = REPO_ROOT / "tasks" / "pronunciation-assessment"


def _load_dotenv(path=None):
    """`.env` 를 읽어 os.environ 에 채운다. 이미 있는 환경변수를 덮어쓰지 않는다."""
    p = Path(path or REPO_ROOT / ".env")
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class _Required:
    """설정되지 않은 경로. 실제로 쓰이는 순간 무엇을 설정해야 하는지 알려준다."""

    def __init__(self, key, what):
        self._key, self._what = key, what

    def _die(self, *_a, **_k):
        raise RuntimeError(
            f"{self._key} 가 설정되지 않았습니다 ({self._what}).\n"
            f"  cp .env.example .env  후 값을 채우거나  export {self._key}=... 하십시오."
        )

    __fspath__ = __str__ = __truediv__ = __getattr__ = _die


def _env(key, what):
    v = os.environ.get(key)
    return Path(v).expanduser() if v else _Required(key, what)


# -- 외부 대용량 (git 에 절대 안 들어간다) ----------------------
DATA_ROOT = _env("SPEECH_AI_DATA", "AIHub 코퍼스와 MDD 체크포인트가 있는 루트")
MODEL_ROOT = _env("SPEECH_AI_MODELS", "XLS-R 사전학습 가중치 디렉터리")
FAIRSEQ_ROOT = _env("FAIRSEQ_ROOT", "editable 설치된 fairseq 소스 루트")

# -- 산출물 (재생성 비쌈. 저장소 밖에 둔다) ---------------------
NAT_OUTPUTS = _env("NATIONALITY_OUTPUTS", "국적 특징 캐시·헤드·ONNX")
NAT_DATA = _env("NATIONALITY_DATA", "국적 split(csv/tsv)")
NAT_RUNS = _env("NATIONALITY_RUNS", "국적 probe/finetune 결과")
MDD_OUTPUTS = _env("MDD_OUTPUTS", "MDD 매니페스트·예측 캐시·metrics")
MDD_ANALYSIS = _env("MDD_ANALYSIS", "L1-prior 분석 산출물")
G2P_TOOLS = _env("G2P_TOOLS", "KoG2P / g2pK 클론 위치")

# -- 자주 쓰는 구체 경로 ----------------------------------------
XLSR_2B = MODEL_ROOT / "xlsr2_2B_1000k.pt"
XLSR_300M = MODEL_ROOT / "xlsr2_300m.pt"
MDD_CKPT = DATA_ROOT / "mdd_ckpt_tag_all_lar" / "checkpoint_best.pt"
KOG2P_DIR = G2P_TOOLS / "KoG2P"
KOG2P_RULEBOOK = KOG2P_DIR / "rulebook.txt"

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")


def ensure_kog2p():
    """KoG2P(`g2p.py`)는 pip 패키지가 아니라 클론이라 sys.path 등록이 필요하다.

    `from g2p import readRules, graph2prono` **앞에서** 호출할 것.
    주의: g2p.py 는 import 시점에 optparse 로 argv 를 파싱한다. 호출부에서
    `sys.argv` 를 잠시 비워야 한다.
    """
    import sys
    p = str(KOG2P_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


__all__ = [n for n in dir() if n.isupper() or n.startswith("ensure_")]
