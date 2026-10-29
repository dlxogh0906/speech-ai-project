# -*- coding: utf-8 -*-
"""배경음 제거(목소리만 남기기) — torchaudio Hybrid Demucs(`HDEMUCS_HIGH_MUSDB_PLUS`). 계획서 6.2 의 사전처리.

영화 클립의 배경음악·효과음은 음높이 추적과 정렬을 망친다. 44.1 kHz 스테레오로 돌려 vocals 만 남기고
16 kHz 모노로 낸다. 새 패키지가 필요 없다(torchaudio 2.1 에 들어 있다). 가중치 319 MB 는 처음 한 번
download.pytorch.org 에서 받는다 — `TORCH_HOME` 을 데이터 디스크로 둘 것.
분리는 10초 조각 + 0.1초 겹침(선형 페이드)으로 돈다 — torchaudio 공식 예제와 같은 방식.
"""
import subprocess

import numpy as np

SR_SEP = 44100


def load_stereo(path, sr=SR_SEP):
    """→ float32 [2, T] (모노 파일이면 두 채널에 복사)."""
    from shared.paths import FFMPEG
    r = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", str(path), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("오디오를 읽지 못했습니다 (%s): %s" % (path, r.stderr.decode("utf-8", "replace")[-300:]))
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2).T.copy()


class Separator:
    def __init__(self, device=None):
        import torch
        import torchaudio
        self.bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        self.model = self.bundle.get_model().eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.vi = list(self.model.sources).index("vocals")

    def split(self, stereo, segment=10.0, overlap=0.1, out_sr=16000):
        """[2, T] 44.1 kHz → (목소리 모노 float32, 나머지 모노 float32) — out_sr 로 바꿔서."""
        import torch
        import torchaudio
        from torchaudio.transforms import Fade
        mix = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))[None].to(self.device)
        ref = mix.mean(1)
        mean, std = ref.mean(), ref.std() + 1e-8
        mix = (mix - mean) / std
        _, C, L = mix.shape
        chunk = int(SR_SEP * segment * (1 + overlap))
        ov = int(overlap * SR_SEP)
        fade = Fade(fade_in_len=0, fade_out_len=ov, fade_shape="linear")
        out = torch.zeros(1, len(self.model.sources), C, L, device=self.device)
        start, end = 0, chunk
        while start < L - ov:
            with torch.no_grad():
                y = fade(self.model(mix[:, :, start:end]))
            out[:, :, :, start:end] += y
            if start == 0:
                fade.fade_in_len = ov
                start += chunk - ov
            else:
                start += chunk
            end += chunk
            if end >= L:
                fade.fade_out_len = 0
        out = out * std + mean
        voc = out[0, self.vi].mean(0)
        rest = out[0].sum(0).mean(0) - voc
        rs = lambda x: torchaudio.functional.resample(x, SR_SEP, out_sr).float().cpu().numpy()
        return rs(voc), rs(rest)
