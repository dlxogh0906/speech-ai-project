"""Audio classification models built around the local XLS-R checkpoint."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import sys
from pathlib import Path

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1, hidden_dim=0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))

    def forward(self, x):
        return self.net(x)


class FairseqXLSRCountryClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        num_classes,
        fairseq_root=None,
        freeze_encoder=True,
        dropout=0.1,
        classifier_hidden_dim=0,
        pooling="mean",
    ):
        super().__init__()
        self.checkpoint_path = str(checkpoint_path)
        self.freeze_encoder = freeze_encoder
        self.pooling = pooling
        self.encoder = self._load_encoder(checkpoint_path, fairseq_root)
        if hasattr(self.encoder, "remove_pretraining_modules"):
            self.encoder.remove_pretraining_modules()
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        input_dim = self._encoder_dim()
        self.classifier = ClassificationHead(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=classifier_hidden_dim,
        )

    @staticmethod
    def _load_encoder(checkpoint_path, fairseq_root):
        if fairseq_root is not None:
            fairseq_root = str(Path(fairseq_root).resolve())
            if fairseq_root not in sys.path:
                sys.path.insert(0, fairseq_root)
        try:
            from fairseq import checkpoint_utils
        except Exception as exc:
            raise ImportError(
                "Could not import fairseq checkpoint utilities. "
                "Use an environment with fairseq and omegaconf installed, for example "
                "`conda run -n fairseq_xlsr python train.py ...`."
            ) from exc

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint_path)])
        if not models:
            raise RuntimeError(f"No model found in checkpoint: {checkpoint_path}")
        return models[0]

    def _encoder_dim(self):
        cfg = getattr(self.encoder, "cfg", None)
        if cfg is not None and hasattr(cfg, "encoder_embed_dim"):
            return int(cfg.encoder_embed_dim)
        args = getattr(self.encoder, "args", None)
        if args is not None and hasattr(args, "encoder_embed_dim"):
            return int(args.encoder_embed_dim)
        raise RuntimeError("Could not infer XLS-R encoder dimension from checkpoint config")

    def forward(self, waveforms, padding_mask=None):
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                features, feature_padding_mask = self._extract_features(waveforms, padding_mask)
        else:
            features, feature_padding_mask = self._extract_features(waveforms, padding_mask)
        pooled = self._pool(features, feature_padding_mask)
        return self.classifier(pooled)

    def _extract_features(self, waveforms, padding_mask):
        output = self.encoder.extract_features(
            source=waveforms,
            padding_mask=padding_mask,
            mask=False,
        )
        if isinstance(output, dict):
            return output["x"], output.get("padding_mask")
        if isinstance(output, tuple):
            features = output[0]
            feature_padding_mask = output[1] if len(output) > 1 else None
            return features, feature_padding_mask
        raise TypeError(f"Unsupported fairseq extract_features output: {type(output)!r}")

    def _pool(self, features, padding_mask):
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        if padding_mask is None:
            return features.mean(dim=1)
        valid = (~padding_mask).to(features.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (features * valid).sum(dim=1) / denom

    def checkpoint_state(self):
        return {
            "backend": "fairseq_xlsr",
            "encoder_checkpoint": self.checkpoint_path,
            "freeze_encoder": self.freeze_encoder,
            "pooling": self.pooling,
            "classifier": self.classifier.state_dict(),
        }

    def load_classifier(self, state_dict):
        self.classifier.load_state_dict(state_dict)


class TinyAudioCountryClassifier(nn.Module):
    """Tiny raw-waveform model for fast DataLoader/training smoke tests."""

    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, waveforms, padding_mask=None):
        del padding_mask
        features = self.net(waveforms.unsqueeze(1))
        return self.classifier(features)

    def checkpoint_state(self):
        return {"backend": "tiny", "model": self.state_dict()}


def default_fairseq_root(model_dir):
    repo_root = Path(model_dir).resolve().parent
    candidate = repo_root / "fairseq"
    return candidate if candidate.exists() else None


def resolve_xlsr_checkpoint(model_dir, checkpoint_name="xlsr2_2B_1000k.pt"):
    model_dir = Path(model_dir)
    checkpoint = model_dir / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"XLS-R checkpoint not found: {checkpoint}")
    return checkpoint


def build_model(
    backend,
    num_classes,
    model_dir=None,
    checkpoint_path=None,
    fairseq_root=None,
    freeze_encoder=True,
    dropout=0.1,
    classifier_hidden_dim=0,
):
    if backend == "tiny":
        return TinyAudioCountryClassifier(num_classes=num_classes, dropout=dropout)
    if backend != "fairseq_xlsr":
        raise ValueError(f"Unsupported model backend: {backend}")
    if checkpoint_path is None:
        checkpoint_path = resolve_xlsr_checkpoint(model_dir)
    if fairseq_root is None and model_dir is not None:
        fairseq_root = default_fairseq_root(model_dir)
    return FairseqXLSRCountryClassifier(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        fairseq_root=fairseq_root,
        freeze_encoder=freeze_encoder,
        dropout=dropout,
        classifier_hidden_dim=classifier_hidden_dim,
    )
