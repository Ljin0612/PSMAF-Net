"""UNIV-original ConvMAE adapter implementing Detectron2's Backbone API."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from detectron2.config import CfgNode as CN
    from detectron2.layers import ShapeSpec
    from detectron2.modeling import BACKBONE_REGISTRY, Backbone
except ImportError as exc:  # pragma: no cover - exercised by the environment check
    raise ImportError(
        "detection.models.univ_backbone requires Detectron2. Install a build "
        "compatible with the active PyTorch and CUDA versions."
    ) from exc

from .feature_adapter import FEATURE_NAMES, FEATURE_STRIDES, UNIVFeatureAdapter


def add_univ_config(cfg: CN) -> None:
    """Register UNIV-specific configuration before merging the YAML file."""
    cfg.MODEL.UNIV = CN()
    cfg.MODEL.UNIV.CHECKPOINT = ""
    cfg.MODEL.UNIV.FREEZE = False
    cfg.MODEL.UNIV.PRETRAIN_IMAGE_SIZE = 224
    cfg.MODEL.UNIV.OUT_CHANNELS = 256
    cfg.MODEL.UNIV.DROP_PATH_RATE = 0.2


def _build_official_encoder(pretrain_image_size: int, drop_path_rate: float) -> nn.Module:
    """Construct the official pretraining ConvViT without changing vendored code."""
    module = importlib.import_module(
        "third_party.UNIV.models.backbone.mcmae.vision_transformer"
    )
    return module.ConvViT(
        img_size=[
            pretrain_image_size,
            pretrain_image_size // 4,
            pretrain_image_size // 8,
        ],
        patch_size=[4, 2, 2],
        in_chans=3,
        num_classes=0,
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
    )


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("UNIV checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "student", "teacher", "backbone"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not isinstance(checkpoint, dict):
        raise TypeError("could not find a state dictionary in UNIV checkpoint")

    prefixes = (
        "module.",
        "student.",
        "backbone.",
        "encoder.",
        "base_model.model.",
    )
    state_dict: dict[str, torch.Tensor] = {}
    for raw_key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    stripped = True
        key = key.replace(".base_layer.", ".")
        state_dict[key] = value
    return state_dict


@BACKBONE_REGISTRY.register()
class UNIVBackbone(Backbone):
    """Official UNIV encoder exposed as ``p2`` through ``p5`` feature maps."""

    def __init__(self, cfg: CN, input_shape: ShapeSpec | None = None) -> None:
        super().__init__()
        del input_shape
        univ_cfg = cfg.MODEL.UNIV
        self.encoder = _build_official_encoder(
            int(univ_cfg.PRETRAIN_IMAGE_SIZE), float(univ_cfg.DROP_PATH_RATE)
        )
        self.feature_adapter = UNIVFeatureAdapter(
            in_channels=(256, 384, 768),
            out_channels=int(univ_cfg.OUT_CHANNELS),
        )
        self._out_features = FEATURE_NAMES
        self._out_feature_channels = {
            name: int(univ_cfg.OUT_CHANNELS) for name in FEATURE_NAMES
        }
        self._out_feature_strides = dict(zip(FEATURE_NAMES, FEATURE_STRIDES))
        self._size_divisibility = 32
        self.checkpoint_report: dict[str, Any] | None = None

        if univ_cfg.CHECKPOINT:
            self.checkpoint_report = self.load_pretrained(univ_cfg.CHECKPOINT)
        self.set_backbone_trainable(not bool(univ_cfg.FREEZE))

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or fine-tune the official UNIV encoder.

        The feature adapter remains trainable so a frozen encoder can still be
        projected into the detector feature space.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self.encoder.eval()

    def train(self, mode: bool = True) -> "UNIVBackbone":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def load_pretrained(self, checkpoint_path: str | Path) -> dict[str, Any]:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"UNIV checkpoint does not exist: {path}")
        # Official UNIV checkpoints include trusted argparse.Namespace metadata,
        # which PyTorch's restricted weights-only unpickler intentionally rejects.
        # This path is supplied explicitly by the user and is therefore treated as
        # a trusted local checkpoint, matching the loader used by upstream UNIV.
        state_dict = _unwrap_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        model_state = self.encoder.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        shape_mismatch = sorted(
            key
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape != value.shape
        )
        incompatible = self.encoder.load_state_dict(compatible, strict=False)
        report = {
            "path": str(path),
            "loaded": len(compatible),
            "missing_keys": sorted(incompatible.missing_keys),
            "unexpected_keys": sorted(set(state_dict) - set(model_state)),
            "shape_mismatch": shape_mismatch,
        }
        print(
            "UNIV checkpoint: "
            f"loaded={report['loaded']} missing={len(report['missing_keys'])} "
            f"unexpected={len(report['unexpected_keys'])} "
            f"shape_mismatch={len(report['shape_mismatch'])} path={path}"
        )
        if not compatible:
            raise RuntimeError(f"no compatible UNIV encoder weights found in {path}")
        return report

    @staticmethod
    def _patch_embed(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply an official PatchEmbed at arbitrary padded detector resolution."""
        x = module.proj(x)
        x = module.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return module.act(x)

    def _encoder_stages(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self._patch_embed(self.encoder.patch_embed1, x)
        x = self.encoder.pos_drop(x)
        for block in self.encoder.blocks1:
            x = block(x)
        p2 = x

        x = self._patch_embed(self.encoder.patch_embed2, x)
        for block in self.encoder.blocks2:
            x = block(x)
        p3 = x

        x = self._patch_embed(self.encoder.patch_embed3, x)
        height, width = x.shape[-2:]
        x = self.encoder.patch_embed4(x.flatten(2).transpose(1, 2))
        position = self.encoder.pos_embed
        source_size = int(position.shape[1] ** 0.5)
        if source_size * source_size != position.shape[1]:
            raise ValueError("UNIV absolute position embedding is not a square grid")
        position = position.reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2)
        if (height, width) != (source_size, source_size):
            position = F.interpolate(
                position, size=(height, width), mode="bicubic", align_corners=False
            )
        position = position.flatten(2).transpose(1, 2)
        x = x + position
        for block in self.encoder.blocks3:
            x = block(x)
        p4 = self.encoder.norm(x).transpose(1, 2).reshape(-1, 768, height, width)
        return p2, p3, p4.contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.feature_adapter(self._encoder_stages(x))
