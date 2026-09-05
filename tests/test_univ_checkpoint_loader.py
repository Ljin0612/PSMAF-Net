import argparse
import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


def test_univ_checkpoint_loader_accepts_namespace_metadata(tmp_path, monkeypatch):
    """Official checkpoints carry an argparse.Namespace beside student weights."""
    detectron2 = types.ModuleType("detectron2")
    config = types.ModuleType("detectron2.config")
    layers = types.ModuleType("detectron2.layers")
    modeling = types.ModuleType("detectron2.modeling")

    class Registry:
        def register(self):
            return lambda cls: cls

    config.CfgNode = object
    layers.ShapeSpec = object
    modeling.BACKBONE_REGISTRY = Registry()
    modeling.Backbone = nn.Module
    monkeypatch.setitem(sys.modules, "detectron2", detectron2)
    monkeypatch.setitem(sys.modules, "detectron2.config", config)
    monkeypatch.setitem(sys.modules, "detectron2.layers", layers)
    monkeypatch.setitem(sys.modules, "detectron2.modeling", modeling)
    sys.modules.pop("detection.models.univ_backbone", None)
    module = importlib.import_module("detection.models.univ_backbone")

    backbone = module.UNIVBackbone.__new__(module.UNIVBackbone)
    nn.Module.__init__(backbone)
    backbone.encoder = nn.Linear(2, 1)
    checkpoint = tmp_path / "official_univ.pth"
    torch.save(
        {
            "args": argparse.Namespace(arch="convvit_base"),
            "student": {
                "module.backbone.weight": torch.ones_like(backbone.encoder.weight),
                "module.backbone.bias": torch.zeros_like(backbone.encoder.bias),
            },
        },
        checkpoint,
    )

    report = backbone.load_pretrained(checkpoint)

    assert report["loaded"] == 2
    assert torch.equal(backbone.encoder.weight, torch.ones_like(backbone.encoder.weight))
