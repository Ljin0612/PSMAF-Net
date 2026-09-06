import json

import pytest

torch = pytest.importorskip("torch")

from detection.models.psmaf_yolov8 import PSMAFYOLOv8, load_yolov8s_weights
from detection.scripts.psmaf_yolo_utils import limit_dataset
from detection.scripts.psmaf_yolov8_utils import (METRIC_KEYS,
                                                  resolve_resume_path)


def test_psmaf_yolov8_forward_and_fusion_shapes():
    model = PSMAFYOLOv8()
    rgb = torch.randn(1, 3, 64, 64)
    features = model.forward_features(rgb, rgb)
    assert [x.shape for x in features] == [(1, 128, 8, 8), (1, 256, 4, 4), (1, 512, 2, 2)]
    assert [x.shape for x in model(rgb, rgb)] == [(1, 70, 8, 8), (1, 70, 4, 4), (1, 70, 2, 2)]


def test_pretrained_loader_safely_skips_unmatched(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"model.0.conv.weight": torch.randn(1), "unknown": torch.randn(2)}, path)
    summary = load_yolov8s_weights(PSMAFYOLOv8(), path, verbose=False)
    assert summary["loaded_keys"] == []
    assert len(summary["skipped_keys"]) == 2
    assert summary["unexpected_keys"] == []


def test_resume_resolution_and_debug_limit(tmp_path):
    checkpoint = tmp_path / "last.pt"; checkpoint.touch()
    assert resolve_resume_path("auto", tmp_path) == checkpoint
    assert len(limit_dataset(list(range(8)), 2)) == 2


def test_stable_metric_keys():
    metrics = {key: {} if key == "per_class_ap" else 0.0 for key in METRIC_KEYS}
    assert tuple(json.loads(json.dumps(metrics))) == METRIC_KEYS
