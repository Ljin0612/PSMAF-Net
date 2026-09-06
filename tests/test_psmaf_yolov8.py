import json

import pytest

torch = pytest.importorskip("torch")

from detection.models.psmaf_yolov8 import PSMAFYOLOv8, load_yolov8s_weights
from detection.scripts.psmaf_yolo_utils import limit_dataset
from detection.scripts.psmaf_yolov8_utils import (METRIC_KEYS,
                                                  evaluate_yolov8,
                                                  resolve_resume_path,
                                                  yolov8_detection_loss)


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


def test_detection_loss_is_finite_scaled_and_differentiable():
    outputs = tuple(torch.zeros(2, 70, size, size, requires_grad=True) for size in (8, 4, 2))
    targets = torch.tensor([[0., 0., .25, .25, .1, .1], [1., 2., .5, .5, .3, .3]])
    components = yolov8_detection_loss(outputs, targets)

    assert set(components) == {"loss", "obj_loss", "box_loss", "cls_loss", "num_pos"}
    assert all(torch.isfinite(components[key]) for key in ("loss", "box_loss", "cls_loss"))
    assert components["cls_loss"].item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    assert components["num_pos"].item() == 2
    assert components["obj_loss"].item() == 0
    components["loss"].backward()
    assert all(output.grad is not None and torch.isfinite(output.grad).all() for output in outputs)


def test_yolov8_evaluator_writes_matching_diagnostics(tmp_path, monkeypatch):
    prediction = torch.tensor([[10., 10., 20., 20., .9, 0.],
                               [30., 30., 40., 40., .8, 1.]])
    monkeypatch.setattr("detection.scripts.psmaf_yolov8_utils.decode_yolov8_outputs",
                        lambda outputs, image_size: [prediction])

    class Model(torch.nn.Module):
        def forward(self, rgb, ir):
            return (torch.zeros(1, 70, 1, 1),) * 3

    batch = {"rgb": torch.zeros(1, 3, 64, 64), "ir": torch.zeros(1, 3, 64, 64),
             "labels": torch.tensor([[0., 0., 15 / 64, 15 / 64, 10 / 64, 10 / 64]])}
    path = tmp_path / "eval_diagnostics.json"
    metrics = evaluate_yolov8(Model(), [batch], torch.device("cpu"), diagnostics_path=path)
    diagnostics = json.loads(path.read_text())

    assert metrics["AP50"] == 1.0
    assert diagnostics["tp50"] == 1
    assert diagnostics["fp50"] == 1
    assert diagnostics["fn50"] == 0
    assert diagnostics["per_class_gt_counts"]["people"] == 1
    assert diagnostics["per_class_prediction_counts"]["car"] == 1
    assert diagnostics["per_class_tp50"]["people"] == 1
    assert diagnostics["per_class_fp50"]["car"] == 1
    assert diagnostics["per_class_fn50"]["people"] == 0


def test_yolov8_training_logs_num_pos(tmp_path):
    from detection.scripts.psmaf_yolo_utils import save_train_log_row

    row = {"epoch": 1, "num_pos": 2.0}
    save_train_log_row(row, tmp_path)
    assert json.loads((tmp_path / "train_log.jsonl").read_text())["num_pos"] == 2.0
    assert "num_pos" in (tmp_path / "train_log.csv").read_text().splitlines()[0]
