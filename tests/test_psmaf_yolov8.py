import csv
import json

import pytest

torch = pytest.importorskip("torch")

from detection.models.psmaf_yolov8 import PSMAFYOLOv8, load_yolov8s_weights
from detection.scripts.psmaf_yolo_utils import limit_dataset
from detection.scripts.psmaf_yolov8_utils import (METRIC_KEYS,
                                                  ModelEMA,
                                                  WarmupCosineScheduler,
                                                  evaluate_yolov8,
                                                  cuda_memory_metrics,
                                                  resolve_resume_path,
                                                  set_backbones_trainable,
                                                  strict_device_check,
                                                  yolov8_detection_loss)


class TinyDualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = torch.nn.Linear(1, 1)
        self.ir_backbone = torch.nn.Linear(1, 1)
        self.fusion = torch.nn.Linear(1, 1)

    def forward(self, rgb, ir):
        return self.fusion(self.rgb_backbone(rgb) + self.ir_backbone(ir))


def test_warmup_cosine_scheduler_changes_learning_rate():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=10, warmup_steps=3, lrf=0.1)
    rates = []
    for _ in range(10):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step(); scheduler.step()
    assert rates[0] < rates[2]
    assert rates[-1] < rates[2]
    assert len(set(rates)) > 3


def test_ema_updates_and_model_can_be_evaluated():
    model = TinyDualModel()
    ema = ModelEMA(model)
    with torch.no_grad():
        model.fusion.weight.add_(1)
    ema.update(model)
    result = ema.ema(torch.ones(1, 1), torch.ones(1, 1))
    assert result.shape == (1, 1)
    assert not ema.ema.training
    assert ema.updates == 1


def test_freeze_backbones_then_unfreeze():
    model = TinyDualModel()
    set_backbones_trainable(model, False)
    assert not any(parameter.requires_grad for parameter in model.rgb_backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    set_backbones_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.rgb_backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.ir_backbone.parameters())


def test_checkpoint_round_trip_restores_ema(tmp_path):
    model = TinyDualModel(); ema = ModelEMA(model)
    with torch.no_grad():
        model.fusion.bias.add_(2)
    ema.update(model)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, path)
    restored = ModelEMA(TinyDualModel())
    restored.load_state_dict(torch.load(path, weights_only=False)["ema"])
    assert restored.updates == ema.updates
    assert all(torch.equal(a, b) for a, b in zip(ema.ema.state_dict().values(),
                                                restored.ema.state_dict().values()))


def test_psmaf_yolov8_forward_and_fusion_shapes():
    model = PSMAFYOLOv8()
    rgb = torch.randn(1, 3, 64, 64)
    features = model.forward_features(rgb, rgb)
    assert [x.shape for x in features] == [(1, 128, 8, 8), (1, 256, 4, 4), (1, 512, 2, 2)]
    assert [x.shape for x in model(rgb, rgb)] == [(1, 70, 8, 8), (1, 70, 4, 4), (1, 70, 2, 2)]
    debug = model.forward_debug(rgb, rgb)
    assert set(debug) == {"rgb_features", "ir_features", "fused_features", "outputs"}


def test_cpu_memory_and_strict_device_diagnostics_are_safe():
    model = TinyDualModel()
    tensor = torch.ones(1, 1)
    strict_device_check(model, torch.device("cpu"), rgb=tensor, outputs=model(tensor, tensor))
    assert set(cuda_memory_metrics("cpu")) == {"cuda_allocated_mib", "cuda_reserved_mib",
                                                "cuda_peak_allocated_mib", "cuda_peak_reserved_mib"}
    assert not any(cuda_memory_metrics("cpu").values())


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
    assert set(diagnostics["confidence_quantiles"]) == {"p25", "p50", "p75", "p90", "p95"}


def test_yolov8_training_logs_num_pos(tmp_path):
    from detection.scripts.psmaf_yolo_utils import save_train_log_row

    row = {"epoch": 1, "avg_total_loss": 3.0, "avg_obj_loss": 1.0,
           "avg_box_loss": 1.0, "avg_cls_loss": 1.0, "num_pos": 2.0,
           "learning_rate": 0.001, "val_precision": 0.5, "val_recall": 0.6,
           "val_AP50": 0.7, "val_mAP50_95": 0.4}
    save_train_log_row(row, tmp_path)

    with (tmp_path / "train_log.csv").open(newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    jsonl_row = json.loads((tmp_path / "train_log.jsonl").read_text())

    assert "num_pos" in csv_row
    assert "num_pos" in jsonl_row
    assert float(csv_row["num_pos"]) == row["num_pos"]
    assert jsonl_row["num_pos"] == row["num_pos"]
