import csv
import json

import pytest
torch = pytest.importorskip("torch")

from core.psmaf import MultiScaleAdaptiveFusion, PSMAFFusion, PseudoSemanticGuidance
from detection.models.psmaf_yolo import PSMAFYOLO, detection_loss
from detection.scripts.psmaf_yolo_utils import (decode_outputs, evaluate, evaluate_detection,
                                                limit_dataset, non_max_suppression, save_train_log_row)
from detection.scripts.train_psmaf_yolo_m3fd import parser, prepare_train_logs


def test_pseudo_semantic_guidance_preserves_spatial_size():
    attention, weights = PseudoSemanticGuidance(16)(torch.randn(2, 16, 9, 11), torch.randn(2, 16, 9, 11))
    assert attention.shape == (2, 1, 9, 11)
    assert weights.shape == (2, 2, 1, 1)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 1, 1, 1))


def test_multiscale_adaptive_fusion_shape():
    x = torch.randn(2, 16, 8, 10)
    assert MultiScaleAdaptiveFusion(16)(x, x).shape == x.shape


@pytest.mark.parametrize("mode", ["psmaf", "add", "concat"])
def test_three_level_fusion(mode):
    features = tuple(torch.randn(1, c, s, s) for c, s in zip((16, 32, 64), (16, 8, 4)))
    result = PSMAFFusion((16, 32, 64), fusion_mode=mode)(features, features)
    assert [x.shape for x in result] == [x.shape for x in features]


def test_model_forward_cpu():
    outputs = PSMAFYOLO(channels=(16, 32, 64))(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert [output.shape for output in outputs] == [(1, 11, 8, 8), (1, 11, 4, 4), (1, 11, 2, 2)]


def _encoded_head(size, active=False, class_id=0, xywh=(.5, .5, .25, .25)):
    head = torch.full((1, 11, size, size), -20.0)
    if active:
        logit = lambda value: torch.logit(torch.tensor(value)).item()
        head[0, :4, 0, 0] = torch.tensor([logit(value) for value in xywh])
        head[0, 4, 0, 0] = 20
        head[0, 5 + class_id, 0, 0] = 20
    return head


@pytest.mark.parametrize("level", range(3), ids=("P3", "P4", "P5"))
def test_correct_prediction_from_each_head_is_true_positive(level):
    heads = [_encoded_head(size) for size in (4, 2, 1)]
    heads[level] = _encoded_head((4, 2, 1)[level], active=True)
    predictions = non_max_suppression(decode_outputs(heads, (100, 100)))
    targets = [{"boxes": torch.tensor([[37.5, 37.5, 62.5, 62.5]]), "classes": torch.tensor([0])}]
    metrics = evaluate_detection(predictions, targets)
    assert metrics["precision"] == metrics["recall"] == metrics["AP50"] == 1.0


@pytest.mark.parametrize("class_id,xywh", [(1, (.5, .5, .25, .25)), (0, (.1, .1, .1, .1))],
                         ids=("wrong-class", "low-IoU"))
def test_invalid_prediction_is_not_true_positive(class_id, xywh):
    heads = [_encoded_head(1, active=True, class_id=class_id, xywh=xywh), _encoded_head(1), _encoded_head(1)]
    predictions = non_max_suppression(decode_outputs(heads, (100, 100)))
    targets = [{"boxes": torch.tensor([[37.5, 37.5, 62.5, 62.5]]), "classes": torch.tensor([0])}]
    metrics = evaluate_detection(predictions, targets)
    assert metrics["recall"] == metrics["AP50"] == 0.0


def test_ap_uses_ranking_and_map_averages_iou_thresholds():
    # The higher-ranked duplicate is a FP at every threshold. The second box is
    # a TP at IoU=.50 but not at stricter thresholds, exercising both behaviors.
    predictions = [torch.tensor([[0., 0., 2., 2., .9, 0.], [0., 0., 8., 10., .8, 0.],
                                 [20., 20., 30., 30., .7, 0.]])]
    targets = [{"boxes": torch.tensor([[0., 0., 10., 10.]]), "classes": torch.tensor([0])}]
    metrics = evaluate_detection(predictions, targets)
    assert metrics["AP50"] == pytest.approx(.5)
    assert metrics["AP50"] != pytest.approx(metrics["precision"] * metrics["recall"])
    assert 0 < metrics["mAP50_95"] < metrics["mAP50"]
    assert list(metrics["per_class_ap"]) == ["people", "car", "bus", "motorcycle", "lamp", "truck"]


def test_detection_loss_returns_components():
    outputs = tuple(torch.randn(1, 11, size, size, requires_grad=True) for size in (4, 2, 1))
    result = detection_loss(outputs, torch.tensor([[0., 0., .5, .5, .1, .1]]), return_components=True)
    assert set(result) == {"loss", "obj_loss", "box_loss", "cls_loss"}
    torch.testing.assert_close(result["loss"], result["obj_loss"] + result["box_loss"] + result["cls_loss"])


def _train_log_row(epoch):
    return {"epoch": epoch, "avg_total_loss": 3., "avg_obj_loss": 1., "avg_box_loss": 1.,
            "avg_cls_loss": 1., "num_pos": 2., "learning_rate": .001, "val_precision": 0.,
            "val_recall": 0., "val_AP50": 0., "val_mAP50_95": 0.}


@pytest.mark.parametrize("extra_args", [[], ["--weights", "pretrained.pt"]], ids=("scratch", "weights"))
def test_fresh_run_resets_existing_train_logs(tmp_path, extra_args):
    save_train_log_row(_train_log_row(99), tmp_path)

    args = parser().parse_args(extra_args)
    prepare_train_logs(tmp_path, resume=args.resume)
    save_train_log_row(_train_log_row(1), tmp_path)

    with (tmp_path / "train_log.csv").open(newline="") as handle:
        assert [row["epoch"] for row in csv.DictReader(handle)] == ["1"]
    assert [row["epoch"] for row in map(json.loads, (tmp_path / "train_log.jsonl").read_text().splitlines())] == [1]


@pytest.mark.parametrize("resume", ["auto", "checkpoint.pt"], ids=("auto", "explicit-checkpoint"))
def test_resumed_run_keeps_existing_train_logs_and_appends(tmp_path, resume):
    save_train_log_row(_train_log_row(1), tmp_path)

    prepare_train_logs(tmp_path, resume=resume)
    save_train_log_row(_train_log_row(2), tmp_path)

    with (tmp_path / "train_log.csv").open(newline="") as handle:
        assert [row["epoch"] for row in csv.DictReader(handle)] == ["1", "2"]
    assert [row["epoch"] for row in map(json.loads, (tmp_path / "train_log.jsonl").read_text().splitlines())] == [1, 2]


def test_train_log_rows_are_valid_csv_and_jsonl(tmp_path):
    expected = _train_log_row(1)
    save_train_log_row(expected, tmp_path)

    with (tmp_path / "train_log.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    jsonl_rows = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert list(csv_rows[0]) == list(expected)
    assert len(csv_rows) == len(jsonl_rows) == 1
    assert jsonl_rows == [expected]


def test_debug_num_images_limits_dataset_size():
    data = list(range(10))
    assert len(limit_dataset(data, 3)) == 3
    assert limit_dataset(data, 0) is data


def test_confidence_threshold_changes_retained_predictions():
    decoded = [torch.tensor([[0., 0., 1., 1., .2, 0.], [2., 2., 3., 3., .8, 0.]])]
    assert len(non_max_suppression(decoded, .1)[0]) == 2
    assert len(non_max_suppression(decoded, .5)[0]) == 1


def test_evaluator_writes_box_diagnostics(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, rgb, ir):
            return (_encoded_head(2, active=True), _encoded_head(1), _encoded_head(1))
    batch = {"rgb": torch.zeros(1, 3, 100, 100), "ir": torch.zeros(1, 3, 100, 100),
             "labels": torch.tensor([[0., 0., .5, .5, .25, .25]])}
    path = tmp_path / "eval_diagnostics.json"
    evaluate(Model(), [batch], torch.device("cpu"), diagnostics_path=path)
    import json
    diagnostics = json.loads(path.read_text())
    assert diagnostics["decoded_boxes"] == 6
    assert diagnostics["boxes_after_confidence"] == 1
    assert diagnostics["boxes_after_nms"] == 1
