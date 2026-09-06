import pytest
torch = pytest.importorskip("torch")

from core.psmaf import MultiScaleAdaptiveFusion, PSMAFFusion, PseudoSemanticGuidance
from detection.models.psmaf_yolo import PSMAFYOLO
from detection.scripts.psmaf_yolo_utils import decode_outputs, evaluate_detection, non_max_suppression


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
