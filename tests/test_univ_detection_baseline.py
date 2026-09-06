from pathlib import Path

from detection.scripts.run_m3fd_univ_fasterrcnn import (
    format_bbox_metrics,
    parse_args,
    resolve_evaluation_results,
)


ROOT = Path(__file__).resolve().parents[1]


def test_univ_runner_exposes_required_arguments():
    args = parse_args(
        [
            "--dataset-root",
            "/data/M3FD",
            "--checkpoint",
            "/models/univ.pth",
            "--freeze-backbone",
            "--amp",
            "--gradient-checkpointing",
            "--epochs",
            "5",
            "--eval-every-epochs",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert args.dataset_root == Path("/data/M3FD")
    assert args.checkpoint == Path("/models/univ.pth")
    assert args.freeze_backbone is True
    assert args.amp is True
    assert args.gradient_checkpointing is True
    assert args.epochs == 5
    assert args.eval_every_epochs == 2
    assert args.device == "cpu"


def test_univ_runner_reports_aggregate_and_per_class_ap():
    metrics = format_bbox_metrics(
        {
            "bbox": {
                "AP": 10.0,
                "AP50": 20.0,
                "AP75": 5.0,
                "APs": 4.0,
                "APm": 8.0,
                "APl": 16.0,
                "AP-people": 11.0,
                "AP-car": 12.0,
            }
        }
    )

    assert metrics["AP"] == 10.0
    assert metrics["AP50"] == 20.0
    assert metrics["AP75"] == 5.0
    assert metrics["APs"] == 4.0
    assert metrics["APm"] == 8.0
    assert metrics["APl"] == 16.0
    assert metrics["AP-people"] == 11.0
    assert metrics["AP-car"] == 12.0
    assert "AP-truck" in metrics


def test_univ_runner_extracts_nested_detectron2_bbox_results():
    metrics = format_bbox_metrics(
        {"bbox": {"AP": 1.0, "AP50": 2.0, "AP75": 3.0}}
    )

    assert metrics["AP"] == 1.0
    assert metrics["AP50"] == 2.0
    assert metrics["AP75"] == 3.0


def test_univ_runner_uses_direct_nested_detectron2_result():
    class TrainerClass:
        @classmethod
        def test(cls, cfg, model):
            raise AssertionError("test fallback should not run")

    trainer = type("Trainer", (), {"model": object()})()
    direct = {"bbox": {"AP": 1.0, "AP50": 2.0}}

    results = resolve_evaluation_results(trainer, TrainerClass, object(), direct)

    assert results == direct


def test_univ_runner_recovers_last_eval_results_when_train_returns_none():
    class TrainerClass:
        @classmethod
        def test(cls, cfg, model):
            raise AssertionError("test fallback should not run")

    expected = {"bbox": {"AP": 3.0}}
    trainer = type(
        "Trainer", (), {"model": object(), "_last_eval_results": expected}
    )()

    results = resolve_evaluation_results(trainer, TrainerClass, object(), None)

    assert results == expected


def test_univ_runner_runs_test_when_train_and_last_eval_results_are_missing():
    cfg = object()
    model = object()
    expected = {"bbox": {"AP": 4.0}}

    class TrainerClass:
        @classmethod
        def test(cls, actual_cfg, actual_model):
            assert actual_cfg is cfg
            assert actual_model is model
            return expected

    trainer = type("Trainer", (), {"model": model})()

    results = resolve_evaluation_results(trainer, TrainerClass, cfg, None)

    assert results == expected


def test_univ_runner_accepts_flat_bbox_results_as_fallback():
    metrics = format_bbox_metrics({"AP": 1.0, "AP50": 2.0, "AP75": 3.0})

    assert metrics["AP"] == 1.0
    assert metrics["AP50"] == 2.0
    assert metrics["AP75"] == 3.0


def test_univ_config_selects_registered_backbone_and_bbox_features():
    config = (ROOT / "configs/detection/univ_fasterrcnn_m3fd.yaml").read_text(
        encoding="utf-8"
    )

    assert "NAME: UNIVBackbone" in config
    assert "IN_FEATURES: [p2, p3, p4, p5]" in config
    assert "SIZES: [[32], [64], [128], [256]]" in config
    assert "MASK_ON: false" in config


def test_univ_adapter_does_not_modify_vendored_source_contract():
    source = (ROOT / "detection/models/univ_backbone.py").read_text(encoding="utf-8")

    assert "third_party.UNIV.models.backbone.mcmae.vision_transformer" in source
    assert "class UNIVBackbone(Backbone):" in source
    assert "with torch.no_grad():" in source
    assert "checkpoint(block, x, use_reentrant=False)" in source
