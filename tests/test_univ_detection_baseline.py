import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from detection.scripts.run_m3fd_univ_fasterrcnn import (
    format_bbox_metrics,
    get_bbox_result,
    has_bbox_metrics,
    parse_args,
    resolve_evaluation_results,
    sanitize_json_numbers,
    should_run_fallback_eval,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_detectron_comm(monkeypatch):
    """Install the optional Detectron2 communication module for unit tests."""
    comm = SimpleNamespace(
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        all_gather=lambda value: [value],
        is_main_process=lambda: True,
    )
    detectron2 = ModuleType("detectron2")
    utils = ModuleType("detectron2.utils")
    utils.comm = comm
    detectron2.utils = utils
    monkeypatch.setitem(sys.modules, "detectron2", detectron2)
    monkeypatch.setitem(sys.modules, "detectron2.utils", utils)
    return comm


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
            "--eval-train",
            "--debug-num-images",
            "8",
            "--save-visualizations",
            "3",
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
    assert args.eval_train is True
    assert args.debug_num_images == 8
    assert args.save_visualizations == 3


@pytest.mark.parametrize(
    "option,value", [("--debug-num-images", "0"), ("--save-visualizations", "-1")]
)
def test_univ_runner_rejects_invalid_debug_counts(option, value):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset-root",
                "/data/M3FD",
                "--checkpoint",
                "/models/univ.pth",
                option,
                value,
            ]
        )


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


def test_univ_runner_uses_direct_nested_detectron2_result(fake_detectron_comm):
    class TrainerClass:
        @classmethod
        def test(cls, cfg, model):
            raise AssertionError("test fallback should not run")

    trainer = type("Trainer", (), {"model": object()})()
    direct = {"bbox": {"AP": 1.0, "AP50": 2.0}}

    results = resolve_evaluation_results(trainer, TrainerClass, object(), direct)

    assert results == direct


def test_univ_runner_recovers_last_eval_results_when_train_returns_none(
    fake_detectron_comm,
):
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


def test_univ_runner_runs_test_when_train_and_last_eval_results_are_missing(
    fake_detectron_comm,
):
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


def test_univ_runner_normalizes_undefined_metrics_for_strict_json():
    results = {"bbox": {"AP": float("nan"), "AP50": float("inf"), "AP75": 3}}

    metrics = format_bbox_metrics(results)

    assert metrics["AP"] is None
    assert metrics["AP50"] is None
    assert metrics["AP75"] == 3.0
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


def test_univ_runner_does_not_treat_non_finite_values_as_evaluation():
    assert has_bbox_metrics({"bbox": {"AP": float("nan")}}) is False
    assert has_bbox_metrics({"bbox": {"AP": "12.3"}}) is False
    assert has_bbox_metrics({"bbox": {"AP": False}}) is False


def test_univ_runner_sanitizes_nested_raw_results_for_json():
    sanitized = sanitize_json_numbers(
        {"bbox": {"AP": float("nan")}, "details": [1, float("-inf")]}
    )

    assert sanitized == {"bbox": {"AP": None}, "details": [1.0, None]}
    assert json.loads(json.dumps(sanitized, allow_nan=False)) == sanitized


def test_univ_runner_recognizes_existing_bbox_metrics(fake_detectron_comm):
    results = {"bbox": {"AP": 1.0, "AP50": 2.0}}

    assert get_bbox_result(results) == results["bbox"]
    assert has_bbox_metrics(results) is True
    assert should_run_fallback_eval(results) is False


def test_univ_runner_fallback_uses_rank_zero_decision(fake_detectron_comm):
    fake_detectron_comm.get_world_size = lambda: 2
    fake_detectron_comm.get_rank = lambda: 1
    fake_detectron_comm.all_gather = lambda value: [
        {"rank": 0, "needs_eval": False},
        value,
    ]

    # Rank 1 has no local evaluator result, but must follow rank 0 and not evaluate.
    assert should_run_fallback_eval({}) is False


def test_univ_runner_final_json_writes_are_main_process_only():
    source = (
        ROOT / "detection/scripts/run_m3fd_univ_fasterrcnn.py"
    ).read_text(encoding="utf-8")
    guarded_block = source.split("if comm.is_main_process():", 1)[1]

    assert 'work_dir / "raw_eval_results.json"' in guarded_block
    assert 'work_dir / "bbox_metrics.json"' in guarded_block
    assert 'work_dir / "train_bbox_metrics.json"' in guarded_block
    assert "save_prediction_visualizations(" in guarded_block


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
