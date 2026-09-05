from pathlib import Path

from detection.scripts.run_m3fd_univ_fasterrcnn import (
    format_bbox_metrics,
    parse_args,
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
                "AP-people": 11.0,
                "AP-car": 12.0,
            }
        }
    )

    assert metrics["AP"] == 10.0
    assert metrics["AP50"] == 20.0
    assert metrics["AP75"] == 5.0
    assert metrics["AP-people"] == 11.0
    assert metrics["AP-car"] == 12.0
    assert "AP-truck" in metrics


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
    assert 'return self.feature_adapter(self._encoder_stages(x))' in source
    assert "MIN_LOAD_RATIO" in source
    assert "LORA_ALPHA" in source
    assert "def _merge_lora_weights(" in source
    assert "loaded_numel" in source
    assert 'required_stem = "patch_embed1.proj.weight"' in source
