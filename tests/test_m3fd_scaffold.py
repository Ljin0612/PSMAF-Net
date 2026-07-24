import argparse
import json
import sys
from pathlib import Path

from detection.scripts.m3fd_maskrcnn_scaffold import (
    detectron2_config_template,
    load_experiment,
    split_file_count,
)


def test_m3fd_config_records_official_detection_setting(tmp_path):
    experiment, _ = load_experiment(
        Path("detection/configs/m3fd_ir_maskrcnn_univ.yaml")
    )

    assert experiment.dataset_name == "M3FD-IR"
    assert experiment.modality == "Infrared"
    assert experiment.head == "MaskRCNN"
    assert experiment.framework == "Detectron2"
    assert experiment.image_size == (1024, 1024)
    assert experiment.fine_tuning_steps == 85000
    assert experiment.optimizer == "AdamW"
    assert experiment.base_learning_rate == 8.0e-5
    assert experiment.weight_decay == 0.1
    assert experiment.batch_size == 4
    assert experiment.train_images == 3360
    assert experiment.test_images == 840

    rendered = detectron2_config_template(experiment, tmp_path / "out")
    assert "MAX_ITER: 85000" in rendered
    assert "IMS_PER_BATCH: 4" in rendered
    assert "OPTIMIZER: AdamW" in rendered


def test_split_file_count_ignores_blank_and_comment_lines(tmp_path):
    split_file = tmp_path / "train.txt"
    split_file.write_text("image_1\n\n# ignored\n image_2 \n", encoding="utf-8")

    assert split_file_count(split_file) == 2


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "detection" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from make_m3fd_ir_coco import convert  # noqa: E402
from make_m3fd_maskrcnn_config import build_config  # noqa: E402
from run_m3fd_maskrcnn_smoke import compute_epoch_schedule  # noqa: E402


def test_epoch_schedule_converts_epochs_to_max_iter():
    schedule = compute_epoch_schedule(
        num_train_images=3360,
        ims_per_batch=1,
        max_iter=10,
        epochs=5,
        eval_every_epochs=1,
        checkpoint_every_epochs=1,
    )

    assert schedule.steps_per_epoch == 3360
    assert schedule.max_iter == 16800
    assert schedule.used_epochs is True


def test_epoch_schedule_converts_eval_and_checkpoint_periods():
    schedule = compute_epoch_schedule(
        num_train_images=3360,
        ims_per_batch=2,
        max_iter=10,
        epochs=12,
        eval_every_epochs=3,
        checkpoint_every_epochs=5,
    )

    assert schedule.steps_per_epoch == 1680
    assert schedule.eval_period == 5040
    assert schedule.checkpoint_period == 8400


def test_epoch_schedule_allows_disabling_periodic_eval():
    schedule = compute_epoch_schedule(
        num_train_images=3360,
        ims_per_batch=1,
        max_iter=10,
        epochs=1,
        eval_every_epochs=0,
        checkpoint_every_epochs=1,
    )

    assert schedule.eval_period == 0


def test_epoch_schedule_keeps_legacy_max_iter_mode():
    schedule = compute_epoch_schedule(
        num_train_images=3360,
        ims_per_batch=1,
        max_iter=10,
        epochs=None,
        eval_every_epochs=1,
        checkpoint_every_epochs=1,
    )

    assert schedule.max_iter == 10
    assert schedule.epochs is None
    assert schedule.used_epochs is False


def test_m3fd_smoke_config_is_bbox_detection_only():
    config = build_config(
        argparse.Namespace(
            train_json="train.json",
            train_image_root="ir/train",
            test_json="test.json",
            test_image_root="ir/test",
            output_dir="out",
            num_classes=6,
            max_iter=10,
            ims_per_batch=1,
            base_lr=8e-5,
            input_size=1024,
            weights=None,
        )
    )

    assert config["model_zoo_config"] == "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    assert (
        config["model_zoo_config"]
        != "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    assert config["mask_on"] is False
    assert config["annotation_type"] == "bbox_only"


def test_m3fd_ir_coco_converter_emits_bbox_only_metadata_and_annotations(tmp_path):
    dataset_root = tmp_path / "M3FD"
    for relative in ("ir", "labels", "meta"):
        (dataset_root / relative).mkdir(parents=True)
    (dataset_root / "meta" / "train.txt").write_text("sample_001\n", encoding="utf-8")
    # Minimal 1x1 PNG header/body; image_size_from_header can read dimensions without Pillow.
    (dataset_root / "ir" / "sample_001.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    )
    (dataset_root / "labels" / "sample_001.txt").write_text(
        "0 0.5 0.5 1.0 1.0\n", encoding="utf-8"
    )

    output_json = tmp_path / "out" / "m3fd_ir_train.json"
    stats = convert(dataset_root, "train", output_json)
    coco = json.loads(output_json.read_text(encoding="utf-8"))

    assert stats["annotation_type"] == "bbox_only"
    assert stats["has_segmentation"] is False
    assert set(coco) == {"info", "licenses", "images", "annotations", "categories"}
    assert (
        coco["info"]["description"]
        == "M3FD bbox detection dataset converted from YOLO labels"
    )
    assert coco["info"]["version"] == "1.0"
    assert coco["info"]["year"] == 2026
    assert coco["info"]["contributor"] == "PSMAF-Net"
    assert coco["info"]["date_created"]
    assert coco["licenses"] == [{"id": 1, "name": "Unknown", "url": ""}]
    assert coco["images"]
    assert coco["annotations"]
    assert "bbox" in coco["annotations"][0]
    assert "segmentation" not in coco["annotations"][0]
    assert [category["id"] for category in coco["categories"]] == [1, 2, 3, 4, 5, 6]
    assert [category["name"] for category in coco["categories"]] == [
        "people",
        "car",
        "bus",
        "motorcycle",
        "lamp",
        "truck",
    ]
    assert coco["annotations"][0]["category_id"] == 1


def test_smoke_runner_explicitly_disables_mask_head():
    source = (SCRIPTS_DIR / "run_m3fd_maskrcnn_smoke.py").read_text(encoding="utf-8")

    assert "cfg.MODEL.MASK_ON = False" in source
    assert 'cfg.MODEL.MASK_ON = bool(config.get("mask_on", False))' in source
    assert "bbox-only," in source
    assert "mask head must stay disabled" in source


def test_smoke_runner_builds_coco_bbox_evaluator():
    source = (SCRIPTS_DIR / "run_m3fd_maskrcnn_smoke.py").read_text(encoding="utf-8")

    assert "class M3FDBBoxSmokeTrainer(DefaultTrainer):" in source
    assert "def build_evaluator" in source
    assert "COCOEvaluator" in source
    assert 'tasks=("bbox",)' in source
    assert 'Path(cfg.OUTPUT_DIR).parent / "eval"' in source
