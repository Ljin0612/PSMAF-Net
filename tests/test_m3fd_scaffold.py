from pathlib import Path

from detection.scripts.m3fd_maskrcnn_scaffold import (
    detectron2_config_template,
    load_experiment,
    split_file_count,
)


def test_m3fd_config_records_official_detection_setting(tmp_path):
    experiment, _ = load_experiment(Path("detection/configs/m3fd_ir_maskrcnn_univ.yaml"))

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
