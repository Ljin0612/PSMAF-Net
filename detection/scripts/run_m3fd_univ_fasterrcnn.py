#!/usr/bin/env python3
"""Train and evaluate the UNIV-original Faster R-CNN baseline on M3FD-IR."""

from __future__ import annotations

import argparse
import json
import math
import sys
from numbers import Real
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.scripts.check_m3fd_detection import check_dataset
from detection.scripts.make_m3fd_ir_coco import convert
from detection.scripts.register_m3fd_detectron2 import THING_CLASSES, register_m3fd_coco
from detection.scripts.run_m3fd_maskrcnn_smoke import (
    compute_epoch_schedule,
    print_schedule,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UNIV-original M3FD-IR Faster R-CNN bbox baseline."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--amp", action="store_true", help="Enable Detectron2 AMP training."
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Checkpoint UNIV blocks3 to reduce fine-tuning activation memory.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/detection/m3fd_univ"))
    parser.add_argument("--ims-per-batch", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override epoch scheduling with an exact iteration count.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Force a 10-iteration checkpoint/forward/loss/backward smoke run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--eval-train",
        action="store_true",
        help="Evaluate the final model on the (possibly debug-limited) training split.",
    )
    parser.add_argument(
        "--debug-num-images",
        type=int,
        default=None,
        metavar="N",
        help="Train on only the first N training images for quick overfitting tests.",
    )
    parser.add_argument(
        "--save-visualizations",
        type=int,
        default=0,
        metavar="N",
        help="Save prediction-vs-GT panels for N test images after evaluation.",
    )
    args = parser.parse_args(argv)
    if args.debug_num_images is not None and args.debug_num_images <= 0:
        parser.error("--debug-num-images must be greater than zero")
    if args.save_visualizations < 0:
        parser.error("--save-visualizations must be non-negative")
    return args


def register_debug_subset(source_name: str, subset_name: str, num_images: int) -> str:
    """Register a deterministic prefix of a Detectron2 dataset."""
    from detectron2.data import DatasetCatalog, MetadataCatalog

    records = DatasetCatalog.get(source_name)[:num_images]
    DatasetCatalog.register(subset_name, lambda records=records: records)
    metadata = MetadataCatalog.get(source_name).as_dict()
    metadata.pop("name", None)
    MetadataCatalog.get(subset_name).set(**metadata)
    print(f"debug subset: {subset_name} uses {len(records)} of " f"{source_name}")
    return subset_name


def save_prediction_visualizations(cfg, model, dataset_name: str, count: int) -> None:
    """Save side-by-side GT (green) and prediction (red, scored) panels."""
    if count <= 0:
        return

    import cv2
    import torch
    from detectron2.data import DatasetCatalog, DatasetMapper
    from detectron2.evaluation.evaluator import inference_context

    output_dir = Path(cfg.OUTPUT_DIR).parent / "debug_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    mapper_cfg = cfg.clone()
    mapper_cfg.defrost()
    mapper_cfg.INPUT.RANDOM_FLIP = "none"
    mapper_cfg.freeze()
    # The training mapper retains transformed annotations as Instances. The model
    # is in inference mode below, so those instances are ignored for prediction.
    mapper = DatasetMapper(mapper_cfg, is_train=True)
    records = DatasetCatalog.get(dataset_name)[:count]

    def draw_boxes(image, boxes, labels, color):
        canvas = image.copy()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    with inference_context(model), torch.no_grad():
        for index, record in enumerate(records):
            mapped = mapper(record)
            prediction = model([mapped])[0]["instances"].to("cpu")
            image = mapped["image"].permute(1, 2, 0).cpu().numpy().copy()
            gt = mapped["instances"].to("cpu")
            gt_labels = [f"GT {THING_CLASSES[int(cls)]}" for cls in gt.gt_classes]
            pred_labels = [
                f"{THING_CLASSES[int(cls)]} {float(score):.2f}"
                for cls, score in zip(prediction.pred_classes, prediction.scores)
            ]
            gt_panel = draw_boxes(image, gt.gt_boxes.tensor, gt_labels, (0, 255, 0))
            pred_panel = draw_boxes(
                image, prediction.pred_boxes.tensor, pred_labels, (0, 0, 255)
            )
            cv2.putText(
                gt_panel,
                "GROUND TRUTH",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                pred_panel,
                "PREDICTIONS",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            panel = cv2.hconcat([gt_panel, pred_panel])
            stem = Path(record["file_name"]).stem
            cv2.imwrite(str(output_dir / f"{index:04d}_{stem}.jpg"), panel)
    print(f"saved {len(records)} prediction visualizations to {output_dir}")


def build_trainer():
    try:
        import torch
        from detectron2.engine import DefaultTrainer
        from detectron2.engine.hooks import HookBase
        from detectron2.evaluation import COCOEvaluator
    except ImportError as exc:
        raise ImportError(
            "Detectron2 is required for the UNIV detection baseline. Install a "
            "build compatible with the active PyTorch and CUDA versions."
        ) from exc

    def log_cuda_memory(label: str) -> None:
        if not torch.cuda.is_available():
            print(f"CUDA memory [{label}]: unavailable (CUDA is not available)")
            return
        device = torch.cuda.current_device()
        mib = 1024**2
        print(
            f"CUDA memory [{label}]: "
            f"allocated={torch.cuda.memory_allocated(device) / mib:.1f} MiB "
            f"reserved={torch.cuda.memory_reserved(device) / mib:.1f} MiB "
            f"peak_allocated={torch.cuda.max_memory_allocated(device) / mib:.1f} MiB"
        )

    class FirstBackwardMemoryHook(HookBase):
        def __init__(self) -> None:
            self.logged = False

        def after_step(self) -> None:
            if not self.logged:
                log_cuda_memory("first backward")
                self.logged = True

    class M3FDUNIVTrainer(DefaultTrainer):
        @classmethod
        def build_model(cls, cfg):
            model = super().build_model(cfg)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            log_cuda_memory("model build")

            handle = None

            def log_first_forward(_module, _inputs, _output):
                nonlocal handle
                log_cuda_memory("first forward")
                if handle is not None:
                    handle.remove()

            handle = model.register_forward_hook(log_first_forward)
            return model

        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.append(FirstBackwardMemoryHook())
            return hooks

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            output_folder = output_folder or str(Path(cfg.OUTPUT_DIR) / "eval")
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            return COCOEvaluator(
                dataset_name,
                tasks=("bbox",),
                distributed=True,
                output_dir=output_folder,
            )

    return M3FDUNIVTrainer


def format_bbox_metrics(results: dict[str, Any] | None) -> dict[str, float | None]:
    """Select the paper-facing aggregate and per-class bbox metrics."""
    bbox = get_bbox_result(results)
    names = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        *(f"AP-{name}" for name in THING_CLASSES),
    )
    return {name: _finite_metric(bbox.get(name)) for name in names}


def _finite_metric(value: Any) -> float | None:
    """Return a finite numeric metric, normalizing undefined COCO values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def sanitize_json_numbers(value: Any) -> Any:
    """Recursively replace non-finite numbers so result JSON stays standards-compliant."""
    if isinstance(value, dict):
        return {key: sanitize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_numbers(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def get_bbox_result(results):
    """Return the bbox metrics mapping from flat or Detectron2-style results."""
    if not isinstance(results, dict):
        return {}
    bbox = results.get("bbox", results)
    return bbox if isinstance(bbox, dict) else {}


def has_bbox_metrics(results):
    """Whether results contain at least one finite standard COCO bbox metric."""
    bbox = get_bbox_result(results)
    return any(
        _finite_metric(bbox.get(key)) is not None for key in ("AP", "AP50", "AP75")
    )


def should_run_fallback_eval(results):
    """Make rank zero's fallback decision authoritative for every worker."""
    # Keep Detectron2 optional when this module is imported for argument parsing.
    from detectron2.utils import comm

    local_needs_eval = not has_bbox_metrics(results)

    if comm.get_world_size() == 1:
        return local_needs_eval

    gathered = comm.all_gather(
        {"rank": comm.get_rank(), "needs_eval": local_needs_eval}
    )
    main_state = next(item for item in gathered if item["rank"] == 0)
    return bool(main_state["needs_eval"])


def resolve_evaluation_results(trainer, trainer_class, cfg, results):
    """Recover evaluation results when ``train`` does not return hook output."""
    if not has_bbox_metrics(results):
        last_eval_results = getattr(trainer, "_last_eval_results", None)
        if has_bbox_metrics(last_eval_results):
            results = last_eval_results
    if should_run_fallback_eval(results):
        results = trainer_class.test(cfg, trainer.model)
    return results


def _check_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNIV checkpoint does not exist: {checkpoint}")
    for split in ("train", "test"):
        errors, warnings, count = check_dataset(dataset_root, "m3fd-rgbt", split)
        print(f"dataset check split={split}: samples={count}")
        for warning in warnings:
            print(f"  warning: {warning}")
        if errors:
            raise ValueError(f"M3FD {split} check failed: {'; '.join(errors)}")
    return dataset_root, checkpoint, work_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root, checkpoint, work_dir = _check_inputs(args)
    train_json = work_dir / "coco" / "m3fd_ir_train.json"
    test_json = work_dir / "coco" / "m3fd_ir_test.json"

    if args.dry_run:
        print("UNIV-original baseline dry run")
        print(f"  dataset_root: {dataset_root}")
        print(f"  checkpoint: {checkpoint}")
        print(f"  freeze_backbone: {args.freeze_backbone}")
        print(f"  amp: {args.amp}")
        print(f"  gradient_checkpointing: {args.gradient_checkpointing}")
        print(f"  eval_train: {args.eval_train}")
        print(f"  debug_num_images: {args.debug_num_images}")
        print(f"  save_visualizations: {args.save_visualizations}")
        print(f"  device: {args.device}")
        print("dry_run: no files generated and no training started")
        return 0

    train_stats = convert(dataset_root, "train", train_json)
    test_stats = convert(dataset_root, "test", test_json)
    print(f"train conversion: {train_stats}")
    print(f"test conversion: {test_stats}")

    exact_iterations = 10 if args.smoke_test else args.max_iter
    num_train_images = int(train_stats["num_images"])
    if args.debug_num_images is not None:
        num_train_images = min(num_train_images, args.debug_num_images)
    schedule = compute_epoch_schedule(
        num_train_images=num_train_images,
        ims_per_batch=args.ims_per_batch,
        max_iter=exact_iterations or 1,
        epochs=None if exact_iterations is not None else args.epochs,
        eval_every_epochs=args.eval_every_epochs,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
    )
    print_schedule(schedule, exact_iterations or 1)

    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
    except ImportError as exc:
        raise ImportError(
            "Detectron2 is required for the UNIV detection baseline."
        ) from exc

    # Import registers UNIVBackbone in Detectron2's BACKBONE_REGISTRY.
    from detection.models.univ_backbone import add_univ_config

    register_m3fd_coco("m3fd_ir_train_univ", str(dataset_root / "ir"), str(train_json))
    register_m3fd_coco("m3fd_ir_test_univ", str(dataset_root / "ir"), str(test_json))
    train_dataset_name = "m3fd_ir_train_univ"
    if args.debug_num_images is not None:
        train_dataset_name = register_debug_subset(
            train_dataset_name, "m3fd_ir_train_univ_debug", args.debug_num_images
        )

    cfg = get_cfg()
    add_univ_config(cfg)
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.merge_from_file(str(REPO_ROOT / "configs/detection/univ_fasterrcnn_m3fd.yaml"))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = ("m3fd_ir_test_univ",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.UNIV.CHECKPOINT = str(checkpoint)
    cfg.MODEL.UNIV.FREEZE = bool(args.freeze_backbone)
    cfg.MODEL.UNIV.GRADIENT_CHECKPOINTING = bool(args.gradient_checkpointing)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ""
    cfg.SOLVER.IMS_PER_BATCH = int(args.ims_per_batch)
    cfg.SOLVER.AMP.ENABLED = bool(args.amp)
    cfg.SOLVER.MAX_ITER = int(schedule.max_iter)
    cfg.SOLVER.CHECKPOINT_PERIOD = int(schedule.checkpoint_period)
    cfg.TEST.EVAL_PERIOD = int(schedule.eval_period)
    cfg.INPUT.MIN_SIZE_TRAIN = (int(args.input_size),)
    cfg.INPUT.MAX_SIZE_TRAIN = int(args.input_size)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_size)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_size)
    cfg.OUTPUT_DIR = str(work_dir / "detectron2_output")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    runtime = {
        "baseline": "UNIV-original",
        "checkpoint": str(checkpoint),
        "freeze_backbone": bool(args.freeze_backbone),
        "amp": bool(args.amp),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "device": args.device,
        "max_iter": schedule.max_iter,
        "epochs": schedule.epochs,
        "eval_period": schedule.eval_period,
        "annotation_type": "bbox_only",
        "eval_train": bool(args.eval_train),
        "debug_num_images": args.debug_num_images,
        "save_visualizations": args.save_visualizations,
    }
    (work_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    trainer_class = build_trainer()
    trainer = trainer_class(cfg)
    trainer.resume_or_load(resume=False)
    results = trainer.train()

    if not has_bbox_metrics(results):
        last_eval_results = getattr(trainer, "_last_eval_results", None)
        if has_bbox_metrics(last_eval_results):
            results = last_eval_results

    if should_run_fallback_eval(results):
        results = trainer_class.test(cfg, trainer.model)

    train_results = None
    if args.eval_train:
        train_eval_cfg = cfg.clone()
        train_eval_cfg.defrost()
        train_eval_cfg.DATASETS.TEST = (train_dataset_name,)
        train_eval_cfg.freeze()
        evaluator = trainer_class.build_evaluator(
            train_eval_cfg,
            train_dataset_name,
            str(work_dir / "detectron2_output" / "eval_train"),
        )
        train_results = trainer_class.test(
            train_eval_cfg, trainer.model, evaluators=[evaluator]
        )

    from detectron2.utils import comm

    if comm.is_main_process():
        print(f"raw evaluation results: {results}")
        metrics = format_bbox_metrics(results)

        (work_dir / "raw_eval_results.json").write_text(
            json.dumps(sanitize_json_numbers(results), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (work_dir / "bbox_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )

        if train_results is not None:
            print(f"raw training-split evaluation results: {train_results}")
            (work_dir / "raw_train_eval_results.json").write_text(
                json.dumps(
                    sanitize_json_numbers(train_results), indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )
            (work_dir / "train_bbox_metrics.json").write_text(
                json.dumps(
                    format_bbox_metrics(train_results), indent=2, allow_nan=False
                ),
                encoding="utf-8",
            )

        visualization_model = getattr(trainer.model, "module", trainer.model)
        save_prediction_visualizations(
            cfg,
            visualization_model,
            "m3fd_ir_test_univ",
            args.save_visualizations,
        )

        print("UNIV-original bbox metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
