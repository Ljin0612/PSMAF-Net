#!/usr/bin/env python3
"""Train PSMAF-YOLO on paired M3FD images."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from detection.datasets import M3FDPairedDataset, paired_collate_fn
from detection.models.psmaf_yolo import PSMAFYOLO, detection_loss
from detection.scripts.psmaf_yolo_utils import (evaluate, limit_dataset, reset_train_logs, save_metrics,
                                                 save_train_log_row, seed_everything)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root"); p.add_argument("--data", default="detection/configs/m3fd_psmaf_yolo.yaml")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="cpu")
    p.add_argument("--weights"); p.add_argument("--project", default="runs/detect"); p.add_argument("--name", default="psmaf-yolo")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", nargs="?", const="auto"); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-period", type=int, default=10); p.add_argument("--fusion-mode", choices=("psmaf", "add", "concat"), default="psmaf")
    p.add_argument("--no-psg", action="store_true"); p.add_argument("--no-msaf", action="store_true")
    p.add_argument("--conf-thres", type=float, default=0.25); p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--eval-train", action="store_true")
    p.add_argument("--debug-num-images", type=int, default=0,
                   help="use only the first N train/val samples for debugging; not for official reporting")
    return p


def prepare_train_logs(output_dir, resume):
    """Reset logs for a fresh run, or preserve them when resuming."""
    if resume:
        print(f"Resuming run; keeping existing training logs: {output_dir}")
    else:
        reset_train_logs(output_dir)
        print(f"Reset training logs for fresh run: {output_dir}")


def main(args=None):
    args = parser().parse_args(args); seed_everything(args.seed)
    config = yaml.safe_load(Path(args.data).read_text()); root = args.dataset_root or config["path"]
    device = torch.device(args.device); output = Path(args.project) / args.name; output.mkdir(parents=True, exist_ok=True)
    train = limit_dataset(M3FDPairedDataset(root, config["train"], args.imgsz), args.debug_num_images)
    val = limit_dataset(M3FDPairedDataset(root, config["val"], args.imgsz), args.debug_num_images)
    kwargs = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=paired_collate_fn)
    train_loader = DataLoader(train, shuffle=True, **kwargs); val_loader = DataLoader(val, **kwargs)
    model = PSMAFYOLO(config["nc"], fusion_mode=args.fusion_mode, use_psg=not args.no_psg, use_msaf=not args.no_msaf).to(device)
    if args.resume and args.weights:
        warnings.warn("--resume takes precedence over --weights", stacklevel=1)
    checkpoint = (output / "last.pt" if args.resume == "auto" else Path(args.resume)) if args.resume else args.weights
    start = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if checkpoint:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        print(f"Loading checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model", state))
        if args.resume:
            start = state.get("epoch", -1) + 1
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            else:
                warnings.warn("checkpoint has no optimizer state; using a fresh optimizer", stacklevel=1)
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            else:
                warnings.warn("checkpoint has no AMP scaler state; using a fresh scaler", stacklevel=1)
    prepare_train_logs(output, args.resume)
    for epoch in range(start, args.epochs):
        model.train()
        sums = {key: 0.0 for key in ("loss", "obj_loss", "box_loss", "cls_loss")}; batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); context = torch.autocast(device.type) if scaler.is_enabled() else nullcontext()
            with context:
                components = detection_loss(model(batch["rgb"].to(device), batch["ir"].to(device)),
                                            batch["labels"].to(device), config["nc"], return_components=True)
                loss = components["loss"]
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            batches += 1
            for key in sums: sums[key] += float(components[key].detach())
        metrics = evaluate(model, val_loader, device, args.conf_thres, args.nms_iou,
                           output / "eval_diagnostics.json")
        if args.eval_train:
            train_metrics = evaluate(model, train_loader, device, args.conf_thres, args.nms_iou)
            save_metrics(train_metrics, output, "train_metrics")
            save_metrics(metrics, output, "val_metrics")
        row = {"epoch": epoch + 1, "avg_total_loss": sums["loss"] / max(batches, 1),
               "avg_obj_loss": sums["obj_loss"] / max(batches, 1),
               "avg_box_loss": sums["box_loss"] / max(batches, 1),
               "avg_cls_loss": sums["cls_loss"] / max(batches, 1),
               "num_pos": None,
               "learning_rate": optimizer.param_groups[0]["lr"], "val_precision": metrics["precision"],
               "val_recall": metrics["recall"], "val_AP50": metrics["AP50"],
               "val_mAP50_95": metrics["mAP50_95"]}
        save_train_log_row(row, output)
        print(" ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "args": vars(args), "metrics": metrics}; torch.save(state, output / "last.pt")
        if args.save_period > 0 and (epoch + 1) % args.save_period == 0: torch.save(state, output / f"epoch{epoch + 1}.pt")
        save_metrics(metrics, output, "metrics")


if __name__ == "__main__": main()
