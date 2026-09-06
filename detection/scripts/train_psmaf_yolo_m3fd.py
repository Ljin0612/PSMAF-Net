#!/usr/bin/env python3
"""Train PSMAF-YOLO on paired M3FD images."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from detection.datasets import M3FDPairedDataset, paired_collate_fn
from detection.models.psmaf_yolo import PSMAFYOLO, detection_loss
from detection.scripts.psmaf_yolo_utils import evaluate, save_metrics, seed_everything


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
    return p


def main(args=None):
    args = parser().parse_args(args); seed_everything(args.seed)
    config = yaml.safe_load(Path(args.data).read_text()); root = args.dataset_root or config["path"]
    device = torch.device(args.device); output = Path(args.project) / args.name; output.mkdir(parents=True, exist_ok=True)
    train = M3FDPairedDataset(root, config["train"], args.imgsz); val = M3FDPairedDataset(root, config["val"], args.imgsz)
    kwargs = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=paired_collate_fn)
    train_loader = DataLoader(train, shuffle=True, **kwargs); val_loader = DataLoader(val, **kwargs)
    model = PSMAFYOLO(config["nc"], fusion_mode=args.fusion_mode, use_psg=not args.no_psg, use_msaf=not args.no_msaf).to(device)
    checkpoint = args.weights or (output / "last.pt" if args.resume else None)
    start = 0
    if checkpoint and Path(checkpoint).is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(state.get("model", state)); start = state.get("epoch", -1) + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    for epoch in range(start, args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True); context = torch.autocast(device.type) if scaler.is_enabled() else nullcontext()
            with context: loss = detection_loss(model(batch["rgb"].to(device), batch["ir"].to(device)), batch["labels"].to(device), config["nc"])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        state = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}; torch.save(state, output / "last.pt")
        if args.save_period > 0 and (epoch + 1) % args.save_period == 0: torch.save(state, output / f"epoch{epoch + 1}.pt")
        save_metrics(evaluate(model, val_loader, device), output, "metrics")


if __name__ == "__main__": main()
