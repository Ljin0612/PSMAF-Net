#!/usr/bin/env python3
"""Evaluate a PSMAF-YOLO checkpoint on an M3FD split."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch, yaml
from torch.utils.data import DataLoader
from detection.datasets import M3FDPairedDataset, paired_collate_fn
from detection.models.psmaf_yolo import PSMAFYOLO
from detection.scripts.psmaf_yolo_utils import evaluate, limit_dataset, save_metrics


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root"); p.add_argument("--data", default="detection/configs/m3fd_psmaf_yolo.yaml")
    p.add_argument("--weights", required=True); p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--batch", type=int, default=8); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--project", default="runs/detect"); p.add_argument("--name", default="psmaf-yolo-eval")
    p.add_argument("--fusion-mode", choices=("psmaf", "add", "concat"), default="psmaf"); p.add_argument("--no-psg", action="store_true"); p.add_argument("--no-msaf", action="store_true")
    p.add_argument("--conf-thres", type=float, default=0.25); p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--debug-num-images", type=int, default=0,
                   help="use only the first N samples for debugging; not for official reporting")
    p.add_argument("--epochs", type=int, default=0, help=argparse.SUPPRESS); p.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS); p.add_argument("--resume", action="store_true", help=argparse.SUPPRESS); p.add_argument("--save-period", type=int, default=-1, help=argparse.SUPPRESS)
    return p


def main(args=None):
    args = parser().parse_args(args); cfg = yaml.safe_load(Path(args.data).read_text()); root = args.dataset_root or cfg["path"]; device = torch.device(args.device)
    model = PSMAFYOLO(cfg["nc"], fusion_mode=args.fusion_mode, use_psg=not args.no_psg, use_msaf=not args.no_msaf).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False); model.load_state_dict(state.get("model", state))
    dataset = limit_dataset(M3FDPairedDataset(root, cfg[args.split], args.imgsz), args.debug_num_images)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, collate_fn=paired_collate_fn)
    output = Path(args.project) / args.name
    metrics = evaluate(model, loader, device, args.conf_thres, args.nms_iou, output / "eval_diagnostics.json")
    save_metrics(metrics, output, "metrics"); print(metrics)


if __name__ == "__main__": main()
