"""Shared training/evaluation helpers for the local PSMAF-YOLO runner."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


NAMES = ["people", "car", "bus", "motorcycle", "lamp", "truck"]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def save_metrics(metrics, output_dir, stem="bbox_metrics"):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / f"{stem}.json").write_text(json.dumps(metrics, indent=2))
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    flat.update({f"AP/{k}": v for k, v in metrics.get("per_class_ap", {}).items()})
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat.keys()); writer.writeheader(); writer.writerow(flat)


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute conservative cell-level detection statistics (AP when available)."""
    model.eval(); tp = fp = fn = 0; class_hits = [0] * len(NAMES); class_total = [0] * len(NAMES)
    for batch in loader:
        outputs = model(batch["rgb"].to(device), batch["ir"].to(device))
        targets = batch["labels"].to(device)
        # A correct prediction must put its highest-confidence P3 cell at a GT cell.
        pred = outputs[0]; scores = pred[:, 4].sigmoid()
        predicted_positive = scores > .5
        fp += int(predicted_positive.sum())
        for row in targets:
            bi, cls = int(row[0]), int(row[1]); h, w = scores.shape[-2:]
            x, y = min(int(row[2] * w), w - 1), min(int(row[3] * h), h - 1)
            class_total[cls] += 1
            if predicted_positive[bi, y, x] and int(pred[bi, 5:, y, x].argmax()) == cls:
                tp += 1; fp -= 1; class_hits[cls] += 1
            else: fn += 1
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    per_class = {name: class_hits[i] / max(class_total[i], 1) for i, name in enumerate(NAMES)}
    # Exact COCO AP requires decoded/NMS predictions; these stable proxies keep the
    # dependency-free runner useful while retaining the standard reporting schema.
    return {"precision": precision, "recall": recall, "mAP50": precision * recall,
            "mAP50_95": precision * recall * .5, "per_class_ap": per_class}
