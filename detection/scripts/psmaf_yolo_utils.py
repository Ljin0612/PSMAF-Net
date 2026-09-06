"""Shared training and standards-based detection evaluation helpers."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


NAMES = ["people", "car", "bus", "motorcycle", "lamp", "truck"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_metrics(metrics, output_dir, stem="metrics"):
    """Write the same real detection metrics to JSON and a one-row CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{stem}.json").write_text(json.dumps(metrics, indent=2))
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    flat.update({f"AP/{k}": v for k, v in metrics.get("per_class_ap", {}).items()})
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)


def xywh_to_xyxy(boxes):
    """Convert centre-x/centre-y/width/height boxes to corner boxes."""
    result = boxes.clone()
    result[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    result[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    result[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    result[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return result


def box_iou(boxes1, boxes2):
    """Return the pairwise IoU matrix for two sets of xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((len(boxes1), len(boxes2)))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp(min=1e-9)


def decode_outputs(outputs, image_size):
    """Decode every P3/P4/P5 head into image-space ``xyxy, score, class`` rows.

    The training loss regresses sigmoid-normalized absolute ``xywh`` values, so
    decoding deliberately uses that same representation rather than a grid or
    anchor transform. Scores are objectness multiplied by the best class
    probability.
    """
    image_h, image_w = image_size
    decoded = [[] for _ in range(outputs[0].shape[0])]
    scale = outputs[0].new_tensor([image_w, image_h, image_w, image_h])
    for head in outputs:
        values = head.permute(0, 2, 3, 1).reshape(head.shape[0], -1, head.shape[1])
        boxes = xywh_to_xyxy(values[..., :4].sigmoid() * scale)
        boxes[..., 0::2].clamp_(0, image_w)
        boxes[..., 1::2].clamp_(0, image_h)
        class_confidence, classes = values[..., 5:].softmax(-1).max(-1)
        scores = values[..., 4].sigmoid() * class_confidence
        for batch_index in range(head.shape[0]):
            decoded[batch_index].append(torch.cat((boxes[batch_index], scores[batch_index, :, None],
                                                   classes[batch_index, :, None].to(boxes.dtype)), 1))
    return [torch.cat(parts, 0) for parts in decoded]


def _nms(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[current].unsqueeze(0), boxes[remaining])[0] <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def non_max_suppression(predictions, confidence_threshold=0.25, iou_threshold=0.45, max_detections=300):
    """Apply confidence filtering and class-aware NMS to decoded predictions."""
    results = []
    for prediction in predictions:
        prediction = prediction[prediction[:, 4] >= confidence_threshold]
        kept = []
        for class_id in prediction[:, 5].unique():
            indices = torch.where(prediction[:, 5] == class_id)[0]
            kept.append(indices[_nms(prediction[indices, :4], prediction[indices, 4], iou_threshold)])
        if kept:
            keep = torch.cat(kept)
            keep = keep[prediction[keep, 4].argsort(descending=True)[:max_detections]]
            prediction = prediction[keep]
        results.append(prediction)
    return results


def compute_ap(recall, precision):
    """Compute interpolated area under a ranked precision-recall curve."""
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _class_ap(predictions, targets, class_id, threshold):
    gt = {i: target["boxes"][target["classes"] == class_id] for i, target in enumerate(targets)}
    total_gt = sum(len(boxes) for boxes in gt.values())
    ranked = []
    for image_id, prediction in enumerate(predictions):
        for row in prediction[prediction[:, 5].long() == class_id]:
            ranked.append((float(row[4]), image_id, row[:4]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched = {i: set() for i in gt}
    tp, fp = [], []
    for _, image_id, box in ranked:
        boxes = gt[image_id]
        ious = box_iou(box.unsqueeze(0), boxes)[0]
        if len(ious):
            best = int(ious.argmax())
            is_tp = float(ious[best]) >= threshold and best not in matched[image_id]
        else:
            is_tp = False
        tp.append(float(is_tp)); fp.append(float(not is_tp))
        if is_tp:
            matched[image_id].add(best)
    tp_cumulative = np.cumsum(tp); fp_cumulative = np.cumsum(fp)
    recall = tp_cumulative / max(total_gt, 1)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1)
    return (compute_ap(recall, precision) if total_gt else 0.0,
            int(tp_cumulative[-1]) if len(tp) else 0, int(fp_cumulative[-1]) if len(fp) else 0, total_gt)


def compute_map(predictions, targets, class_names=NAMES, iou_thresholds=None):
    """Compute AP from globally ranked predictions and one-to-one IoU matching."""
    thresholds = tuple(np.arange(0.50, 0.96, 0.05) if iou_thresholds is None else iou_thresholds)
    present = [c for c in range(len(class_names)) if any((t["classes"] == c).any() for t in targets)]
    per_threshold = {threshold: [] for threshold in thresholds}
    per_class_ap = {}
    tp50 = fp50 = gt_count = 0
    for class_id, name in enumerate(class_names):
        aps = []
        for threshold in thresholds:
            ap, tp, fp, count = _class_ap(predictions, targets, class_id, threshold)
            aps.append(ap)
            if abs(threshold - 0.5) < 1e-9:
                tp50 += tp; fp50 += fp; gt_count += count
            if class_id in present:
                per_threshold[threshold].append(ap)
        # Per-class AP follows the COCO convention and averages IoU=.50:.95.
        per_class_ap[name] = float(np.mean(aps))
    maps = [float(np.mean(per_threshold[t])) if per_threshold[t] else 0.0 for t in thresholds]
    precision = tp50 / max(tp50 + fp50, 1)
    recall = tp50 / max(gt_count, 1)
    return {"precision": precision, "recall": recall, "AP50": maps[0], "mAP50": maps[0],
            "mAP50_95": float(np.mean(maps)), "per_class_ap": per_class_ap}


def evaluate_detection(predictions, targets, class_names=NAMES, iou_thresholds=None):
    """Public entry point for evaluating decoded/NMS detection lists."""
    return compute_map(predictions, targets, class_names, iou_thresholds)


@torch.no_grad()
def evaluate(model, loader, device, confidence_threshold=0.25, nms_iou_threshold=0.45):
    """Run multi-head inference and standards-based detection evaluation."""
    model.eval()
    all_predictions, all_targets = [], []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        outputs = model(rgb, batch["ir"].to(device))
        predictions = non_max_suppression(decode_outputs(outputs, rgb.shape[-2:]), confidence_threshold,
                                          nms_iou_threshold)
        labels = batch["labels"].to(device)
        scale = labels.new_tensor([rgb.shape[-1], rgb.shape[-2], rgb.shape[-1], rgb.shape[-2]])
        for batch_index in range(rgb.shape[0]):
            rows = labels[labels[:, 0] == batch_index]
            boxes = xywh_to_xyxy(rows[:, 2:6] * scale) if len(rows) else labels.new_empty((0, 4))
            all_targets.append({"boxes": boxes, "classes": rows[:, 1].long()})
        all_predictions.extend(predictions)
    return evaluate_detection(all_predictions, all_targets, NAMES)
