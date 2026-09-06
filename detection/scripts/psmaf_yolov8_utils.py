"""Loss, decoding, evaluation, and checkpoint helpers for PSMAF-YOLOv8."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from detection.scripts.psmaf_yolo_utils import (NAMES, _class_ap, evaluate_detection,
                                                 non_max_suppression,
                                                 xywh_to_xyxy)


METRIC_KEYS = ("precision", "recall", "AP50", "mAP50", "mAP50_95", "per_class_ap")


def resolve_resume_path(resume, output_dir):
    """Resolve ``--resume``, including its Ultralytics-style ``auto`` form."""
    if not resume:
        return None
    path = Path(output_dir) / "last.pt" if str(resume) == "auto" else Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def _distribution(head, reg_max=16):
    logits = head[:, :4 * reg_max].view(head.shape[0], 4, reg_max, *head.shape[-2:])
    bins = torch.arange(reg_max, device=head.device, dtype=head.dtype).view(1, 1, reg_max, 1, 1)
    return (logits.softmax(2) * bins).sum(2)


def decode_yolov8_outputs(outputs, image_size, strides=(8, 16, 32), reg_max=16):
    """Decode all YOLOv8 DFL heads to ``xyxy, confidence, class`` rows."""
    decoded = [[] for _ in range(outputs[0].shape[0])]
    image_h, image_w = image_size
    for head, stride in zip(outputs, strides):
        distances = _distribution(head, reg_max)
        _, _, h, w = distances.shape
        gy, gx = torch.meshgrid(torch.arange(h, device=head.device), torch.arange(w, device=head.device), indexing="ij")
        cx, cy = (gx + .5) * stride, (gy + .5) * stride
        boxes = torch.stack((cx - distances[:, 0] * stride, cy - distances[:, 1] * stride,
                             cx + distances[:, 2] * stride, cy + distances[:, 3] * stride), -1)
        boxes[..., 0::2].clamp_(0, image_w); boxes[..., 1::2].clamp_(0, image_h)
        confidence, classes = head[:, 4 * reg_max:].sigmoid().max(1)
        for i in range(head.shape[0]):
            decoded[i].append(torch.cat((boxes[i].reshape(-1, 4), confidence[i].reshape(-1, 1),
                                         classes[i].reshape(-1, 1).to(head.dtype)), 1))
    return [torch.cat(parts) for parts in decoded]


def yolov8_detection_loss(outputs, targets, nc=6, strides=(8, 16, 32), reg_max=16):
    """Anchor-free YOLOv8-style DFL/box/class loss with centre assignment."""
    cls_loss = outputs[0].new_zeros(()); box_loss = outputs[0].new_zeros(())
    cls_elements = 0; positives = 0
    for head, stride in zip(outputs, strides):
        b, _, h, w = head.shape
        cls_target = head.new_zeros((b, nc, h, w))
        for row in targets:
            bi, cls = int(row[0]), int(row[1]); x, y, bw, bh = row[2:]
            gx = min(int(x * w), w - 1); gy = min(int(y * h), h - 1)
            # Assign by object scale to the most suitable pyramid level.
            preferred = 0 if bw * bh < .02 else 1 if bw * bh < .12 else 2
            if stride != strides[preferred]: continue
            cls_target[bi, cls, gy, gx] = 1; positives += 1
            center_x, center_y = gx + .5, gy + .5
            target = torch.stack((center_x - (x * w - bw * w / 2), center_y - (y * h - bh * h / 2),
                                  x * w + bw * w / 2 - center_x, y * h + bh * h / 2 - center_y)).clamp(0, reg_max - 1.01)
            logits = head[bi, :4 * reg_max, gy, gx].view(4, reg_max)
            lower = target.floor().long(); upper = lower + 1
            box_loss = box_loss + ((upper - target) * F.cross_entropy(logits, lower, reduction="none") +
                                   (target - lower) * F.cross_entropy(logits, upper, reduction="none")).mean()
        class_logits = head[:, 4 * reg_max:]
        cls_loss = cls_loss + F.binary_cross_entropy_with_logits(class_logits, cls_target, reduction="sum")
        cls_elements += class_logits.numel()
    # Classification supervises every class at every cell, so its denominator
    # must include negative cells too.  Normalizing this sum by positives made
    # the loss grow with image resolution and overwhelmed box optimization.
    cls_loss = cls_loss / max(cls_elements, 1)
    box_loss = box_loss / max(positives, 1)
    obj_loss = outputs[0].new_zeros(())  # YOLOv8 folds object confidence into class logits.
    total = 7.5 * box_loss + .5 * cls_loss
    num_pos = outputs[0].new_tensor(positives)
    return {"loss": total, "obj_loss": obj_loss, "box_loss": box_loss,
            "cls_loss": cls_loss, "num_pos": num_pos}


@torch.no_grad()
def evaluate_yolov8(model, loader, device, confidence_threshold=.25, nms_iou_threshold=.45,
                    diagnostics_path=None):
    model.eval(); predictions, targets = [], []
    decoded_count = filtered_count = nms_count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device); labels = batch["labels"].to(device)
        decoded = decode_yolov8_outputs(model(rgb, batch["ir"].to(device)), rgb.shape[-2:])
        decoded_count += sum(len(x) for x in decoded)
        filtered_count += sum(int((x[:, 4] >= confidence_threshold).sum()) for x in decoded)
        selected = non_max_suppression(decoded, confidence_threshold, nms_iou_threshold)
        nms_count += sum(len(x) for x in selected); predictions.extend(selected)
        scale = labels.new_tensor([rgb.shape[-1], rgb.shape[-2], rgb.shape[-1], rgb.shape[-2]])
        for i in range(len(rgb)):
            rows = labels[labels[:, 0] == i]
            targets.append({"boxes": xywh_to_xyxy(rows[:, 2:6] * scale), "classes": rows[:, 1].long()})
    metrics = evaluate_detection(predictions, targets, NAMES)
    metrics = {key: metrics[key] for key in METRIC_KEYS}
    if diagnostics_path:
        class_results = [_class_ap(predictions, targets, class_id, .5)
                         for class_id in range(len(NAMES))]
        tp50 = [result[1] for result in class_results]
        fp50 = [result[2] for result in class_results]
        gt_counts = [result[3] for result in class_results]
        fn50 = [gt - tp for gt, tp in zip(gt_counts, tp50)]
        prediction_counts = [sum(int((rows[:, 5].long() == class_id).sum()) for rows in predictions)
                             for class_id in range(len(NAMES))]
        path = Path(diagnostics_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"decoded_boxes": decoded_count, "boxes_after_confidence": filtered_count,
                                    "boxes_after_nms": nms_count, "images": len(targets),
                                    "gt_boxes": sum(gt_counts), "tp50": sum(tp50),
                                    "fp50": sum(fp50), "fn50": sum(fn50),
                                    "per_class_gt_counts": dict(zip(NAMES, gt_counts)),
                                    "per_class_prediction_counts": dict(zip(NAMES, prediction_counts)),
                                    "per_class_tp50": dict(zip(NAMES, tp50)),
                                    "per_class_fp50": dict(zip(NAMES, fp50)),
                                    "per_class_fn50": dict(zip(NAMES, fn50))}, indent=2))
    return metrics
