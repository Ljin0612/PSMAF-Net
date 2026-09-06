"""A compact repository-local PSMAF-YOLO detector.

The implementation keeps the standard YOLO separation of backbone pyramid,
PAN/FPN neck, and anchor-free detection head while exposing paired inputs.
"""

import torch
from torch import nn
from torch.nn import functional as F

from core.psmaf import PSMAFFusion


class Conv(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(nn.Conv2d(c1, c2, k, s, k // 2, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))


class YOLOBackbone(nn.Module):
    """Small YOLO-style backbone returning stride 8, 16 and 32 features."""
    def __init__(self, in_channels=3, channels=(64, 128, 256)):
        super().__init__()
        c3, c4, c5 = channels
        self.stem = nn.Sequential(Conv(in_channels, c3 // 2, 3, 2), Conv(c3 // 2, c3, 3, 2), Conv(c3, c3, 3, 2))
        self.p4 = nn.Sequential(Conv(c3, c4, 3, 2), Conv(c4, c4))
        self.p5 = nn.Sequential(Conv(c4, c5, 3, 2), Conv(c5, c5))

    def forward(self, x):
        p3 = self.stem(x)
        p4 = self.p4(p3)
        return p3, p4, self.p5(p4)


class YOLONeckHead(nn.Module):
    """Top-down YOLO feature neck and three anchor-free prediction heads."""
    def __init__(self, channels=(64, 128, 256), nc=6):
        super().__init__()
        c3, c4, c5 = channels
        self.reduce5, self.fuse4 = Conv(c5, c4, 1), Conv(2 * c4, c4)
        self.reduce4, self.fuse3 = Conv(c4, c3, 1), Conv(2 * c3, c3)
        self.heads = nn.ModuleList([nn.Conv2d(c, nc + 5, 1) for c in channels])

    def forward(self, features):
        p3, p4, p5 = features
        n4 = self.fuse4(torch.cat((p4, F.interpolate(self.reduce5(p5), size=p4.shape[-2:], mode="nearest")), 1))
        n3 = self.fuse3(torch.cat((p3, F.interpolate(self.reduce4(n4), size=p3.shape[-2:], mode="nearest")), 1))
        return tuple(head(feature) for head, feature in zip(self.heads, (n3, n4, p5)))


class PSMAFYOLO(nn.Module):
    """Dual-backbone YOLO detector with PSMAF immediately before its neck."""
    def __init__(self, nc=6, channels=(64, 128, 256), fusion_mode="psmaf", use_psg=True, use_msaf=True):
        super().__init__()
        self.nc = nc
        self.rgb_backbone = YOLOBackbone(3, channels)
        self.ir_backbone = YOLOBackbone(3, channels)
        self.fusion = PSMAFFusion(channels, use_psg, use_msaf, fusion_mode)
        self.neck_head = YOLONeckHead(channels, nc)

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError("paired RGB and IR input tensors must have identical shapes")
        return self.neck_head(self.fusion(self.rgb_backbone(rgb), self.ir_backbone(ir)))


def detection_loss(outputs, targets, nc=6, return_components=False):
    """Lightweight anchor-free training loss for normalized YOLO targets."""
    total = outputs[0].new_zeros(())
    total_obj = outputs[0].new_zeros(())
    total_box = outputs[0].new_zeros(())
    total_cls = outputs[0].new_zeros(())
    for level, prediction in enumerate(outputs):
        b, _, h, w = prediction.shape
        obj_target = prediction.new_zeros((b, h, w))
        selected = targets[(targets[:, 4] * targets[:, 5] * (4 ** level) < 0.08) &
                           (targets[:, 4] * targets[:, 5] * (4 ** level) >= 0.005)] if targets.numel() else targets
        # Always assign very small objects to P3 and very large objects to P5.
        if targets.numel():
            area = targets[:, 4] * targets[:, 5]
            mask = area < .02 if level == 0 else (area >= .02) & (area < .12) if level == 1 else area >= .12
            selected = targets[mask]
        box_loss = prediction.new_zeros(())
        cls_loss = prediction.new_zeros(())
        for row in selected:
            bi, cls = int(row[0]), int(row[1])
            gx, gy = min(int(row[2] * w), w - 1), min(int(row[3] * h), h - 1)
            obj_target[bi, gy, gx] = 1
            cell = prediction[bi, :, gy, gx]
            box_loss = box_loss + F.smooth_l1_loss(cell[:4].sigmoid(), row[2:6])
            cls_loss = cls_loss + F.cross_entropy(cell[5:].unsqueeze(0), torch.tensor([cls], device=cell.device))
        obj_loss = F.binary_cross_entropy_with_logits(prediction[:, 4], obj_target)
        normalizer = max(len(selected), 1)
        box_loss, cls_loss = box_loss / normalizer, cls_loss / normalizer
        total_obj, total_box, total_cls = total_obj + obj_loss, total_box + box_loss, total_cls + cls_loss
        total = total + obj_loss + box_loss + cls_loss
    if return_components:
        return {"loss": total, "obj_loss": total_obj, "box_loss": total_box, "cls_loss": total_cls}
    return total
