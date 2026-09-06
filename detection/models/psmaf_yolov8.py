"""YOLOv8-s PSMAF detector for aligned RGB/infrared image pairs.

The blocks and depth/width layout mirror the Ultralytics YOLOv8-s detection
model.  They live in this repository so inference and unit tests do not depend
on an Ultralytics installation; local Ultralytics ``yolov8s.pt`` checkpoints
can nevertheless be imported with :func:`load_yolov8s_weights`.
"""

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from core.psmaf import PSMAFFusion


def autopad(k, p=None):
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Ultralytics-compatible Conv-BN-SiLU block."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1, self.cv2 = Conv(c1, c2, 3), Conv(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """YOLOv8 CSP block, preserving Ultralytics parameter names."""
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        parts.extend(block(parts[-1]) for block in self.m)
        return self.cv2(torch.cat(parts, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1, self.cv2 = Conv(c1, c1 // 2, 1), Conv(2 * c1, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x); y1 = self.m(x); y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLOv8sBackbone(nn.Module):
    """Standard YOLOv8-s backbone returning stride 8/16/32 features."""
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList((
            Conv(3, 32, 3, 2), Conv(32, 64, 3, 2), C2f(64, 64, 1, True),
            Conv(64, 128, 3, 2), C2f(128, 128, 2, True),
            Conv(128, 256, 3, 2), C2f(256, 256, 2, True),
            Conv(256, 512, 3, 2), C2f(512, 512, 1, True), SPPF(512, 512, 5)))

    def forward(self, x):
        pyramid = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in (4, 6, 9): pyramid.append(x)
        return tuple(pyramid)


class Detect(nn.Module):
    """YOLOv8 decoupled distributional box/classification head."""
    def __init__(self, nc=6, ch=(128, 256, 512), reg_max=16):
        super().__init__(); self.nc, self.reg_max = nc, reg_max
        self.no = nc + reg_max * 4
        c2, c3 = max(64, ch[0] // 4, reg_max * 4), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for c in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(c, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for c in ch)

    def forward(self, features):
        return tuple(torch.cat((self.cv2[i](x), self.cv3[i](x)), 1) for i, x in enumerate(features))


class YOLOv8NeckHead(nn.Module):
    def __init__(self, nc=6):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f12 = C2f(768, 256, 1); self.c2f15 = C2f(384, 128, 1)
        self.down16 = Conv(128, 128, 3, 2); self.c2f18 = C2f(384, 256, 1)
        self.down19 = Conv(256, 256, 3, 2); self.c2f21 = C2f(768, 512, 1)
        self.detect = Detect(nc)

    def forward(self, features):
        p3, p4, p5 = features
        n4 = self.c2f12(torch.cat((self.up(p5), p4), 1))
        n3 = self.c2f15(torch.cat((self.up(n4), p3), 1))
        o4 = self.c2f18(torch.cat((self.down16(n3), n4), 1))
        o5 = self.c2f21(torch.cat((self.down19(o4), p5), 1))
        return self.detect((n3, o4, o5))


class PSMAFYOLOv8(nn.Module):
    """Dual YOLOv8-s backbones, PSMAF pyramid fusion, and YOLOv8 head."""
    strides = (8, 16, 32)
    channels = (128, 256, 512)

    def __init__(self, nc=6, fusion_mode="psmaf", use_psg=True, use_msaf=True):
        super().__init__(); self.nc = nc
        self.rgb_backbone = YOLOv8sBackbone(); self.ir_backbone = YOLOv8sBackbone()
        self.fusion = PSMAFFusion(self.channels, use_psg, use_msaf, fusion_mode)
        self.neck_head = YOLOv8NeckHead(nc)

    def forward_features(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError("paired RGB and IR input tensors must have identical shapes")
        return self.fusion(self.rgb_backbone(rgb), self.ir_backbone(ir))

    def forward(self, rgb, ir):
        return self.neck_head(self.forward_features(rgb, ir))


def _checkpoint_state(checkpoint):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    obj = obj.get("ema") or obj.get("model") or obj if isinstance(obj, Mapping) else obj
    if isinstance(obj, nn.Module): return obj.float().state_dict()
    if isinstance(obj, Mapping): return obj
    raise TypeError("checkpoint does not contain a model or state dictionary")


def load_yolov8s_weights(model, weights, verbose=True):
    """Partially import a *local* official YOLOv8-s checkpoint.

    Both modality backbones receive the same pretrained backbone parameters.
    Neck/head keys are translated from the official sequential model indices.
    """
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"YOLOv8-s weights not found: {path}. Download yolov8s.pt separately and provide its local path; automatic downloads are disabled.")
    source = _checkpoint_state(path); destination = model.state_dict(); candidates = {}
    for key, value in source.items():
        key = key.removeprefix("module.")
        if key.startswith("model."):
            bits = key.split("."); index = int(bits[1]); tail = ".".join(bits[2:])
            if index <= 9:
                candidates[f"rgb_backbone.model.{index}.{tail}"] = value
                candidates[f"ir_backbone.model.{index}.{tail}"] = value
            else:
                names = {12: "c2f12", 15: "c2f15", 16: "down16", 18: "c2f18", 19: "down19", 21: "c2f21", 22: "detect"}
                if index in names: candidates[f"neck_head.{names[index]}.{tail}"] = value
        else:
            candidates[key] = value
    loaded = {k: v for k, v in candidates.items() if k in destination and destination[k].shape == v.shape}
    skipped = sorted(k for k in candidates if k not in loaded)
    result = model.load_state_dict(loaded, strict=False)
    summary = {"loaded_keys": sorted(loaded), "skipped_keys": skipped,
               "missing_keys": sorted(result.missing_keys), "unexpected_keys": sorted(result.unexpected_keys)}
    if verbose:
        print("YOLOv8-s weight loading summary: " + ", ".join(f"{k}={len(v)}" for k, v in summary.items()))
    return summary
