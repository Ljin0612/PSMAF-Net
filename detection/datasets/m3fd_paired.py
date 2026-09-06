"""Paired RGB/infrared M3FD dataset using native YOLO text annotations."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class M3FDPairedDataset(Dataset):
    """Load aligned ``vi``/``ir`` images and normalized YOLO labels."""

    def __init__(self, root, split="train", imgsz=640, transform=None):
        self.root = Path(root)
        split_file = Path(split)
        if not split_file.is_file():
            root_relative = self.root / split_file
            split_file = root_relative if root_relative.is_file() else self.root / "meta" / (split if str(split).endswith(".txt") else f"{split}.txt")
        if not split_file.is_file():
            raise FileNotFoundError(f"M3FD split file not found: {split_file}")
        self.ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def _path(self, folder, sample_id, extensions=(".png", ".jpg", ".jpeg", ".bmp")):
        raw = Path(sample_id)
        candidates = [self.root / folder / raw, self.root / folder / raw.name]
        if not raw.suffix:
            candidates = [path.with_suffix(ext) for path in candidates for ext in extensions]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"no {folder} file for sample {sample_id!r}")

    def __getitem__(self, index):
        sample_id = self.ids[index]
        rgb_path = self._path("vi", sample_id)
        ir_path = self._path("ir", sample_id)
        label_path = self.root / "labels" / Path(sample_id).with_suffix(".txt").name
        rgb = Image.open(rgb_path).convert("RGB")
        ir = Image.open(ir_path).convert("RGB")
        rgb = TF.to_tensor(TF.resize(rgb, self.imgsz, antialias=True))
        ir = TF.to_tensor(TF.resize(ir, self.imgsz, antialias=True))
        labels = []
        if label_path.is_file():
            for number, line in enumerate(label_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                values = [float(value) for value in line.split()]
                if len(values) != 5 or not 0 <= int(values[0]) < 6:
                    raise ValueError(f"invalid YOLO label at {label_path}:{number}")
                labels.append(values)
        target = torch.tensor(labels, dtype=torch.float32).reshape(-1, 5)
        sample = {"rgb": rgb, "ir": ir, "labels": target, "image_id": sample_id,
                  "rgb_path": str(rgb_path), "ir_path": str(ir_path)}
        return self.transform(sample) if self.transform else sample


def paired_collate_fn(batch):
    """Stack paired images and prefix labels with their batch index."""
    labels = []
    for index, sample in enumerate(batch):
        if sample["labels"].numel():
            labels.append(torch.cat((torch.full((len(sample["labels"]), 1), index), sample["labels"]), dim=1))
    return {
        "rgb": torch.stack([sample["rgb"] for sample in batch]),
        "ir": torch.stack([sample["ir"] for sample in batch]),
        "labels": torch.cat(labels) if labels else torch.empty((0, 6)),
        "image_id": [sample["image_id"] for sample in batch],
    }


def sanity_check(root, split="train", imgsz=640):
    """Load one pair and return its shapes for quick installation checks."""
    sample = M3FDPairedDataset(root, split, imgsz)[0]
    return {key: tuple(sample[key].shape) for key in ("rgb", "ir", "labels")}
