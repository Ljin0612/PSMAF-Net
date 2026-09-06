from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
from PIL import Image

from detection.datasets import M3FDPairedDataset


def test_dataset_loads_pair(tmp_path):
    for folder in ("vi", "ir", "labels", "meta"): (tmp_path / folder).mkdir()
    Image.new("RGB", (20, 10), "red").save(tmp_path / "vi" / "one.png")
    Image.new("L", (20, 10), 100).save(tmp_path / "ir" / "one.png")
    (tmp_path / "labels" / "one.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    (tmp_path / "meta" / "train.txt").write_text("one\n")
    sample = M3FDPairedDataset(tmp_path, "train", 32)[0]
    assert sample["rgb"].shape == sample["ir"].shape == (3, 32, 32)
    torch.testing.assert_close(sample["labels"], torch.tensor([[1., .5, .5, .2, .4]]))


def test_real_dataset_when_provided():
    root = Path("/home/jinlei/database/M3FD_Detection")
    if not root.exists(): pytest.skip("M3FD path is not available")
    assert M3FDPairedDataset(root, "train")[0]["rgb"].shape[0] == 3
