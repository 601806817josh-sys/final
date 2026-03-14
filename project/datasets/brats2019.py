import itertools
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class BraTS2019(Dataset):
    """BraTS loader for preprocessed H5 volumes listed in train/val split files."""

    def __init__(self, base_dir, split="train", num=None, transform=None):
        self.base_dir = Path(base_dir)
        self.transform = transform

        split_file = self.base_dir / ("train.txt" if split == "train" else "val.txt")
        with split_file.open("r", encoding="utf-8") as f:
            image_list = [line.strip().split(",")[0] for line in f]

        self.image_list = image_list[:num] if num is not None else image_list
        print(f"Loaded {len(self.image_list)} samples from {split_file.name}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        with h5py.File(self.base_dir / "data" / f"{image_name}.h5", "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]
        sample = {"image": image, "label": label.astype(np.uint8)}
        return self.transform(sample) if self.transform else sample


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        ow, oh, od = self.output_size

        if label.shape[0] <= ow or label.shape[1] <= oh or label.shape[2] <= od:
            pw = max((ow - label.shape[0]) // 2 + 3, 0)
            ph = max((oh - label.shape[1]) // 2 + 3, 0)
            pd = max((od - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant")

        w, h, d = image.shape
        w1 = np.random.randint(0, w - ow)
        h1 = np.random.randint(0, h - oh)
        d1 = np.random.randint(0, d - od)

        return {
            "image": image[w1 : w1 + ow, h1 : h1 + oh, d1 : d1 + od],
            "label": label[w1 : w1 + ow, h1 : h1 + oh, d1 : d1 + od],
        }


class RandomRotFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        k = np.random.randint(0, 4)
        image, label = np.rot90(image, k), np.rot90(label, k)
        axis = np.random.randint(0, 2)
        return {"image": np.flip(image, axis=axis).copy(), "label": np.flip(label, axis=axis).copy()}


class ToTensor:
    def __call__(self, sample):
        image = sample["image"].reshape(1, *sample["image"].shape).astype(np.float32)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(sample["label"]).long(),
        }


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = np.random.permutation(self.primary_indices)
        secondary_iter = itertools.chain.from_iterable(
            np.random.permutation(self.secondary_indices) for _ in itertools.repeat(None)
        )
        return (
            list(primary_batch) + list(secondary_batch)
            for primary_batch, secondary_batch in zip(
                _grouper(primary_iter, self.primary_batch_size),
                _grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def _grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
