import os
from dataclasses import dataclass
from typing import List, Tuple

import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from torch.utils.data import Dataset


def build_train_transforms(spatial_size: Tuple[int, int] = (256, 256)) -> Compose:
    """Build training transforms with geometric and intensity augmentation."""
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            Resized(keys=["mask"], spatial_size=spatial_size, mode="nearest"),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
            RandGaussianNoised(keys=["image"], mean=0.0, std=0.01, prob=0.3),
            ToTensord(keys=["image", "mask"]),
        ]
    )


def build_val_transforms(spatial_size: Tuple[int, int] = (256, 256)) -> Compose:
    """Build validation transforms without stochastic augmentation."""
    return Compose(
        [
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=spatial_size, mode="area"),
            Resized(keys=["mask"], spatial_size=spatial_size, mode="nearest"),
            ToTensord(keys=["image", "mask"]),
        ]
    )


@dataclass(frozen=True)
class SliceIndex:
    image_file: str
    mask_file: str
    slice_idx: int


class Nifti2DSliceDataset(Dataset):
    """
    Build a 2D-slice dataset from paired image/mask NIfTI volumes.

    Naming rule:
    - image: <case>.nii.gz
    - mask:  <case>_gt.nii.gz
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Compose = None,
        slice_axis: int = 2,
        drop_empty_mask_slices: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.slice_axis = slice_axis
        self.drop_empty_mask_slices = drop_empty_mask_slices

        self.pairs = self._build_pairs()
        self.slice_map = self._build_slice_map()

        if not self.slice_map:
            raise RuntimeError(
                "No training samples were built. Check data paths, naming convention, and slice axis."
            )

    def _build_pairs(self) -> List[Tuple[str, str]]:
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_files = sorted(
            [file_name for file_name in os.listdir(self.image_dir) if file_name.endswith(".nii.gz")]
        )
        pairs = []

        for image_name in image_files:
            base_name = image_name.replace(".nii.gz", "")
            mask_name = f"{base_name}_gt.nii.gz"
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                pairs.append((image_name, mask_name))

        if not pairs:
            raise RuntimeError(
                "No matched image/mask pairs found. Expected mask naming pattern: <image>_gt.nii.gz"
            )

        return pairs

    def _build_slice_map(self) -> List[SliceIndex]:
        slice_map: List[SliceIndex] = []

        for image_file, mask_file in self.pairs:
            image_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            image_img = nib.load(image_path)
            mask_img = nib.load(mask_path)

            if image_img.shape != mask_img.shape:
                raise ValueError(
                    f"Shape mismatch for pair ({image_file}, {mask_file}): "
                    f"{image_img.shape} vs {mask_img.shape}"
                )

            num_slices = image_img.shape[self.slice_axis]

            for slice_idx in range(num_slices):
                if self.drop_empty_mask_slices:
                    mask_2d = np.take(np.asanyarray(mask_img.dataobj), slice_idx, axis=self.slice_axis)
                    if np.max(mask_2d) == 0:
                        continue

                slice_map.append(SliceIndex(image_file=image_file, mask_file=mask_file, slice_idx=slice_idx))

        return slice_map

    def __len__(self) -> int:
        return len(self.slice_map)

    def __getitem__(self, idx: int):
        sample_idx = self.slice_map[idx]
        image_path = os.path.join(self.image_dir, sample_idx.image_file)
        mask_path = os.path.join(self.mask_dir, sample_idx.mask_file)

        image_img = nib.load(image_path)
        mask_img = nib.load(mask_path)

        image = np.asanyarray(image_img.dataobj, dtype=np.float32)
        mask = np.asanyarray(mask_img.dataobj, dtype=np.int64)

        image_2d = np.take(image, sample_idx.slice_idx, axis=self.slice_axis)
        mask_2d = np.take(mask, sample_idx.slice_idx, axis=self.slice_axis)

        # Add channel axis: [H, W] -> [1, H, W]
        image_2d = np.expand_dims(image_2d, axis=0)
        mask_2d = np.expand_dims(mask_2d, axis=0)

        sample = {"image": image_2d, "mask": mask_2d}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample["image"], sample["mask"]