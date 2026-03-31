from monai.transforms import (
    Compose,
    ScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Resized
)

# Training transforms with resizing, augmentation, and intensity variations
train_transforms = Compose([
    ScaleIntensityd(keys=["image"]),

    # Resize image and mask
    Resized(keys=["image"], spatial_size=(256, 256), mode="area"),
    Resized(keys=["mask"], spatial_size=(256, 256), mode="nearest"),

    # Geometric augmentations
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),

    # Intensity augmentations for grayscale images
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
    RandGaussianNoised(keys=["image"], mean=0.0, std=0.01, prob=0.3),

    # Convert to PyTorch tensors
    ToTensord(keys=["image", "mask"])
])

# Validation transforms (resize but no augmentation)
val_transforms = Compose([
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(256, 256), mode="area"),
    Resized(keys=["mask"], spatial_size=(256, 256), mode="nearest"),
    ToTensord(keys=["image", "mask"])
])

####################################################################################

class Nifti2DSliceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, slice_axis=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.slice_axis = slice_axis

        # ✅ Explicitly match files by base name
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
        self.pairs = []
        for f in self.image_files:
            base = f.replace(".nii.gz", "")
            mask_f = f"{base}_gt.nii.gz"
            mask_path = os.path.join(mask_dir, mask_f)
            if os.path.exists(mask_path):
                self.pairs.append((f, mask_f))
            else:
                print(f"⚠️ Missing mask for {f}, skipping it.")

        # ✅ Precompute slice map correctly
        self.slice_map = []
        for img_file, mask_file in self.pairs:
            img = nib.load(os.path.join(image_dir, img_file)).get_fdata()
            num_slices = img.shape[self.slice_axis]
            for s in range(num_slices):
                self.slice_map.append((img_file, mask_file, s))

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, idx):
        img_file, mask_file, slice_idx = self.slice_map[idx]
        image = nib.load(os.path.join(self.image_dir, img_file)).get_fdata().astype(np.float32)
        mask = nib.load(os.path.join(self.mask_dir, mask_file)).get_fdata().astype(np.int64)

        image_2d = np.take(image, slice_idx, axis=self.slice_axis)
        mask_2d = np.take(mask, slice_idx, axis=self.slice_axis)

        image_2d = np.expand_dims(image_2d, axis=0)
        mask_2d = np.expand_dims(mask_2d, axis=0)

        sample = {"image": image_2d, "mask": mask_2d}
        if self.transform:
            sample = self.transform(sample)
        return sample["image"], sample["mask"]
        
        
test_imagedir = '.../images' # dir to .nii files
test_masksdir = '.../masks' # dir to .nii files

test_dataset = Nifti2DSliceDataset(
    image_dir=test_imagedir,
    mask_dir=test_masksdir,
    transform=val_transforms,   # just use validation transforms
    slice_axis=2
)

# Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
print(f"Test dataset size: {len(test_dataset)} slices")