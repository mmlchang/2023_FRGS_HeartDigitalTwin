import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from monai.losses import DiceLoss
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.unet2d import UNet2D
from utils.preprocess import Nifti2DSliceDataset, build_train_transforms, build_val_transforms


def set_seed(seed: int) -> None:
    """Set deterministic seeds to improve reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multiclass_dice(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    """Compute mean Dice score over classes from model logits and integer labels."""
    pred = torch.argmax(logits, dim=1)
    dice_scores = []

    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (pred_class * target_class).sum()
        denominator = pred_class.sum() + target_class.sum()

        score = (2.0 * intersection + eps) / (denominator + eps)
        dice_scores.append(score)

    return torch.stack(dice_scores).mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Run one training epoch and return aggregated loss and Dice."""
    model.train()

    running_loss = 0.0
    running_dice = 0.0

    for images, masks in loader:
        images = images.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.long).squeeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss_ce = ce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks.unsqueeze(1))
        loss = loss_ce + loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += multiclass_dice(logits, masks, num_classes)

    num_batches = max(1, len(loader))
    return {
        "loss": running_loss / num_batches,
        "dice": running_dice / num_batches,
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Run one validation epoch and return aggregated loss and Dice."""
    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long).squeeze(1)

            logits = model(images)
            loss_ce = ce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks.unsqueeze(1))
            loss = loss_ce + loss_dice

            running_loss += loss.item()
            running_dice += multiclass_dice(logits, masks, num_classes)

    num_batches = max(1, len(loader))
    return {
        "loss": running_loss / num_batches,
        "dice": running_dice / num_batches,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_dice: float) -> None:
    """Save full training state so training can be resumed later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D UNet (Mamba encoder) on NIfTI slices and export .pth")

    # Dataset arguments
    parser.add_argument("--train-image-dir", type=str, required=True, help="Path to train image .nii.gz files")
    parser.add_argument("--train-mask-dir", type=str, required=True, help="Path to train mask .nii.gz files")
    parser.add_argument("--val-image-dir", type=str, required=True, help="Path to val image .nii.gz files")
    parser.add_argument("--val-mask-dir", type=str, required=True, help="Path to val mask .nii.gz files")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model and output arguments
    parser.add_argument("--num-classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize size for input slices")
    parser.add_argument("--slice-axis", type=int, default=2, help="Axis used to slice 3D volumes")
    parser.add_argument(
        "--drop-empty-mask-slices",
        action="store_true",
        help="Skip slices whose mask is completely zero",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument(
        "--best-name",
        type=str,
        default="unet.pth",
        help="Filename for best model state_dict",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume training",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = build_train_transforms(spatial_size=(args.image_size, args.image_size))
    val_transform = build_val_transforms(spatial_size=(args.image_size, args.image_size))

    train_dataset = Nifti2DSliceDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        transform=train_transform,
        slice_axis=args.slice_axis,
        drop_empty_mask_slices=args.drop_empty_mask_slices,
    )
    val_dataset = Nifti2DSliceDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        transform=val_transform,
        slice_axis=args.slice_axis,
        drop_empty_mask_slices=False,
    )

    print(f"Train slices: {len(train_dataset)}")
    print(f"Val slices: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet2D(in_channels=1, num_classes=args.num_classes, pretrained_encoder=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / args.best_name
    last_ckpt_path = output_dir / "last_checkpoint.pth"

    start_epoch = 1
    best_val_dice = -1.0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
            print(f"Resumed from checkpoint: {resume_path}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, device, args.num_classes)
        val_metrics = validate_one_epoch(model, val_loader, ce_loss, dice_loss, device, args.num_classes)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val_dice)

        # Save the best model as pure state_dict (.pth) for clean inference loading.
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path} (val_dice={best_val_dice:.4f})")

    print(f"Training complete. Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model (.pth): {best_model_path}")


if __name__ == "__main__":
    main()
