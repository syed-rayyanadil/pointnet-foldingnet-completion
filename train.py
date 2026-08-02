"""
Training script for Point Cloud Completion.

Usage:
    python train.py --category chair --epochs 5 --batch_size 32 --lr 1e-4
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from data.dataset import ModelNetDataset
from src.models import PointCloudCompletion, chamfer_loss


def get_device(requested: str = "auto"):
    """Select the best available device, or use the one explicitly requested."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS can fail in sandboxed environments; default to cpu for safety
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Point Cloud Completion model")
    parser.add_argument(
        "--category", type=str, default="chair", help="ModelNet40 category (default: chair)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pointnet",
        choices=["pointnet", "pointnet++"],
        help="Encoder architecture (default: pointnet)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    return parser.parse_args()


def train(args):
    device = get_device(args.device)
    print(f"Using device: {device}")

    dataset = ModelNetDataset(
        root_dir="data/ModelNet40", category=args.category
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"Dataset: {args.category} - {len(dataset)} samples ({len(loader)} batches/epoch)")

    model = PointCloudCompletion(encoder_type=args.encoder).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.encoder} encoder - {total_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_loss = float("inf")

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (partial, complete) in enumerate(loader):
            partial = partial.to(device)
            complete = complete.to(device)

            optimizer.zero_grad()
            predicted = model(partial)
            loss = chamfer_loss(predicted, complete)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(
                    f"  Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{args.epochs} complete | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model checkpoint to {best_path}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved periodic checkpoint to {ckpt_path}")

        print()

    print(f"Training finished. Best Chamfer Loss: {best_loss:.6f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
