"""
Evaluation script for Point Cloud Completion model.

Loads a trained model checkpoint and evaluates Chamfer Distance on the test split.
Outputs mean Chamfer Distance and standard deviation.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from data.dataset import ModelNetDataset
from src.models import PointCloudCompletion, chamfer_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Point Cloud Completion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ModelNet40",
        help="Path to ModelNet40 dataset root",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["chair"],
        help="Categories to evaluate (default: chair)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pointnet++",
        choices=["pointnet", "pointnet++"],
        help="Encoder architecture (default: pointnet++)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def evaluate(args):
    device = torch.device(args.device)
    print(f"Evaluation Device: {device}")

    # Load test dataset
    dataset = ModelNetDataset(
        root_dir=args.data_dir,
        category=args.categories,
        split="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    categories_str = ", ".join(args.categories)
    print(f"Test Dataset: [{categories_str}] - {len(dataset)} samples ({len(loader)} batches)")

    # Load model
    model = PointCloudCompletion(encoder_type=args.encoder).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    model.eval()

    cd_losses = []

    print("\nEvaluating on test split...")
    with torch.no_grad():
        for batch_idx, (partial, complete) in enumerate(loader):
            partial = partial.to(device)
            complete = complete.to(device)

            output = model(partial)

            if isinstance(output, dict):
                pred = output["predicted_cloud"]
            else:
                pred = output

            loss = chamfer_loss(pred, complete)
            cd_losses.append(loss.item())

    cd_losses = torch.tensor(cd_losses)
    mean_cd = cd_losses.mean().item()
    std_cd = cd_losses.std().item()

    print("\n" + "=" * 45)
    print(f" EVALUATION RESULTS [{categories_str.upper()}]")
    print("=" * 45)
    print(f" Test Samples       : {len(dataset)}")
    print(f" Mean Chamfer Loss  : {mean_cd:.6f}")
    print(f" Std Chamfer Loss   : {std_cd:.6f}")
    print("=" * 45 + "\n")

    return mean_cd, std_cd


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
