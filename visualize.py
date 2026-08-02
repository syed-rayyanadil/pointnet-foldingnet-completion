"""
Visualization script for Point Cloud Completion.

Loads a trained checkpoint, runs inference on a test sample, and
plots a 3-panel comparison: Partial → Predicted → Ground Truth.

Usage:
    python visualize.py
    python visualize.py --checkpoint checkpoints/best_model.pth --category chair --sample_idx 0
"""

import argparse

import matplotlib.pyplot as plt
import torch

from data.dataset import ModelNetDataset
from src.models import PointCloudCompletion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize point cloud completion results"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="chair",
        help="ModelNet40 category (default: chair)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pointnet",
        choices=["pointnet", "pointnet++"],
        help="Encoder architecture used during training (default: pointnet)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the sample to visualize (default: 0)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save the figure to this path instead of showing it",
    )
    return parser.parse_args()


def plot_point_cloud(ax, points, title, color="steelblue"):
    """Plot a single point cloud on a 3D axis."""
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        s=1,
        alpha=0.6,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set equal aspect ratio for all axes
    max_range = max(
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


def visualize(args):
    device = torch.device("cpu")

    # Load dataset
    dataset = ModelNetDataset(
        root_dir="data/ModelNet40", category=args.category
    )
    print(f"Dataset: {args.category} — {len(dataset)} samples")

    if args.sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx {args.sample_idx} out of range (dataset has {len(dataset)} samples)"
        )

    # Get a sample
    partial, complete = dataset[args.sample_idx]
    print(f"Sample {args.sample_idx}: partial {partial.shape}, complete {complete.shape}")

    # Load model
    model = PointCloudCompletion(encoder_type=args.encoder).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Inference
    with torch.no_grad():
        partial_input = partial.unsqueeze(0).to(device)  # [1, 1024, 3]
        predicted = model(partial_input)                  # [1, 2048, 3]

    # Convert to numpy for plotting
    partial_np = partial.numpy()
    predicted_np = predicted.squeeze(0).numpy()
    complete_np = complete.numpy()

    # Plot 3-panel comparison
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    plot_point_cloud(ax1, partial_np, f"Partial Input\n({partial_np.shape[0]} points)", color="coral")

    ax2 = fig.add_subplot(132, projection="3d")
    plot_point_cloud(ax2, predicted_np, f"Predicted Completion\n({predicted_np.shape[0]} points)", color="mediumseagreen")

    ax3 = fig.add_subplot(133, projection="3d")
    plot_point_cloud(ax3, complete_np, f"Ground Truth\n({complete_np.shape[0]} points)", color="steelblue")

    fig.suptitle(
        f"Point Cloud Completion — {args.category} (sample #{args.sample_idx})",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    visualize(args)
