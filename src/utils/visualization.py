import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_uncertainty_panels(
    partial_cloud,
    mean_completion,
    variance_heatmap,
    title: str = "Point Cloud Completion with Uncertainty",
) -> plt.Figure:
    """
    Render a 1x3 side-by-side 3D subplot showing:
        Panel 1 — Partial Input
        Panel 2 — Mean Predicted Completion (coloured by variance)
        Panel 3 — Uncertainty Heatmap (point size scaled by variance)

    Args:
        partial_cloud:    [N, 3] tensor or numpy array — partial input cloud
        mean_completion:  [M, 3] tensor or numpy array — mean predicted completion
        variance_heatmap: [M]    tensor or numpy array — per-point variance values

    Returns:
        matplotlib.figure.Figure
    """
    # Convert to numpy if tensors are passed
    if isinstance(partial_cloud, torch.Tensor):
        partial_cloud = partial_cloud.detach().cpu().numpy()
    if isinstance(mean_completion, torch.Tensor):
        mean_completion = mean_completion.detach().cpu().numpy()
    if isinstance(variance_heatmap, torch.Tensor):
        variance_heatmap = variance_heatmap.detach().cpu().numpy()

    partial_cloud = np.asarray(partial_cloud)
    mean_completion = np.asarray(mean_completion)
    variance_heatmap = np.asarray(variance_heatmap)

    # Normalise variance to [0, 1] for colour mapping
    v_min, v_max = variance_heatmap.min(), variance_heatmap.max()
    if v_max - v_min > 1e-8:
        variance_norm = (variance_heatmap - v_min) / (v_max - v_min)
    else:
        variance_norm = np.zeros_like(variance_heatmap)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Panel 1: Partial Input ────────────────────────────────────────────
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(
        partial_cloud[:, 0],
        partial_cloud[:, 1],
        partial_cloud[:, 2],
        c="coral",
        s=2,
        alpha=0.7,
    )
    ax1.set_title(f"Partial Input\n({len(partial_cloud)} pts)", fontsize=11)
    _set_equal_axes(ax1, partial_cloud)

    # ── Panel 2: Mean Completion (coloured by variance) ───────────────────
    ax2 = fig.add_subplot(132, projection="3d")
    sc = ax2.scatter(
        mean_completion[:, 0],
        mean_completion[:, 1],
        mean_completion[:, 2],
        c=variance_norm,
        cmap="plasma",
        s=2,
        alpha=0.8,
    )
    ax2.set_title(f"Mean Completion\n({len(mean_completion)} pts)", fontsize=11)
    _set_equal_axes(ax2, mean_completion)
    fig.colorbar(sc, ax=ax2, shrink=0.5, pad=0.1, label="Uncertainty")

    # ── Panel 3: Uncertainty Heatmap (point size ∝ variance) ─────────────
    ax3 = fig.add_subplot(133, projection="3d")
    # Scale point sizes: low variance → small dots, high variance → large dots
    point_sizes = 1 + variance_norm * 10
    ax3.scatter(
        mean_completion[:, 0],
        mean_completion[:, 1],
        mean_completion[:, 2],
        c=variance_norm,
        cmap="hot",
        s=point_sizes,
        alpha=0.8,
    )
    ax3.set_title("Uncertainty Heatmap\n(size ∝ variance)", fontsize=11)
    _set_equal_axes(ax3, mean_completion)

    plt.tight_layout()
    return fig


def _set_equal_axes(ax, points: np.ndarray):
    """Set equal aspect ratio on a 3D axis based on the point cloud bounds."""
    ranges = points.max(axis=0) - points.min(axis=0)
    max_range = ranges.max() if ranges.max() > 0 else 1.0
    mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
