import torch
from pytorch3d.loss import chamfer_distance


def chamfer_loss(pred, gt):
    """
    Compute Chamfer Distance between two point clouds using PyTorch3D.

    Args:
        pred: [B, N, 3] predicted point cloud
        gt: [B, N, 3] ground truth point cloud

    Returns:
        scalar loss value
    """
    loss, _ = chamfer_distance(pred, gt)
    return loss
