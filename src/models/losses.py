import torch
import importlib

try:
    chamfer_distance = importlib.import_module("pytorch3d.loss").chamfer_distance
except ImportError:
    chamfer_distance = None


def chamfer_loss(pred, gt):
    """
    Compute Chamfer Distance between two point clouds using PyTorch3D.

    Args:
        pred: [B, N, 3] predicted point cloud
        gt: [B, N, 3] ground truth point cloud

    Returns:
        scalar loss value
    """
    if chamfer_distance is not None:
        loss, _ = chamfer_distance(pred, gt)
        return loss

    pairwise_distances = torch.cdist(pred, gt)
    pred_to_gt = pairwise_distances.min(dim=2).values.mean()
    gt_to_pred = pairwise_distances.min(dim=1).values.mean()
    return pred_to_gt + gt_to_pred
