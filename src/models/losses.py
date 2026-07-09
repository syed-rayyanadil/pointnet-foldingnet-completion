import torch
import torch.nn as nn


def chamfer_loss(pred, gt):
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred: [B, N, 3] predicted point cloud
        gt: [B, N, 3] ground truth point cloud
    
    Returns:
        scalar loss value
    """
    # Compute pairwise distances
    # pred: [B, N, 3], gt: [B, M, 3]
    # Output: [B, N, M] distances
    
    B, N, _ = pred.shape
    M = gt.shape[1]
    
    # Reshape for broadcasting: pred [B, N, 1, 3], gt [B, 1, M, 3]
    pred_exp = pred.unsqueeze(2)  # [B, N, 1, 3]
    gt_exp = gt.unsqueeze(1)      # [B, 1, M, 3]
    
    # Compute L2 distances: [B, N, M]
    distances = torch.sqrt(torch.sum((pred_exp - gt_exp) ** 2, dim=3) + 1e-8)
    
    # Chamfer: min distance from each pred point to any gt point
    # + min distance from each gt point to any pred point
    min_pred_to_gt = torch.min(distances, dim=2)[0]  # [B, N]
    min_gt_to_pred = torch.min(distances, dim=1)[0]  # [B, M]
    
    # Average over both directions
    loss = torch.mean(min_pred_to_gt) + torch.mean(min_gt_to_pred)
    
    return loss
