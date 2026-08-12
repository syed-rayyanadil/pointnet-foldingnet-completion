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


def kl_divergence_loss(mu, log_sigma):
    """
    Compute KL divergence between the learned latent distribution
    and the standard normal distribution N(0, I).

    KL = -0.5 * sum(1 + log_sigma - mu^2 - exp(log_sigma))

    Args:
        mu:        [B, latent_dim] mean of the latent distribution
        log_sigma: [B, latent_dim] log-variance of the latent distribution

    Returns:
        scalar loss value (mean over the batch)
    """
    return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1).mean()
