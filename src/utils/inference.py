"""
Inference utilities for Monte Carlo uncertainty estimation.

Uses the stochastic reparameterization of the VAE encoder: running the same
partial cloud through the model K times produces K different completions
(because eps is re-sampled each forward pass). The variance across those K
completions gives a per-point uncertainty estimate.
"""

import torch


def get_uncertainty_predictions(
    model: torch.nn.Module,
    partial_cloud: torch.Tensor,
    k: int = 20,
) -> tuple:
    """
    Run Monte Carlo sampling to estimate per-point uncertainty.

    The model is run K times on the same partial input. Because the VAE
    encoder re-samples the noise vector (eps) on every forward pass, each
    run produces a slightly different completion. The spread of those K
    completions is used as the uncertainty estimate.

    Args:
        model:         The trained PointCloudCompletion model (in eval mode).
        partial_cloud: [1, N, 3] or [N, 3] partial point cloud tensor.
        k:             Number of stochastic forward passes (default: 20).

    Returns:
        mean_completion: [2048, 3]  — mean predicted completion across K runs.
        point_variance:  [2048]     — scalar variance per point, averaged over
                                      the (x, y, z) channels.
    """
    model.eval()

    # Ensure input has a batch dimension: [1, N, 3]
    if partial_cloud.dim() == 2:
        partial_cloud = partial_cloud.unsqueeze(0)

    device = next(model.parameters()).device
    partial_cloud = partial_cloud.to(device)

    completions = []

    with torch.no_grad():
        for _ in range(k):
            output = model(partial_cloud)

            # Handle both deterministic (tensor) and VAE (dict) outputs
            if isinstance(output, dict):
                predicted = output["predicted_cloud"]  # [1, 2048, 3]
            else:
                predicted = output                     # [1, 2048, 3]

            completions.append(predicted.squeeze(0))   # [2048, 3]

    # Stack into [K, 2048, 3]
    completions = torch.stack(completions, dim=0)

    # Mean completion: [2048, 3]
    mean_completion = completions.mean(dim=0)

    # Variance per point: [K, 2048, 3] → variance over K → [2048, 3]
    # Then average over (x, y, z) channels → [2048] scalar per point
    point_variance = completions.var(dim=0).mean(dim=-1)

    return mean_completion, point_variance
