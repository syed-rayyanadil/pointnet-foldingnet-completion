import torch
import torch.nn as nn

from .encoder import PointNetEncoder, PointNetPPEncoder
from .decoder import FoldingNetDecoder


class PointCloudCompletion(nn.Module):
    """
    End-to-end point cloud completion model.

    Takes a partial point cloud [B, N, 3] and predicts
    the complete point cloud [B, 2048, 3].

    For deterministic encoders (pointnet):
        forward() returns a plain tensor [B, 2048, 3]

    For VAE encoders (pointnet++):
        forward() returns a dict with keys:
            'predicted_cloud', 'mu', 'log_sigma'
    """

    def __init__(self, encoder_type: str = "pointnet") -> None:
        super().__init__()

        if encoder_type == "pointnet":
            self.encoder = PointNetEncoder()
            self.is_vae = False
        elif encoder_type == "pointnet++":
            self.encoder = PointNetPPEncoder()
            self.is_vae = True
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                "Choose 'pointnet' or 'pointnet++'."
            )

        self.decoder = FoldingNetDecoder()

    def forward(self, partial: torch.Tensor):
        """
        Args:
            partial: [B, N, 3] partial point cloud input

        Returns:
            - Deterministic encoder: predicted_cloud tensor [B, 2048, 3]
            - VAE encoder: dict with 'predicted_cloud', 'mu', 'log_sigma'
        """
        encoded = self.encoder(partial)

        if self.is_vae:
            latent, mu, log_sigma = encoded
            predicted_cloud = self.decoder(latent)
            return {
                "predicted_cloud": predicted_cloud,
                "mu": mu,
                "log_sigma": log_sigma,
            }
        else:
            predicted_cloud = self.decoder(encoded)
            return predicted_cloud


# Alias for compatibility
PointCloudCompletionNet = PointCloudCompletion
