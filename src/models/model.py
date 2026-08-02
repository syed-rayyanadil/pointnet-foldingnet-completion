import torch
import torch.nn as nn

from .encoder import PointNetEncoder, PointNetPPEncoder
from .decoder import FoldingNetDecoder


class PointCloudCompletion(nn.Module):
    """
    End-to-end point cloud completion model.

    Takes a partial point cloud [B, N, 3] and predicts
    the complete point cloud [B, 2048, 3].
    """

    def __init__(self, encoder_type: str = "pointnet") -> None:
        super().__init__()

        if encoder_type == "pointnet":
            self.encoder = PointNetEncoder()
        elif encoder_type == "pointnet++":
            self.encoder = PointNetPPEncoder()
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                "Choose 'pointnet' or 'pointnet++'."
            )

        self.decoder = FoldingNetDecoder()

    def forward(self, partial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial: [B, N, 3] partial point cloud input

        Returns:
            [B, 2048, 3] predicted complete point cloud
        """
        # Encode partial cloud to global feature vector [B, 1024]
        latent = self.encoder(partial)

        # Decode latent vector to complete point cloud [B, 2048, 3]
        complete = self.decoder(latent)

        return complete
