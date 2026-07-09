import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input shape [B, N, 3], got {tuple(x.shape)}")

        x = x.transpose(1, 2)  # [B, 3, N]
        x = self.mlp(x)       # [B, 1024, N]
        x = torch.max(x, dim=2)[0]
        return x