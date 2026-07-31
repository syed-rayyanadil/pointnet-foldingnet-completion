import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, global_max_pool, radius


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius_value, nsample, in_channels, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius_value = radius_value
        self.nsample = nsample
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, mlp[0]),
            nn.BatchNorm1d(mlp[0]),
            nn.ReLU(inplace=True),
            *[
                layer
                for index in range(1, len(mlp))
                for layer in (
                    nn.Linear(mlp[index - 1], mlp[index]),
                    nn.BatchNorm1d(mlp[index]),
                    nn.ReLU(inplace=True),
                )
            ],
        )
        self.conv = PointNetConv(local_nn=self.mlp)

    def forward(self, pos, features, batch):
        points_per_cloud = pos.shape[0] // (int(batch.max().item()) + 1)
        try:
            sample_index = fps(pos, batch, ratio=self.npoint / points_per_cloud)
        except ImportError:
            sample_index = torch.cat(
                [
                    torch.where(batch == batch_id)[0][: self.npoint]
                    for batch_id in torch.unique(batch, sorted=True)
                ]
            )
        sampled_pos = pos[sample_index]
        sampled_batch = batch[sample_index]
        try:
            edge_index = radius(
                pos,
                sampled_pos,
                self.radius_value,
                batch,
                sampled_batch,
                max_num_neighbors=self.nsample,
            )
        except ImportError:
            source_indices = []
            target_indices = []
            for batch_id in torch.unique(batch, sorted=True):
                source = torch.where(batch == batch_id)[0]
                target = torch.where(sampled_batch == batch_id)[0]
                distances = torch.cdist(sampled_pos[target], pos[source])
                for target_offset, target_index in enumerate(target):
                    neighbors = torch.where(
                        distances[target_offset] <= self.radius_value
                    )[0]
                    if neighbors.numel() == 0:
                        neighbors = distances[target_offset].argmin().view(1)
                    neighbors = neighbors[: self.nsample]
                    source_indices.append(source[neighbors])
                    target_indices.append(target_index.expand(neighbors.numel()))
            edge_index = torch.stack(
                [torch.cat(source_indices), torch.cat(target_indices)]
            )
        sampled_features = self.conv(
            (features, None) if features is not None else None,
            (pos, sampled_pos),
            edge_index,
        )
        return sampled_pos, sampled_features, sampled_batch


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


class PointNetPPEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius_value=0.2,
            nsample=32,
            in_channels=0,
            mlp=[64, 64, 128],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius_value=0.4,
            nsample=64,
            in_channels=128,
            mlp=[128, 128, 256],
        )
        self.sa3_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected input shape [B, N, 3], got {tuple(x.shape)}")

        batch_size, num_points, _ = x.shape
        pos = x.reshape(batch_size * num_points, 3)
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_points)

        pos, features, batch = self.sa1(pos, None, batch)
        pos, features, batch = self.sa2(pos, features, batch)
        features = global_max_pool(features, batch)
        return self.sa3_mlp(features)