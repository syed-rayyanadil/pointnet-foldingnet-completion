import torch
import torch.nn as nn


class FoldingNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # Create fixed 2D grid that produces 2048 points (64 × 32)
        grid_y = torch.linspace(-1, 1, 32)
        grid_x = torch.linspace(-1, 1, 64)
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        self.register_buffer('grid', torch.stack([grid_xx, grid_yy], dim=-1).view(-1, 2))
        
        # MLP that takes concatenated [grid(2) + feature(1024)] and outputs [3]
        self.mlp = nn.Sequential(
            nn.Linear(1024 + 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != 1024:
            raise ValueError(f"Expected input shape [B, 1024], got {tuple(x.shape)}")
        
        B = x.shape[0]
        # Expand grid for batch: [2048, 2] → [B, 2048, 2]
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # [B, 2048, 2]
        
        # Expand features: [B, 1024] → [B, 2048, 1024]
        features = x.unsqueeze(1).expand(-1, self.grid.shape[0], -1)  # [B, 2048, 1024]
        
        # Concatenate: [B, 2048, 2] + [B, 2048, 1024] → [B, 2048, 1026]
        combined = torch.cat([grid, features], dim=-1)
        
        # Reshape for MLP: [B*2048, 1026]
        combined_flat = combined.view(-1, 1024 + 2)
        
        # Apply MLP: [B*2048, 1026] → [B*2048, 3]
        output_flat = self.mlp(combined_flat)
        
        # Reshape back: [B, 2048, 3]
        output = output_flat.view(B, self.grid.shape[0], 3)
        
        return output
