"""
Standalone verification script for the VAE PointNet++ encoder.

Usage:
    python test_vae_encoder.py
"""

import torch
from src.models.encoder import PointNetPPEncoder


def verify_vae_encoder():
    # 1. Instantiate the updated VAE encoder
    encoder = PointNetPPEncoder()
    encoder.eval()

    # 2. Create a random partial point cloud tensor of shape [1, 2048, 3]
    input_cloud = torch.randn(1, 2048, 3)

    # 3. Pass the exact same tensor through the encoder twice
    with torch.no_grad():
        z1, mu1, log_sigma1 = encoder(input_cloud)
        z2, mu2, log_sigma2 = encoder(input_cloud)

    # 4. Assert that mu1 and mu2 are identical, but z1 and z2 are different
    torch.testing.assert_close(mu1, mu2, msg="mu1 and mu2 should be identical for the same input")
    torch.testing.assert_close(log_sigma1, log_sigma2, msg="log_sigma1 and log_sigma2 should be identical")
    
    assert not torch.equal(z1, z2), "z1 and z2 should be different due to stochastic reparameterization"

    # 5. Print the mean absolute difference to confirm stochasticity
    mean_abs_diff = (z1 - z2).abs().mean().item()
    print("Verification Successful!")
    print(f"Mean Absolute Difference |z1 - z2|: {mean_abs_diff:.6f}")


if __name__ == "__main__":
    verify_vae_encoder()
