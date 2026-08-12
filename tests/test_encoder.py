import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoder import PointNetEncoder, PointNetPPEncoder


def test_encoder_output_shape():
    torch.manual_seed(0)
    model = PointNetEncoder()
    points = torch.randn(4, 2048, 3)

    features = model(points)

    assert features.shape == (4, 1024)
    assert features.dtype == torch.float32


def test_encoder_output_is_finite():
    torch.manual_seed(1)
    model = PointNetEncoder()
    points = torch.randn(2, 2048, 3)

    features = model(points)

    assert torch.isfinite(features).all()


def test_pointnetpp_encoder_vae():
    torch.manual_seed(42)
    model = PointNetPPEncoder()
    points = torch.randn(2, 512, 3)

    outputs = model(points)
    
    # Check outputs format
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    
    z, mu, log_sigma = outputs
    
    # Verify shapes
    assert z.shape == (2, 1024)
    assert mu.shape == (2, 1024)
    assert log_sigma.shape == (2, 1024)
    
    # Verify finiteness
    assert torch.isfinite(z).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_sigma).all()
    
    # Verify that reparameterization standard deviation acts as expected
    # z should not be exactly equal to mu since eps is random
    assert not torch.equal(z, mu)

