import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoder import PointNetEncoder


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
