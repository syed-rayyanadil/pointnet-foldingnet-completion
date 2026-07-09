import sys
from pathlib import Path

import torch
from pytorch3d.loss import chamfer_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoder import PointNetEncoder
from src.models.decoder import FoldingNetDecoder
from src.models.losses import chamfer_loss


def test_loss_identical():
    """Test: identical clouds should have loss near 0"""
    pred = torch.randn(4, 2048, 3)
    gt = pred.clone()
    
    loss = chamfer_loss(pred, gt)
    print(f"✓ Identical clouds: loss={loss.item():.6f} (should be ~0)")
    assert loss.item() < 1e-3, f"Loss should be near 0 for identical clouds, got {loss.item()}"


def test_loss_different():
    """Test: different clouds should have loss > 0"""
    pred = torch.randn(4, 2048, 3)
    gt = torch.randn(4, 2048, 3)
    
    loss = chamfer_loss(pred, gt)
    print(f"✓ Different clouds: loss={loss.item():.6f} (should be > 0)")
    assert loss.item() > 0, f"Loss should be > 0 for different clouds, got {loss.item()}"


def test_smoke_train():
    """Smoke test: train on 1 batch for 5 steps, verify loss decreases"""
    torch.manual_seed(42)
    encoder = PointNetEncoder()
    decoder = FoldingNetDecoder()
    
    # Create dummy batch
    partial = torch.randn(4, 1024, 3)
    complete = torch.randn(4, 2048, 3)
    
    # Setup optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    losses = []
    print("\nTraining for 5 steps:")
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        features = encoder(partial)
        pred = decoder(features)
        
        # Loss
        loss = chamfer_loss(pred, complete)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step}: loss={loss.item():.6f}")
    
    # Verify loss is decreasing (mostly)
    print(f"\nInitial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Total decrease: {(losses[0] - losses[-1]):.6f}")
    
    # Loss should decrease overall (allow some noise)
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print("✓ Loss decreasing confirmed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chamfer Loss Function")
    print("=" * 60)
    
    test_loss_identical()
    test_loss_different()
    test_smoke_train()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Pipeline is differentiable and working.")
    print("=" * 60)
