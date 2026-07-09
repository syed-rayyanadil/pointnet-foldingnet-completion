import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoder import PointNetEncoder
from src.models.decoder import FoldingNetDecoder
from data.dataset import ModelNetDataset


def test_decoder_output_shape():
    """Test that decoder outputs correct shape [B, 2048, 3]"""
    torch.manual_seed(0)
    decoder = FoldingNetDecoder()
    features = torch.randn(4, 1024)
    
    output = decoder(features)
    
    assert output.shape == (4, 2048, 3)
    assert output.dtype == torch.float32


def test_decoder_is_finite():
    """Test that decoder output has no NaN or Inf values"""
    torch.manual_seed(1)
    decoder = FoldingNetDecoder()
    features = torch.randn(2, 1024)
    
    output = decoder(features)
    
    assert torch.isfinite(output).all()


def test_end_to_end_pipeline():
    """Test full pipeline: partial → encoder → decoder → completion"""
    torch.manual_seed(42)
    encoder = PointNetEncoder()
    decoder = FoldingNetDecoder()
    
    # Create dummy batch
    partial = torch.randn(4, 1024, 3)
    
    # Forward through encoder
    features = encoder(partial)
    assert features.shape == (4, 1024)
    
    # Forward through decoder
    completion = decoder(features)
    assert completion.shape == (4, 2048, 3)
    assert torch.isfinite(completion).all()


def test_end_to_end_with_dataloader():
    """Test pipeline with actual DataLoader"""
    torch.manual_seed(42)
    encoder = PointNetEncoder()
    decoder = FoldingNetDecoder()
    
    dataset = ModelNetDataset(root_dir="data/ModelNet40", category="chair")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    for partial_batch, complete_batch in loader:
        assert partial_batch.shape == (4, 1024, 3)
        assert complete_batch.shape == (4, 2048, 3)
        
        # Encoder
        features = encoder(partial_batch)
        assert features.shape == (4, 1024)
        
        # Decoder
        prediction = decoder(features)
        assert prediction.shape == (4, 2048, 3)
        assert torch.isfinite(prediction).all()
        assert torch.isfinite(features).all()
        
        break  # Just test first batch


if __name__ == "__main__":
    test_decoder_output_shape()
    print("✓ Decoder output shape test passed")
    
    test_decoder_is_finite()
    print("✓ Decoder finite values test passed")
    
    test_end_to_end_pipeline()
    print("✓ End-to-end pipeline test passed")
    
    test_end_to_end_with_dataloader()
    print("✓ End-to-end DataLoader test passed")
    
    print("\n✓ All tests passed!")
