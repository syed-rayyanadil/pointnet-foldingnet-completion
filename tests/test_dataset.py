import sys
from pathlib import Path
from torch.utils.data import DataLoader

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import ModelNetDataset


CATEGORIES = ["chair", "table", "airplane"]


def test_dataset_shapes_and_normalization():
    for category in CATEGORIES:
        dataset = ModelNetDataset(root_dir="data/ModelNet40", category=category)
        assert len(dataset) > 0, f"{category} dataset should not be empty"
        print(f"{category}: {len(dataset)} samples")

        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (2048, 3)
        assert sample.dtype == torch.float32
        assert torch.isfinite(sample).all()
        assert sample.min() >= -1.0 - 1e-6
        assert sample.max() <= 1.0 + 1e-6

def test_dataloader():
    dataset = ModelNetDataset(root_dir="data/ModelNet40", category="chair")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in loader:
        print(f"Batch shape: {batch.shape}")  # Should be [4, 2048, 3]
        assert batch.shape == (4, 2048, 3)
        break  # Just test first batch
    
    print("✓ DataLoader works!")

if __name__ == "__main__":
    test_dataset_shapes_and_normalization()
    test_dataloader()  # ADD THIS
    print("✓ All tests passed!")