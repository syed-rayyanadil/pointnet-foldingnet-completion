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

        partial, complete = dataset[0]
        assert isinstance(partial, torch.Tensor)
        assert isinstance(complete, torch.Tensor)
        assert partial.shape == (1024, 3)
        assert complete.shape == (2048, 3)
        assert partial.dtype == torch.float32
        assert complete.dtype == torch.float32
        assert torch.isfinite(partial).all()
        assert torch.isfinite(complete).all()
        assert partial.min() >= -1.0 - 1e-6
        assert partial.max() <= 1.0 + 1e-6

def test_dataloader():
    dataset = ModelNetDataset(root_dir="data/ModelNet40", category="chair")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for partial_batch, complete_batch in loader:
        print(f"Batch shapes: partial {partial_batch.shape}, complete {complete_batch.shape}")
        assert partial_batch.shape == (4, 1024, 3)
        assert complete_batch.shape == (4, 2048, 3)
        break
    
    print("✓ DataLoader works!")

if __name__ == "__main__":
    test_dataset_shapes_and_normalization()
    test_dataloader()  # ADD THIS
    print("✓ All tests passed!")