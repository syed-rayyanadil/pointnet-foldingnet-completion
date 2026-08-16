import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

np.random.seed(42) 

def load_off(filename):
    with open(filename, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("OFF"):
            raise ValueError("Not a valid OFF header")

        header_tokens = header.split()
        if len(header_tokens) > 1:
            n = int(header_tokens[1])
            if len(header_tokens) < 3:
                raise ValueError("Invalid OFF header")
        else:
            counts = f.readline().strip().split()
            if len(counts) < 3:
                raise ValueError("Invalid OFF vertex count line")
            n = int(counts[0])

        vertices = []
        for _ in range(n):
            vals = [float(s) for s in f.readline().strip().split()]
            vertices.append(vals[:3])  # Only take XYZ, ignore extra columns (e.g. color)

        return np.array(vertices, dtype=np.float32)

class ModelNetDataset(Dataset):
    def __init__(self, root_dir, category, num_points=2048, split="train"):
        """
        Args:
            root_dir: path to the ModelNet40 root directory
            category: a single category string (e.g. 'chair') or
                      a list of category strings (e.g. ['chair', 'table', 'airplane'])
            num_points: number of points to sample per object
            split: 'train' or 'test' dataset split (default: 'train')
        """
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.split = split

        categories = [category] if isinstance(category, str) else list(category)
        self.category = categories

        self.file_paths = []
        for cat in categories:
            cat_files = sorted((self.root_dir / cat / self.split).glob("*.off"))
            if not cat_files:
                raise FileNotFoundError(f"No .off files found in {self.root_dir / cat / self.split}")
            self.file_paths.extend(cat_files)

        if not self.file_paths:
            raise FileNotFoundError(f"No .off files found for categories: {categories}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        points = load_off(str(file_path))

        if points.shape[0] == 0:
            raise ValueError(f"No vertices found in {file_path}")

        points = points[:, :3]

        centroid = points.mean(axis=0)
        points = points - centroid

        scale = np.linalg.norm(points, axis=1).max()
        if scale > 0:
            points = points / scale
        else:
            points = np.zeros_like(points)

        if len(points) >= self.num_points:
            chosen = np.random.choice(len(points), self.num_points, replace=False)
        else:
            chosen = np.random.choice(len(points), self.num_points, replace=True)

        complete_points = torch.from_numpy(points[chosen]).float()
        
        # Create partial point cloud by randomly dropping 50% of points
        num_partial = self.num_points // 2
        partial_indices = np.random.choice(self.num_points, num_partial, replace=False)
        partial_points = complete_points[partial_indices]
        
        return partial_points, complete_points
