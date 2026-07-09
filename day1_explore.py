import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_off(filename):
    with open(filename, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise('Not a valid OFF header')
        n, _, _ = tuple([int(s) for s in f.readline().strip().split(' ')])
        vertices = []
        for i in range(n):
            vertices.append([float(s) for s in f.readline().strip().split()])
        return np.array(vertices)

# Load
points = load_off("data/ModelNet40/chair/train/chair_0001.off")
print(f"Before normalization: min={points.min()}, max={points.max()}")

# Normalize to unit sphere
centroid = points.mean(axis=0)
points -= centroid
max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
points /= max_dist

print(f"After normalization: min={points.min()}, max={points.max()}")

# Visualize with matplotlib
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Chair Point Cloud')
plt.show()
