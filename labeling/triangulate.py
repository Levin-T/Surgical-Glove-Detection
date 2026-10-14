import torch


class Triangulator:
    """Turns 2D keypoints from several cameras into 3D points."""

    def __init__(self, cameras):
        self.P = torch.stack([camera.projection_matrix() for camera in cameras])  # (V, 3, 4)

    def triangulate(self, points_2d, weights):
        """points_2d: (V, J, 2) pixels for V cameras and J joints.
        weights: (V, J) trust per point, 0 = not seen in that camera.
        Returns (J, 3) world points; NaN for joints seen by fewer than two cameras."""
        points_2d = torch.as_tensor(points_2d, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        points_3d = torch.full((points_2d.shape[1], 3), float("nan"))
        for j in range(points_2d.shape[1]):
            seen = weights[:, j] > 0
            if seen.sum() < 2:
                continue
            u = points_2d[seen, j, 0:1]
            v = points_2d[seen, j, 1:2]
            w = weights[seen, j][:, None]
            P = self.P[seen]
            A = torch.cat([w * (u * P[:, 2] - P[:, 0]),
                           w * (v * P[:, 2] - P[:, 1])])
            X = torch.linalg.svd(A).Vh[-1]
            points_3d[j] = X[:3] / X[3]
        return points_3d
