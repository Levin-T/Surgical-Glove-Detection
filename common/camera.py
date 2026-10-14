import json

import torch


class Camera:
    """A calibrated pinhole camera.
    R and t map world points into the camera: x_cam = R @ x_world + t (meters)."""

    def __init__(self, K, R, t):
        self.K = torch.as_tensor(K, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
        self.t = torch.as_tensor(t, dtype=torch.float32)

    def projection_matrix(self):
        """The 3x4 matrix K [R | t] that maps world points to pixels."""
        return self.K @ torch.cat([self.R, self.t[:, None]], dim=1)

    def project(self, points):
        """World points (N, 3) -> pixel coordinates (N, 2)."""
        uvw = (points @ self.R.T + self.t) @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3]


def load_cameras(path):
    """Reads a session's calibration.json and returns one Camera per entry."""
    with open(path) as f:
        data = json.load(f)
    return [Camera(c["K"], c["R"], c["t"]) for c in data["cameras"]]
