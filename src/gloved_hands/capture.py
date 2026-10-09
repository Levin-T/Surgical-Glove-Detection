"""Interface to the capture pipeline, which is implemented separately.

``Recording`` is a stub: it defines what the rest of the code needs from one recorded take.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """Calibrated pinhole camera. A world point X (metres) maps to camera coordinates as R @ X + t."""

    name: str
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @property
    def P(self) -> np.ndarray:
        """(3, 4) projection matrix."""
        return self.K @ np.hstack([self.R, self.t[:, None]])

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return points @ self.R.T + self.t

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (..., 3) to pixels (..., 2)."""
        uvw = self.to_camera(points) @ self.K.T
        return uvw[..., :2] / uvw[..., 2:]

    def backproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) with z-depth (N,) in metres to world points (N, 3)."""
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.K).T
        return (rays * depth[:, None] - self.t) @ self.R


@dataclass
class Frame:
    """One synchronised instant. Lists follow the order of ``Recording.cameras``."""

    color: list[np.ndarray]  # (H, W, 3) uint8 RGB, undistorted
    depth: list[np.ndarray]  # (H, W) float32 metres, aligned to color; 0 where there is no depth
    manus: dict[str, np.ndarray] = field(default_factory=dict)  # side -> (21, 3) MANUS joints, joint order of mano.py


class Recording:
    """One synchronised, calibrated take of the rig. Stub: implement it on top of the capture pipeline."""

    session: str  # unique id of the take
    subject: str  # pseudonym, e.g. "S03"
    task: str
    cameras: list[Camera]

    def __init__(self, root: Path, session: str) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Frame:
        raise NotImplementedError
