"""Checks that the HandFitter recovers a known MANO hand from noisy 2D keypoints in four cameras.
Run from the project root: python -m tests.test_fit   (or: pytest tests)"""
import math
from pathlib import Path

import torch

from common.camera import Camera
from common.mano import Hand
from labeling.fit import HandFitter

MODELS = Path(__file__).resolve().parent.parent / "models"


def look_at(position, target):
    """A camera at `position` looking at `target`, with a D455-like lens."""
    position, target = torch.tensor(position), torch.tensor(target)
    z = target - position
    z = z / z.norm()
    x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, 1.0]))
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])
    return Camera([[600, 0, 424], [0, 600, 240], [0, 0, 1]], R, -R @ position)


def test_fit_recovers_hand():
    # four cameras around the table, tilted down onto the hand
    cameras = [look_at([0.6 * math.cos(a), 0.6 * math.sin(a), 0.55], [0.0, 0.0, 0.12])
               for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
    hand = Hand(str(MODELS), "right")
    shape = torch.zeros(10)

    # a known hand: random finger bends, turned and placed in front of the cameras
    torch.manual_seed(0)
    true_pose = torch.randn(45) * 0.3
    true_rotation = torch.tensor([0.3, -0.2, 1.0])
    true_translation = torch.tensor([0.0, 0.0, 0.12])
    true_joints = hand(true_pose[None], shape[None], true_rotation[None], true_translation[None])[0][0]

    # what the cameras would report: projected joints plus 2 pixels of noise
    points = torch.stack([camera.project(true_joints) for camera in cameras]) + torch.randn(4, 21, 2) * 2.0
    weights = torch.ones(4, 21)

    # start from a flat hand that is slightly turned and shifted, as step 3 would give
    start = {"pose": torch.zeros(45), "rotation": true_rotation + 0.1, "translation": true_translation + 0.01}
    fitter = HandFitter(hand, cameras)
    before = fitter.joints(start["pose"], shape, start["rotation"], start["translation"])
    result = fitter.fit(points, weights, shape, start)

    error_before = (before - true_joints).norm(dim=-1).mean().item() * 1000
    error_after = (result["joints"] - true_joints).norm(dim=-1).mean().item() * 1000
    print(f"joint error before: {error_before:.1f} mm")
    print(f"joint error after:  {error_after:.1f} mm")
    print("pixel error per camera:", [round(e, 1) for e in result["errors"].tolist()])
    assert error_after < 10.0
    assert error_after < error_before / 3


if __name__ == "__main__":
    test_fit_recovers_hand()
