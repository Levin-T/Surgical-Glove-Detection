"""MANO with the 21 joints in OpenPose order, the order HaMeR, WiLoR and DexYCB use."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINT_NAMES = ["wrist"] + [f"{finger}{k}" for finger in FINGERS for k in (1, 2, 3, 4)]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
FINGERTIP_VERTICES = [744, 320, 443, 554, 671]  # thumb .. pinky, the vertices HaMeR and WiLoR use
MIRROR = np.array([1.0, -1.0, -1.0])  # axis-angle of the same rotation seen in a mirrored image


@dataclass
class HandParams:
    """MANO parameters of T frames, axis-angle; the hand pose is relative to the flat hand."""

    global_orient: np.ndarray  # (T, 3)
    hand_pose: np.ndarray  # (T, 45)
    betas: np.ndarray  # (10,), shared by all frames
    transl: np.ndarray  # (T, 3) metres


class Mano(nn.Module):
    """MANO layer of one side: parameters to vertices (B, 778, 3) and joints (B, 21, 3).

    The left model is used as released, without the shape-direction fix of smplx issue 48, because DexYCB's
    left hands were fitted with it."""

    def __init__(self, model_dir: str, side: str) -> None:
        super().__init__()
        import smplx

        self.layer = smplx.MANO(model_dir, is_rhand=side == "right", use_pca=False, flat_hand_mean=True)

    def forward(self, global_orient: Tensor, hand_pose: Tensor, betas: Tensor, transl: Tensor) -> tuple[Tensor, Tensor]:
        out = self.layer(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
        joints = torch.cat([out.joints, out.vertices[:, FINGERTIP_VERTICES]], dim=1)[:, MANO_TO_OPENPOSE]
        return out.vertices, joints
