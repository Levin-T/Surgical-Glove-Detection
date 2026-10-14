import smplx
import torch
from smplx.lbs import batch_rodrigues
from smplx.vertex_ids import vertex_ids

# RTMPose / OpenPose order: wrist, then thumb, index, middle, ring, little finger (4 joints each)
MANO_TO_KEYPOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
FINGERTIP_VERTICES = list(vertex_ids["mano"].values())  # thumb, index, middle, ring, little


class Hand:
    """The MANO hand model for one side, returning 21 joints in RTMPose's order."""

    def __init__(self, model_folder, side):
        self.side = side
        self.layer = smplx.MANOLayer(model_folder, is_rhand=(side == "right"), use_pca=False)
        if side == "left":  # known sign error in MANO_LEFT's first shape direction
            right = smplx.MANOLayer(model_folder, is_rhand=True, use_pca=False)
            if torch.sum(torch.abs(self.layer.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
                self.layer.shapedirs[:, 0, :] *= -1

    def __call__(self, pose, shape, rotation, translation):
        """pose (B, 45): finger joint rotations; shape (B, 10): hand size and proportions;
        rotation (B, 3) and translation (B, 3): orientation and position of the whole hand.
        All rotations are axis-angle. Returns joints (B, 21, 3) and mesh vertices (B, 778, 3), in meters."""
        B = pose.shape[0]
        finger_matrices = batch_rodrigues(pose.reshape(-1, 3)).reshape(B, 15, 3, 3)
        hand_matrix = batch_rodrigues(rotation.reshape(-1, 3)).reshape(B, 1, 3, 3)
        out = self.layer(hand_pose=finger_matrices, betas=shape, global_orient=hand_matrix, transl=translation)
        joints = torch.cat([out.joints[:, :16], out.vertices[:, FINGERTIP_VERTICES]], dim=1)
        return joints[:, MANO_TO_KEYPOINT_ORDER], out.vertices
