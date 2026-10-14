import torch
from scipy.spatial.transform import Rotation

PALM = [0, 5, 9, 13, 17]  # wrist and finger bases: they barely move when the fingers bend


class HandInitializer:
    """Places a MANO hand onto triangulated 3D joints, as the starting point for the fit."""

    def __init__(self, hand):
        self.hand = hand

    def initialize(self, target, shape):
        """target: (21, 3) triangulated joints, NaN where unknown; shape: (10,) hand shape.
        Returns rotation (3,) and translation (3,), or None if too little of the palm is known."""
        known = [j for j in PALM if not torch.isnan(target[j]).any()]
        if len(known) < 3:
            return None
        zeros = torch.zeros(1, 3)
        pose = torch.zeros(1, 45)
        template = self.hand(pose, shape[None], zeros, zeros)[0][0]

        a = template[known] - template[known].mean(0)
        b = target[known] - target[known].mean(0)
        U, _, Vh = torch.linalg.svd(a.T @ b)
        d = torch.sign(torch.det(Vh.T @ U.T))
        R = Vh.T @ torch.diag(torch.tensor([1.0, 1.0, d])) @ U.T
        rotation = torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec(), dtype=torch.float32)

        rotated = self.hand(pose, shape[None], rotation[None], zeros)[0][0]
        translation = (target[known] - rotated[known]).mean(0)
        return rotation, translation
