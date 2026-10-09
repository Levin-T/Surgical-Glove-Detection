"""Per-view evidence for labelling: hand boxes (WiLoR's detector), HaMeR estimates and SAM 2 glove masks."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import default_collate

from gloved_hands.config import PathsConfig
from gloved_hands.mano import MIRROR
from gloved_hands.model import load_pretrained


@dataclass
class HandEstimate:
    """One hand in one view. HaMeR only initialises the fit and is never used as a label."""

    score: float  # detector confidence
    keypoints: np.ndarray  # (21, 2) pixels
    mask: np.ndarray  # (H, W) bool, glove mask
    global_orient: np.ndarray  # (3,) axis-angle, camera frame
    hand_pose: np.ndarray  # (45,)
    betas: np.ndarray  # (10,)


class ViewEstimator:
    """The most confident right and left hand in an image."""

    def __init__(self, paths: PathsConfig, device: str) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from ultralytics import YOLO

        self.detector = YOLO(paths.detector)
        self.hamer = load_pretrained("hamer", paths).to(device).eval()
        self.segmenter = SAM2ImagePredictor.from_pretrained(paths.sam2, device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> dict[str, HandEstimate]:
        """Estimates by side ("right", "left") for an RGB image."""
        from wilor.datasets.vitdet_dataset import ViTDetDataset  # HaMeR's crops, without its debug print

        bgr = np.ascontiguousarray(image[..., ::-1])  # the detector and HaMeR's crops expect BGR
        boxes = self.detector(bgr, conf=0.3, verbose=False)[0].boxes
        best: dict[str, tuple[np.ndarray, float]] = {}
        for box, label, score in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            side = "right" if label == 1 else "left"
            if side not in best or score > best[side][1]:
                best[side] = (box, float(score))
        if not best:
            return {}

        sides = list(best)
        right = np.array([side == "right" for side in sides], dtype=np.float32)
        crops = ViTDetDataset(
            self.hamer.cfg, bgr, np.stack([best[side][0] for side in sides]), right, rescale_factor=2.0
        )
        batch = default_collate([crops[i] for i in range(len(sides))])
        out = self.hamer({"img": batch["img"].to(self.device)})
        keypoints = out["pred_keypoints_2d"].float().cpu().numpy()  # normalised crop coordinates
        keypoints[..., 0] *= np.where(right > 0, 1.0, -1.0)[:, None]  # left hands were mirrored in the crop
        keypoints = keypoints * batch["box_size"].numpy()[:, None, None] + batch["box_center"].numpy()[:, None]
        mano = {name: value.float().cpu().numpy() for name, value in out["pred_mano_params"].items()}

        self.segmenter.set_image(image)
        estimates = {}
        for i, side in enumerate(sides):
            mirror = np.ones(3) if side == "right" else MIRROR  # HaMeR saw left hands mirrored, as right hands
            orient = Rotation.from_matrix(mano["global_orient"][i].reshape(3, 3)).as_rotvec() * mirror
            pose = Rotation.from_matrix(mano["hand_pose"][i].reshape(15, 3, 3)).as_rotvec() * mirror
            box, score = best[side]
            masks, _, _ = self.segmenter.predict(box=box, multimask_output=False)
            estimates[side] = HandEstimate(
                score, keypoints[i], masks[0] > 0, orient, pose.reshape(45), mano["betas"][i]
            )
        return estimates
