"""Hand crops with their labels in the crop's camera frame, saved as one .npz file per crop."""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from gloved_hands.capture import Camera
from gloved_hands.mano import MIRROR

CROP_SIZE = 256  # the models' input; their ViT sees the central 192 columns
CROP_SCALE = 2.0  # crop side / the joints' box grown to the ViT's 3:4 shape, as WiLoR's demo does with detector boxes
MEAN = 255 * np.array([0.485, 0.456, 0.406])  # input normalisation of HaMeR and WiLoR
STD = 255 * np.array([0.229, 0.224, 0.225])


@dataclass
class Sample:
    """A square hand crop. Joints and MANO parameters are in the crop's camera frame. Left hands are
    stored mirrored, as right hands, because HaMeR and WiLoR only predict right hands."""

    key: str
    image: np.ndarray  # (CROP_SIZE, CROP_SIZE, 3) uint8 RGB
    K: np.ndarray  # (3, 3) intrinsics of the crop
    joints: np.ndarray  # (21, 3) metres; NaN where unknown (keyframes)
    mano: dict[str, np.ndarray] | None  # global_orient (3,), hand_pose (45,), betas (10,)
    meta: dict[str, str]

    @classmethod
    def crop(
        cls,
        key: str,
        image: np.ndarray,
        camera: Camera,
        joints: np.ndarray,
        side: str,
        meta: dict[str, str],
        mano: dict[str, np.ndarray] | None = None,
    ) -> "Sample":
        """The crop around the projected joints. ``joints`` and ``mano`` are in the world frame of ``camera``."""
        pixels = camera.project(joints)
        pixels = pixels[np.isfinite(pixels).all(-1)]
        low, high = pixels.min(0), pixels.max(0)
        w, h = high - low
        width = max(int(round(CROP_SCALE * max(4 / 3 * w, h))), 1)  # the box grown to 3:4, then its longer side
        x0 = int(round((low[0] + high[0]) / 2 - width / 2))
        y0 = int(round((low[1] + high[1]) / 2 - width / 2))
        patch = _patch(image, x0, y0, width)
        crop = cv2.resize(patch, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

        # where an image pixel lands in the crop, with pixel centres as cv2.resize places them
        s = CROP_SIZE / width
        to_crop = np.array(
            [
                [s, 0, s * (0.5 - x0) - 0.5],
                [0, s, s * (0.5 - y0) - 0.5],
                [0, 0, 1],
            ]
        )
        if mano is not None:
            orient = Rotation.from_matrix(camera.R) * Rotation.from_rotvec(mano["global_orient"])
            mano = {**mano, "global_orient": orient.as_rotvec()}
        sample = cls(key, crop, to_crop @ camera.K, camera.to_camera(joints), mano, {**meta, "side": side})
        if side == "left":
            return sample.mirrored()
        return sample

    def mirrored(self) -> "Sample":
        """The same hand in the horizontally flipped crop: a left hand becomes a right hand."""
        image = np.ascontiguousarray(self.image[:, ::-1])
        K = self.K.copy()
        K[0, 2] = CROP_SIZE - 1 - K[0, 2]
        joints = self.joints * [-1, 1, 1]
        mano = None
        if self.mano is not None:
            pose = self.mano["hand_pose"].reshape(15, 3) * MIRROR
            mano = {**self.mano, "global_orient": self.mano["global_orient"] * MIRROR, "hand_pose": pose.reshape(45)}
        return Sample(self.key, image, K, joints, mano, self.meta)

    @property
    def keypoints(self) -> np.ndarray:
        """The joints projected into the crop, (21, 2) pixels."""
        uvw = self.joints @ self.K.T
        return uvw[:, :2] / uvw[:, 2:]

    def save(self, directory: Path) -> None:
        fields: dict[str, Any] = {"key": self.key, "meta": json.dumps(self.meta), "image": self.image}
        fields |= {"K": self.K, "joints": self.joints, **(self.mano or {})}
        np.savez_compressed(directory / (self.key.replace("/", "_") + ".npz"), **fields)

    @classmethod
    def load(cls, path: Path) -> "Sample":
        with np.load(path) as f:
            mano = None
            if "betas" in f:
                mano = {name: f[name] for name in ("global_orient", "hand_pose", "betas")}
            return cls(str(f["key"]), f["image"], f["K"], f["joints"], mano, json.loads(str(f["meta"])))


def write_samples(samples: Iterable[Sample], directory: Path) -> int:
    """Save the samples in ``directory``, replacing those of an earlier build; returns how many there are."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.npz"):
        old.unlink()
    count = 0
    for sample in samples:
        sample.save(directory)
        count += 1
    return count


class SampleFolder(Dataset[dict[str, Any]]):
    """The samples saved in a directory, each turned into a model example by ``transform``."""

    def __init__(self, directory: Path, transform: Callable[[Sample], dict[str, Any]]) -> None:
        self.files = sorted(directory.glob("*.npz"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(Sample.load(self.files[index]))


def to_example(sample: Sample, angle_deg: float = 0.0, scale: float = 1.0) -> dict[str, Any]:
    """Model input and targets. A rotation or scale is applied about the crop centre (augmentation); the 3D
    targets are rolled about the camera axis to match, as in HaMeR's training. K is only kept for evaluation."""
    centre = ((CROP_SIZE - 1) / 2, (CROP_SIZE - 1) / 2)
    affine = cv2.getRotationMatrix2D(centre, angle_deg, scale)
    image = cv2.warpAffine(sample.image, affine, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)
    keypoints = sample.keypoints @ affine[:, :2].T + affine[:, 2]
    roll = Rotation.from_euler("z", -angle_deg, degrees=True)  # the image rotation, in camera coordinates
    example = {
        "key": sample.key,
        "meta": sample.meta,
        "image": _float(((image - MEAN) / STD).transpose(2, 0, 1)),
        "keypoints": _float(keypoints / CROP_SIZE - 0.5),  # normalised crop coordinates, as the models predict
        "joints": _float(roll.apply(sample.joints)),
        "K": _float(sample.K),
    }
    if sample.mano is not None:
        orient = roll * Rotation.from_rotvec(sample.mano["global_orient"])
        example["global_orient"] = _float(orient.as_rotvec())
        example["hand_pose"] = _float(sample.mano["hand_pose"])
        example["betas"] = _float(sample.mano["betas"])
    return example


def random_example(sample: Sample, rotation_deg: float, scale_jitter: float) -> dict[str, Any]:
    """A training example, randomly rotated and scaled. Uses torch's generator, which DataLoader seeds per worker."""
    angle = rotation_deg * (2 * torch.rand(1).item() - 1)
    scale = 1 + scale_jitter * (2 * torch.rand(1).item() - 1)
    return to_example(sample, angle, scale)


def _patch(image: np.ndarray, x0: int, y0: int, width: int) -> np.ndarray:
    """The square region of ``image`` with top-left corner (x0, y0); black outside the image."""
    patch = np.zeros((width, width, 3), np.uint8)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + width, image.shape[1]), min(y0 + width, image.shape[0])
    if xa < xb and ya < yb:
        patch[ya - y0 : yb - y0, xa - x0 : xb - x0] = image[ya:yb, xa:xb]
    return patch


def _float(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)
