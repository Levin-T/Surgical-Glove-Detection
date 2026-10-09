"""Training and test examples (hand crops with labels in the crop's camera frame) and their webdataset shards."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import webdataset as wds
from scipy.spatial.transform import Rotation

from gloved_hands.capture import Camera
from gloved_hands.mano import MIRROR

CROP_SIZE = 256  # the models' input; their ViT sees the central 192 columns
CROP_SCALE = 2.0  # crop side / the joints' box grown to the ViT's 3:4 shape, as WiLoR's demo does with detector boxes
SHARD_SIZE = 100  # small shards, so that the shuffle buffer mixes many recordings into every batch
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
    meta: dict[str, Any]

    @classmethod
    def crop(
        cls,
        key: str,
        image: np.ndarray,
        camera: Camera,
        joints: np.ndarray,
        side: str,
        meta: dict[str, Any],
        mano: dict[str, np.ndarray] | None = None,
    ) -> "Sample":
        """The crop around the projected joints. ``joints`` and ``mano`` are in the world frame of ``camera``."""
        pixels = camera.project(joints)
        known = np.isfinite(pixels).all(-1)
        low, high = pixels[known].min(0), pixels[known].max(0)
        w, h = high - low
        width = max(int(round(CROP_SCALE * max(4 / 3 * w, h))), 1)
        x0, y0 = (int(c) for c in np.round((low + high) / 2 - width / 2))
        crop = cv2.resize(_patch(image, x0, y0, width), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        s = CROP_SIZE / width
        to_crop = np.array([[s, 0, s * (0.5 - x0) - 0.5], [0, s, s * (0.5 - y0) - 0.5], [0, 0, 1]])  # as cv2.resize
        if mano is not None:
            orient = Rotation.from_matrix(camera.R) * Rotation.from_rotvec(mano["global_orient"])
            mano = {**mano, "global_orient": orient.as_rotvec()}
        sample = cls(key, crop, to_crop @ camera.K, camera.to_camera(joints), mano, {**meta, "side": side})
        return sample.mirrored() if side == "left" else sample

    def mirrored(self) -> "Sample":
        """The same hand in the horizontally flipped crop: a left hand becomes a right hand."""
        K = self.K.copy()
        K[0, 2] = self.image.shape[1] - 1 - K[0, 2]
        mano = None
        if self.mano is not None:
            pose = (self.mano["hand_pose"].reshape(15, 3) * MIRROR).reshape(45)
            mano = {**self.mano, "global_orient": self.mano["global_orient"] * MIRROR, "hand_pose": pose}
        return Sample(self.key, np.ascontiguousarray(self.image[:, ::-1]), K, self.joints * [-1, 1, 1], mano, self.meta)

    @property
    def keypoints(self) -> np.ndarray:
        """The joints projected into the crop, (21, 2) pixels."""
        uvw = self.joints @ self.K.T
        return uvw[:, :2] / uvw[:, 2:]

    def to_wds(self) -> dict[str, Any]:
        arrays = {"K": self.K, "joints": self.joints, **(self.mano or {})}
        return {
            "__key__": self.key.replace(".", "_"),  # webdataset splits keys at dots
            "png": self.image,
            "json": self.meta,
            "npz": {name: np.asarray(value, np.float32) for name, value in arrays.items()},
        }

    @classmethod
    def from_wds(cls, item: dict[str, Any]) -> "Sample":
        arrays = item["npz"]
        mano = {name: arrays[name] for name in ("global_orient", "hand_pose", "betas")} if "betas" in arrays else None
        return cls(item["__key__"], item["png"], arrays["K"], arrays["joints"], mano, item["json"])


def _patch(image: np.ndarray, x0: int, y0: int, width: int) -> np.ndarray:
    """The square region of ``image`` with top-left corner (x0, y0); black outside the image."""
    patch = np.zeros((width, width, 3), np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + width, image.shape[1]), min(y0 + width, image.shape[0])
    if xa < xb and ya < yb:
        patch[ya - y0 : yb - y0, xa - x0 : xb - x0] = image[ya:yb, xa:xb]
    return patch


def to_example(sample: Sample, angle_deg: float = 0.0, scale: float = 1.0) -> dict[str, Any]:
    """Model input and targets. A rotation or scale is applied about the crop centre (augmentation); the 3D
    targets are rolled about the camera axis to match, as in HaMeR's training. K is only kept for evaluation."""
    affine = cv2.getRotationMatrix2D(((CROP_SIZE - 1) / 2, (CROP_SIZE - 1) / 2), angle_deg, scale)
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
        example["global_orient"] = _float((roll * Rotation.from_rotvec(sample.mano["global_orient"])).as_rotvec())
        example["hand_pose"] = _float(sample.mano["hand_pose"])
        example["betas"] = _float(sample.mano["betas"])
    return example


def random_example(sample: Sample, rotation_deg: float, scale_jitter: float) -> dict[str, Any]:
    """Training example with a random rotation and scale."""
    angle, scale = np.random.uniform(-rotation_deg, rotation_deg), np.random.uniform(1 - scale_jitter, 1 + scale_jitter)
    return to_example(sample, angle, scale)


def write_shards(samples: Iterable[Sample], directory: Path) -> int:
    """The samples as directory/000000.tar, 000001.tar, ..., replacing earlier shards there; returns their count."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.tar"):
        old.unlink()
    count = 0
    with wds.ShardWriter(str(directory / "%06d.tar"), maxcount=SHARD_SIZE, verbose=0) as sink:
        for sample in samples:
            sink.write(sample.to_wds())
            count += 1
    return count


def read_shards(directory: Path, shuffle: bool = False) -> wds.WebDataset:
    """The samples of a shard directory, shuffled for training."""
    urls = sorted(str(path) for path in directory.glob("*.tar"))
    dataset = wds.WebDataset(urls, shardshuffle=len(urls) if shuffle else False, empty_check=False)
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.decode("rgb8").map(Sample.from_wds)


def _float(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)
