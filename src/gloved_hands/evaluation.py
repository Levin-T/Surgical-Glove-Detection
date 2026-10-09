"""Scoring a model on test shards (RQ1)."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gloved_hands.config import Config
from gloved_hands.metrics import hand_errors
from gloved_hands.model import HandPoseModel, load_pretrained
from gloved_hands.samples import read_shards, to_example


class Evaluator:
    """One model, either fine-tuned (``cfg.adapter``) or zero-shot (``cfg.model``), scored crop by crop."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.adapter:
            self.net = HandPoseModel.from_adapter(cfg.adapter, cfg.train, cfg.paths).net
        else:
            self.net = load_pretrained(cfg.model, cfg.paths)
        self.net.to(self.device).eval()

    @torch.no_grad()
    def evaluate(self, directory: Path) -> list[dict[str, Any]]:
        """One row per crop: its meta data and its errors."""
        rows = []
        loader = DataLoader(
            read_shards(directory).map(to_example),
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.workers,
        )
        for batch in loader:
            out = self.net.forward_step({"img": batch["image"].to(self.device)})
            pred_cam = out["pred_cam"].float().cpu().numpy()
            translation = camera_translation(pred_cam, batch["K"].numpy(), batch["image"].shape[-1])
            predicted = out["pred_keypoints_3d"].float().cpu().numpy() + translation[:, None]
            for i, key in enumerate(batch["key"]):
                meta = {name: values[i] for name, values in batch["meta"].items()}
                rows.append({"key": key, **meta, **hand_errors(predicted[i], batch["joints"][i].numpy())})
        return rows


def camera_translation(pred_cam: np.ndarray, K: np.ndarray, size: int) -> np.ndarray:
    """The weak-perspective camera (s, tx, ty) of HaMeR and WiLoR as a translation (B, 3) in the crop's real
    camera, i.e. HaMeR's cam_crop_to_full with the crop's own focal length and principal point."""
    scaled = size * pred_cam[:, 0]
    tx = pred_cam[:, 1] + 2 * (size / 2 - K[:, 0, 2]) / scaled
    ty = pred_cam[:, 2] + 2 * (size / 2 - K[:, 1, 2]) / scaled
    return np.stack([tx, ty, 2 * K[:, 0, 0] / scaled], -1)
