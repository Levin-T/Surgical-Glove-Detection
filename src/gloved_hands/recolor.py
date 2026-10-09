"""Arm B's glove appearance: bare-hand pixels recoloured to look like a glove recorded in the rig."""

from pathlib import Path

import cv2
import numpy as np


class GloveRecolorer:
    """Colour statistics (Lab) of one glove variant, from rig images of the glove alone."""

    def __init__(self, directory: Path) -> None:
        """From the <name>.png images in ``directory`` and their <name>_mask.png glove masks."""
        images = sorted(path for path in directory.glob("*.png") if not path.stem.endswith("_mask"))
        masks = [read_rgb(path.with_name(f"{path.stem}_mask.png"))[..., 0] > 127 for path in images]
        pixels = np.concatenate([_lab(read_rgb(path))[mask] for path, mask in zip(images, masks)])
        self.mean, self.std = pixels.mean(0), np.maximum(pixels.std(0), 1.0)

    def __call__(self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The masked hand in glove colours: the hand's shading at the glove's lightness, skin texture
        suppressed, some gloss. Hue, texture suppression and gloss are sampled per image."""
        weight = mask.astype(np.float32)
        lightness = _lab(image)[..., 0]
        smooth = cv2.GaussianBlur(lightness * weight, (0, 0), 3) / np.maximum(cv2.GaussianBlur(weight, (0, 0), 3), 1e-6)
        lightness = lightness + rng.uniform(0.5, 0.9) * (smooth - lightness)  # suppress skin texture
        inside = lightness[mask]
        lightness = self.mean[0] + (lightness - inside.mean()) * self.std[0] / max(inside.std(), 1e-3)
        highlight = np.clip((lightness - self.mean[0]) / (2 * self.std[0]), 0, 1) ** 2
        lightness = np.clip(lightness + rng.uniform(0.1, 0.5) * highlight * (100 - lightness), 0, 100)  # gloss
        a, b = self.mean[1:] + rng.normal(0, 0.5, 2) * self.std[1:]
        glove = _rgb(np.stack([lightness, np.full_like(lightness, a), np.full_like(lightness, b)], -1))
        alpha = cv2.GaussianBlur(weight, (0, 0), 1.5)[..., None]  # soft edge
        return np.clip(alpha * glove + (1 - alpha) * image, 0, 255).astype(np.uint8)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB) * 255
