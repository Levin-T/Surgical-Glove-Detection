"""Human-annotated keyframes: exported for CVAT, read back as COCO keypoints, triangulated."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gloved_hands.capture import Recording
from gloved_hands.geometry import triangulate
from gloved_hands.mano import JOINT_NAMES


@dataclass
class Keyframes:
    """2D clicks per (session, frame, side) and camera: (21, 2) pixels, NaN where a joint is not marked visible."""

    clicks: dict[tuple[str, int, str], dict[str, np.ndarray]]

    @staticmethod
    def export(recording: Recording, frames: Iterable[int], directory: Path) -> None:
        """Images to annotate, named <session>/<camera>/<frame>.png so the annotations map back to frames."""
        for index in frames:
            for camera, image in zip(recording.cameras, recording[index].color):
                path = directory / recording.session / camera.name / f"{index:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @classmethod
    def load(cls, path: Path) -> "Keyframes":
        """A CVAT export in COCO keypoints format: one skeleton per hand, the category name contains "left" or
        "right", the point names are those of ``gloved_hands.mano.JOINT_NAMES``."""
        coco = json.loads(path.read_text())
        files = {image["id"]: Path(image["file_name"]) for image in coco["images"]}
        categories = {category["id"]: category for category in coco["categories"]}
        clicks: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
        for annotation in coco["annotations"]:
            file, category = files[annotation["image_id"]], categories[annotation["category_id"]]
            side = "left" if "left" in category["name"] else "right"
            points = np.full((21, 2), np.nan)
            for name, (x, y, visibility) in zip(category["keypoints"], np.reshape(annotation["keypoints"], (-1, 3))):
                if visibility == 2:  # visible; guesses of occluded joints are not a reference
                    points[JOINT_NAMES.index(name)] = (x, y)
            clicks.setdefault((file.parts[-3], int(file.stem), side), {})[file.parts[-2]] = points
        return cls(clicks)

    def triangulate(self, recording: Recording) -> dict[tuple[int, str], np.ndarray]:
        """World joints (21, 3) per (frame, side) of a recording; NaN where fewer than two cameras saw a joint."""
        result = {}
        for (session, frame, side), per_camera in self.clicks.items():
            cameras = [camera for camera in recording.cameras if camera.name in per_camera]
            if session != recording.session or len(cameras) < 2:
                continue
            points = np.stack([per_camera[camera.name] for camera in cameras])
            seen = np.isfinite(points).all(-1).astype(float)
            projections = np.stack([camera.P for camera in cameras])
            result[(frame, side)] = triangulate(projections, np.nan_to_num(points), seen)
        return result
