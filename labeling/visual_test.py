"""Draws a session's labels onto its images so you can see whether they sit on the hands.

Reads data/labeled/<session>.npz and draws every label (the fitted joints projected into that
camera) as a hand skeleton onto its image: right hands red, left hands blue. A camera image
without a skeleton means that hand was dropped there (not seen or fit error too large).

Writes outputs/visual_test/<session>/<frame>.png, all cameras side by side, and the same
frames as <session>.mp4.

python -m labeling.visual_test <session>
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # the project folder

BONES = [(0, 1), (1, 2), (2, 3), (3, 4),  # thumb
         (0, 5), (5, 6), (6, 7), (7, 8),  # index
         (0, 9), (9, 10), (10, 11), (11, 12),  # middle
         (0, 13), (13, 14), (14, 15), (15, 16),  # ring
         (0, 17), (17, 18), (18, 19), (19, 20)]  # little finger
COLORS = {True: (0, 0, 255), False: (255, 0, 0)}  # BGR: right red, left blue
SCALE = 0.5  # the side-by-side images are shrunk to keep the video small
FPS = 15


class VisualTest:
    def __init__(self, session):
        self.session = session
        self.raw = ROOT / "data" / "raw"
        self.output = ROOT / "outputs" / "visual_test" / session
        labels = np.load(ROOT / "data" / "labeled" / f"{session}.npz")
        self.labels = defaultdict(list)  # image path -> [(is_right, keypoints)]
        for image, is_right, keypoints in zip(labels["image"], labels["is_right"], labels["keypoints_2d"]):
            self.labels[str(image)].append((bool(is_right), keypoints))
        self.cameras = sorted(p.name for p in (self.raw / session).glob("cam*"))
        self.frames = len(list((self.raw / session / "cam0" / "color").glob("*.png")))

    def draw(self, image, is_right, keypoints):
        color = COLORS[is_right]
        points = [tuple(int(round(v)) for v in p) for p in keypoints]
        for a, b in BONES:
            cv2.line(image, points[a], points[b], color, 2)
        for p in points:
            cv2.circle(image, p, 3, color, -1)
        cv2.putText(image, "R" if is_right else "L", points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def frame_image(self, frame):
        """All cameras of one frame side by side, with their labels drawn on."""
        views = []
        for camera in self.cameras:
            path = f"{self.session}/{camera}/color/{frame:06d}.png"
            image = cv2.imread(str(self.raw / path))
            for is_right, keypoints in self.labels.get(path, []):
                self.draw(image, is_right, keypoints)
            cv2.putText(image, f"{camera}  frame {frame}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            views.append(cv2.resize(image, None, fx=SCALE, fy=SCALE))
        return np.hstack(views)

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        video = None
        for frame in range(self.frames):
            image = self.frame_image(frame)
            cv2.imwrite(str(self.output / f"{frame:06d}.png"), image)
            if video is None:
                size = (image.shape[1], image.shape[0])
                video = cv2.VideoWriter(str(self.output / f"{self.session}.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
            video.write(image)
        video.release()
        print(f"{len(self.labels)} labeled images over {self.frames} frames and {len(self.cameras)} cameras")
        print(f"Saved to {self.output}")


if __name__ == "__main__":
    VisualTest(sys.argv[1]).run()