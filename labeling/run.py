import sys
from pathlib import Path

import cv2
import torch

from common.camera import load_cameras
from common.mano import Hand
from labeling.detector import HandDetector
from labeling.export import LabelExporter
from labeling.fit import HandFitter
from labeling.initialize import HandInitializer
from labeling.keypoints import KeypointDetector
from labeling.triangulate import Triangulator

ROOT = Path(__file__).resolve().parent.parent  # the project folder
MODELS = ROOT / "models"
SIDES = ("left", "right")


class LabelingPipeline:
    """Runs all labeling steps over one recorded session and saves its labels."""

    def __init__(self, session, subject, max_missed_frames=5):
        self.session = session
        self.subject = subject
        self.folder = ROOT / "data" / "raw" / session
        self.cameras = load_cameras(self.folder / "calibration.json")
        self.detector = HandDetector(str(MODELS / "detector.pt"))
        self.keypoints = KeypointDetector()
        self.triangulator = Triangulator(self.cameras)
        hands = {side: Hand(str(MODELS), side) for side in SIDES}
        self.initializers = {side: HandInitializer(hands[side]) for side in SIDES}
        self.fitters = {side: HandFitter(hands[side], self.cameras) for side in SIDES}
        self.exporter = LabelExporter(session, self.cameras)
        self.shape = torch.zeros(10)  # average hand shape until we fit one per subject
        self.max_missed_frames = max_missed_frames
        self.last_box = {}  # (camera, side) -> [box, frames since it was last detected]
        self.previous = {side: None for side in SIDES}  # last frame's fit for each hand

    def num_frames(self):
        return len(list((self.folder / "cam0" / "color").glob("*.png")))

    def image(self, camera, frame):
        return cv2.imread(str(self.folder / f"cam{camera}" / "color" / f"{frame:06d}.png"))

    def box(self, camera, side, found):
        """This frame's box for a hand, or the last one if it was missed only briefly."""
        if side in found:
            self.last_box[camera, side] = [found[side]["box"], 0]
        elif (camera, side) in self.last_box and self.last_box[camera, side][1] < self.max_missed_frames:
            self.last_box[camera, side][1] += 1
        else:
            return None
        return self.last_box[camera, side][0]

    def keypoints_2d(self, frame):
        """Steps 1 and 2 for all cameras: 2D keypoints and weights for each hand."""
        V = len(self.cameras)
        points = {side: torch.zeros(V, 21, 2) for side in SIDES}
        weights = {side: torch.zeros(V, 21) for side in SIDES}
        for camera in range(V):
            image = self.image(camera, frame)
            found = self.detector.detect(image)
            for side in SIDES:
                box = self.box(camera, side, found)
                if box is None:
                    continue
                keypoints, confidences = self.keypoints.predict(image, box)
                points[side][camera] = torch.as_tensor(keypoints)
                weights[side][camera] = torch.as_tensor(confidences)
        return points, weights

    def start(self, side, points, weights):
        """Where the fit starts: last frame's result, or a fresh placement (step 3)."""
        if self.previous[side] is not None:
            return self.previous[side], self.previous[side]["joints"]
        joints_3d = self.triangulator.triangulate(points, weights)
        placed = self.initializers[side].initialize(joints_3d, self.shape)
        if placed is None:
            return None, None
        rotation, translation = placed
        return {"pose": torch.zeros(45), "rotation": rotation, "translation": translation}, None

    def run_frame(self, frame):
        points, weights = self.keypoints_2d(frame)
        for side in SIDES:
            start, previous_joints = self.start(side, points[side], weights[side])
            if start is None:  # not enough of the hand seen to place it
                self.previous[side] = None
                continue
            fit = self.fitters[side].fit(points[side], weights[side], self.shape, start, previous_joints)  # step 4
            self.exporter.add(frame, side, fit)                                                            # steps 5, 6
            good = fit["errors"].nanmean() < self.exporter.max_error
            self.previous[side] = fit if good else None  # a bad fit is not used as the next start

    def run(self):
        for frame in range(self.num_frames()):
            self.run_frame(frame)
        output = ROOT / "data" / "labeled" / f"{self.session}.npz"
        self.exporter.save(output, self.shape, self.subject)
        return output


if __name__ == "__main__":
    # python -m labeling.run <session> <subject>
    print("saved", LabelingPipeline(sys.argv[1], sys.argv[2]).run())
