"""Calibrates the rig: where every camera sits, all in one world frame (cam0's).

Capture: takes SHOTS synchronized image sets, one every INTERVAL_S seconds, with a countdown and
a beep. Between beeps, move the board to a new place and angle and hold it still (or lay it down)
when the beep sounds. Every shot should show the board to at least two cameras, a few shots to
all of them. Vary height and tilt; a board that only lies flat on the table gives weak poses.

Solve: finds the board pose in every image (solvePnP with the factory intrinsics and no
distortion, the pinhole model the labeling pipeline uses), chains the cameras to cam0, then
refines all camera and board poses together so the board corners reproject as well as possible
in every image at once.

Writes data/calibration/<name>/calibration.json in the format labeling/run.py reads
(x_cam = R @ x_world + t, meters, world = cam0) and prints how good it is. Copy that file into
every session recorded without moving the cameras.

python "camera capturing/calibrate.py" <name>           capture and solve
python "camera capturing/calibrate.py" <name> --solve   solve again from the saved images
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from board import Board

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.camera import Camera  # noqa: E402
from labeling.triangulate import Triangulator  # noqa: E402

SHOTS = 12
INTERVAL_S = 6  # time to move the board between shots
GOOD_PX, OK_PX = 1.0, 2.0  # reprojection error per camera
GOOD_MM, OK_MM = 1.0, 2.0  # error of the triangulated square edges


def to_matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.ravel(rvec)).as_matrix()
    T[:3, 3] = np.ravel(tvec)
    return T


def to_vector(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def verdict(value, good, ok):
    return "good" if value < good else "ok" if value < ok else "BAD"


def evaluate(board, cameras, seen, num_shots):
    """How well a calibration explains the board shots, measured the way the labeling pipeline works:
    board corners are triangulated with the pipeline's Triangulator, then
    reprojected into each camera (pixels) and their square edges compared with the true size (mm).

    cameras: one Camera per cam<k>. seen: {(camera, shot): {corner id: pixel}}. Returns report lines."""
    V, n = len(cameras), len(board.corners)
    triangulator = Triangulator(cameras)
    edge_errors, px_errors = [], {k: [] for k in range(V)}
    for s in range(num_shots):
        points = np.zeros((V, n, 2))
        weights = np.zeros((V, n))
        for k in range(V):
            for i, pixel in seen.get((k, s), {}).items():
                points[k, i], weights[k, i] = pixel, 1
        world = triangulator.triangulate(points, weights)
        found = {i: world[i].numpy().astype(np.float64) for i in range(n) if weights[:, i].sum() >= 2}
        edge_errors += board.square_errors_mm(found)
        for k in range(V):
            ids = [i for i in found if weights[k, i]]
            if ids:
                projected = cameras[k].project(world[ids]).numpy()
                px_errors[k] += list(np.linalg.norm(projected - points[k, ids], axis=1))
    lines = []
    for k in range(V):
        rms = float(np.sqrt(np.mean(np.square(px_errors[k])))) if px_errors[k] else float("nan")
        shots = sum((k, s) in seen for s in range(num_shots))
        lines.append(f"cam{k}: board in {shots}/{num_shots} shots, "
                     f"reprojection {rms:.2f} px ({verdict(rms, GOOD_PX, OK_PX)})")
    mean_mm = float(np.mean(edge_errors)) if edge_errors else float("nan")
    lines.append(f"square edges: {mean_mm:.2f} mm mean, {max(edge_errors, default=float('nan')):.2f} mm max "
                 f"over {len(edge_errors)} edges ({verdict(mean_mm, GOOD_MM, OK_MM)})")
    return lines


class ShotCapture:
    """Takes the timed shots and saves them with the intrinsics."""

    def __init__(self, folder):
        self.folder = folder
        self.board = Board()

    def run(self):
        from rig import Rig  # needs the cameras; --solve works without them
        rig = Rig()
        try:
            rig.start()
            cameras = rig.cameras()
            with open(self.folder / "cameras.json", "w") as f:
                json.dump(cameras, f, indent=2)
            for k in range(len(cameras)):
                (self.folder / f"cam{k}").mkdir(parents=True)
            for shot in range(SHOTS):
                for second in range(INTERVAL_S, 0, -1):
                    print(f"\rShot {shot + 1}/{SHOTS}: move the board, {second} ", end="", flush=True)
                    time.sleep(1)
                rig.drop_queued()
                framesets, _ = rig.grab()
                counts = []
                for k, serial in enumerate(rig.serials):
                    image = np.asanyarray(framesets[serial].get_color_frame().get_data())
                    cv2.imwrite(str(self.folder / f"cam{k}" / f"shot_{shot:02d}.png"), image)
                    counts.append(len(self.board.detect(image)))
                print(f"\a\rShot {shot + 1}/{SHOTS}: corners per camera {counts}          ")
        finally:
            rig.stop()


class RigCalibrator:
    """Turns the saved shots into one camera pose per camera."""

    def __init__(self, folder):
        self.folder = folder
        self.board = Board()
        with open(folder / "cameras.json") as f:
            self.cameras = json.load(f)
        self.K = [np.array(c["K"]) for c in self.cameras]
        self.V = len(self.cameras)
        self.shots = sorted(p.stem for p in (folder / "cam0").glob("shot_*.png"))
        self.seen = {}  # (camera, shot index) -> {corner id: pixel}
        for k in range(self.V):
            for s, shot in enumerate(self.shots):
                found = self.board.detect(cv2.imread(str(folder / f"cam{k}" / f"{shot}.png")))
                if found:
                    self.seen[k, s] = found

    def board_pose(self, k, s):
        """Board -> camera k in shot s, from that image alone."""
        ids = sorted(self.seen[k, s])
        pixels = np.array([self.seen[k, s][i] for i in ids])
        _, rvec, tvec = cv2.solvePnP(self.board.corners[ids], pixels, self.K[k], None)
        return to_matrix(rvec, tvec)

    def initial_guess(self):
        """Chains every camera to cam0 through the shot both saw best, and places the board in each shot."""
        single = {key: self.board_pose(*key) for key in self.seen}
        cams = {0: np.eye(4)}  # world (= cam0) -> camera k
        while len(cams) < self.V:
            options = [(min(len(self.seen[j, s]), len(self.seen[k, s])), k, j, s)
                       for (k, s) in self.seen if k not in cams
                       for j in cams if (j, s) in self.seen]
            if not options:
                missing = [k for k in range(self.V) if k not in cams]
                raise SystemExit(f"cam {missing} never saw the board together with a calibrated camera")
            _, k, j, s = max(options)
            cams[k] = single[k, s] @ np.linalg.inv(single[j, s]) @ cams[j]
        boards = {}  # shot -> board -> world
        for (k, s) in sorted(self.seen, key=lambda key: -len(self.seen[key])):
            if s not in boards:
                boards[s] = np.linalg.inv(cams[k]) @ single[k, s]
        return [cams[k] for k in range(self.V)], boards

    def residuals(self, x, shots):
        """Reprojection errors (pixels) of all seen board corners, for cameras 1.. and board poses x."""
        cams = [np.eye(4)] + [to_matrix(x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6]) for i in range(self.V - 1)]
        offset = 6 * (self.V - 1)
        errors = []
        for n, s in enumerate(shots):
            board = to_matrix(x[offset + 6 * n:offset + 6 * n + 3], x[offset + 6 * n + 3:offset + 6 * n + 6])
            for k in range(self.V):
                if (k, s) not in self.seen:
                    continue
                ids = sorted(self.seen[k, s])
                T = cams[k] @ board
                points = self.board.corners[ids] @ T[:3, :3].T + T[:3, 3]
                uv = points @ self.K[k].T
                errors.append((uv[:, :2] / uv[:, 2:3] - np.array([self.seen[k, s][i] for i in ids])).ravel())
        return np.concatenate(errors)

    def solve(self):
        cams, boards = self.initial_guess()
        shots = sorted(boards)
        x0 = np.concatenate([to_vector(T) for T in cams[1:]] + [to_vector(boards[s]) for s in shots])
        x = least_squares(self.residuals, x0, args=(shots,), loss="huber", f_scale=1.0).x
        return [np.eye(4)] + [to_matrix(x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6]) for i in range(self.V - 1)]

    def save(self, cams, lines):
        data = {
            "cameras": [{
                "serial": c["serial"],
                "K": c["K"],
                "R": cams[k][:3, :3].tolist(),
                "t": cams[k][:3, 3].tolist(),
                "depth_scale": c["depth_scale"],
            } for k, c in enumerate(self.cameras)],
            "world": "cam0",
            "board": {"squares": list(self.board.board.getChessboardSize()), "square_m": self.board.square},
            "report": lines,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.folder / "calibration.json", "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    folder = ROOT / "data" / "calibration" / sys.argv[1]
    if "--solve" not in sys.argv:
        if folder.exists():
            raise SystemExit(f"{folder} already exists, pick a new name or add --solve")
        folder.mkdir(parents=True)
        ShotCapture(folder).run()
    calibrator = RigCalibrator(folder)
    cams = calibrator.solve()
    cameras = [Camera(calibrator.K[k], T[:3, :3], T[:3, 3]) for k, T in enumerate(cams)]
    lines = evaluate(calibrator.board, cameras, calibrator.seen, len(calibrator.shots))
    calibrator.save(cams, lines)
    print("\n".join(lines))
    print(f"Saved {folder / 'calibration.json'}")
