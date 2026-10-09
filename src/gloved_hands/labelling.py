"""Labelling a recording: per-view estimates, a chunked multi-view fit and per-frame quality control."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gloved_hands.capture import Camera, Frame, Recording
from gloved_hands.config import Config
from gloved_hands.estimators import HandEstimate, ViewEstimator
from gloved_hands.fitting import FitResult, MultiViewFitter, Observations
from gloved_hands.geometry import triangulate
from gloved_hands.mano import HandParams, Mano

Evidence = list[tuple[Frame, list[HandEstimate | None]]]  # per frame: the frame and each camera's estimate


@dataclass
class Labels:
    """The fitted hand of one side over a recording, in the world frame."""

    session: str
    side: str
    frames: np.ndarray  # (T,) frame indices in the recording
    params: HandParams
    joints: np.ndarray  # (T, 21, 3) metres
    reprojection_px: np.ndarray  # (T,)
    depth_mm: np.ndarray  # (T,)
    passed: np.ndarray  # (T,) bool: passed quality control

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = {name: value for name, value in vars(self).items() if name != "params"}
        np.savez_compressed(path, **fields, **vars(self.params))

    @classmethod
    def load(cls, path: Path) -> "Labels":
        with np.load(path) as f:
            return cls(
                session=str(f["session"]),
                side=str(f["side"]),
                frames=f["frames"],
                params=HandParams(f["global_orient"], f["hand_pose"], f["betas"], f["transl"]),
                joints=f["joints"],
                reprojection_px=f["reprojection_px"],
                depth_mm=f["depth_mm"],
                passed=f["passed"],
            )


class Labeller:
    """MANO labels for one hand of a recording, fitted chunk by chunk with a shape shared by all chunks."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.fit
        self.estimator = ViewEstimator(cfg.paths, cfg.fit.device)
        self.hands = {side: Mano(cfg.paths.mano, side).to(cfg.fit.device) for side in cfg.fit.sides}
        self.rng = np.random.default_rng(cfg.seed)

    def label(self, recording: Recording, side: str) -> Labels:
        cfg = self.cfg
        fitter = MultiViewFitter(self.hands[side], recording.cameras, cfg)
        candidates = np.arange(0, len(recording), cfg.frame_stride)
        frames: list[np.ndarray] = []
        results: list[FitResult] = []
        previous: HandParams | None = None
        for start in range(0, len(candidates), cfg.chunk_size):
            observed = self.observe(recording, candidates[start : start + cfg.chunk_size], side, previous)
            if observed is None:
                continue
            kept, obs, init = observed
            result = fitter.fit(obs, init, fit_shape=previous is None, previous=previous)
            frames.append(kept)
            results.append(result)
            previous = result.params
        if not results:
            raise ValueError(f"the {side} hand is never seen by two cameras in {recording.session}")

        reprojection = np.concatenate([r.reprojection_px for r in results])
        depth = np.concatenate([r.depth_mm for r in results])
        params = HandParams(
            global_orient=np.concatenate([r.params.global_orient for r in results]),
            hand_pose=np.concatenate([r.params.hand_pose for r in results]),
            betas=results[0].params.betas,
            transl=np.concatenate([r.params.transl for r in results]),
        )
        # quality control; a frame without depth points (NaN) is judged on reprojection alone
        passed = (reprojection <= cfg.max_reprojection_px) & (np.isnan(depth) | (depth <= cfg.max_depth_mm))
        joints = np.concatenate([r.joints for r in results])
        return Labels(recording.session, side, np.concatenate(frames), params, joints, reprojection, depth, passed)

    def observe(
        self, recording: Recording, frames: np.ndarray, side: str, previous: HandParams | None
    ) -> tuple[np.ndarray, Observations, HandParams] | None:
        """Evidence and an initial hand for the frames in which at least two cameras see the hand.
        The shape comes from ``previous`` when given, else from HaMeR."""
        kept, evidence = [], []
        for index in frames:
            frame = recording[int(index)]
            estimates = [self.estimator(image).get(side) for image in frame.color]
            if sum(e is not None for e in estimates) >= 2:
                kept.append(int(index))
                evidence.append((frame, estimates))
        if not kept:
            return None
        if previous is not None:
            betas = previous.betas
        else:
            betas = np.mean([e.betas for _, estimates in evidence for e in estimates if e is not None], axis=0)
        cameras = recording.cameras
        return np.array(kept), self._observations(cameras, evidence), self._initial(cameras, evidence, side, betas)

    def _observations(self, cameras: list[Camera], evidence: Evidence) -> Observations:
        T, V, M, P = len(evidence), len(cameras), self.cfg.mask_points, self.cfg.depth_points
        obs = Observations(
            keypoints=np.zeros((T, V, 21, 2)),
            confidence=np.zeros((T, V)),
            mask_points=np.zeros((T, V, M, 2)),
            mask_valid=np.zeros((T, V, M), bool),
            depth_points=np.zeros((T, V, P, 3)),
            depth_valid=np.zeros((T, V, P), bool),
        )
        for t, (frame, estimates) in enumerate(evidence):
            for v, (camera, estimate) in enumerate(zip(cameras, estimates)):
                if estimate is None:
                    continue
                obs.keypoints[t, v], obs.confidence[t, v] = estimate.keypoints, estimate.score
                pixels = self._subsample(np.argwhere(estimate.mask)[:, ::-1], M)
                obs.mask_points[t, v, : len(pixels)], obs.mask_valid[t, v, : len(pixels)] = pixels, True
                points = self._depth_points(camera, frame.depth[v], estimate.mask)
                obs.depth_points[t, v, : len(points)], obs.depth_valid[t, v, : len(points)] = points, True
        return obs

    def _depth_points(self, camera: Camera, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """World points of the eroded mask interior: depth at the silhouette edge mixes hand and background."""
        size = 2 * self.cfg.erosion_px + 1
        interior = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
        pixels = self._subsample(np.argwhere(interior & (depth > 0))[:, ::-1], self.cfg.depth_points)
        return camera.backproject(pixels.astype(float), depth[pixels[:, 1], pixels[:, 0]])

    def _initial(self, cameras: list[Camera], evidence: Evidence, side: str, betas: np.ndarray) -> HandParams:
        """Articulation and orientation of the most confident view, translation from the triangulated wrist."""
        T = len(evidence)
        global_orient, hand_pose, wrist = np.zeros((T, 3)), np.zeros((T, 45)), np.zeros((T, 3))
        projections = np.stack([c.P for c in cameras])
        for t, (_, estimates) in enumerate(evidence):
            v, best = max(((v, e) for v, e in enumerate(estimates) if e is not None), key=lambda item: item[1].score)
            orient = Rotation.from_matrix(cameras[v].R).inv() * Rotation.from_rotvec(best.global_orient)  # to world
            global_orient[t], hand_pose[t] = orient.as_rotvec(), best.hand_pose
            scores = np.array([[e.score if e is not None else 0.0] for e in estimates])
            wrists = np.stack([e.keypoints[:1] if e is not None else np.zeros((1, 2)) for e in estimates])
            wrist[t] = triangulate(projections, wrists, scores)[0]
        return HandParams(global_orient, hand_pose, betas, wrist - self._wrist_at_origin(side, betas))

    @torch.no_grad()
    def _wrist_at_origin(self, side: str, betas: np.ndarray) -> np.ndarray:
        """Wrist position at zero translation; for MANO it depends on the shape only."""
        zeros = torch.zeros(1, 3, device=self.cfg.device)
        shape = torch.tensor(betas[None], dtype=torch.float32, device=self.cfg.device)
        _, joints = self.hands[side](zeros, torch.zeros(1, 45, device=self.cfg.device), shape, zeros)
        return joints[0, 0].cpu().double().numpy()

    def _subsample(self, rows: np.ndarray, count: int) -> np.ndarray:
        """At most ``count`` of the rows, drawn at random."""
        return rows[self.rng.choice(len(rows), size=min(count, len(rows)), replace=False)]
