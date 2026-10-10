"""Joint MANO fit over all cameras: one hand per frame, one shape for the whole recording."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor, nn

from gloved_hands.capture import Camera
from gloved_hands.config import FitConfig
from gloved_hands.geometry import rotvec_to_matrix
from gloved_hands.mano import HandParams


@dataclass
class Observations:
    """Per-view evidence for T frames and V cameras. Arrays are padded; the masks mark real entries."""

    keypoints: np.ndarray  # (T, V, 21, 2) pixels
    confidence: np.ndarray  # (T, V) detector score; 0 where the camera did not see the hand
    mask_points: np.ndarray  # (T, V, M, 2) pixels inside the glove mask
    mask_valid: np.ndarray  # (T, V, M) bool
    depth_points: np.ndarray  # (T, V, P, 3) world points from the eroded mask interior
    depth_valid: np.ndarray  # (T, V, P) bool

    def without_view(self, view: int) -> "Observations":
        """The same evidence with one camera removed (leave-one-view-out)."""
        confidence = self.confidence.copy()
        mask_valid = self.mask_valid.copy()
        depth_valid = self.depth_valid.copy()
        confidence[:, view] = 0
        mask_valid[:, view] = False
        depth_valid[:, view] = False
        return replace(self, confidence=confidence, mask_valid=mask_valid, depth_valid=depth_valid)


@dataclass
class FitResult:
    params: HandParams
    joints: np.ndarray  # (T, 21, 3) world
    reprojection_px: np.ndarray  # (T,) mean joint error over the cameras that saw the hand
    depth_mm: np.ndarray  # (T,) median distance of the depth points to the mesh; NaN without depth


def geman_mcclure(squared: Tensor, sigma: float) -> Tensor:
    """Robust kernel in [0, 1): far outliers stop pulling."""
    return squared / (squared + sigma**2)


def weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def squared_distances(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise squared distances (..., N, M) between points a (..., N, D) and b (..., M, D)."""
    centre = a.detach().mean(-2, keepdim=True)  # centring keeps float32 precise; the distances do not change
    a, b = a - centre, b - centre
    squared = a.square().sum(-1)[..., :, None] + b.square().sum(-1)[..., None, :] - 2 * a @ b.transpose(-1, -2)
    return squared.clamp_min(0)


class MultiViewFitter:
    """Fits MANO to all cameras at once: robust reprojection, silhouette, depth and temporal smoothness."""

    def __init__(self, hand: nn.Module, cameras: list[Camera], cfg: FitConfig) -> None:
        self.hand = hand  # MANO layer on cfg.device
        self.cfg = cfg
        self.R = self._tensor(np.stack([camera.R for camera in cameras]))  # (V, 3, 3)
        self.t = self._tensor(np.stack([camera.t for camera in cameras]))  # (V, 3)
        self.focal = self._tensor(np.stack([camera.K.diagonal()[:2] for camera in cameras]))  # (V, 2)
        self.centre = self._tensor(np.stack([camera.K[:2, 2] for camera in cameras]))  # (V, 2)

    def fit(
        self, obs: Observations, init: HandParams, fit_shape: bool, previous: HandParams | None = None
    ) -> FitResult:
        """Fit T frames from ``init``. ``previous`` (the preceding chunk) keeps consecutive chunks smooth."""
        evidence = {name: self._tensor(value) for name, value in vars(obs).items()}
        params = {name: self._tensor(value).requires_grad_() for name, value in vars(init).items()}
        params["betas"].requires_grad_(fit_shape)

        # first place the hand on the 2D keypoints (orientation and translation), then fit everything with all terms
        self._optimise(params, ["global_orient"], evidence, previous, full=False, steps=self.cfg.iters_placement)
        free = ["global_orient", "hand_pose", "betas"] if fit_shape else ["global_orient", "hand_pose"]
        self._optimise(params, free, evidence, previous, full=True, steps=self.cfg.iters_full)
        return self._result(params, evidence)

    def project(self, points: Tensor) -> Tensor:
        """World points (T, N, 3) into every camera: pixels (T, V, N, 2)."""
        in_camera = points[:, None] @ self.R.transpose(1, 2) + self.t[:, None]  # (T, V, N, 3)
        xy = in_camera[..., :2] / in_camera[..., 2:].clamp_min(1e-3)
        return xy * self.focal[:, None] + self.centre[:, None]

    def _optimise(
        self,
        params: dict[str, Tensor],
        free: list[str],
        evidence: dict[str, Tensor],
        previous: HandParams | None,
        full: bool,
        steps: int,
    ) -> None:
        """Adam on the ``free`` parameters and the translation, which moves in metres and gets its own step size."""
        groups = [
            {"params": [params[name] for name in free], "lr": self.cfg.lr},
            {"params": [params["transl"]], "lr": self.cfg.lr_translation},
        ]
        optimiser = torch.optim.Adam(groups)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
        for _ in range(steps):
            optimiser.zero_grad()
            self._loss(params, evidence, previous, full).backward()
            optimiser.step()
            schedule.step()

    def _loss(
        self, params: dict[str, Tensor], evidence: dict[str, Tensor], previous: HandParams | None, full: bool
    ) -> Tensor:
        cfg = self.cfg
        vertices, joints = self._hand(params)
        loss = self._reprojection(joints, evidence)
        if full:
            loss = loss + cfg.w_silhouette * self._silhouette(vertices, evidence)
            loss = loss + cfg.w_depth * self._depth(vertices, evidence)
            loss = loss + self._smoothness(params, previous)
            loss = loss + cfg.w_pose_prior * params["hand_pose"].square().sum(-1).mean()
            loss = loss + cfg.w_shape_prior * params["betas"].square().sum()
        return loss

    def _reprojection(self, joints: Tensor, evidence: dict[str, Tensor]) -> Tensor:
        """Robust keypoint error, weighted by the detector's confidence in each view."""
        squared = (self.project(joints) - evidence["keypoints"]).square().sum(-1)  # (T, V, 21)
        weights = evidence["confidence"][..., None].expand_as(squared)
        return weighted_mean(geman_mcclure(squared, self.cfg.sigma_px), weights)

    def _silhouette(self, vertices: Tensor, evidence: dict[str, Tensor]) -> Tensor:
        """Symmetric chamfer between the projected vertices and the glove-mask pixels (no renderer needed)."""
        valid = evidence["mask_valid"]  # (T, V, M)
        seen = valid.any(-1)  # (T, V)
        squared = squared_distances(self.project(vertices), evidence["mask_points"])  # (T, V, vertices, M)
        squared = squared.masked_fill(~valid[:, :, None], float("inf"))  # padding is never the nearest point

        # every vertex should fall inside the mask ...
        vertex_to_mask = squared.min(-1).values.masked_fill(~seen[..., None], 0)  # (T, V, vertices)
        inside = weighted_mean(geman_mcclure(vertex_to_mask, self.cfg.sigma_px).mean(-1), seen.float())
        # ... and every mask pixel should be covered by the mesh
        mask_to_vertex = squared.min(-2).values.masked_fill(~valid, 0)  # (T, V, M)
        covered = weighted_mean(geman_mcclure(mask_to_vertex, self.cfg.sigma_px), valid.float())
        return inside + covered

    def _depth(self, vertices: Tensor, evidence: dict[str, Tensor]) -> Tensor:
        """Distance of the depth points to the mesh (to its nearest vertex): places the hand in depth."""
        points = evidence["depth_points"].flatten(1, 2)  # (T, V * P, 3)
        valid = evidence["depth_valid"].flatten(1, 2)
        squared = squared_distances(vertices, points).min(-2).values  # (T, V * P)
        return weighted_mean(geman_mcclure(squared, self.cfg.sigma_depth_m), valid.float())

    def _smoothness(self, params: dict[str, Tensor], previous: HandParams | None) -> Tensor:
        """Squared change from frame to frame, of the rotations (as matrices: axis-angle jumps at 180 degrees)
        and of the translation."""
        rotations = torch.cat([params["global_orient"], params["hand_pose"]], -1)  # (T, 48)
        transl = params["transl"]
        if previous is not None:  # continue from the last frame of the preceding chunk
            last_rotation = np.concatenate([previous.global_orient[-1], previous.hand_pose[-1]])
            rotations = torch.cat([self._tensor(last_rotation)[None], rotations])
            transl = torch.cat([self._tensor(previous.transl[-1])[None], transl])
        if len(transl) < 2:
            return transl.sum() * 0
        matrices = rotvec_to_matrix(rotations.reshape(len(rotations), 16, 3))
        rotation_change = (matrices[1:] - matrices[:-1]).square().sum((1, 2, 3)).mean()
        translation_change = (transl[1:] - transl[:-1]).square().sum(-1).mean()
        return self.cfg.w_smooth_rotation * rotation_change + self.cfg.w_smooth_translation * translation_change

    def _hand(self, params: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        frames = len(params["transl"])
        betas = params["betas"].expand(frames, -1)
        return self.hand(params["global_orient"], params["hand_pose"], betas, params["transl"])

    @torch.no_grad()
    def _result(self, params: dict[str, Tensor], evidence: dict[str, Tensor]) -> FitResult:
        vertices, joints = self._hand(params)

        seen = evidence["confidence"] > 0  # (T, V)
        error = (self.project(joints) - evidence["keypoints"]).norm(dim=-1).mean(-1)  # (T, V)
        reprojection = (error * seen).sum(1) / seen.sum(1).clamp_min(1)

        valid = evidence["depth_valid"].flatten(1, 2)
        distance = squared_distances(vertices, evidence["depth_points"].flatten(1, 2)).min(-2).values.sqrt()
        depth = distance.masked_fill(~valid, float("nan")).nanmedian(-1).values * 1000

        fitted = HandParams(**{name: value.detach().cpu().double().numpy() for name, value in params.items()})
        return FitResult(fitted, joints.cpu().double().numpy(), reprojection.cpu().numpy(), depth.cpu().numpy())

    def _tensor(self, value: np.ndarray) -> Tensor:
        """A copy on the fitting device, so optimising never writes into the caller's arrays."""
        dtype = torch.bool if value.dtype == bool else torch.float32
        return torch.tensor(value, dtype=dtype, device=self.cfg.device)
