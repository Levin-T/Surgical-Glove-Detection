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
        confidence, mask_valid, depth_valid = self.confidence.copy(), self.mask_valid.copy(), self.depth_valid.copy()
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


def squared_distances(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise squared distances (..., N, M) between a (..., N, D) and b (..., M, D)."""
    centre = a.detach().mean(-2, keepdim=True)  # better float32 precision; the distances do not change
    a, b = a - centre, b - centre
    return (
        a.square().sum(-1)[..., :, None] + b.square().sum(-1)[..., None, :] - 2 * a @ b.transpose(-1, -2)
    ).clamp_min(0)


class MultiViewFitter:
    """Fits MANO to all cameras at once: robust reprojection, silhouette, depth and temporal smoothness."""

    def __init__(self, hand: nn.Module, cameras: list[Camera], cfg: FitConfig) -> None:
        self.hand = hand  # on cfg.device
        self.cfg = cfg
        self.K = self._tensor(np.stack([camera.K for camera in cameras]))
        self.R = self._tensor(np.stack([camera.R for camera in cameras]))
        self.t = self._tensor(np.stack([camera.t for camera in cameras]))

    def fit(
        self, obs: Observations, init: HandParams, fit_shape: bool, previous: HandParams | None = None
    ) -> FitResult:
        """Fit T frames from ``init``. ``previous`` (the preceding chunk) keeps consecutive chunks smooth."""
        o = {name: self._tensor(value) for name, value in vars(obs).items()}
        p = {name: self._tensor(value).requires_grad_() for name, value in vars(init).items()}
        p["betas"].requires_grad_(fit_shape)
        # first place the hand (orientation, translation) on the 2D keypoints, then fit everything with all terms
        self._optimise(p, ["global_orient"], o, previous, full=False, iterations=self.cfg.iters_placement)
        free = ["global_orient", "hand_pose"] + (["betas"] if fit_shape else [])
        self._optimise(p, free, o, previous, full=True, iterations=self.cfg.iters_full)
        return self._result(p, o)

    def project(self, points: Tensor) -> Tensor:
        """World points (T, N, 3) into every camera: pixels (T, V, N, 2)."""
        camera = torch.einsum("vij,tnj->tvni", self.R, points) + self.t[None, :, None]
        xy = camera[..., :2] / camera[..., 2:].clamp_min(1e-3)
        return torch.einsum("vij,tvnj->tvni", self.K[:, :2, :2], xy) + self.K[None, :, None, :2, 2]

    def _optimise(
        self,
        p: dict[str, Tensor],
        free: list[str],
        o: dict[str, Tensor],
        previous: HandParams | None,
        full: bool,
        iterations: int,
    ) -> None:
        """Adam on the ``free`` parameters and the translation, which moves in metres and gets its own step size."""
        groups = [
            {"params": [p[name] for name in free], "lr": self.cfg.lr},
            {"params": [p["transl"]], "lr": self.cfg.lr_translation},
        ]
        optimiser = torch.optim.Adam(groups)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, iterations)
        for _ in range(iterations):
            optimiser.zero_grad()
            self._loss(p, o, previous, full).backward()
            optimiser.step()
            schedule.step()

    def _loss(self, p: dict[str, Tensor], o: dict[str, Tensor], previous: HandParams | None, full: bool) -> Tensor:
        cfg = self.cfg
        vertices, joints = self._hand(p)
        loss = self._reprojection(joints, o)
        if full:
            loss = (
                loss
                + cfg.w_silhouette * self._silhouette(vertices, o)
                + cfg.w_depth * self._depth(vertices, o)
                + self._smoothness(p, previous)
                + cfg.w_pose_prior * p["hand_pose"].square().sum(-1).mean()
                + cfg.w_shape_prior * p["betas"].square().sum()
            )
        return loss

    def _reprojection(self, joints: Tensor, o: dict[str, Tensor]) -> Tensor:
        squared = (self.project(joints) - o["keypoints"]).square().sum(-1)  # (T, V, 21)
        weight = o["confidence"][..., None]
        return (weight * geman_mcclure(squared, self.cfg.sigma_px)).sum() / (21 * weight.sum()).clamp_min(1e-6)

    def _silhouette(self, vertices: Tensor, o: dict[str, Tensor]) -> Tensor:
        """Symmetric chamfer between the projected vertices and the glove-mask pixels (no renderer needed)."""
        valid = o["mask_valid"]  # (T, V, M)
        seen = valid.any(-1)  # (T, V)
        squared = squared_distances(self.project(vertices), o["mask_points"])  # (T, V, vertices, M)
        squared = squared.masked_fill(~valid[:, :, None], float("inf"))
        vertex_to_mask = squared.min(-1).values.masked_fill(~seen[..., None], 0)  # every vertex inside the mask
        mask_to_vertex = squared.min(-2).values.masked_fill(~valid, 0)  # every mask pixel covered by the mesh
        sigma = self.cfg.sigma_px
        inside = (geman_mcclure(vertex_to_mask, sigma).mean(-1) * seen).sum() / seen.sum().clamp_min(1)
        covered = (geman_mcclure(mask_to_vertex, sigma) * valid).sum() / valid.sum().clamp_min(1)
        return inside + covered

    def _depth(self, vertices: Tensor, o: dict[str, Tensor]) -> Tensor:
        """Point-to-surface distance (to the nearest vertex) of the depth points: places the hand in depth."""
        points, valid = o["depth_points"].flatten(1, 2), o["depth_valid"].flatten(1, 2)
        squared = squared_distances(vertices, points).min(-2).values  # (T, V * P)
        return (geman_mcclure(squared, self.cfg.sigma_depth_m) * valid).sum() / valid.sum().clamp_min(1)

    def _smoothness(self, p: dict[str, Tensor], previous: HandParams | None) -> Tensor:
        rotations, transl = torch.cat([p["global_orient"], p["hand_pose"]], -1), p["transl"]
        if previous is not None:  # continue from the last frame of the preceding chunk
            last = np.r_[previous.global_orient[-1], previous.hand_pose[-1]]
            rotations = torch.cat([self._tensor(last)[None], rotations])
            transl = torch.cat([self._tensor(previous.transl[-1])[None], transl])
        if len(transl) < 2:
            return transl.sum() * 0
        R = rotvec_to_matrix(rotations.reshape(len(rotations), 16, 3))  # on matrices: axis-angle jumps at 180 degrees
        return (
            self.cfg.w_smooth_rotation * (R[1:] - R[:-1]).square().sum((1, 2, 3)).mean()
            + self.cfg.w_smooth_translation * (transl[1:] - transl[:-1]).square().sum(-1).mean()
        )

    def _hand(self, p: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        frames = len(p["transl"])
        return self.hand(p["global_orient"], p["hand_pose"], p["betas"].expand(frames, -1), p["transl"])

    @torch.no_grad()
    def _result(self, p: dict[str, Tensor], o: dict[str, Tensor]) -> FitResult:
        vertices, joints = self._hand(p)
        seen = o["confidence"] > 0
        error = (self.project(joints) - o["keypoints"]).norm(dim=-1).mean(-1)  # (T, V)
        reprojection = (error * seen).sum(1) / seen.sum(1).clamp_min(1)
        valid = o["depth_valid"].flatten(1, 2)
        distance = squared_distances(vertices, o["depth_points"].flatten(1, 2)).min(-2).values.sqrt()
        depth = distance.masked_fill(~valid, float("nan")).nanmedian(-1).values * 1000
        params = HandParams(**{name: value.detach().cpu().double().numpy() for name, value in p.items()})
        return FitResult(params, joints.cpu().double().numpy(), reprojection.cpu().numpy(), depth.cpu().numpy())

    def _tensor(self, value: np.ndarray) -> Tensor:
        """A copy on the fitting device, so optimising never writes into the caller's arrays."""
        return torch.tensor(value, dtype=torch.bool if value.dtype == bool else torch.float32, device=self.cfg.device)
