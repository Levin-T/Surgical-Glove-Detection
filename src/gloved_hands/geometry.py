"""Rotations, triangulation and Procrustes alignment."""

import numpy as np
import torch
from torch import Tensor


def rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    """Axis-angle (..., 3) to rotation matrices (..., 3, 3); smooth at zero rotation."""
    theta = (rotvec.square().sum(-1) + 1e-12).sqrt()[..., None, None]
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sin(theta) / theta * skew + (1 - torch.cos(theta)) / theta**2 * skew @ skew


def triangulate(projections: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted linear triangulation of N points seen by V cameras.

    projections (V, 3, 4), points (V, N, 2) pixels, weights (V, N) with 0 for "not seen".
    Returns (N, 3), NaN where fewer than two cameras see a point.
    """
    result = np.full((points.shape[1], 3), np.nan)
    for n in range(points.shape[1]):
        seen = weights[:, n] > 0
        if seen.sum() < 2:
            continue
        P, (u, v), w = projections[seen], points[seen, n].T, weights[seen, n][:, None]
        A = np.concatenate([w * (u[:, None] * P[:, 2] - P[:, 0]), w * (v[:, None] * P[:, 2] - P[:, 1])])
        X = np.linalg.svd(A)[2][-1]
        result[n] = X[:3] / X[3]
    return result


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Source points (N, 3) after the similarity transform that best fits them to target (N, 3)."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    s, t = source - source_mean, target - target_mean
    U, S, Vt = np.linalg.svd(s.T @ t)
    d = np.array([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])  # no reflections
    rotation = (U * d) @ Vt
    scale = (S * d).sum() / (s**2).sum()
    return scale * s @ rotation + target_mean
