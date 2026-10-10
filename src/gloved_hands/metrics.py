"""RQ1 metrics in millimetres. Root error (placement), RR-MPJPE (plus rotation) and PA-MPJPE (articulation only)
separate the error components."""

from typing import Any

import numpy as np

from gloved_hands.geometry import procrustes_align
from gloved_hands.mano import JOINT_NAMES


def hand_errors(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Errors of one hand, joints (21, 3) in metres. Target joints may be NaN (not annotated)."""
    known = np.isfinite(target).all(-1)
    p, t = predicted[known], target[known]
    per_joint = np.full(21, np.nan)
    if known.sum() >= 4:  # enough for the alignment
        per_joint[known] = _distance_mm(procrustes_align(p, t), t)
    return {  # NaN where the root, or too few joints, are known
        "mpjpe": float(_distance_mm(p, t).mean()),
        "root_error": float(_distance_mm(predicted[0], target[0])),
        "rr_mpjpe": float(_distance_mm(p - predicted[0], t - target[0]).mean()),
        "pa_mpjpe": float(per_joint[known].mean()),
        **{f"pa_{name}": float(error) for name, error in zip(JOINT_NAMES, per_joint)},
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Means over the rows (crops), and the area under the 3D-PCK curve (0 to 50 mm) of the per-joint PA errors."""
    summary = {}
    for name in ("mpjpe", "root_error", "rr_mpjpe", "pa_mpjpe"):
        summary[name] = float(np.nanmean([row[name] for row in rows]))

    errors = np.array([row[f"pa_{name}"] for row in rows for name in JOINT_NAMES])
    errors = errors[np.isfinite(errors)]
    pck = [(errors <= threshold).mean() for threshold in np.linspace(0, 50, 101)]  # share of joints within
    summary["pck_auc"] = float(np.mean(pck))
    summary["crops"] = len(rows)
    return summary


def _distance_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances in millimetres between points in metres."""
    return np.linalg.norm(a - b, axis=-1) * 1000
