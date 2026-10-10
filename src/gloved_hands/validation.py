"""The validation layers of the labels (RQ2), with decreasing dependence on the labelling pipeline. Every layer
judges the labels that passed quality control, the ones that reach training."""

from typing import Any

import numpy as np

from gloved_hands.capture import Recording
from gloved_hands.fitting import MultiViewFitter
from gloved_hands.keyframes import Keyframes
from gloved_hands.labelling import Labeller, Labels
from gloved_hands.metrics import hand_errors


class LabelValidator:
    def __init__(self, labeller: Labeller) -> None:
        self.labeller = labeller

    def leave_one_view_out(self, recording: Recording, labels: Labels) -> list[dict[str, Any]]:
        """Layer 1, internal consistency: refit without each camera in turn, from HaMeR as in labelling, and project
        into the camera left out. Blind to errors that all cameras share; the start and the shape use all cameras."""
        cfg = self.labeller.cfg
        fitter = MultiViewFitter(self.labeller.hands[labels.side], recording.cameras, cfg)
        passed = labels.frames[labels.passed]
        rows = []
        starts = np.linspace(0, max(len(passed) - cfg.chunk_size, 0), cfg.lovo_chunks).astype(int)
        for start in np.unique(starts):  # evenly spaced chunks
            chunk = passed[start : start + cfg.chunk_size]
            observed = self.labeller.observe(recording, chunk, labels.side, labels.params)
            if observed is None:
                continue
            frames, obs, init = observed
            for v, camera in enumerate(recording.cameras):
                result = fitter.fit(obs.without_view(v), init, fit_shape=False)
                pixels = camera.project(result.joints)  # (T, 21, 2)
                errors = np.linalg.norm(pixels - obs.keypoints[:, v], axis=-1).mean(-1)
                for frame, error, confidence in zip(frames, errors, obs.confidence[:, v]):
                    if confidence > 0:  # this camera saw the hand
                        rows.append({"frame": int(frame), "camera": camera.name, "error_px": float(error)})
        return rows

    @staticmethod
    def keyframes(recording: Recording, labels: Labels, keyframes: Keyframes) -> list[dict[str, Any]]:
        """Layer 2: the labels against triangulated human annotations, which do not depend on HaMeR."""
        rows_of = {int(frame): row for row, frame in enumerate(labels.frames) if labels.passed[row]}
        rows = []
        for (frame, side), joints in keyframes.triangulate(recording).items():
            if side == labels.side and frame in rows_of:
                rows.append({"frame": frame, **hand_errors(labels.joints[rows_of[frame]], joints)})
        return rows

    @staticmethod
    def manus(recording: Recording, labels: Labels) -> list[dict[str, Any]]:
        """Layer 3: articulation against the MANUS glove. Procrustes-aligned, so no registration is needed;
        agreement is necessary but not sufficient, as MANUS joint angles are model-derived."""
        rows = []
        for row in np.flatnonzero(labels.passed):
            frame = int(labels.frames[row])
            reference = recording[frame].manus.get(labels.side)
            if reference is not None:
                errors = hand_errors(labels.joints[row], reference)
                articulation = {name: value for name, value in errors.items() if name.startswith("pa_")}
                rows.append({"frame": frame, **articulation})
        return rows
