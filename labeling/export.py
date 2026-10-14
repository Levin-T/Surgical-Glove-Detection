import numpy as np
from scipy.spatial.transform import Rotation


class LabelExporter:
    """Steps 5 and 6: keeps the good fits and stores one label per camera image."""

    def __init__(self, session, cameras, max_error=10.0, min_views=3):
        self.session = session
        self.cameras = cameras
        self.max_error = max_error
        self.min_views = min_views
        self.rows = {"image": [], "is_right": [], "hand_pose": [], "global_orient": [], "keypoints_2d": []}

    def add(self, frame, side, fit):
        """fit: the HandFitter's result for one hand in one frame."""
        errors = fit["errors"]
        seen = ~errors.isnan()
        if seen.sum() < self.min_views:  # step 5: too few cameras saw this hand
            return
        hand_rotation = Rotation.from_rotvec(fit["rotation"].numpy())
        for k, camera in enumerate(self.cameras):
            if not seen[k] or errors[k] > self.max_error:  # step 5: the fit doesn't match this image
                continue
            orient = Rotation.from_matrix(camera.R.numpy()) * hand_rotation
            self.rows["image"].append(f"{self.session}/cam{k}/color/{frame:06d}.png")
            self.rows["is_right"].append(side == "right")
            self.rows["hand_pose"].append(fit["pose"].numpy())
            self.rows["global_orient"].append(orient.as_rotvec())
            self.rows["keypoints_2d"].append(camera.project(fit["joints"]).numpy())

    def save(self, path, shape, subject):
        """Writes all kept labels of the session into one .npz file."""
        np.savez(path, **{name: np.array(values) for name, values in self.rows.items()},
                 shape=np.asarray(shape), subject=np.array(subject))
