import torch


class HandFitter:
    """Step 4: bends and moves a MANO hand until it matches the 2D keypoints in all cameras."""

    def __init__(self, hand, cameras, smooth_weight=1.0, pose_weight=1.0, iterations=100):
        self.hand = hand
        self.cameras = cameras
        self.smooth_weight = smooth_weight
        self.pose_weight = pose_weight
        self.iterations = iterations

    def joints(self, pose, shape, rotation, translation):
        """MANO joints (21, 3) for one hand."""
        return self.hand(pose[None], shape[None], rotation[None], translation[None])[0][0]

    def reprojection_loss(self, joints, points_2d, weights):
        """Weighted mean squared distance, in pixels, between projected joints and keypoints."""
        loss = 0.0
        for camera, observed, w in zip(self.cameras, points_2d, weights):
            projected = camera.project(joints)
            loss = loss + (w * (projected - observed).pow(2).sum(-1)).sum()
        return loss / weights.sum().clamp(min=1e-6)

    def fit(self, points_2d, weights, shape, start, previous_joints=None):
        """points_2d (V, 21, 2) and weights (V, 21): keypoints per camera, weight 0 = not seen.
        shape (10,): this subject's hand shape.
        start: dict with pose (45,), rotation (3,), translation (3,), e.g. the previous
        frame's result, or step 3's placement with all-zero pose.
        previous_joints (21, 3): last frame's fitted joints, or None for the first frame.
        Returns the fitted pose, rotation, translation and joints, plus the remaining
        error per camera in pixels (NaN for cameras that didn't see the hand)."""
        points_2d = torch.as_tensor(points_2d, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        pose = start["pose"].clone().requires_grad_(True)
        rotation = start["rotation"].clone().requires_grad_(True)
        translation = start["translation"].clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([pose, rotation, translation], lr=1.0,
                                      max_iter=self.iterations, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            joints = self.joints(pose, shape, rotation, translation)
            loss = self.reprojection_loss(joints, points_2d, weights)
            if previous_joints is not None:  # movement since last frame, in millimeters
                loss = loss + self.smooth_weight * ((joints - previous_joints) * 1000).pow(2).sum(-1).mean()
            loss = loss + self.pose_weight * pose.pow(2).mean()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            joints = self.joints(pose, shape, rotation, translation)
            errors = []
            for camera, observed, w in zip(self.cameras, points_2d, weights):
                seen = w > 0
                distance = (camera.project(joints) - observed).norm(dim=-1)
                errors.append(distance[seen].mean() if seen.any() else torch.tensor(float("nan")))
        return {"pose": pose.detach(), "rotation": rotation.detach(), "translation": translation.detach(),
                "joints": joints, "errors": torch.stack(errors)}
