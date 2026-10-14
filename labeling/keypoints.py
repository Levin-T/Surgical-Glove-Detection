from rtmlib import RTMPose

RTMPOSE_HAND = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")


class KeypointDetector:
    """Finds the 21 hand joints inside a hand box with RTMPose's hand model."""

    def __init__(self, model=RTMPOSE_HAND, min_confidence=0.1, device="cpu"):
        self.model = RTMPose(onnx_model=model, model_input_size=(256, 256),
                             backend="onnxruntime", device=device)
        self.min_confidence = min_confidence

    def predict(self, image, box):
        """image: (H, W, 3) BGR uint8 array; box: [x1, y1, x2, y2] from the HandDetector.
        Returns keypoints (21, 2) in pixels and confidences (21,).
        Joints below min_confidence get confidence 0, so the fit ignores them."""
        keypoints, confidences = self.model(image, bboxes=[box])
        keypoints, confidences = keypoints[0], confidences[0]
        confidences[confidences < self.min_confidence] = 0.0
        confidences[0] = 1.0  # wrist: always trusted, as in Fischer et al.
        return keypoints, confidences
