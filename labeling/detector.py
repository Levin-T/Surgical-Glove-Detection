from ultralytics import YOLO


class HandDetector:
    """Finds hands in an image with WiLoR's YOLO hand detector."""

    def __init__(self, weights_path, min_confidence=0.3, device="cpu"):
        self.model = YOLO(weights_path).to(device)
        self.min_confidence = min_confidence

    def detect(self, image):
        """image: (H, W, 3) BGR uint8 array, as read by OpenCV."""
        result = self.model(image, conf=self.min_confidence, verbose=False)[0]
        boxes = result.boxes.xyxy.tolist()        # [[x1, y1, x2, y2], ...] in pixels
        confidences = result.boxes.conf.tolist()  # [0.91, 0.47, ...]
        sides = ["right" if c == 1 else "left" for c in result.boxes.cls.tolist()]

        hands = {}
        for box, confidence, side in zip(boxes, confidences, sides):
            if side not in hands or confidence > hands[side]["confidence"]:
                hands[side] = {"box": box, "confidence": confidence}
        return hands
