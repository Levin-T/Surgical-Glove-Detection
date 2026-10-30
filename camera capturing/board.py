"""The printed ChArUco board used to calibrate the rig (same board as Henri's config.py)."""

import cv2
import cv2.aruco as aruco
import numpy as np

SQUARES = (7, 5)  # squares across, squares down
SQUARE_M = 0.114  # edge of one black square; measure it on the print, it sets the scale of everything
MARKER_M = 0.091  # edge of one marker
DICTIONARY = aruco.DICT_4X4_50
MIN_CORNERS = 6  # fewer corners than this and the board counts as not seen


class Board:
    def __init__(self):
        dictionary = aruco.getPredefinedDictionary(DICTIONARY)
        self.board = aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M, dictionary)
        self.detector = aruco.CharucoDetector(self.board)
        self.corners = self.board.getChessboardCorners().astype(np.float64)  # (24, 3) meters, board frame
        self.square = SQUARE_M

    def detect(self, image):
        """{corner id: pixel (2,)}, or {} if the board is not seen well enough."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            return {}
        return {int(i): c.astype(np.float64) for i, c in zip(ids.flatten(), corners.reshape(-1, 2))}

    def neighbours(self):
        """Pairs of corner ids that lie one square apart."""
        per_row = SQUARES[0] - 1
        pairs = []
        for i in range(len(self.corners)):
            if (i + 1) % per_row:
                pairs.append((i, i + 1))
            if i + per_row < len(self.corners):
                pairs.append((i, i + per_row))
        return pairs

    def square_errors_mm(self, points):
        """points: {corner id: 3D point in meters}. How far each measured square edge is off, in mm."""
        return [abs(np.linalg.norm(points[a] - points[b]) - self.square) * 1000
                for a, b in self.neighbours() if a in points and b in points]
