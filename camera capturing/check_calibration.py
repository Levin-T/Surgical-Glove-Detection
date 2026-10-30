"""Checks a calibration against a board shot recorded later, e.g. at the end of a recording day.

Record the board with record_session.py (<session> 0 takes one frame), then run this with the
calibration in use. If the numbers are as good as right after calibrating, no camera has moved.

python "camera capturing/check_calibration.py" <calibration.json> <session>
"""

import sys
from pathlib import Path

import cv2

from board import Board
from calibrate import ROOT, evaluate
from common.camera import load_cameras


if __name__ == "__main__":
    cameras = load_cameras(Path(sys.argv[1]))
    folder = ROOT / "data" / "raw" / sys.argv[2]
    board = Board()
    seen = {}
    for k in range(len(cameras)):
        found = board.detect(cv2.imread(str(folder / f"cam{k}" / "color" / "000000.png")))
        if found:
            seen[k, 0] = found
    print("\n".join(evaluate(board, cameras, seen, 1)))
