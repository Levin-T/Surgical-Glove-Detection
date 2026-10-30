"""Records one synchronized session from the GRATA rig into data/raw/<session>/.

Uses the same camera setup as capture_test.py. Every frameset whose timestamps agree
within MAX_JITTER_MS is saved for all cameras under one frame number, so frames stay
consecutive and match across cameras. Depth is aligned to color and saved as raw
16-bit PNG. A background thread writes the images so the capture loop keeps up.

Layout (what labeling/run.py reads):
  data/raw/<session>/cam<k>/color/<frame:06d>.png
  data/raw/<session>/cam<k>/depth/<frame:06d>.png
  data/raw/<session>/meta.json        serials, intrinsics, depth scale, settings
  data/raw/<session>/timestamps.csv   frame, then one hardware timestamp per camera
cam0 is the lowest serial. calibration.json (R, t) comes from the calibration project.

python "camera capturing/record_session.py" <session>             until Ctrl+C
python "camera capturing/record_session.py" <session> <seconds>   stops after that long (0 = one frame)
A countdown runs before recording starts, and the terminal beeps at start and stop.
"""

import json
import queue
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from capture_test import (COLOR_HEIGHT, COLOR_WIDTH, DEPTH_HEIGHT, DEPTH_WIDTH, DISPARITY_SHIFT,
                          FPS, MASTER, MAX_JITTER_MS, ROOT, SERIALS, SLAVE, WARMUP_S, configure,
                          enable_advanced_mode, set_disparity_shift)

COUNTDOWN_S = 5  # time for the subject to get into position


class ImageWriter:
    """Writes images on a background thread so saving does not slow down capturing."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.work)
        self.thread.start()

    def work(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            path, image = job
            cv2.imwrite(str(path), image)

    def write(self, path, image):
        self.jobs.put((path, image))

    def close(self):
        self.jobs.put(None)
        self.thread.join()


class SessionRecorder:
    def __init__(self, session):
        self.folder = ROOT / "data" / "raw" / session
        if self.folder.exists():
            raise SystemExit(f"{self.folder} already exists, pick a new session name")
        self.cameras = sorted(SERIALS)  # cam<k> order
        self.pipelines = {}
        self.align = rs.align(rs.stream.color)
        self.writer = ImageWriter()

    def start(self):
        devices = enable_advanced_mode(rs.context())
        missing = [sn for sn in SERIALS if sn not in devices]
        if missing:
            raise SystemExit(f"cameras not detected: {', '.join(missing)}")
        for index, serial in enumerate(SERIALS):
            configure(devices[serial], MASTER if index == 0 else SLAVE)
            set_disparity_shift(devices[serial], DISPARITY_SHIFT)

        ctx = rs.context()
        for serial in reversed(SERIALS):  # slaves first, master last
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
            pipe = rs.pipeline(ctx)
            pipe.start(cfg)
            self.pipelines[serial] = pipe

        for k in range(len(self.cameras)):
            (self.folder / f"cam{k}" / "color").mkdir(parents=True)
            (self.folder / f"cam{k}" / "depth").mkdir(parents=True)
        self.save_meta()
        print(f"Warming up for {WARMUP_S:.0f} s")
        time.sleep(WARMUP_S)

    def save_meta(self):
        cameras = []
        for serial in self.cameras:
            profile = self.pipelines[serial].get_active_profile()
            i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            cameras.append({
                "serial": serial,
                "K": [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]],
                "dist": list(i.coeffs),
                "depth_scale": depth_scale,
            })
        meta = {
            "cameras": cameras,
            "master": SERIALS[0],
            "color_size": [COLOR_WIDTH, COLOR_HEIGHT],
            "depth_size": [DEPTH_WIDTH, DEPTH_HEIGHT],
            "depth_aligned_to_color": True,
            "disparity_shift": DISPARITY_SHIFT,
            "fps": FPS,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.folder / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def countdown(self):
        for second in range(COUNTDOWN_S, 0, -1):
            print(f"Starting in {second}")
            time.sleep(1)

    def record(self, seconds=None):
        max_frames = None if seconds is None else max(1, round(seconds * FPS))
        frame, skipped = 0, 0
        self.countdown()
        with open(self.folder / "timestamps.csv", "w") as stamps:
            stamps.write("frame," + ",".join(f"cam{k}" for k in range(len(self.cameras))) + "\n")
            for pipe in self.pipelines.values():  # drop frames that queued up during the countdown
                while pipe.poll_for_frames():
                    pass
            print("\aRecording" + (f" for {seconds:g} s" if seconds else ", press Ctrl+C to stop"))
            started = time.time()
            try:
                while max_frames is None or frame < max_frames:
                    framesets = {sn: self.pipelines[sn].wait_for_frames(5000) for sn in self.cameras}
                    times = [framesets[sn].get_color_frame().get_timestamp() for sn in self.cameras]
                    if max(times) - min(times) > MAX_JITTER_MS:
                        skipped += 1
                        continue
                    for k, serial in enumerate(self.cameras):
                        aligned = self.align.process(framesets[serial])
                        color = np.asanyarray(aligned.get_color_frame().get_data()).copy()
                        depth = np.asanyarray(aligned.get_depth_frame().get_data()).copy()
                        self.writer.write(self.folder / f"cam{k}" / "color" / f"{frame:06d}.png", color)
                        self.writer.write(self.folder / f"cam{k}" / "depth" / f"{frame:06d}.png", depth)
                    stamps.write(f"{frame}," + ",".join(f"{t:.3f}" for t in times) + "\n")
                    frame += 1
                    if frame % FPS == 0:
                        elapsed = time.time() - started
                        total = f" / {seconds:g}" if seconds else ""
                        print(f"\r  {elapsed:5.0f}{total} s  {frame} frames  {frame / elapsed:4.1f} fps  "
                              f"{skipped} skipped  {self.writer.jobs.qsize()} waiting   ", end="", flush=True)
            except KeyboardInterrupt:
                pass
        print(f"\n\aStopped: {frame} frames ({frame / FPS:.1f} s), {skipped} skipped for jitter")

    def stop(self):
        for pipe in self.pipelines.values():
            pipe.stop()
        print("Writing the remaining images")
        self.writer.close()
        print(f"Saved to {self.folder}")


if __name__ == "__main__":
    recorder = SessionRecorder(sys.argv[1])
    try:
        recorder.start()
        recorder.record(float(sys.argv[2]) if len(sys.argv) > 2 else None)
    finally:
        recorder.stop()