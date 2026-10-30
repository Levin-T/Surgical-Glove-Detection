"""Starts the GRATA cameras with the settings from capture_test.py and hands out synchronized framesets."""

import time

import pyrealsense2 as rs

from capture_test import (COLOR_HEIGHT, COLOR_WIDTH, DEPTH_HEIGHT, DEPTH_WIDTH, DISPARITY_SHIFT, FPS,
                          MASTER, MAX_JITTER_MS, SERIALS, SLAVE, WARMUP_S, configure,
                          enable_advanced_mode, set_disparity_shift)


class Rig:
    def __init__(self):
        self.serials = sorted(SERIALS)  # cam<k> order: cam0 is the lowest serial
        self.pipelines = {}

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
        print(f"Warming up for {WARMUP_S:.0f} s")
        time.sleep(WARMUP_S)

    def cameras(self):
        """Factory color intrinsics and depth scale per camera, in cam<k> order."""
        cameras = []
        for serial in self.serials:
            profile = self.pipelines[serial].get_active_profile()
            i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            cameras.append({
                "serial": serial,
                "K": [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]],
                "dist": list(i.coeffs),
                "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
            })
        return cameras

    def drop_queued(self):
        """Throws away frames that piled up while nobody was reading."""
        for pipe in self.pipelines.values():
            while pipe.poll_for_frames():
                pass

    def grab(self):
        """The next framesets whose timestamps agree within MAX_JITTER_MS: ({serial: frameset}, times)."""
        while True:
            framesets = {sn: self.pipelines[sn].wait_for_frames(5000) for sn in self.serials}
            times = [framesets[sn].get_color_frame().get_timestamp() for sn in self.serials]
            if max(times) - min(times) <= MAX_JITTER_MS:
                return framesets, times

    def stop(self):
        for pipe in self.pipelines.values():
            pipe.stop()
