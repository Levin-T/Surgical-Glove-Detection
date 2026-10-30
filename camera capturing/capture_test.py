"""Captures one synchronized test image set from the GRATA rig.

Configures the cameras as in sync_test.py (first serial master, the others slaves),
starts the slaves first and the master last, waits for the sync to settle and keeps
the first frameset whose timestamps agree within MAX_JITTER_MS. Color streams at
1280x720 and depth at 640x360 with a raised disparity shift for close range, then
depth is aligned to color. Saves color,
raw 16-bit depth (millimeters, looks black in an image viewer) and a colorized depth
preview for every camera to outputs/test_capture/ and prints the jitter.

python "camera capturing/capture_test.py"
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "test_capture"
SERIALS = ["336522303322", "213622251480", "336522302876"]  # master first
COLOR_WIDTH, COLOR_HEIGHT = 1280, 720
DEPTH_WIDTH, DEPTH_HEIGHT = 640, 360  # minimum distance scales with resolution, about 30-35 cm here
DISPARITY_SHIFT = 25  # moves the whole depth window closer: max_z = fx * baseline / shift
BASELINE_M = 0.095  # D455 stereo baseline
SEARCH_RANGE_PX = 126  # disparity search range added to the shift for the minimum distance
RECONNECT_TIMEOUT_S = 30.0
FPS = 15
WARMUP_S = 3.0
MAX_JITTER_MS = 2.0
MAX_ATTEMPTS = 60
MASTER, SLAVE = 1, 2


def configure(device, mode):
    """Sets the sync mode and the options the hardware lock needs."""
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.inter_cam_sync_mode):
            sensor.set_option(rs.option.inter_cam_sync_mode, mode)
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(rs.option.auto_exposure_priority, 0)
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)


def find_devices(ctx):
    return {d.get_info(rs.camera_info.serial_number): d for d in ctx.query_devices()}


def enable_advanced_mode(ctx):
    """Turns advanced mode on for the rig cameras and returns the fresh devices.

    Switching it on makes a camera disconnect and reconnect on USB, so the old
    device handles are stale and the cameras are queried again once they are back.
    """
    toggled = False
    for serial, device in find_devices(ctx).items():
        if serial in SERIALS:
            advanced = rs.rs400_advanced_mode(device)
            if not advanced.is_enabled():
                advanced.toggle_advanced_mode(True)
                toggled = True
    if toggled:
        print("Advanced mode switched on, waiting for the cameras to reconnect")
        time.sleep(5)

    deadline = time.time() + RECONNECT_TIMEOUT_S
    while time.time() < deadline:
        devices = find_devices(ctx)
        if all(sn in devices for sn in SERIALS) and all(
                rs.rs400_advanced_mode(devices[sn]).is_enabled() for sn in SERIALS):
            return devices
        time.sleep(1)
    return find_devices(ctx)


def set_disparity_shift(device, shift):
    """Sets the disparity shift in the depth table and returns the value read back."""
    advanced = rs.rs400_advanced_mode(device)
    table = advanced.get_depth_table()
    table.disparityShift = shift
    advanced.set_depth_table(table)
    return advanced.get_depth_table().disparityShift


def main():
    ctx = rs.context()
    devices = enable_advanced_mode(ctx)
    missing = [sn for sn in SERIALS if sn not in devices]
    if missing:
        print(f"FAIL cameras not detected: {', '.join(missing)}")
        return 1

    for index, serial in enumerate(SERIALS):
        mode = MASTER if index == 0 else SLAVE
        configure(devices[serial], mode)
        shift = set_disparity_shift(devices[serial], DISPARITY_SHIFT)
        print(f"{serial}: sync mode {mode}, disparity shift {shift}")

    pipelines = {}
    try:
        for serial in reversed(SERIALS):
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
            pipe = rs.pipeline(ctx)
            profile = pipe.start(cfg)
            pipelines[serial] = pipe
            fx = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics().fx
            near = fx * BASELINE_M / (DISPARITY_SHIFT + SEARCH_RANGE_PX)
            far = fx * BASELINE_M / DISPARITY_SHIFT if DISPARITY_SHIFT else float("inf")
            print(f"{serial}: depth fx {fx:.0f} px, estimated depth window "
                  f"{near * 100:.0f} to {far * 100:.0f} cm")

        print(f"Warming up for {WARMUP_S:.0f} s")
        time.sleep(WARMUP_S)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            framesets = {sn: pipelines[sn].wait_for_frames(5000) for sn in SERIALS}
            stamps = [fs.get_color_frame().get_timestamp() for fs in framesets.values()]
            jitter = max(stamps) - min(stamps)
            if jitter < MAX_JITTER_MS:
                break
            print(f"  attempt {attempt}: jitter {jitter:.2f} ms, too large, retrying")
        else:
            print(f"FAIL no frameset within {MAX_JITTER_MS} ms after {MAX_ATTEMPTS} attempts")
            return 1

        OUTPUT.mkdir(parents=True, exist_ok=True)
        align = rs.align(rs.stream.color)
        for serial, frames in framesets.items():
            frames = align.process(frames)
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            cv2.imwrite(str(OUTPUT / f"cam_{serial}_color.png"), color)
            cv2.imwrite(str(OUTPUT / f"cam_{serial}_depth.png"), depth)
            preview = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET)
            preview[depth == 0] = 0
            cv2.imwrite(str(OUTPUT / f"cam_{serial}_depth_preview.png"), preview)
            print(f"{serial}: color mean {color.mean():.0f}/255, "
                  f"{np.count_nonzero(depth) / depth.size * 100:.0f}% depth valid")

        print(f"Jitter {jitter:.3f} ms (attempt {attempt})")
        print(f"Saved to {OUTPUT}")
        return 0
    finally:
        for pipe in pipelines.values():
            pipe.stop()


if __name__ == "__main__":
    sys.exit(main())
