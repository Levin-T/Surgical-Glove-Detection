"""Checks the RealSense cameras that are plugged in, one or more.

Lists every detected device, starts a stream on each one, waits for a frameset
and reports whether color and depth arrive. Saves one preview image per camera
to outputs/camera_check/. Works with any number of cameras, including one.

Reads each camera's inter-cam sync mode and does not change it. If the cameras
are wired as a sync rig (a master plus slaves), they are started in the same
order as sync_test.py: slaves first, master last, so the slaves get the trigger.

python "camera capturing/check_cameras.py"
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "camera_check"
WIDTH, HEIGHT, FPS = 1280, 720, 15
TIMEOUT_S = 5.0
WARMUP_S = 3.0
STANDALONE, MASTER, SLAVE, FULL_SLAVE = 0, 1, 2, 3


def sync_mode(device):
    """The camera's inter-cam sync mode, or None if it has no such option."""
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.inter_cam_sync_mode):
            return int(sensor.get_option(rs.option.inter_cam_sync_mode))
    return None


def start_order(mode):
    """Slaves start first and the master last, as in sync_test.py."""
    if mode in (SLAVE, FULL_SLAVE):
        return 0
    if mode == MASTER:
        return 2
    return 1


def check_frames(pipe, serial, mode, name, firmware):
    """Waits for one frameset from a started camera and returns (ok, report lines, color, depth)."""
    lines = [f"{name}  serial {serial}  firmware {firmware}"]
    lines.append(f"  inter-cam sync mode {mode if mode is not None else 'not supported'}")
    try:
        frames = pipe.wait_for_frames(int(TIMEOUT_S * 1000))
    except RuntimeError as error:
        lines.append(f"  FAIL no frameset within {TIMEOUT_S:.0f} s: {error}")
        return False, lines, None, None

    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        lines.append("  FAIL frameset is missing color or depth")
        return False, lines, None, None

    color_image = np.asanyarray(color.get_data())
    depth_image = np.asanyarray(depth.get_data())
    valid = float(np.count_nonzero(depth_image)) / depth_image.size
    lines.append(f"  color {color_image.shape[1]}x{color_image.shape[0]}, "
                 f"mean brightness {color_image.mean():.0f}/255")
    lines.append(f"  depth {depth_image.shape[1]}x{depth_image.shape[0]}, "
                 f"{valid * 100:.0f}% of pixels valid")
    ok = valid > 0.05 and color_image.mean() > 5
    if not ok:
        lines.append("  FAIL image is black or depth is empty (lens cap, exposure, or USB power?)")
    return ok, lines, color_image, depth_image


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ctx = rs.context()
    devices = list(ctx.query_devices())
    print(f"pyrealsense2 {rs.__version__}: {len(devices)} RealSense device(s) detected")
    if not devices:
        print("FAIL no camera found. Check the USB cable, port (USB 3), and power.")
        return 1

    modes = [sync_mode(device) for device in devices]
    order = sorted(range(len(devices)), key=lambda i: start_order(modes[i]))
    pipelines = {}
    for i in order:
        device = devices[i]
        serial = device.get_info(rs.camera_info.serial_number)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        pipe = rs.pipeline(ctx)
        try:
            pipe.start(cfg)
            pipelines[i] = pipe
        except RuntimeError as error:
            print(f"  {serial}: could not start streams: {error}")

    print(f"Started {len(pipelines)} of {len(devices)} camera(s), warming up for {WARMUP_S:.0f} s")
    time.sleep(WARMUP_S)

    all_ok = True
    for index, device in enumerate(devices):
        print()
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        firmware = device.get_info(rs.camera_info.firmware_version)
        if index not in pipelines:
            all_ok = False
            print(f"{name}  serial {serial}  firmware {firmware}\n  FAIL stream did not start")
            continue
        ok, lines, color, depth = check_frames(pipelines[index], serial, modes[index], name, firmware)
        all_ok &= ok
        print("\n".join(lines))
        print("  OK" if ok else "")
        if color is not None:
            preview = np.hstack([color, cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)])
            cv2.imwrite(str(OUTPUT / f"camera{index}.png"), preview)

    for pipe in pipelines.values():
        pipe.stop()

    print(f"\nPreviews saved to {OUTPUT}")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
