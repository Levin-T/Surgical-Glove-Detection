import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# List your camera serial numbers (Master first, followed by Slaves)
camera_serials = ["336522303322", "213622251480", "336522302876"]
pipelines = []
configs = []

os.makedirs("captured_images", exist_ok=True)

# 1. Initialize Context and configure hardware sync BEFORE streaming
ctx = rs.context()
devices = {dev.get_info(rs.camera_info.serial_number): dev for dev in ctx.query_devices()}

for idx, sn in enumerate(camera_serials):
    if sn not in devices:
        print(f"Error: Camera {sn} not detected on USB.")
        continue
        
    device = devices[sn]
    sensors = device.query_sensors()
    
    # Mode 1 for Master (index 0), Mode 2 (or 3) for Slaves
    sync_mode = 1 if idx == 0 else 2 
    
    for sensor in sensors:
        if sensor.supports(rs.option.inter_cam_sync_mode):
            sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(rs.option.auto_exposure_priority, 0) # Mandatory for hardware lock
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1) # Force precise time mapping
            
    print(f"Camera {sn} hardware sync configured to Mode {sync_mode}.")

# 2. Start pipelines (Initialize from last Slave up to the Master)
for idx in reversed(range(len(camera_serials))):
    sn = camera_serials[idx]
    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_device(sn)
    
    # Force 30 FPS to ensure stable hardware trigger pulses
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    pipe.start(cfg)
    pipelines.append(pipe)

# Re-align list order so index 0 matches the Master camera
pipelines.reverse()

print("\nWarming up streams for 3 seconds to let hardware clocks stabilize...")
time.sleep(3)

print("Starting hardware-synchronized capture loop. Press Ctrl+C to stop.")
try:
    capture_count = 0
    frame_counter = 0
    
    while True:
        synced_set = {}
        frame_counter += 1
        
        # Pull frames from all pipelines
        for idx, pipe in enumerate(pipelines):
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            # Get the exact Global Hardware Timestamp (in milliseconds)
            synced_set[idx] = {
                "timestamp": color_frame.get_timestamp(),
                "color": np.asanyarray(color_frame.get_data()),
                "depth": np.asanyarray(depth_frame.get_data())
            }
            
        # Verify we received frames from all cameras
        if len(synced_set) == len(camera_serials):
            timestamps = [synced_set[i]["timestamp"] for i in range(len(camera_serials))]
            ts_jitter = max(timestamps) - min(timestamps)
            
            # Since hardware works, the jitter threshold (< 2ms) will now easily pass
            if ts_jitter < 2.0:
                # Downsample step: Only save every 2nd hardware frame (30 FPS / 2 = 15 FPS capture rate)
                if frame_counter % 2 == 0:
                    print(f"Saving Set {capture_count} | Hardware Jitter: {ts_jitter:.3f} ms")
                    
                    for idx in range(len(camera_serials)):
                        cam_sn = camera_serials[idx]
                        # Save RGB
                        cv2.imwrite(f"captured_images/set_{capture_count}_cam_{cam_sn}_color.png", synced_set[idx]["color"])
                        # Save Depth (Visualized as grayscale for verification)
                        depth_visual = cv2.convertScaleAbs(synced_set[idx]["depth"], alpha=0.03)
                        cv2.imwrite(f"captured_images/set_{capture_count}_cam_{cam_sn}_depth.png", depth_visual)
                        
                    capture_count += 1
            else:
                # Occasional USB bus hiccups can cause a dropped package, skip silently
                pass

except KeyboardInterrupt:
    print("\nStopping all camera streams safely...")
finally:
    for pipe in pipelines:
        pipe.stop()
    print("Streams stopped. Your images are ready in 'captured_images/'.")
