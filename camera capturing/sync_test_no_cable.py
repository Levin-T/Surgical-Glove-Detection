import pyrealsense2 as rs
import numpy as np
import time

# List your camera serial numbers (Order does not matter since sync is disabled)
camera_serials = ["336522303322", "213622251480", "336522302876"]
pipelines = []
configs = []

# 1. Initialize Context and strip all hardware/software sync mechanisms
ctx = rs.context()
devices = {dev.get_info(rs.camera_info.serial_number): dev for dev in ctx.query_devices()}

print("--- CONFIGURING CAMERAS FOR NON-SYNCED BASELINE TEST ---")
for idx, sn in enumerate(camera_serials):
    if sn not in devices:
        print(f"Error: Camera {sn} not detected on USB.")
        continue
        
    device = devices[sn]
    sensors = device.query_sensors()
    
    for sensor in sensors:
        if sensor.supports(rs.option.inter_cam_sync_mode):
            sensor.set_option(rs.option.inter_cam_sync_mode, 0) # 0 = OFF (Independent Clocks)
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(rs.option.auto_exposure_priority, 0)
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 0) # 0 = OFF (Use Host System Arrival Time)
            
    print(f"Camera {sn} running completely independently.")

# 2. Start pipelines normally
for sn in camera_serials:
    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_device(sn)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    pipe.start(cfg)
    pipelines.append(pipe)

print("\nWarming up streams for 3 seconds...")
time.sleep(3)

print("Starting non-synced measurement loop. Press Ctrl+C to stop.\n")
try:
    sample_count = 0
    jitter_samples = []
    
    while True:
        synced_set = {}
        
        # Grab frames as they arrive at the operating system
        for idx, pipe in enumerate(pipelines):
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
                
            # Without global time, this returns the raw host OS system arrival timestamp
            synced_set[idx] = {
                "timestamp": color_frame.get_timestamp()
            }
            
        if len(synced_set) == len(camera_serials):
            timestamps = [synced_set[i]["timestamp"] for i in range(len(camera_serials))]
            
            # Calculate the drift/jitter between frame arrivals
            non_synced_jitter = max(timestamps) - min(timestamps)
            jitter_samples.append(non_synced_jitter)
            sample_count += 1
            
            # Print metrics every 15 frames (~0.5 seconds) to avoid terminal spam
            if sample_count % 15 == 0:
                avg_jitter = sum(jitter_samples) / len(jitter_samples)
                print(f"Sample {sample_count:4d} | Current Jitter: {non_synced_jitter:6.3f} ms | Rolling Avg Jitter: {avg_jitter:6.3f} ms")
                
            # Sleep slightly to match your target 15 FPS evaluation loop
            time.sleep(0.066)

except KeyboardInterrupt:
    print("\n\nStopping measurement...")
finally:
    for pipe in pipelines:
        pipe.stop()
        
    if jitter_samples:
        print("\n--- FINAL BENCHMARK RESULTS ---")
        print(f"Total frame sets measured: {len(jitter_samples)}")
        print(f"Minimum observed jitter  : {min(jitter_samples):.3f} ms")
        print(f"Maximum observed jitter  : {max(jitter_samples):.3f} ms")
        print(f"Average non-synced jitter: {sum(jitter_samples)/len(jitter_samples):.3f} ms")
