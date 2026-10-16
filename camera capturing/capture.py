import pyrealsense2 as rs
import numpy as np
import cv2
import time

def setup_synchronized_cameras():
    # 1. Initialize context to find connected devices
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) == 0:
        print("Error: No RealSense devices detected.")
        return
        
    print(f"Detected {len(devices)} RealSense device(s).")
    
    pipelines = []
    
    # 2. Configure pipelines and set Master/Slave modes
    for i, dev in enumerate(devices):
        serial_number = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        
        # Initialize pipeline for this specific serial number
        pipe = rs.pipeline(context)
        cfg = rs.config()
        cfg.enable_device(serial_number)
        
        # Configure streams (Note: keep streams matching across synchronized units)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Retrieve the first depth sensor to set the hardware sync option
        depth_sensor = dev.first_depth_sensor()
        
        if depth_sensor.supports(rs.option.inter_cam_sync_mode):
            # Designate the first camera as Master, all subsequent cameras as Slaves
            if i == 0:
                sync_mode = 1.0  # Master
                mode_str = "MASTER"
            else:
                sync_mode = 2.0  # Slave
                mode_str = "SLAVE"
                
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
            print(f"[{i}] {name} (S/N: {serial_number}) -> Configured as {mode_str}")
        else:
            print(f"Warning: Camera {serial_number} does not support inter-cam synchronization.")
        
        pipelines.append((pipe, cfg, serial_number))
        
    return pipelines

def start_capture_loop(pipelines):
    active_pipelines = []
    
    # 3. Start pipelines (always start Slaves FIRST, then start the Master last)
    # This ensures Slave cameras are ready and listening for the Master's trigger pulse.
    try:
        # Sort so that slaves (index 1 to N) start before master (index 0)
        sorted_pipelines = list(enumerate(pipelines))
        sorted_pipelines.sort(key=lambda x: x[0], reverse=True) # Slave first
        
        for idx, (pipe, cfg, sn) in sorted_pipelines:
            profile = pipe.start(cfg)
            active_pipelines.append((idx, pipe, sn))
            print(f"Pipeline started for device index [{idx}] ({sn})")
            
        print("\nAll pipelines started. Capturing frames... (Press 'q' in visualizer to quit)")
        time.sleep(1.0) # Warm up sensors
        
        while True:
            frame_bundle = {}
            
            # Polling frames from each camera sequentially
            # Since hardware sync handles hardware-level shutter alignment,
            # wait_for_frames() grabs the corresponding matching frame buffers.
            for idx, pipe, sn in active_pipelines:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                    
                # Store frames paired with camera metadata for timestamp verification
                frame_bundle[idx] = {
                    "color": np.asanyarray(color_frame.get_data()),
                    "depth": np.asanyarray(depth_frame.get_data()),
                    "hw_timestamp": frames.get_timestamp(), # Milliseconds (sensor time)
                    "sn": sn
                }
                
            # Render images side-by-side if they were successfully polled
            if len(frame_bundle) == len(pipelines):
                output_panels = []
                for idx in sorted(frame_bundle.keys()):
                    cam_data = frame_bundle[idx]
                    color_img = cam_data["color"]
                    
                    # Optional: Print out the timestamps to verify sync drift
                    # For perfectly hardware-synced boards, these numbers should stay closely aligned.
                    # print(f"Cam {idx} TS: {cam_data['hw_timestamp']:.2f}", end=" | ")
                    
                    # Overlay camera metadata on visual panels
                    cv2.putText(color_img, f"Cam {idx} (SN: {cam_data['sn']})", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    output_panels.append(color_img)
                # print() # newline for log clean-up
                
                # Stack images horizontally-wise and display
                combined_view = np.hstack(output_panels)
                cv2.imshow("Synchronous Multicam Stream", combined_view)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean shutdown of all running pipelines
        print("\nStopping all pipelines...")
        for idx, pipe, sn in active_pipelines:
            pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_configs = setup_synchronized_cameras()
    if camera_configs:
        start_capture_loop(camera_configs)