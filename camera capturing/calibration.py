# --- START OF MODIFIED FILE CameraCalibtartion_Charuco.py ---

import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np

# --- CHARUCO BOARD SETUP ---
# Update these to match the board you print
SQUARES_VERTICALLY = 3 # Number of squares vertically
SQUARES_HORIZONTALLY = 4 # Number of squares horizontally
square_size = 0.18 # in meters
marker_size = 0.135 # The size of the ArUco marker in meters

# This is the dictionary of ArUco markers we are using
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Create the ChArUco board object
# This object holds the layout of your board and is used for detection
board = aruco.CharucoBoard(
    (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), 
    square_size, 
    marker_size, 
    aruco_dict
)

charuco_detector = aruco.CharucoDetector(board)

# Other script parameters
output_file = "calibration_d405.npz"

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_device("336222300607") #336222300607 = 4, 337122300608 = 10, 336522303249 = 16, 336522302876 = 22
config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

# --- Variables for calibration ---
all_corners = [] # All image points from all frames
all_ids = [] # All corner IDs from all frames
img_id = 0

print("Using ChArUco board detection.")
print("Drücke 's' um ein Musterbild zu speichern. ESC zum Abbrechen.")

window_title = "Kalibrierung D405 (ChArUco)"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        # --- DETECTION LOGIC ---
        # First, detect the ArUco markers
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        found = False
        # If we found at least one marker, try to interpolate the board corners
        if charuco_ids is not None and len(charuco_ids) > 3:
            found = True
            aruco.drawDetectedCornersCharuco(disp, charuco_corners, charuco_ids)

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and found:
            print(f"[{img_id}] Muster erkannt und gespeichert.")
            # Append the detected corners and their IDs
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            img_id += 1

        elif key == 27: # ESC key
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

if len(all_corners) < 5:
    print("Nicht genug Bilder gespeichert. Mindestens 5 empfohlen.")
else:
    print(f"Kalibriere mit {len(all_corners)} Bildern...")
    
    # The ChArUco calibration function is slightly different
    # It uses the IDs to match the 3D and 2D points, which is very reliable

    all_object_points = []
    all_image_points = []

    # Ensure that the number of object points and image points match for each image.
    for i in range(len(all_ids)):
        # Get the 3D coordinates of the corners from the board layout
        obj_points_for_img = board.getChessboardCorners()[all_ids[i].flatten()]
        
        # Check if we have enough points for this image
        if len(obj_points_for_img) >= 4:
            all_object_points.append(obj_points_for_img)
            all_image_points.append(all_corners[i])

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_object_points,
        imagePoints=all_image_points,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None
    )

    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("✅ Kalibrierung abgeschlossen.")
    print("📷 Kamera-Matrix:\n", camera_matrix)
    print("🔧 Verzerrung:\n", dist_coeffs)
    print(f"📈 RMS-Reprojektionsfehler: {ret:.4f} Pixel")
    print(f"💾 Gespeichert in: {output_file}")