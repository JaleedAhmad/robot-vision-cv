import cv2
import numpy as np

# -------------------------------
# 1. Initialize camera
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# 2. Load ArUco dictionary & parameters
# -------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# -------------------------------
# 3. Load pre-calibrated camera parameters
# Replace these with your own calibration results
# -------------------------------
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)

dist_coeffs = np.array([0.1, -0.25, 0.001, 0.0, 0.0])  # Example distortion coefficients

# Define the marker side length in meters (adjust as per your marker size)
marker_length = 0.05  # 5 cm marker

# -------------------------------
# 4. Real-time detection loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Draw markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            rvec, tvec = rvecs[i][0], tvecs[i][0]
            marker_id = ids[i][0]

            # Draw 3D axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

            # Print 3D position
            print(f"Marker ID: {marker_id} | Position (x, y, z): {tvec}")

            # Display the coordinates on the frame
            pos_text = f"ID:{marker_id}  x:{tvec[0]:.2f}  y:{tvec[1]:.2f}  z:{tvec[2]:.2f}"
            cv2.putText(frame, pos_text, 
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('ArUco Pose Estimation', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 5. Clean up
# -------------------------------
cap.release()
cv2.destroyAllWindows()
