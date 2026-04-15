import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Create the detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the frame
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Iterate through all detected markers
        for i in range(len(ids)):
            corner = corners[i][0]
            marker_id = ids[i][0]

            # Compute centroid
            cx = int(np.mean(corner[:, 0]))
            cy = int(np.mean(corner[:, 1]))

            # Display the marker ID and centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {marker_id} ({cx},{cy})",
                        (corner[0][0], corner[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Print centroid coordinates
            print(f"Marker ID: {marker_id}, Centroid: ({cx}, {cy})")

    # Show the frame
    cv2.imshow('ArUco Marker Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
