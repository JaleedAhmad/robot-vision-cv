import cv2
import numpy as np

cap = cv2.VideoCapture(0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner[0]
            cx, cy = int(pts[:,0].mean()), int(pts[:,1].mean())
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.putText(frame, f"ID {ids[i][0]} ({cx},{cy})", (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
