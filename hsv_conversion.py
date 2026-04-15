import cv2
import numpy as np

cap = cv2.VideoCapture(0)
lower = np.array([35, 100, 100])   # Example: green lower HSV
upper = np.array([85, 255, 255])   # Example: green upper HSV

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

        if len(approx) >= 8:
            shape = "Circle"
        elif len(approx) == 4:
            shape = "Square"
        else:
            continue

        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"{shape} ({cx},{cy})", (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
