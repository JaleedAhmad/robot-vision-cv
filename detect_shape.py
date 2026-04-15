import cv2
import numpy as np

# --- Adjustable parameters ---
# Define the HSV color range for the target color (example: red)
# You can adjust these values using an HSV color picker
lower_color = np.array([0, 120, 70])   # Lower HSV bound for red
upper_color = np.array([10, 255, 255]) # Upper HSV bound for red

# If you want to detect green instead, use:
# lower_color = np.array([35, 100, 100])
# upper_color = np.array([85, 255, 255])

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # --- Convert to HSV color space ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Apply color mask ---
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # --- Remove noise using morphological operations ---
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # --- Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip small areas / noise
            continue

        # --- Approximate contour to polygon ---
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # --- Compute centroid ---
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        shape_type = "Unidentified"

        # --- Classify shape based on number of sides ---
        if len(approx) == 4:
            shape_type = "Square"
        else:
            # Compute circularity to detect circles
            circularity = (4 * np.pi * area) / (peri * peri)
            if 0.7 < circularity <= 1.2:
                shape_type = "Circle"

        # --- Draw contours and centroid ---
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"{shape_type} ({cX}, {cY})", (cX - 50, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Print coordinates ---
        print(f"Detected {shape_type} at: x={cX}, y={cY}")

    # --- Display result ---
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # --- Break loop on 'q' key press ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
