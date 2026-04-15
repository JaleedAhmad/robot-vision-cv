import cv2
import numpy as np

# -----------------------------
# Configuration
# -----------------------------

# Known real-world width of the target (in cm)
KNOWN_WIDTH_CM = 5.0   # adjust this to match your target

# Pre-calibrated focal length (in pixels)
# You can compute this once using a reference object at a known distance:
# focal_length = (measured_pixel_width * known_distance_cm) / real_width_cm
FOCAL_LENGTH = 700.0   # example value; adjust after calibration

# HSV color range for red target
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

# Uncomment for green
# lower_color = np.array([35, 100, 100])
# upper_color = np.array([85, 255, 255])

# -----------------------------
# Distance Estimation Function
# -----------------------------
def estimate_distance(real_width_cm, pixel_width, focal_length):
    """Estimate distance (z) in cm using pinhole camera model."""
    if pixel_width == 0:
        return None
    return (real_width_cm * focal_length) / pixel_width


# -----------------------------
# Start Webcam Capture
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    # Convert to HSV and mask by color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Clean mask (reduce noise)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # ignore small blobs
            continue

        # Approximate contour and classify shape
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Bounding box (for width in pixels)
        x, y, w, h = cv2.boundingRect(contour)

        # Compute centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        shape_type = "Unidentified"

        if len(approx) == 4:
            shape_type = "Square"
        else:
            circularity = (4 * np.pi * area) / (peri * peri)
            if 0.7 < circularity <= 1.2:
                shape_type = "Circle"

        # Estimate distance z (in cm)
        z = estimate_distance(KNOWN_WIDTH_CM, w, FOCAL_LENGTH)
        if z is not None:
            z_text = f"{z:.2f} cm"
        else:
            z_text = "N/A"

        # Draw and label
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"{shape_type}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"(x={cX}, y={cY}, z={z_text})", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Print coordinates and distance
        print(f"Detected {shape_type}: (x={cX}, y={cY}, z={z_text})")

    # Show video and mask
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
