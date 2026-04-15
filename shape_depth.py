import cv2
import numpy as np
import pyrealsense2 as rs  # Intel RealSense SDK

# -----------------------------
# Intel RealSense Configuration
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

# Configure depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Align depth map to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# -----------------------------
# Color Detection Settings
# -----------------------------
# Example: red color range (in HSV)
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

# Uncomment for green
# lower_color = np.array([35, 100, 100])
# upper_color = np.array([85, 255, 255])

print("Press 'q' to quit...")

while True:
    # Wait for frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Convert to HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Mask color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        # Approximate and classify shape
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

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

        # -----------------------------
        # Depth Extraction (RealSense)
        # -----------------------------
        depth_value = depth_frame.get_distance(cX, cY)  # depth in meters

        # Draw results
        cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
        cv2.circle(color_image, (cX, cY), 5, (255, 0, 0), -1)
        text = f"{shape_type} (x={cX}, y={cY}, z={depth_value:.3f}m)"
        cv2.putText(color_image, text, (cX - 50, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Print to console
        print(f"Detected {shape_type}: x={cX}, y={cY}, z={depth_value:.3f} m")

    # Display color and mask
    cv2.imshow("Color Frame", color_image)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
