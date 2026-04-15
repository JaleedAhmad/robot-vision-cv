# Robot Vision and Computer Vision Scripts

A collection of lightweight Python scripts for common robot vision tasks using OpenCV. These scripts cover ArUco marker detection, pose estimation, shape detection, and distance measurement.

## Scripts Overview
- **`detect_aruco.py`**: Detects ArUco markers (DICT_4X4_50) and displays their centroids.
- **`aruco_pose.py`**: Performs pose estimation for ArUco markers.
- **`detect_shape.py`**: Generic shape detection using contours.
- **`shape_depth.py` & `shape_distance.py`**: Estimates depth and distance of detected objects from the camera.
- **`hsv_conversion.py`**: Helper script for HSV color space manipulation.

## Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd robot-vision-cv
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Usage
Run any specific script using Python:
```bash
python detect_aruco.py
```

---
*Created by [Your Name] for Shaheer.*
