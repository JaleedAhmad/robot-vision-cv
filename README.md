<div align="center">

# 🤖 Robot Vision & Computer Vision Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

*A lightweight, robust collection of Python scripts for fundamental robot vision tasks.*

</div>

---

## 📌 Overview

This repository provides a foundational toolkit for implementing core computer vision algorithms commonly used in robotics. Utilizing the power of **OpenCV**, these scripts offer real-time capabilities for marker tracking, geometric shape detection, distance measurement, and spatial pose estimation using a monocular camera setup.

## 🏗️ Core Architecture & Scripts

The project is modularized into specific feature-focused scripts, allowing for easy integration into larger robotics pipelines:

### 1. ArUco Marker Systems
- **`detect_aruco.py`**: Identifies standard ArUco markers (`DICT_4X4_50`) in real-time and computes their 2D image centroids.
- **`aruco_pose.py`**: Executes full 3D pose estimation for detected ArUco markers, rendering real-time coordinate axes and outputting translation vectors (x, y, z) relative to the camera.

### 2. Shape & Object Detection
- **`detect_shape.py`**: Implements HSV color-space masking, morphological noise reduction, and contour approximation to dynamically classify geometric shapes (e.g., squares, circles) and track their centroids.
- **`hsv_conversion.py`**: A utility tool for calibrating and fine-tuning HSV color bounds required for robust object masking under varying lighting conditions.

### 3. Spatial Estimation
- **`shape_distance.py`**: Utilizes the pinhole camera model to estimate the real-world distance (z-axis) of detected shapes based on a pre-calibrated focal length and known physical object dimensions.
- **`shape_depth.py`**: Provides alternative depth computation mechanics for detected contour clusters.

## 📁 Repository Structure

```text
robot-vision-cv/
├── README.md                 # Project documentation
├── .gitignore                # Version control exclusions
├── aruco_pose.py             # 3D pose estimation script
├── detect_aruco.py           # ArUco marker detection
├── detect_shape.py           # Shape recognition script
├── hsv_conversion.py         # Color space tuning utility
├── shape_depth.py            # Depth estimation
├── shape_distance.py         # Distance measurement (pinhole model)
├── testing.py                # Sandbox for experimental code
└── tracking.py               # Object tracking logic
```

## 🚀 Setup Instructions

### Prerequisites

Ensure you have a modern version of Python installed (3.8 or higher recommended). A webcam or external camera is required to test the live feed capabilities.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd robot-vision-cv
```

### 2. Environment Configuration

It is highly recommended to isolate dependencies using a virtual environment:

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required computer vision and mathematical libraries:

```bash
pip install opencv-python opencv-contrib-python numpy
```
*(Note: `opencv-contrib-python` is required for full ArUco module support depending on your OpenCV version).*

## ⚡ Execution

Run any of the modular scripts directly via Python. Press the **`q`** key while focused on the video window to terminate the process.

**Example: Running Pose Estimation**
```bash
python aruco_pose.py
```

**Example: Running Shape Distance Estimation**
```bash
python shape_distance.py
```

---

*Engineered with precision for Shaheer.*
