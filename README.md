# Real-Time Object Detection Using OpenCV

## Project Overview
This project implements a real-time object detection system using Python and OpenCV.  
The system detects:
- Faces
- Full Bodies
- Eyes
- Vehicle License Plates
- And also highlights moving objects using contour detection.

Additionally, a **beep sound** is played whenever any object is detected, providing an instant alert feature.

---

## Features
- Real-time detection from webcam
- Multi-object detection (Face, Body, Eye, License Plate)
- Contour detection for moving objects
- Beep sound alert on detection
- Lightweight, fast, and runs smoothly without GPU

---

## Technologies Used
- Python 3
- OpenCV
- NumPy
- winsound (for beep alerts, Windows only)

---

## How to Run
1. Install required libraries:
   ```bash
   pip install opencv-python numpy
