# Vehicle Counter Application Documentation

## Overview
This document provides a comprehensive guide to the AI-based Vehicle Counter application, which uses YOLOv8 for vehicle detection and counting in video streams or files.

## Table of Contents
1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [Technical Implementation](#technical-implementation)
6. [Troubleshooting](#troubleshooting)
7. [Future Enhancements](#future-enhancements)

## Features
- Real-time vehicle detection and counting
- Support for video file input
- RTSP camera stream support
- Live statistics display
- User-friendly GUI interface
- Multi-threaded processing for smooth operation

## System Requirements
- Python 3.7+
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLOv8
- Tkinter
- PIL (Python Imaging Library)

## Installation
1. Clone or download the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. The application will automatically download the YOLOv8 model on first run

## Usage Guide

### 1. Launching the Application
Run the application by executing:
```
python vehicle_counter.py
```

### 2. Using Video File Input
1. Click "Browse" to select a video file
2. Click "Start Processing" to begin vehicle detection
3. View live statistics in the statistics panel

### 3. Using RTSP Camera
1. Enter the RTSP URL in the format: `rtsp://username:password@ip:port/stream`
2. Click "Connect Camera" to establish the connection
3. The application will automatically start processing the stream

## Technical Implementation

### Core Components
1. **YOLOv8 Model**
   - Pre-trained on COCO dataset
   - Optimized for real-time object detection
   - Handles multiple vehicle classes

2. **Multi-threaded Processing**
   - Separate threads for:
     - Video capture
     - Model inference
     - GUI updates

3. **Vehicle Tracking**
   - Implements tracking using bounding box positions
   - Maintains vehicle count history
   - Prevents double-counting

### Code Structure
- `VehicleCounterApp` class: Main application class
- UI components created using Tkinter
- Separate methods for file handling, camera connection, and video processing

## Troubleshooting

### Common Issues
1. **Model not loading**
   - Check internet connection for model download
   - Verify write permissions in the application directory

2. **Video not playing**
   - Verify video file format support
   - Check if codecs are properly installed

3. **Camera connection issues**
   - Verify RTSP URL format
   - Check network connectivity
   - Ensure camera credentials are correct

## Future Enhancements
- Add support for multiple camera feeds
- Implement vehicle classification (car, truck, bus, etc.)
- Add speed estimation
- Export statistics to CSV/Excel
- Add support for more video formats
- Implement a web interface

## Support
For any issues or feature requests, please open an issue in the project repository.

---
*Documentation generated on: 2025-03-23*
*Version: 1.0.0*
