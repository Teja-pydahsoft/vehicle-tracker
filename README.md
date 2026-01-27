# Ultra-Modern Multi-Camera Vehicle Detection System

## 🚀 Quick Start
**Run the application with a single command:**
```bash
python main.py
```

This single file (`main.py`) handles EVERYTHING:
- Starts the **Ultra-Modern Desktop UI**
- Autostarts the **API Server** (for web dashboard)
- Manages **Multi-Camera Processing** (`vehicle_counter.py`)
- Initializes the **Database**

---

## ✨ Key Features
- **Beautiful UI**: Smooth rounded borders, gradients, and modern design.
- **5 Integrated Tabs**: 
  - **Dashboard**: Real-time overview of traffic.
  - **Camera Grid**: View all 4 camera feeds.
  - **Configuration**: Set RTSP URLs and enable/disable cameras.
  - **History**: Search past detection logs with filters.
  - **Analytics**: View system reports (placeholder for future charts).
- **Multi-Camera Support**: Process up to 4 cameras simultaneously.
- **Robust Database**: All detections are logged to `gate_log.db`.

---

## 🔧 Configuration
1. Go to the **Configuration** tab.
2. Enter the **RTSP URL** for each camera (e.g., `rtsp://admin:pass@192.168.1.10:554/stream`).
3. Check **Enable this camera**.
4. Click **Test Connection** (optional).
5. Click **Save** in the toolbar.

## 📹 Starting Cameras
- Click **▶ Start All** in the toolbar.
- Or go to **Dashboard** and start individual cameras.

## 📦 Dependencies
Ensure you have the following installed:
- Python 3.8+
- `tkinter` (usually included with Python)
- `ultralytics` (YOLOv8)
- `opencv-python`
- `pillow`
- `requests`

## ❓ Troubleshooting
- **Cameras not starting?** Check your RTSP stream URLs in VLC player first.
- **No data in dashboard?** Ensure `vehicle_counter.py` is in the same directory.
- **Database error?** Delete `gate_log.db` to reset the database.

---
**Enjoy your Smart Traffic Management System!**
