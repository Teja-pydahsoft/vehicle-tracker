# Multi-Camera Vehicle Detection System - Quick Start Guide

## 📦 What You've Got

I've created a complete **Multi-Camera Dashboard and Configuration System** for processing 4 cameras simultaneously. Here's what's included:

### 📁 New Files Created

1. **`dashboard/multi_camera.html`** - Main dashboard interface
2. **`dashboard/multi_camera.css`** - Modern dark-themed styling
3. **`dashboard/multi_camera.js`** - Full functionality and API integration
4. **`multi_camera_api.py`** - Flask backend API server
5. **`LAUNCH_MULTI_CAMERA.bat`** - Easy launch script
6. **`MULTI_CAMERA_README.md`** - Complete documentation
7. **`camera_config.example.json`** - Configuration template

## 🚀 How to Start

### Option 1: Quick Launch (Recommended)
```bash
# Double-click this file:
LAUNCH_MULTI_CAMERA.bat
```

### Option 2: Manual Launch
```bash
# Open terminal in the project directory
cd "C:\Users\Ashok Kumar\Desktop\Vehical Detection"

# Start the API server
python multi_camera_api.py

# Open browser to:
http://localhost:5000
```

## 🎯 Key Features

### 1. **Live Dashboard**
- Real-time statistics for all 4 cameras
- Total IN/OUT vehicle counts
- Active camera monitoring
- Average FPS across all cameras
- Recent detection activity feed

### 2. **Camera Grid View**
- Simultaneous display of all 4 camera feeds
- 2x2 grid layout
- Individual camera controls (Start/Stop)
- Real-time FPS per camera
- Fullscreen mode

### 3. **Configuration Panel**
For each camera (1-4):
- **Camera Name**: Custom naming (e.g., "Main Gate", "Exit Gate")
- **RTSP URL**: Stream configuration
  ```
  Format: rtsp://username:password@ip:port/stream
  Example: rtsp://admin:pydah@123@192.168.1.101:554/stream1
  ```
- **Enable/Disable**: Toggle camera on/off
- **Test Connection**: Verify camera accessibility

Global Settings:
- Detection confidence threshold (0-100%)
- Frame processing rate (1-5)
- Auto-restart on failure
- OCR enable/disable

### 4. **History & Logs**
- Advanced filtering:
  - By camera (1-4 or All)
  - By vehicle type (Car, Truck, Bus, Motorcycle)
  - By license plate number
  - By date range
- Export to CSV/PDF
- Detailed detection records with timestamps

### 5. **Analytics**
- Hourly traffic flow charts
- Vehicle type distribution
- Camera performance metrics
- Daily trend analysis

## 🔧 Camera Configuration Steps

### Step 1: Access Configuration
1. Start the system (using `LAUNCH_MULTI_CAMERA.bat`)
2. Open browser to `http://localhost:5000`
3. Click **Configuration** in the sidebar

### Step 2: Configure Each Camera

**Camera 1 - Main Gate Entry:**
```
Name: Main Gate Entry
RTSP URL: rtsp://admin:password@192.168.1.101:554/stream1
Status: Enabled
```

**Camera 2 - Main Gate Exit:**
```
Name: Main Gate Exit
RTSP URL: rtsp://admin:password@192.168.1.102:554/stream1
Status: Enabled
```

**Camera 3 - Parking Entry:**
```
Name: Parking Entry
RTSP URL: rtsp://admin:password@192.168.1.103:554/stream1
Status: Enabled
```

**Camera 4 - Parking Exit:**
```
Name: Parking Exit
RTSP URL: rtsp://admin:password@192.168.1.104:554/stream1
Status: Enabled
```

### Step 3: Test Connections
- Click **Test Connection** for each camera
- Verify green success message
- Fix any connection issues before proceeding

### Step 4: Save Configuration
- Click **Save Configuration** button
- Settings are stored in database
- Configuration persists across restarts

## 🎮 Operating the System

### Starting Cameras

**Individual Camera:**
1. Go to **Live Dashboard**
2. Find the camera card
3. Click **Start** button
4. Status changes to "ACTIVE" (green)

**All Cameras at Once:**
1. Go to **Camera Grid** view
2. Click **Start All** button at top
3. All enabled cameras start simultaneously

### Monitoring Live Feeds

1. Navigate to **Camera Grid** view
2. See all 4 cameras in 2x2 grid
3. Each shows:
   - Camera name
   - Real-time FPS
   - Live video feed

### Viewing Statistics

**Dashboard View:**
- **Total IN**: All vehicles entering (all cameras combined)
- **Total OUT**: All vehicles exiting (all cameras combined)
- **Active Cameras**: Currently running cameras (e.g., "3/4")
- **Average FPS**: Performance metric across all cameras

**Per-Camera Stats:**
- Individual IN/OUT counts
- Current FPS
- Connection status

### Searching History

1. Click **History** tab
2. Set filters:
   - **Camera**: Select 1, 2, 3, 4, or All
   - **Vehicle Type**: Car, Truck, Bus, Motorcycle
   - **Plate Number**: Search specific plate
   - **Date Range**: Start and end dates
3. Click **Search**
4. Export results: **Export CSV** or **Export PDF**

## 🎨 Dashboard Interface

### Color Coding
- **Green**: IN direction, Active status, Success
- **Red**: OUT direction, Inactive status, Errors
- **Orange**: Warnings, Maintenance needed
- **Blue**: Information, FPS metrics
- **Purple**: Primary actions, Navigation

### Status Indicators
- **ACTIVE** (Green): Camera running and processing
- **INACTIVE** (Gray): Camera stopped
- **MAINTENANCE** (Orange): Camera needs attention
- **OFFLINE** (Red): Camera connection lost

## 📊 Database Structure

All data is stored in `gate_log.db`:

**vehicle_logs table:**
- `id`: Unique log ID
- `camera_id`: Which camera (1-4)
- `timestamp`: Detection time
- `vehicle_type`: Car, Truck, Bus, Motorcycle
- `track_id`: Tracking ID
- `direction`: IN or OUT
- `confidence`: Detection confidence (0-1)
- `plate_number`: License plate (if detected)

**camera_config table:**
- `id`: Camera ID (1-4)
- `name`: Camera name
- `rtsp_url`: Stream URL
- `enabled`: 1 or 0
- `position`: Location description

## 🔌 API Integration

The system exposes REST API endpoints:

```http
GET /api/cameras              # Get all cameras
GET /api/cameras/1            # Get camera 1
POST /api/cameras             # Update config
POST /api/camera/1/start      # Start camera 1
POST /api/camera/1/stop       # Stop camera 1
GET /api/stats                # Get statistics
GET /api/logs                 # Get detection logs
GET /api/analytics/hourly     # Hourly data
GET /api/analytics/daily      # Daily data
```

## 🛠️ Troubleshooting

### Camera Won't Connect
**Check:**
- RTSP URL format: `rtsp://user:pass@ip:port/stream`
- Camera is powered on
- Network connectivity
- Correct credentials
- Firewall settings

**Test manually:**
```bash
ffplay rtsp://admin:password@192.168.1.101:554/stream1
```

### Low FPS
**Solutions:**
- Increase processing rate (every 2nd or 3rd frame)
- Reduce camera resolution
- Check CPU/GPU usage
- Verify network bandwidth

### Dashboard Not Loading
**Check:**
- API server is running
- Port 5000 is not blocked
- Browser console (F12) for errors
- Correct URL: `http://localhost:5000`

## 📈 Performance Tips

**For 4 Cameras Simultaneously:**

**Recommended Settings:**
- Processing Rate: Every 2nd frame
- Confidence: 30-40%
- Resolution: 1280x720 (not Full HD)
- Enable GPU if available

**System Requirements:**
- CPU: Intel i7 or equivalent
- RAM: 16GB recommended
- GPU: NVIDIA with CUDA (optional)
- Network: Gigabit Ethernet

## 🔄 Integration with Existing System

The multi-camera dashboard works alongside `vehicle_counter.py`:

1. **Shared Database**: Both use `gate_log.db`
2. **Independent**: Can run separately or together
3. **API Access**: Dashboard reads from same database

## 📱 Responsive Design

The dashboard works on:
- **Desktop**: Full features, optimal experience
- **Tablet**: Adapted layout, touch-friendly
- **Mobile**: Compact view, essential features

## 🎯 Next Steps

1. **Configure Cameras**: Add your RTSP URLs
2. **Test Connections**: Verify each camera works
3. **Start System**: Begin monitoring
4. **Review Logs**: Check detection accuracy
5. **Adjust Settings**: Optimize for your setup

## 📞 Support

For issues:
1. Check troubleshooting section
2. Review API logs in console
3. Check browser console (F12)
4. Verify camera connectivity

---

**System Version**: 2.0.0  
**Created**: January 25, 2026  
**Dashboard URL**: http://localhost:5000  
**API Port**: 5000
