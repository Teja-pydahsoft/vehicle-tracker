# Multi-Camera Vehicle Detection Dashboard

A comprehensive web-based dashboard for managing and monitoring up to 4 simultaneous camera streams with AI-powered vehicle detection, tracking, and license plate recognition.

## 🌟 Features

### Dashboard Views

1. **Live Dashboard**
   - Real-time statistics (Total IN/OUT, Active Cameras, Average FPS)
   - Camera status overview with individual controls
   - Recent detection activity feed
   - Visual trend indicators

2. **Camera Grid View**
   - Simultaneous display of all 4 camera feeds
   - Individual camera controls (Start/Stop)
   - Real-time FPS monitoring
   - Fullscreen mode support

3. **Configuration Panel**
   - Individual camera setup (Name, RTSP URL, Enable/Disable)
   - Test camera connections
   - Global settings:
     - Detection confidence threshold
     - Frame processing rate
     - Auto-restart on failure
     - OCR enable/disable

4. **History & Logs**
   - Advanced filtering (Camera, Vehicle Type, Plate Number, Date Range)
   - Export to CSV/PDF
   - Detailed detection records

5. **Analytics**
   - Hourly traffic flow charts
   - Vehicle type distribution
   - Camera performance metrics
   - Daily trend analysis

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install flask flask-cors opencv-python numpy
```

### Installation

1. **Navigate to the project directory:**
```bash
cd "C:\Users\Ashok Kumar\Desktop\Vehical Detection"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the API server:**
```bash
python multi_camera_api.py
```

4. **Access the dashboard:**
   - Open your browser and navigate to: `http://localhost:5000`

## 📋 Configuration

### Camera Setup

1. Navigate to the **Configuration** tab in the dashboard
2. For each camera (1-4), configure:
   - **Camera Name**: Descriptive name (e.g., "Main Gate", "Exit Gate")
   - **RTSP URL**: Camera stream URL
     ```
     Format: rtsp://username:password@ip_address:port/stream
     Example: rtsp://admin:pydah@123@192.168.1.101:554/stream1
     ```
   - **Status**: Enable or Disable the camera
3. Click **Test Connection** to verify camera accessibility
4. Click **Save Configuration** to persist settings

### RTSP URL Examples

**Hikvision:**
```
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
```

**Generic:**
```
rtsp://username:password@ip:port/stream
```

### Global Settings

- **Detection Confidence Threshold**: Minimum confidence (0-100%) for vehicle detection
- **Frame Processing Rate**: Process every Nth frame (1-5) to optimize performance
- **Auto-Restart on Failure**: Automatically restart cameras if connection is lost
- **Enable OCR**: Turn on/off license plate recognition

## 🎯 Usage

### Starting Cameras

**Individual Camera:**
1. Go to **Live Dashboard** or **Camera Grid**
2. Click **Start** button on the desired camera card
3. Monitor the status indicator (changes to green when active)

**All Cameras:**
1. Go to **Camera Grid** view
2. Click **Start All** button in the top controls
3. All enabled cameras will start simultaneously

### Viewing Live Feeds

1. Navigate to **Camera Grid** view
2. Active camera feeds will display in a 2x2 grid
3. Each feed shows:
   - Camera name
   - Real-time FPS
   - Live video stream

### Monitoring Statistics

**Dashboard View:**
- **Total IN**: Cumulative vehicles entering
- **Total OUT**: Cumulative vehicles exiting
- **Active Cameras**: Number of running cameras
- **Average FPS**: Performance across all cameras

**Per-Camera Stats:**
- Individual IN/OUT counts
- Current FPS
- Connection status

### Searching History

1. Navigate to **History** tab
2. Apply filters:
   - **Camera**: Select specific camera or "All"
   - **Vehicle Type**: Filter by Car, Truck, Bus, Motorcycle
   - **Plate Number**: Search by license plate
   - **Date Range**: Select start and end dates
3. Click **Search** to apply filters
4. Export results using **Export CSV** or **Export PDF**

## 🔧 API Endpoints

### Camera Management

```http
GET /api/cameras
```
Get all camera configurations with current stats

```http
GET /api/cameras/<camera_id>
```
Get specific camera details

```http
POST /api/cameras
```
Update camera configurations
```json
{
  "cameras": [
    {
      "id": 1,
      "name": "Main Gate",
      "rtsp_url": "rtsp://...",
      "enabled": true
    }
  ]
}
```

```http
POST /api/camera/<camera_id>/start
```
Start a specific camera

```http
POST /api/camera/<camera_id>/stop
```
Stop a specific camera

### Statistics & Logs

```http
GET /api/stats
```
Get overall system statistics

```http
GET /api/logs?camera_id=1&vehicle_type=car&limit=100
```
Get detection logs with filters

```http
GET /api/vehicle-types
```
Get list of detected vehicle types

### Analytics

```http
GET /api/analytics/hourly
```
Get hourly traffic data

```http
GET /api/analytics/daily
```
Get daily traffic trends

### Settings

```http
GET /api/settings
```
Get system settings

```http
POST /api/settings
```
Update system settings

```http
POST /api/test-camera
```
Test camera connection
```json
{
  "rtsp_url": "rtsp://..."
}
```

## 📊 Database Schema

### vehicle_logs
```sql
CREATE TABLE vehicle_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER,
    timestamp TEXT,
    vehicle_type TEXT,
    track_id INTEGER,
    direction TEXT,
    confidence REAL,
    plate_number TEXT
);
```

### camera_config
```sql
CREATE TABLE camera_config (
    id INTEGER PRIMARY KEY,
    name TEXT,
    rtsp_url TEXT,
    enabled INTEGER,
    position TEXT
);
```

### settings
```sql
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

## 🎨 UI Features

### Modern Design
- Dark theme with vibrant accent colors
- Glassmorphism effects
- Smooth animations and transitions
- Responsive layout (Desktop, Tablet, Mobile)

### Interactive Elements
- Hover effects on cards and buttons
- Real-time status indicators
- Animated notifications
- Progress indicators

### Accessibility
- Clear visual hierarchy
- High contrast text
- Keyboard navigation support
- Screen reader friendly

## 🔒 Security Considerations

1. **RTSP Credentials**: Store securely, avoid hardcoding
2. **API Access**: Consider adding authentication for production
3. **Database**: Use proper permissions and backups
4. **Network**: Ensure cameras are on a secure network

## 🐛 Troubleshooting

### Camera Won't Connect

**Check:**
- RTSP URL format is correct
- Camera is powered on and network accessible
- Credentials are correct
- Firewall isn't blocking the connection
- Camera supports the RTSP protocol

**Test manually:**
```bash
ffplay rtsp://username:password@ip:port/stream
```

### Low FPS

**Solutions:**
- Increase **Frame Processing Rate** (process every 2nd or 3rd frame)
- Reduce camera resolution in camera settings
- Ensure sufficient CPU/GPU resources
- Check network bandwidth

### Database Errors

**Reset database:**
```bash
# Backup first
copy gate_log.db gate_log.backup.db

# Delete and restart server (will recreate)
del gate_log.db
python multi_camera_api.py
```

### Dashboard Not Loading

**Check:**
- API server is running (`python multi_camera_api.py`)
- No port conflicts (port 5000)
- Browser console for errors (F12)
- Correct URL: `http://localhost:5000`

## 📈 Performance Optimization

### For 4 Cameras Simultaneously

**Recommended Settings:**
- **Processing Rate**: Every 2nd or 3rd frame
- **Confidence Threshold**: 30-40%
- **Resolution**: 640x480 or 1280x720 (not Full HD)
- **Hardware**: GPU acceleration if available

**System Requirements:**
- **CPU**: Intel i5 or equivalent (i7 recommended for 4 cameras)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Network**: Gigabit Ethernet for multiple HD streams

## 🔄 Integration with Existing System

The multi-camera dashboard can work alongside the existing `vehicle_counter.py`:

1. **Shared Database**: Both systems use `gate_log.db`
2. **API Access**: Dashboard reads from the same database
3. **Independent Operation**: Can run simultaneously or separately

## 📝 Future Enhancements

- [ ] Live video streaming to dashboard
- [ ] Advanced analytics with charts (Chart.js integration)
- [ ] Email/SMS alerts for specific events
- [ ] Cloud storage integration
- [ ] Mobile app companion
- [ ] AI-powered anomaly detection
- [ ] Multi-user access with roles
- [ ] Backup and restore functionality

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API logs in the console
3. Check browser console (F12) for errors
4. Verify camera connectivity

## 📄 License

This project is part of the Smart Gate Vehicle Tracking System.

---

**Version**: 2.0.0  
**Last Updated**: January 25, 2026  
**Author**: Pydah Soft Solutions
