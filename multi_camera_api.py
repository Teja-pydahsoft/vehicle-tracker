"""
Multi-Camera Vehicle Detection API Server
Handles 4 simultaneous camera streams with vehicle detection and tracking
"""

from flask import Flask, jsonify, request, send_from_directory, session, Response
from flask_cors import CORS
import sqlite3
import json
import threading
import queue
import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import hashlib
import secrets
import os
import urllib.parse
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='dashboard')
app.secret_key = secrets.token_hex(16) # Secret key for sessions
CORS(app)

# ===== Configuration =====
DB_PATH = 'gate_log.db'
CONFIG_PATH = 'camera_config.json'

# Camera processors storage
camera_processors = {}
camera_queues = {}
camera_stats = {
    1: {'status': 'inactive', 'fps': 0, 'in': 0, 'out': 0},
    2: {'status': 'inactive', 'fps': 0, 'in': 0, 'out': 0},
    3: {'status': 'inactive', 'fps': 0, 'in': 0, 'out': 0},
    4: {'status': 'inactive', 'fps': 0, 'in': 0, 'out': 0}
}

class StreamManager:
    """Manages a single RTSP connection and shares frames with multiple web clients."""
    def __init__(self, camera_id, url):
        self.camera_id = camera_id
        self.url = url
        self.frame = None
        self.stopped = False
        self.thread = None
        self.last_access = time.time()

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stopped = False
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(f"StreamManager {self.camera_id} started.")

    def stop(self):
        self.stopped = True
        logger.info(f"StreamManager {self.camera_id} stopping.")

    def _run(self):
        # Determine connection string
        relay_path = os.path.join(os.getcwd(), "shared_frames", f"cam_{self.camera_id}.jpg")
        proc_url = sanitize_rtsp_url(self.url)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        
        while not self.stopped:
            # 1. Check RELAY (Shared from Desktop App)
            if os.path.exists(relay_path):
                # Check if file is "fresh" (modified in last 3 seconds)
                if time.time() - os.path.getmtime(relay_path) < 3.0:
                    with open(relay_path, "rb") as f:
                        self.frame = f.read()
                    time.sleep(0.04) # Match ~25fps
                    continue

            # 2. Fallback to RTSP (If desktop app is not running)
            cap = cv2.VideoCapture(proc_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.warning(f"API Cam {self.camera_id} connection failed. Retrying...")
                time.sleep(5)
                continue

            while not self.stopped:
                # Check for relay again - if Desktop starts, we switch back
                if os.path.exists(relay_path) and time.time() - os.path.getmtime(relay_path) < 1.0:
                    break # Back to relay mode

                success, frame = cap.read()
                if not success:
                    break
                
                # Pre-process for web (Fast)
                frame = cv2.resize(frame, (640, 480))
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret:
                    self.frame = buffer.tobytes()
                
                # Auto-shutdown if no one is watching for 60s
                if time.time() - self.last_access > 60:
                    self.stopped = True
                    break
                
                time.sleep(0.02) # Cap at ~50fps

            cap.release()
            if not self.stopped: time.sleep(2)

    def get_frame(self):
        self.last_access = time.time()
        return self.frame

# Global stream managers
streams = {}

# ===== Database Functions =====
def get_db_connection():
    """Get database connection"""
    db_path = os.path.abspath(DB_PATH)
    logger.debug(f"API: Connecting to DB at {db_path}")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize database with multi-camera support"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create enhanced vehicle logs table with camera_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            timestamp TEXT,
            vehicle_type TEXT,
            track_id INTEGER,
            direction TEXT,
            confidence REAL,
            plate_number TEXT
        )
    ''')
    
    # Create camera configuration table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_config (
            id INTEGER PRIMARY KEY,
            name TEXT,
            rtsp_url TEXT,
            enabled INTEGER,
            position TEXT,
            status TEXT DEFAULT 'inactive'
        )
    ''')
    
    # Migration: add status if not exists
    try:
        cursor.execute('ALTER TABLE camera_config ADD COLUMN status TEXT DEFAULT "inactive"')
    except: pass
    
    # Create settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Create users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')

     # Check if admin exists, if not create default
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        # Default admin:admin123 (In production use proper hashing like bcrypt, here using SHA256 for simplicity)
        default_pass = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                      ('admin', default_pass, 'admin'))

    # Migration: Add vehicle_state to vehicle_logs if not exists
    try:
        cursor.execute('ALTER TABLE vehicle_logs ADD COLUMN vehicle_state TEXT')
    except sqlite3.OperationalError:
        pass # Column likely exists

    conn.commit()
    conn.close()
    logger.info("Database initialized with multi-camera support, auth, and state tracking")

# ===== Configuration Management =====
def load_camera_config():
    """Load camera configuration from database or file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM camera_config")
        cameras = cursor.fetchall()
        conn.close()
        
        if cameras:
            return [dict(camera) for camera in cameras]
        else:
            # Return default configuration
            return [
                {'id': 1, 'name': 'Camera 1 - Main Gate', 'rtsp_url': '', 'enabled': 1, 'position': 'main_gate'},
                {'id': 2, 'name': 'Camera 2 - Exit Gate', 'rtsp_url': '', 'enabled': 1, 'position': 'exit_gate'},
                {'id': 3, 'name': 'Camera 3 - Parking Entry', 'rtsp_url': '', 'enabled': 1, 'position': 'parking_entry'},
                {'id': 4, 'name': 'Camera 4 - Parking Exit', 'rtsp_url': '', 'enabled': 1, 'position': 'parking_exit'}
            ]
    except Exception as e:
        logger.error(f"Error loading camera config: {e}")
        return []

def save_camera_config(cameras):
    """Save camera configuration to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for camera in cameras:
            cursor.execute('''
                INSERT OR REPLACE INTO camera_config (id, name, rtsp_url, enabled, position)
                VALUES (?, ?, ?, ?, ?)
            ''', (camera['id'], camera['name'], camera['rtsp_url'], 
                  camera.get('enabled', 1), camera.get('position', '')))
        
        conn.commit()
        conn.close()
        logger.info("Camera configuration saved")
        return True
    except Exception as e:
        logger.error(f"Error saving camera config: {e}")
        return False

# ===== API Routes =====

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Missing credentials'}), 400
            
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, role FROM users WHERE username=? AND password_hash=?", (username, hashed_pw))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['logged_in'] = True
            return jsonify({'success': True, 'role': user['role'], 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if session.get('logged_in'):
        return jsonify({'authenticated': True, 'role': session.get('role')})
    return jsonify({'authenticated': False}), 401

@app.route('/')
def index():
    """Serve the multi-camera dashboard"""
    return send_from_directory('dashboard', 'multi_camera.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('dashboard', path)

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all camera configurations"""
    cameras = load_camera_config()
    
    # Add current stats to each camera
    for camera in cameras:
        camera_id = camera['id']
        if camera_id in camera_stats:
            camera.update(camera_stats[camera_id])
    
    return jsonify(cameras)

@app.route('/api/cameras/<int:camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get specific camera configuration"""
    cameras = load_camera_config()
    camera = next((c for c in cameras if c['id'] == camera_id), None)
    
    if camera:
        if camera_id in camera_stats:
            camera.update(camera_stats[camera_id])
        return jsonify(camera)
    else:
        return jsonify({'error': 'Camera not found'}), 404

@app.route('/api/cameras', methods=['POST'])
def update_cameras():
    """Update camera configurations"""
    try:
        cameras = request.json.get('cameras', [])
        if save_camera_config(cameras):
            return jsonify({'success': True, 'message': 'Configuration saved'})
        else:
            return jsonify({'success': False, 'message': 'Failed to save configuration'}), 500
    except Exception as e:
        logger.error(f"Error updating cameras: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/<int:camera_id>/start', methods=['POST'])
def start_camera(camera_id):
    """Start a specific camera"""
    try:
        cameras = load_camera_config()
        camera = next((c for c in cameras if c['id'] == camera_id), None)
        
        logger.info(f"API Attempting to start Cam {camera_id}. Found config: {camera}")
        
        if not camera:
            return jsonify({'success': False, 'message': 'Camera not found in database'}), 404
        
        # Check both rtsp_url and rtspUrl mappings
        url = camera.get('rtsp_url') or camera.get('rtspUrl')
        
        if not url:
            return jsonify({'success': False, 'message': f'RTSP URL for Camera {camera_id} is empty in database.'}), 400
        
        # Start camera processing (implement actual processing logic)
        camera_stats[camera_id]['status'] = 'active'
        logger.info(f"Started camera {camera_id}: {camera['name']}")
        
        # PERSIST TO DB for sync
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE camera_config SET status='active' WHERE id=?", (camera_id,))
            conn.commit()
            conn.close()
        except: pass
        
        return jsonify({'success': True, 'message': f'Camera {camera_id} started'})
    except Exception as e:
        logger.error(f"Error starting camera {camera_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/<int:camera_id>/stop', methods=['POST'])
def stop_camera(camera_id):
    """Stop a specific camera"""
    try:
        camera_stats[camera_id]['status'] = 'inactive'
        camera_stats[camera_id]['fps'] = 0
        
        # Stop internal stream if any
        if camera_id in camera_processors:
            camera_processors[camera_id]['active'] = False
        
        # PERSIST TO DB for sync
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE camera_config SET status='inactive' WHERE id=?", (camera_id,))
            conn.commit()
            conn.close()
        except: pass
        
        logger.info(f"Stopped camera {camera_id}")
        return jsonify({'success': True, 'message': f'Camera {camera_id} stopped'})
    except Exception as e:
        logger.error(f"Error stopping camera {camera_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

def sanitize_rtsp_url(url):
    if not isinstance(url, str) or not url.startswith('rtsp://'):
        return url
    try:
        prefix = 'rtsp://'
        url_stripped = url[len(prefix):]
        if '@' not in url_stripped: return url
        auth_part, host_part = url_stripped.rsplit('@', 1)
        if ':' not in auth_part: return url
        user, password = auth_part.split(':', 1)
        encoded_pass = urllib.parse.quote(password)
        return f"{prefix}{user}:{encoded_pass}@{host_part}"
    except:
        return url

def generate_frames(camera_id):
    """Serve frames from the shared StreamManager."""
    while True:
        if camera_id not in streams:
            # Try to start it if DB says it should be active
            cameras = load_camera_config()
            cam_data = next((c for c in cameras if c['id'] == camera_id), None)
            if cam_data and cam_data.get('status') == 'active' and cam_data.get('rtsp_url'):
                streams[camera_id] = StreamManager(camera_id, cam_data['rtsp_url'])
                streams[camera_id].start()
            else:
                time.sleep(1)
                continue

        try:
            frame = streams[camera_id].get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Streaming error for camera {camera_id}: {e}")
        
        time.sleep(0.04) # Serve at ~25fps

@app.route('/api/camera/<int:camera_id>/feed')
def camera_feed(camera_id):
    """MJPEG streaming endpoint"""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total counts by direction
    cursor.execute('''
        SELECT direction, COUNT(*) as count
        FROM vehicle_logs
        GROUP BY direction
    ''')
    summary = [dict(row) for row in cursor.fetchall()]
    
    # Get counts by camera
    cursor.execute('''
        SELECT camera_id, direction, COUNT(*) as count
        FROM vehicle_logs
        GROUP BY camera_id, direction
    ''')
    by_camera = [dict(row) for row in cursor.fetchall()]
    
    # Get counts by vehicle type
    cursor.execute('''
        SELECT vehicle_type, COUNT(*) as count
        FROM vehicle_logs
        GROUP BY vehicle_type
    ''')
    by_type = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    # Combined camera status
    db_camera_stats = {}
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, status FROM camera_config")
    rows = cursor.fetchall()
    conn.close()
    
    active_count = 0
    for row in rows:
        cid = row['id']
        st = row['status'] or 'inactive'
        if st == 'active': active_count += 1
        
        # Merge DB status with live FPS/Counts
        live = camera_stats.get(cid, {'fps': 0, 'in': 0, 'out': 0})
        db_camera_stats[cid] = {
            'camera_id': cid,
            'name': row['name'],
            'status': st,
            'fps': live.get('fps', 0),
            'in': live.get('in', 0),
            'out': live.get('out', 0)
        }

    return jsonify({
        'summary': summary,
        'by_camera': by_camera,
        'by_type': by_type,
        'active_cameras': active_count,
        'camera_stats': db_camera_stats
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get vehicle detection logs with filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get query parameters
        camera_id = request.args.get('camera_id', 'all')
        vehicle_type = request.args.get('vehicle_type', 'all')
        plate_number = request.args.get('plate_number', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        limit = request.args.get('limit', 100, type=int)
        
        # Build query
        query = "SELECT * FROM vehicle_logs WHERE 1=1"
        params = []
        
        if camera_id != 'all':
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        if vehicle_type != 'all' and vehicle_type != 'All':
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)
        
        if plate_number:
            query += " AND plate_number LIKE ?"
            params.append(f'%{plate_number}%')
        
        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicle-types', methods=['GET'])
def get_vehicle_types():
    """Get list of detected vehicle types"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT vehicle_type FROM vehicle_logs WHERE vehicle_type IS NOT NULL")
        types = [row[0] for row in cursor.fetchall()]
        conn.close()
        return jsonify(types)
    except Exception as e:
        logger.error(f"Error fetching vehicle types: {e}")
        return jsonify([])

@app.route('/api/analytics/hourly', methods=['GET'])
def get_hourly_analytics():
    """Get hourly traffic analytics with filters"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        camera_id = request.args.get('camera_id', 'all')
        vehicle_type = request.args.get('vehicle_type', 'all')
        direction = request.args.get('direction', 'all')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count,
                direction
            FROM vehicle_logs
            WHERE 1=1
        '''
        params = []
        
        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date)
        elif not end_date:
            # Default to today if no range provided
            query += " AND DATE(timestamp) = DATE('now')"
            
        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date)
            
        if camera_id != 'all' and camera_id != 'All':
            query += " AND camera_id = ?"
            params.append(camera_id)
            
        if vehicle_type != 'all' and vehicle_type != 'All':
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)
            
        if direction != 'all' and direction != 'All':
            query += " AND direction = ?"
            params.append(direction)
            
        query += " GROUP BY hour, direction ORDER BY hour"
        
        cursor.execute(query, params)
        data = [dict(row) for row in cursor.fetchall()]
        
        # Calculate Peak and Low Hours
        peak_hour = None
        low_hour = None
        max_count = -1
        min_count = float('inf')
        
        # Aggregate by hour (ignoring direction for peak calcs)
        hourly_totals = {}
        for row in data:
            h = row['hour']
            c = row['count']
            hourly_totals[h] = hourly_totals.get(h, 0) + c
            
        if hourly_totals:
            peak_hour = max(hourly_totals, key=hourly_totals.get)
            low_hour = min(hourly_totals, key=hourly_totals.get)
            max_count = hourly_totals[peak_hour]
            min_count = hourly_totals[low_hour]
            
        conn.close()
        
        return jsonify({
            'hourly_data': data,
            'stats': {
                'peak_hour': peak_hour,
                'peak_count': max_count,
                'low_hour': low_hour,
                'low_count': min_count
            }
        })
    except Exception as e:
        logger.error(f"Error fetching hourly analytics: {e}")
        return jsonify({'hourly_data': [], 'stats': {}})

@app.route('/api/analytics/daily', methods=['GET'])
def get_daily_analytics():
    """Get daily traffic analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count,
                direction
            FROM vehicle_logs
            WHERE DATE(timestamp) >= DATE('now', '-30 days')
            GROUP BY date, direction
            ORDER BY date
        ''')
        
        data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching daily analytics: {e}")
        return jsonify([])

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get system settings"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        settings = {row['key']: row['value'] for row in cursor.fetchall()}
        conn.close()
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error fetching settings: {e}")
        return jsonify({})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update system settings"""
    try:
        settings = request.json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for key, value in settings.items():
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES (?, ?)
            ''', (key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Settings updated'})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/test-camera', methods=['POST'])
def test_camera():
    """Test camera connection"""
    try:
        rtsp_url = request.json.get('rtsp_url')
        
        if not rtsp_url:
            return jsonify({'success': False, 'message': 'RTSP URL required'}), 400
        
        # Try to open the camera
        cap = cv2.VideoCapture(rtsp_url)
        success = cap.isOpened()
        
        if success:
            # Try to read a frame
            ret, frame = cap.read()
            success = ret
        
        cap.release()
        
        if success:
            return jsonify({'success': True, 'message': 'Camera connection successful'})
        else:
            return jsonify({'success': False, 'message': 'Failed to connect to camera'}), 400
    except Exception as e:
        logger.error(f"Error testing camera: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ===== Main =====
def run():
    # Initialize database
    init_database()
    
    # Start server
    logger.info("Starting Multi-Camera Vehicle Detection API Server")
    logger.info("Dashboard available at: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    run()
