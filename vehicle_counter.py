import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import time
import queue
import threading
import torch
from ultralytics import YOLO
from collections import defaultdict, Counter
import psutil
import logging
from collections import deque
from tkinter.scrolledtext import ScrolledText
import sqlite3
from datetime import datetime
import easyocr
import cv2
import urllib.parse
import calendar
import requests
import subprocess
import tempfile
import zipfile
import shutil

# Application Versioning
CURRENT_VERSION = "v1.0.1" 
GITHUB_REPO = "Teja-pydahsoft/vehicle-tracker"

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZoneConfigDialog(tk.Toplevel):
    def __init__(self, parent, initial_w, initial_h, update_callback):
        super().__init__(parent)
        self.title("Zone Configuration")
        self.geometry("300x200")
        self.resizable(False, False)
        self.update_callback = update_callback
        
        # Center dialog
        try:
            x = parent.winfo_rootx() + parent.winfo_width()//2 - 150
            y = parent.winfo_rooty() + parent.winfo_height()//2 - 100
            self.geometry(f"+{x}+{y}")
        except:
            pass
        
        # Inputs
        ttk.Label(self, text="Zone Width (px):").grid(row=0, column=0, padx=10, pady=10)
        self.w_var = tk.StringVar(value=str(initial_w))
        self.w_entry = ttk.Entry(self, textvariable=self.w_var)
        self.w_entry.grid(row=0, column=1, padx=10, pady=10)
        self.w_entry.bind('<KeyRelease>', self.on_change)
        
        ttk.Label(self, text="Zone Height (px):").grid(row=1, column=0, padx=10, pady=10)
        self.h_var = tk.StringVar(value=str(initial_h))
        self.h_entry = ttk.Entry(self, textvariable=self.h_var)
        self.h_entry.grid(row=1, column=1, padx=10, pady=10)
        self.h_entry.bind('<KeyRelease>', self.on_change)
        
        ttk.Button(self, text="Update Preview", command=self.on_change).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(self, text="Confirm & Close", command=self.destroy).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Trigger initial preview
        self.after(100, self.on_change)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
    def on_change(self, event=None):
        try:
            w_str = self.w_var.get()
            h_str = self.h_var.get()
            
            if not w_str or not h_str: return
            
            w = int(w_str)
            h = int(h_str)
            self.update_callback(w, h)
        except ValueError:
            pass # Ignore incomplete input

def sanitize_rtsp_url(url):
    """
    Sanitize RTSP URL by encoding special characters in the password.
    Example: rtsp://admin:pydah@123@192.168.1.1 -> rtsp://admin:pydah%40123@192.168.1.1
    """
    if not isinstance(url, str) or not url.startswith('rtsp://'):
        return url
    
    try:
        # Basic structure: rtsp://[user[:password]@]host[:port]/path
        prefix = 'rtsp://'
        url_stripped = url[len(prefix):]
        
        if '@' not in url_stripped:
            return url
            
        # The last '@' before the host/path part separates credentials
        # We split by '@' from the right to isolate the credentials from the host part
        parts = url_stripped.rsplit('@', 1)
        if len(parts) != 2:
            return url
            
        creds, host_part = parts
        
        if ':' in creds:
            user, password = creds.split(':', 1)
            # Only encode password if it's not already encoded
            if '%' not in password:
                encoded_password = urllib.parse.quote(password)
                sanitized = f"{prefix}{user}:{encoded_password}@{host_part}"
                logger.info(f"RTSP URL sanitized (password encoded)")
                return sanitized
        
        return url
    except Exception as e:
        logger.error(f"Error sanitizing RTSP URL: {e}")
        return url

class VideoProcessor:
    """
    Handles video capture and model inference.
    Uses a separate reader thread to ensure RTSP stability and zero lag.
    """
    def __init__(self, model, source, result_queue, stop_event, device='cpu', start_frame=0):
        self.model = model
        self.source = source
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = device
        self.start_frame = start_frame
        self.cap = None
        self.frame_buffer = queue.Queue(maxsize=1) # Only keep the freshest frame
    
    def _reader(self):
        """Dedicated thread for reading frames as fast as possible."""
        try:
            logger.info(f"Starting Video Reader with source: {self.source}")
            proc_source = sanitize_rtsp_url(self.source)
            
            is_rtsp = isinstance(self.source, str) and self.source.startswith('rtsp://')
            
            if is_rtsp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
                logger.info("RTSP Optimization: Forcing TCP transport")
            else:
                 # Clear RTSP options for local files to avoid issues
                 if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                     del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

            self.cap = cv2.VideoCapture(proc_source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                logger.error(f"Could not open video source: {proc_source}")
                return

            # RESUME LOGIC: Seek to start frame if file
            if not is_rtsp and self.start_frame > 0:
                logger.info(f"Resuming video from frame {self.start_frame}")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            # Only set small buffer for RTSP to reduce latency. detailed file reading doesn't need this restrictive buffer.
            if is_rtsp:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Re-initialize frame buffer with small size for RTSP (Low Latency)
                self.frame_buffer = queue.Queue(maxsize=2)
            else:
                # Re-initialize frame buffer with larger size for File (Smoothness)
                self.frame_buffer = queue.Queue(maxsize=64)
            
            retry_count = 0
            max_retries = 30 # Allow some retries for RTSP glitches

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    if is_rtsp:
                        retry_count += 1
                        if retry_count < max_retries:
                            if retry_count % 5 == 0:
                                logger.warning(f"RTSP Glitch: Retrying frame read ({retry_count}/{max_retries})...")
                                if retry_count >= 15:
                                    logger.info("Heavy corruption detected, re-opening stream...")
                                    self.cap.release()
                                    time.sleep(1.0)
                                    self.cap = cv2.VideoCapture(proc_source, cv2.CAP_FFMPEG)
                            time.sleep(0.05)
                            continue
                        else:
                            logger.info("End of stream or connection lost")
                            break
                    else:
                        # Local File: Loop video for testing
                        logger.info("End of video file reached, re-starting logic...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                
                # Check current position for resume logic
                # We can't easily push this back to main thread per frame, but we rely on main thread counting or inference
                
                retry_count = 0 
                
                if is_rtsp:
                    # RTSP MODE: Drop frames if queue is full (Keep Real-Time)
                    if not self.frame_buffer.full():
                        self.frame_buffer.put(frame)
                    else:
                        try:
                            self.frame_buffer.get_nowait() # Remove old
                            self.frame_buffer.put(frame)   # Add new
                        except: pass
                else:
                    # FILE MODE: BLOCK if queue is full (Process Every Frame)
                    # This slows down the 'reader' to match 'processing' speed
                    while not self.stop_event.is_set():
                        try:
                            self.frame_buffer.put(frame, timeout=0.5)
                            break
                        except queue.Full:
                            continue 


        except Exception as e:
            logger.error(f"Reader thread error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            logger.info("Reader thread stopped")

    def run(self):
        # Start capture thread
        reader_thread = threading.Thread(target=self._reader, daemon=True)
        reader_thread.start()
        
        while not self.stop_event.is_set():
            try:
                # Wait for the next available frame (blocking with timeout)
                try:
                    frame = self.frame_buffer.get(timeout=1.0)
                except queue.Empty:
                    if not reader_thread.is_alive(): break
                    continue

                # Skip processing if queue is already full (keeps latency minimal)
                if self.result_queue.full():
                    continue

                # Run Inference
                try:
                    start_time = time.time()
                    results = self.model.track(
                        frame,
                        persist=True,
                        tracker='custom_tracker.yaml',
                        device=self.device,
                        verbose=False,
                        imgsz=640,  # Optimized for speed (Standard YOLO size)
                        conf=0.25,  # Standard confidence
                        iou=0.45,   # Standard IOU
                        half=(self.device == 'cuda'),
                        max_det=50,
                        classes=[2, 3, 5, 7] # Filter: car, motorcycle, bus, truck
                    )
                    inference_time = (time.time() - start_time) * 1000
                    logger.info(f"Frame processed in {inference_time:.1f}ms")
                    
                    # Put result in queue
                    self.result_queue.put((frame.copy(), results[0] if results else None))
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    continue

            except Exception as e:
                logger.error(f"VideoProcessor run loop error: {e}")
                time.sleep(0.1)

        logger.info("Inference processor stopped")

class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vehicle Counter - Robust Threading")
        self.root.geometry("1200x900")
        
        # Configure Styles
        self.setup_styles()
        
        # Check for CUDA        # Data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Application State
        self.model = None
        self.processor_thread = None
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue(maxsize=2) # Small buffer to keep latency low
        self.is_running = False
        self.current_source = None # Track the active source (video/image)
        
        # Stats
        # Counting State
        self.zone_points = []
        self.drawing_zone = False
        self.zone_defined = False
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Use a Dict of Sets to track unique IDs per class
        # { 'Car': {'in': {1, 2}, 'out': {5}}, ... }
        self.counted_ids = defaultdict(lambda: {'in': set(), 'out': set()})
        self.total_counts = defaultdict(lambda: {'in': 0, 'out': 0})
        self.track_history = defaultdict(int)
        # Track usage history for Robust Counting
        # { track_id: {'min_y': val, 'max_y': val, 'start_y': val} }
        self.track_data = defaultdict(lambda: {'min_y': float('inf'), 'max_y': float('-inf'), 'start_y': None})
        # Track previous positions to detect direction
        self.prev_positions = {}  # {track_id: (cx, cy)}
        self.frame_count_fps = 0
        self.last_fps_update = time.time()
        self.current_frame = None # Store for redrawing zone
        self.video_position = 0 # Track current frame index for resume
        
        # Database Initialization
        self.init_db()
        
        # Initialize OCR in background
        # Initialize OCR in background
        self.reader = None
        self.ocr_queue = queue.Queue()
        self.plate_cache = {} # Cache for Last Known Plate: {track_id: plate_text}
        
        # Start background threads
        threading.Thread(target=self.init_ocr, daemon=True).start()
        threading.Thread(target=self.ocr_worker_loop, daemon=True).start()
        
        # UI
        self.create_widgets()
        self.load_settings()
        self.update_status("Initializing... Loading Model...")
        
        # Start DB Polling for logs
        self.poll_db_logs()
        
        # Check for Updates (New Feature)
        self.root.after(2000, self.check_for_updates)
        
        # Load Model Async
        # Force usage of standard YOLOv8n model
        initial_model = 'yolov8n.pt'
        threading.Thread(target=self.load_model, args=(initial_model,), daemon=True).start()

    def check_for_updates(self):
        """Check GitHub for new releases"""
        if not getattr(sys, 'frozen', False):
            return # Only update frozen exe
            
        try:
            logger.info("Checking for updates...")
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get('tag_name', '')
                
                if latest_version and latest_version != CURRENT_VERSION:
                    logger.info(f"Update available: {latest_version}")
                    msg = f"A new version ({latest_version}) is available!\n\nCurrent Version: {CURRENT_VERSION}\n\nDo you want to download and update now?"
                    if messagebox.askyesno("Update Available", msg):
                        # Find asset
                        assets = data.get('assets', [])
                        download_url = None
                        for asset in assets:
                            if asset['name'].endswith('.zip') and 'VehicleCounter' in asset['name']:
                                download_url = asset['browser_download_url']
                                break
                        
                        if download_url:
                            self.perform_update(download_url, latest_version)
                        else:
                            messagebox.showerror("Update Error", "Could not find a valid update file in the release.")
            
        except Exception as e:
            logger.error(f"Update check failed: {e}")

    def perform_update(self, url, version):
        """Download and install update"""
        try:
            self.update_status(f"Downloading update {version}...")
            
            # 1. Download Zip
            r = requests.get(url, stream=True)
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "update.zip")
            
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.update_status("Installing update...")
            
            # 2. Extract
            extract_path = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # 3. Find inner folder if nested (e.g. dist/VehicleCounter)
            # We assume the zip contains the CONTENTS of the app folder or the folder itself
            # Let's find the exe to be sure
            new_exe_path = None
            source_root = extract_path
            
            for root, dirs, files in os.walk(extract_path):
                if "VehicleCounter.exe" in files:
                    source_root = root
                    new_exe_path = os.path.join(root, "VehicleCounter.exe")
                    break
            
            if not new_exe_path:
                raise Exception("Invalid update package: VehicleCounter.exe not found")

            # 4. Create Updater Batch Script
            # We need to close this app, wait, delete old files, copy new files, restart
            current_exe = sys.executable
            current_dir = os.path.dirname(current_exe)
            
            updater_bat = os.path.join(temp_dir, "updater.bat")
            
            bat_content = f"""
@echo off
echo Updating VehicleCounter to {version}...
timeout /t 3 /nobreak > NUL
echo Copying files...
xcopy /E /H /Y "{source_root}\\*" "{current_dir}\\" > NUL
echo Cleaning up...
start "" "{current_exe}"
del "{updater_bat}"
"""
            with open(updater_bat, 'w') as f:
                f.write(bat_content)
                
            # 5. Launch Updater and Exit
            subprocess.Popen([updater_bat], shell=True)
            self.root.quit()
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            messagebox.showerror("Update Error", f"Failed to update: {e}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam') # 'clam' allows more custom color configurations than 'vista'
        
        # Corporate Color Palette
        self.colors = {
            'bg_main': '#F4F6F9',       # Light Blue-Gray background
            'bg_card': '#FFFFFF',       # White Card background
            'primary': '#2C3E50',       # Dark Blue Header
            'accent': '#3498DB',        # Bright Blue Buttons/Highlights
            'text_main': '#2C3E50',     # Dark Text
            'text_light': '#7F8C8D',    # Gray Text
            'success': '#27AE60',       # Green
            'danger': '#E74C3C',        # Red
            'warning': '#F39C12'        # Orange
        }
        
        # Configure Standard Elements
        style.configure("TFrame", background=self.colors['bg_main'])
        style.configure("Card.TFrame", background=self.colors['bg_card'], relief="flat")
        
        # Labels
        style.configure("TLabel", background=self.colors['bg_main'], foreground=self.colors['text_main'], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.colors['bg_card'], foreground=self.colors['text_main'], font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.colors['primary'], foreground="white", font=("Segoe UI", 14, "bold"), padding=10)
        style.configure("SubHeader.TLabel", background=self.colors['bg_card'], foreground=self.colors['text_main'], font=("Segoe UI", 12, "bold"))
        style.configure("BigStat.TLabel", background=self.colors['bg_card'], foreground=self.colors['accent'], font=("Segoe UI", 24, "bold"))
        
        # Buttons
        style.configure("TButton", 
                        font=("Segoe UI", 10, "bold"), 
                        background=self.colors['accent'], 
                        foreground="white", 
                        borderwidth=0, 
                        focuscolor="none", 
                        padding=8)
        style.map("TButton", 
                  background=[('active', '#2980B9'), ('disabled', '#BDC3C7')],
                  foreground=[('disabled', '#7F8C8D')])
        
        style.configure("Danger.TButton", background=self.colors['danger'])
        style.map("Danger.TButton", background=[('active', '#C0392B')])

        # Notebook (Tabs) - Modern Navigation Bar Style
        style.configure("TNotebook", background=self.colors['bg_main'], borderwidth=0, tabmargins=[0, 10, 0, 0])
        style.configure("TNotebook.Tab", 
                        padding=[30, 12], 
                        font=("Segoe UI", 11, "bold"),
                        background="white",
                        foreground=self.colors['text_light'],
                        borderwidth=0,
                        focuscolor=self.colors['bg_main']) # Remove focus ring
                        
        style.map("TNotebook.Tab", 
                  background=[("selected", self.colors['primary']), ('active', '#ECF0F1')], 
                  foreground=[("selected", "white"), ('active', self.colors['primary'])])
        
        # Treeview (Log Table)
        style.configure("Treeview", 
                        background="white", 
                        foreground=self.colors['text_main'], 
                        fieldbackground="white", 
                        rowheight=30,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading", 
                        background=self.colors['primary'], 
                        foreground="white", 
                        font=("Segoe UI", 10, "bold"),
                        relief="flat")
        style.map("Treeview", background=[("selected", self.colors['accent'])], foreground=[("selected", "white")])
        
        # Labelframes
        style.configure("TLabelframe", background=self.colors['bg_main'], borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=self.colors['bg_main'], foreground=self.colors['text_main'], font=("Segoe UI", 10, "bold")) 
        
        # Card Labelframes (White background)
        style.configure("Card.TLabelframe", background=self.colors['bg_card'], borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=self.colors['bg_card'], foreground=self.colors['text_main'], font=("Segoe UI", 10, "bold"))
        
    def init_ocr(self):
        try:
            logger.info("Initializing EasyOCR (Background)...")
            
            # Detect if running as EXE (frozen) or script
            if getattr(sys, 'frozen', False):
                # PyInstaller Bundled Path
                base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.abspath(__file__))
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Look for models in 'models/ocr' relative to the app
            local_model_dir = os.path.join(base_path, 'models', 'ocr')
            
            if os.path.exists(local_model_dir):
                logger.info(f"Using bundled OCR models from: {local_model_dir}")
                self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), 
                                           model_storage_directory=local_model_dir,
                                           download_enabled=False)
            else:
                logger.info("Using default EasyOCR storage")
                self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                
            logger.info("EasyOCR Initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

    def create_widgets(self):
        self.root.state('zoomed')
        
        # Main Theme background
        self.root.configure(bg=self.colors['bg_main'])
        
        # Header
        header_frame = ttk.Frame(self.root, style="Header.TLabel") # Uses primary color
        header_frame.pack(fill=tk.X)
        
        logo_lbl = ttk.Label(header_frame, text="GATE SYSTEM - VEHICLE TRACKER", style="Header.TLabel")
        logo_lbl.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Power Button (Exit)
        ttk.Button(header_frame, text="EXIT APP", command=self.on_close, style="Danger.TButton").pack(side=tk.RIGHT, padx=20)
        
        # Tabbed Layout
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # --- TAB 1: Live View ---
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="  LIVE DASHBOARD  ")
        
        live_container = ttk.Frame(self.live_tab)
        live_container.pack(fill=tk.BOTH, expand=True)
        
        # Left: Video (70% Width)
        self.video_frame = ttk.Frame(live_container)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(self.video_frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Double-Button-1>", self.open_zone_dialog)
        
        # Right: Quick Controls & Live Stats (30%)
        live_right_panel = ttk.Frame(live_container, width=400)
        live_right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        
        # Controls Group (Card Style)
        self.controls_frame = ttk.LabelFrame(live_right_panel, text="System Control", padding=15, style="Card.TLabelframe")
        self.controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.controls_frame, text="Video Source:", style="Card.TLabel").pack(anchor=tk.W)
        source_box = ttk.Frame(self.controls_frame, style="Card.TFrame")
        source_box.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(source_box, textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_btn = ttk.Button(source_box, text="...", width=4, command=self.browse_file)
        self.browse_btn.pack(side=tk.RIGHT)
        
        self.rtsp_url = tk.StringVar(value="rtsp://")
        
        btn_box = ttk.Frame(self.controls_frame, style="Card.TFrame")
        btn_box.pack(fill=tk.X, pady=15)
        self.start_btn = ttk.Button(btn_box, text="START SYSTEM", command=self.start_file_processing, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(btn_box, text="STOP", command=self.stop_processing, style="Danger.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Last Plate Detected Display
        self.plate_frame = ttk.LabelFrame(live_right_panel, text="Last Detected Plate", padding=20, style="Card.TLabelframe")
        self.plate_frame.pack(fill=tk.X, pady=15)
        
        self.last_plate_label = ttk.Label(self.plate_frame, text="NO PLATE", style="BigStat.TLabel", anchor="center")
        self.last_plate_label.pack(fill=tk.X)
        self.last_plate_type = ttk.Label(self.plate_frame, text="Waiting...", style="Card.TLabel", anchor="center", font=("Segoe UI", 10, "italic"))
        self.last_plate_type.pack(fill=tk.X)
        
        # Live Stats Group
        self.stats_frame = ttk.LabelFrame(live_right_panel, text="Vehicle Counts", padding=15, style="Card.TLabelframe")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.count_labels = {}
        
        # Add scrollable canvas for stats if many classes
        self.stats_canvas = tk.Canvas(self.stats_frame, bg=self.colors['bg_card'], highlightthickness=0)
        self.stats_scrollbar = ttk.Scrollbar(self.stats_frame, orient="vertical", command=self.stats_canvas.yview)
        self.scrollable_stats_frame = ttk.Frame(self.stats_canvas, style="Card.TFrame")
        
        self.stats_scroll_window = self.stats_canvas.create_window((0, 0), window=self.scrollable_stats_frame, anchor="nw")
        
        self.stats_canvas.configure(yscrollcommand=self.stats_scrollbar.set)
        
        self.stats_canvas.pack(side="left", fill="both", expand=True)
        self.stats_scrollbar.pack(side="right", fill="y")
        
        self.scrollable_stats_frame.bind(
            "<Configure>",
            lambda e: self.stats_canvas.configure(
                scrollregion=self.stats_canvas.bbox("all")
            )
        )

            
        # FPS Label at bottom of right panel
        self.fps_label = ttk.Label(live_right_panel, text="FPS: 0.0", font=("Arial", 10))
        self.fps_label.pack(side=tk.BOTTOM, pady=10)
        
        # --- TAB 2: LOGS & HISTORY ---
        self.logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_tab, text="  GATE LOGS / HISTORY  ")
        
        logs_container = ttk.Frame(self.logs_tab)
        logs_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Filter Bar
        filter_bar = ttk.Frame(logs_container)
        filter_bar.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(filter_bar, text="Type:").pack(side=tk.LEFT, padx=5)
        self.filter_type_var = tk.StringVar(value="All")
        self.filter_type_combo = ttk.Combobox(filter_bar, textvariable=self.filter_type_var, width=12)
        self.filter_type_combo['values'] = ("All", "Car", "Truck", "Bus", "Motorcycle", "Auto")
        self.filter_type_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_bar, text="Plate:").pack(side=tk.LEFT, padx=5)
        self.filter_plate_var = tk.StringVar()
        ttk.Entry(filter_bar, textvariable=self.filter_plate_var, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_bar, text="Date:").pack(side=tk.LEFT, padx=5)
        self.filter_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.date_entry = ttk.Entry(filter_bar, textvariable=self.filter_date_var, width=12)
        self.date_entry.pack(side=tk.LEFT, padx=5)
        self.date_entry.bind("<Button-1>", lambda e: self.show_calendar())
        
        ttk.Button(filter_bar, text="Search Logs", command=self.load_history_logs).pack(side=tk.LEFT, padx=15)
        
        # Log Table
        self.tree_frame = ttk.Frame(logs_container)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("id", "type", "plate", "direction", "time", "conf")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=20)
        
        self.tree.heading("id", text="Track ID")
        self.tree.heading("type", text="Vehicle Type")
        self.tree.heading("plate", text="License Plate")
        self.tree.heading("direction", text="Direction")
        self.tree.heading("time", text="Timestamp")
        self.tree.heading("conf", text="Confidence")
        
        self.tree.column("id", width=80)
        self.tree.column("type", width=100)
        self.tree.column("plate", width=150)
        self.tree.column("direction", width=100)
        self.tree.column("time", width=200)
        self.tree.column("conf", width=100)
        
        self.tree_scroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tree_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # --- TAB 3: SYSTEM SETTINGS ---
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="  SYSTEM SETTINGS  ")
        
        settings_container = ttk.Frame(self.settings_tab)
        settings_container.pack(fill=tk.BOTH, expand=True, padx=50, pady=30)
        
        # Stream Config
        stream_frame = ttk.LabelFrame(settings_container, text="Stream Configuration", padding=20)
        stream_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stream_frame, text="Default RTSP URL:").pack(anchor=tk.W)
        ttk.Entry(stream_frame, textvariable=self.rtsp_url, width=60).pack(fill=tk.X, pady=5)
        ttk.Label(stream_frame, text="Example: rtsp://username:password@ip_address:port/stream", font=("Segoe UI", 8), foreground="gray").pack(anchor=tk.W)
        
        # Action Buttons
        actions_frame = ttk.Frame(settings_container)
        actions_frame.pack(fill=tk.X, pady=30)
        
        ttk.Button(actions_frame, text="SAVE & APPLY SETTINGS", command=self.save_all_settings).pack(side=tk.LEFT, padx=5)
        
        # Status Bar at very bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress.start(10)

    def load_model(self, model_path=None):
        try:
            if model_path is None:
                model_path = 'models/yolov8n.pt'
                if not os.path.exists('models'):
                    os.makedirs('models')
                
                # Download if needed (simple check)
                if not os.path.exists(model_path):
                    self.root.after(0, lambda: self.update_status("Downloading Model..."))
                    YOLO('yolov8n.pt').export() # Triggers download
            
            self.root.after(0, lambda: self.update_status(f"Loading {os.path.basename(model_path)}..."))
            self.model = YOLO(model_path)
            if self.device == 'cuda':
                self.model.to('cuda')
                
                # Warmup
                self.root.after(0, lambda: self.update_status("Warming up GPU (Tracking)..."))
                logger.info("Warming up GPU with tracker...")
                dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
                for _ in range(3):
                    # Warmup track specifically as it initializes different components than predict
                    self.model.track(dummy_input, persist=True, device='cuda', verbose=False, half=True)
                logger.info("GPU Warmup Complete")
                
            logger.info(f"Model loaded: {model_path} on {self.device.upper()}")
            
            # Update UI for classes
            self.root.after(0, self.setup_stats_ui)
            
            # Enable buttons on main thread
            self.root.after(0, self.enable_ui)
            self.root.after(0, lambda: self.update_status(f"Model Ready: {os.path.basename(model_path)} ({self.device.upper()})"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
        finally:
             self.root.after(0, self.progress.stop)
             self.root.after(0, self.progress.destroy)

    def init_db(self):
        """Initialize local SQLite database for logging"""
        try:
            self.conn = sqlite3.connect('gate_log.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    vehicle_type TEXT,
                    track_id INTEGER,
                    direction TEXT,
                    confidence REAL,
                    plate_number TEXT
                )
            ''')
            # Migration: Add plate_number if it doesn't exist
            try:
                self.cursor.execute('ALTER TABLE vehicle_logs ADD COLUMN plate_number TEXT')
            except sqlite3.OperationalError:
                pass # Column already exists
            
            # Create settings table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            self.conn.commit()
            
            logger.info("Database initialized with settings support")
        except Exception as e:
            logger.error(f"Database error: {e}")

    def load_settings(self):
        """Load system settings from database"""
        try:
            self.cursor.execute("SELECT key, value FROM settings")
            rows = self.cursor.fetchall()
            settings = {row[0]: row[1] for row in rows}
            
            # Apply to UI
            if 'rtsp_url' in settings:
                self.rtsp_url.set(settings['rtsp_url'])
            
            logger.info(f"Settings loaded: {settings}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def save_setting(self, key, value):
        """Save a single setting to database"""
        try:
            self.cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            self.conn.commit()
            logger.info(f"Setting saved: {key}={value}")
        except Exception as e:
            logger.error(f"Failed to save setting {key}: {e}")

    def is_valid_indian_plate(self, text):
        """Helper to validate if text matches strong Indian plate format"""
        import re
        if not text: return False
        return bool(re.match(r'^[A-Z]{2}\d{1,2}[A-Z]+\d{4}$', text))

    def get_best_plate(self, track_id):
        """Find the most frequent/stable plate for a track_id"""
        if track_id not in self.plate_history or not self.plate_history[track_id]:
            return None
        
        # Get most common plate from history
        counts = Counter(self.plate_history[track_id])
        best_plate, count = counts.most_common(1)[0]
        return best_plate

    def ocr_worker_loop(self):
        """Background thread to process OCR tasks sequentially"""
        while not self.stop_event.is_set():
            try:
                task = self.ocr_queue.get(timeout=1.0)
                
                if task['type'] == 'scan' or task['type'] == 'log':
                    img = task['image']
                    track_id = task['track_id']
                    
                    text, conf = self.run_single_ocr(img)
                    
                    if text:
                        # LOGIC: Trust "Strong" (Regex-validated) matches over history of weak matches
                        is_strong = (conf > 0.9)
                        
                        # Add to history for voting
                        if not hasattr(self, 'plate_history'): self.plate_history = defaultdict(list)
                        
                        # If HIGH CONFIDENCE, it overrides previous low-confidence noise
                        # Check if we already have a strong history
                        current_history = self.plate_history[track_id]
                        has_strong_in_history = any(self.is_valid_indian_plate(p) for p in current_history)
                        
                        if is_strong:
                             # If this is the first strong match, CLEAR the old noise
                             if not has_strong_in_history:
                                 self.plate_history[track_id] = [text] # Reset to this good one
                                 logger.info(f"Strong Plate Override for #{track_id}: {text}")
                             else:
                                 self.plate_history[track_id].append(text)
                        else:
                             # Low confidence: Only add if we DON'T have a locked-in strong plate yet
                             if not has_strong_in_history:
                                 self.plate_history[track_id].append(text)

                        # Keep history limited to last 10 reads
                        if len(self.plate_history[track_id]) > 10:
                            self.plate_history[track_id].pop(0)

                        # Get voted "Best" plate
                        stable_plate = self.get_best_plate(track_id)
                        
                        # Update cache
                        self.plate_cache[track_id] = stable_plate
                        
                        # Update Last Detected Plate UI
                        if stable_plate:
                             try:
                                self.root.after(0, lambda p=stable_plate, t=task['vehicle_type']: self.update_last_plate_ui(p, t))
                             except: pass
                             
                        # If simple logging task
                        if task['type'] == 'log':
                            self.log_to_db(task['vehicle_type'], track_id, task['direction'], task['conf'], stable_plate)
                            
                self.ocr_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"OCR Worker Error: {e}")

    def run_single_ocr(self, img):
        """Helper to run actual EasyOCR on a cropped image (Background Thread)"""
        if self.reader is None or img is None or img.size == 0:
            return None, None
            
        try:
            # OPTIMIZATION: Crop to remove top 30% (Removes Roof/Header text like 'PYDAH')
            h, w = img.shape[:2]
            # Safety check for very small images
            if h > 50:
                # Skip top 10% (Reduced from 30% to avoid cutting high plates)
                crop_y = int(h * 0.10) 
                plate_roi = img[crop_y:h, :]
            else:
                plate_roi = img

            # Preprocessing
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            
            # 1. Noise Reduction
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
            
            # 2. Contrast Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(bfilter)
            
            # 3. Binarization (Black & White) - Critical for sharp text
            # Use simple thresholding or adaptive
            _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run OCR with paragraph=True to handle multi-line plates (common in India)
            # detail=1 gives boxes, but paragraph=True merges lines automatically
            ocr_result = self.reader.readtext(enhanced, detail=1, paragraph=True)
            
            best_text = None
            best_conf = 0.0
            
            for (bbox, text) in ocr_result:
                 logger.info(f"DEBUG OCR RAW: {text}") # Debug print
                 
                 # Clean up: "AP 39 UH 7092" -> "AP39UH7092"
                 clean_text = "".join(filter(str.isalnum, text)).upper()
                 
                 # FILTER: Remove common noise words instead of discarding the whole text
                 ignored_words = ["PYDAH", "TATA", "ASHOK", "LEYLAND", "INDIA", "STOP", "HORN", "SOUND", "KEEP", "DISTANCE", "SPEED", "40KM", "KERALA", "STATE", "TOURIST", "LBLATA", "MARCOPOLO", "BHARATBENZ"]
                 for ignored in ignored_words:
                     clean_text = clean_text.replace(ignored, "")
                 
                 # CORRECTION: Fix common State Code errors (e.g., GP -> AP)
                 # Map of Invalid -> Valid
                 state_corrections = {
                    "GP": "AP", "6P": "AP", "8P": "AP", "4P": "AP", "MP":"AP", "JP": "AP", # Misreads for AP
                    "T5": "TS", "I5": "TS", # Misreads for TS
                    "KA": "KA", "TN": "TN", "KL": "KL", "OD": "OD", "MH": "MH"
                 }
                 
                 # Check first 2 chars
                 if len(clean_text) >= 2:
                     first_two = clean_text[:2]
                     if first_two in state_corrections:
                         clean_text = state_corrections[first_two] + clean_text[2:]

                 # ADVANCED FORMAT CORRECTION: Handle AP 39 34 7092 -> AP 39 UW 7092
                 # Standard format: 2 Letters + 2 Numbers + 1-2 Letters + 4 Numbers
                 # If we have enough chars, try to enforce this
                 import re
                 # This regex looks for chunks attempting to match the format
                 if len(clean_text) >= 8:
                     # Check the "Series" part (Chars 5 and 6 approximately)
                     # Example: AP39 | 34 | 7092 -> The '34' should be letters
                     
                     # Helper to swap digit->char
                     def digit_to_char(c):
                         mapping = {'0':'O', '1':'I', '2':'Z', '3':'W', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', '9':'P'}
                         return mapping.get(c, c)
                         
                     # If format effectively looks like 4 blocks merged or just a long string
                     # We assume first 4 are State+Dist (AP39)
                     # Last 4 are Number (7092)
                     # Middle part MUST be alpha
                     
                     prefix = clean_text[:4] # AP39
                     suffix = clean_text[-4:] # 7092
                     middle = clean_text[4:-4] # 34 or UW
                     
                     if len(middle) > 0 and len(middle) <= 3:
                         new_middle = ""
                         for char in middle:
                             if char.isdigit():
                                 new_middle += digit_to_char(char)
                             else:
                                 new_middle += char
                         clean_text = prefix + new_middle + suffix
                         logger.info(f"Corrected Plate Format (Middle): {clean_text}")

                     # NEW: Enforce Last 4 to be Digits (Suffix)
                     suffix = clean_text[-4:]
                     if len(clean_text) >= 8 and not suffix.isdigit():
                         # Helper to swap char->digit
                         def char_to_digit(c):
                             mapping = {'O':'0', 'D':'0', 'I':'1', 'Z':'2', 'S':'5', 'G':'6', 'T':'7', 'B':'8', 'P':'9', 'A':'4'}
                             return mapping.get(c, c)
                        
                         new_suffix = ""
                         for char in suffix:
                             if not char.isdigit():
                                  new_suffix += char_to_digit(char)
                             else:
                                  new_suffix += char
                         
                         clean_text = clean_text[:-4] + new_suffix
                         logger.info(f"Corrected Plate Format (Suffix): {clean_text}")

                     if len(clean_text) >= 5:
                         
                         # STRICT REGEX VALIDATION for Indian Plates
                         # Format: 2 Letters (State) + 1-2 Digits (Dist) + 1-3 Letters (Series) + 4 Digits (Num)
                         # Example: AP 39 UW 7092
                         if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]+\d{4}$', clean_text):
                             # Excellent match
                             best_text = clean_text
                             best_conf = 0.95
                             logger.info(f"Strong Plate Match: {best_text}")
                             break
                         
                         # Fallback: Check digit count and structure looser
                         digit_count = sum(c.isdigit() for c in clean_text)
                         alpha_count = sum(c.isalpha() for c in clean_text)
                         
                         # Heuristic: Valid plate usually has balanced mix. Garbage mostly letters.
                         # RAISED THRESHOLD: Require at least 3 digits and 3 letters to accept a non-regex result
                         if digit_count >= 3 and alpha_count >= 3:
                             best_text = clean_text
                             best_conf = 0.6
                             # Keep searching for a better regex match, but store this just in case
                         
                         # If it looks like garbage "LBLATA..." (mostly letters, few digits at end?), skip
                         # Strict skipping for noise
                         if alpha_count > digit_count * 2: 
                             continue 
            
            return best_text, best_conf
            
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return None, None

    def log_to_db(self, vehicle_type, track_id, direction, confidence, plate_number=None):
        """Save OR UPDATE a detection event to the database"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if we already logged this track_id recently (to avoid duplicates or improve existing log)
            # We look for a log with same track_id today
            self.cursor.execute("SELECT id, plate_number FROM vehicle_logs WHERE track_id=? AND timestamp LIKE ? ORDER BY id DESC LIMIT 1", 
                              (track_id, f"{timestamp[:10]}%"))
            existing = self.cursor.fetchone()
            
            if existing:
                row_id, existing_plate = existing
                # If we have a new plate and the old one was None or different, UPDATE it
                # We prioritize the "Latest" reading as it might be closer/better
                if plate_number and plate_number != existing_plate:
                     self.cursor.execute("UPDATE vehicle_logs SET plate_number=?, confidence=? WHERE id=?", 
                                       (plate_number, confidence, row_id))
                     self.conn.commit()
                     logger.info(f"DB Update: #{track_id} plate updated to {plate_number}")
                else:
                    # Duplicate or no improvement, ignore
                    pass
            else:
                # INSERT NEW
                self.cursor.execute('''
                    INSERT INTO vehicle_logs (timestamp, vehicle_type, track_id, direction, confidence, plate_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, vehicle_type, track_id, direction, confidence, plate_number))
                self.conn.commit()
                logger.info(f"DB Log: {vehicle_type} #{track_id} {direction} {f'[{plate_number}]' if plate_number else ''}")
                
        except Exception as e:
            logger.error(f"Failed to log to DB: {e}")

    def update_last_plate_ui(self, plate_text, vehicle_type):
        """Update the UI labels for the last detected plate"""
        try:
            self.last_plate_label.config(text=plate_text, foreground=self.colors['accent'])
            self.last_plate_type.config(text=f"Detected Vehicle: {vehicle_type}", foreground=self.colors['success'])
        except Exception as e:
            pass

    def setup_stats_ui(self):
        # Clear existing
        for widget in self.scrollable_stats_frame.winfo_children():
            widget.destroy()
        self.count_labels.clear()
        
        # Define relevant vehicle classes (COCO indices)
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        relevant_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                
        self.active_classes = relevant_classes
        
        # Sort by name for display
        sorted_items = sorted(self.active_classes.items(), key=lambda x: x[1])
                
        for cls_id, cls_name in sorted_items:
            row = ttk.Frame(self.scrollable_stats_frame)
            row.pack(fill=tk.X, pady=2, padx=5)
            
            # Icon/Name
            ttk.Label(row, text=f"{cls_name}:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            
            # Count Value
            lbl = ttk.Label(row, text="0 (In:0 Out:0)", font=("Arial", 10), foreground="#007acc")
            lbl.pack(side=tk.RIGHT)
            self.count_labels[cls_name] = lbl
            
            # Reset counters
            self.counted_ids.clear()
            self.total_counts.clear()

    def load_custom_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            self.progress = ttk.Progressbar(self.root, mode='indeterminate')
            self.progress.pack(side=tk.BOTTOM, fill=tk.X)
            self.progress.start(10)
            threading.Thread(target=self.load_model, args=(path,), daemon=True).start()


    def save_all_settings(self):
        """Save all current settings to database"""
        self.save_setting('rtsp_url', self.rtsp_url.get())
        messagebox.showinfo("Success", "Settings saved and applied!")

    def enable_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        # self.connect_btn.config(state=tk.NORMAL) # Removed in tabbed layout
        self.browse_btn.config(state=tk.NORMAL)
        self.update_status("Model Ready")

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video/Image", "*.mp4 *.avi *.jpg *.png *.mkv")])
        if path:
            self.file_path.set(path)
            # Don't enable start button yet - wait for zone to be defined
            # self.start_btn.config(state=tk.NORMAL)
            
            # Preview first frame for zone drawing
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.preview_video_frame(path)
            elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.preview_image(path)

    def preview_image(self, path):
        img = cv2.imread(path)
        if img is not None:
            self.current_frame = img
            self.display_frame(img, is_video=False)
            self.update_status("Image loaded. DOUBLE-CLICK video to set counting zone.")

    def preview_video_frame(self, path):
        """Show first frame of video for zone drawing"""
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            self.current_frame = frame
            self.display_frame(frame, is_video=True)
            self.update_status("Video loaded. DOUBLE-CLICK video to set counting zone, then click Start Processing.")
        else:
            self.update_status("Failed to load video preview.")

    def start_file_processing(self):
        # 1. Check for local file
        path = self.file_path.get()
        if path:
            self.start_processing(path)
            return
            
        # 2. Check for RTSP URL
        url = self.rtsp_url.get()
        if url and url != "rtsp://":
            self.start_processing(url)
            return
            
        messagebox.showwarning("Warning", "Please select a video file or configure an RTSP URL in Settings.")

    def start_processing(self, source):
        # If switching sources, reset video position
        if source != self.current_source:
             self.video_position = 0
        
        self.current_source = source
        
        # NOTE: Do NOT reset is_running here, wait until cleared
        if self.is_running:
            self.stop_processing()
            # Small delay to let thread die
            time.sleep(0.5)
            
        self.stop_event.clear()
        
        # Flush queue
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except queue.Empty: break
            
        # Reset Counting State
        self.counted_ids.clear()
        self.total_counts.clear()
        self.track_data.clear()
        self.prev_positions = {}
        
        # CLEAR CACHES to prevent stale data on restart
        if hasattr(self, 'plate_cache'): self.plate_cache.clear()
        if hasattr(self, 'plate_history'): self.plate_history.clear()
        if hasattr(self, 'last_plate_label'): self.last_plate_label.config(text="NO PLATE")
        if hasattr(self, 'last_plate_type'): self.last_plate_type.config(text="Waiting...")
        
        # Calculate Zone Center Line (in frame coords) for crossing check
        self.zone_center_y = None
        if self.zone_defined and len(self.zone_points) == 2 and hasattr(self, 'scale') and self.scale > 0:
             z_y1_c = self.zone_points[0][1]
             z_y2_c = self.zone_points[1][1]
             # Convert to frame coords
             z_y1 = (z_y1_c - self.offset_y) / self.scale
             z_y2 = (z_y2_c - self.offset_y) / self.scale
             self.zone_center_y = (z_y1 + z_y2) / 2
             logger.info(f"Zone Center Y Line at: {self.zone_center_y:.1f}")
            
        # Start Thread
        self.processor_thread = threading.Thread(
            target=VideoProcessor(self.model, source, self.result_queue, self.stop_event, self.device, start_frame=self.video_position).run,
            daemon=True
        )
        self.processor_thread.start()
        self.is_running = True
        
        # Start Polling
        self.poll_results()
        self.update_status(f"Processing: {source}")

    def stop_processing(self):
        self.stop_event.set()
        self.is_running = False
        self.update_status("Stopped.")

    def poll_db_logs(self):
        """Periodically check DB for new logs to update the Treeview if we are in Tab 2"""
        if self.notebook.index("current") == 1: # Tab index 1 is Logs
            self.load_history_logs()
            self.refresh_vehicle_types()
        self.root.after(5000, self.poll_db_logs)

    def refresh_vehicle_types(self):
        """Update the ComboBox with unique vehicle types from DB"""
        try:
            self.cursor.execute("SELECT DISTINCT vehicle_type FROM vehicle_logs ORDER BY vehicle_type")
            types = [r[0].capitalize() for r in self.cursor.fetchall() if r[0]]
            current_values = list(self.filter_type_combo['values'])
            new_values = ["All"] + sorted(list(set(types)))
            if set(current_values) != set(new_values):
                self.filter_type_combo['values'] = new_values
        except:
            pass

    def get_available_dates(self):
        """Fetch unique dates that have vehicle logs from the database"""
        try:
            self.cursor.execute("SELECT DISTINCT substr(timestamp, 1, 10) FROM vehicle_logs ORDER BY timestamp DESC")
            return [r[0] for r in self.cursor.fetchall() if r[0]]
        except Exception as e:
            logger.error(f"Error fetching available dates: {e}")
            return []

    def show_calendar(self):
        """Show a visual calendar picker popup"""
        top = tk.Toplevel(self.root)
        top.title("Select Date")
        top.geometry("300x320")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()
        
        # Center popup
        x = self.root.winfo_x() + 100
        y = self.root.winfo_y() + 100
        top.geometry(f"+{x}+{y}")
        
        # State for month/year
        try:
            current_date = datetime.strptime(self.filter_date_var.get(), "%Y-%m-%d")
        except:
            current_date = datetime.now()
            
        month_var = tk.IntVar(value=current_date.month)
        year_var = tk.IntVar(value=current_date.year)
        
        available_dates = self.get_available_dates()
        
        def refresh_calendar():
            # Clear previous grid
            for child in calendar_frame.winfo_children():
                child.destroy()
            
            m = month_var.get()
            y = year_var.get()
            
            # Header
            header_str = f"{calendar.month_name[m]} {y}"
            header_label.config(text=header_str)
            
            # Weekdays
            days_labels = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
            for i, day in enumerate(days_labels):
                lbl = ttk.Label(calendar_frame, text=day, font=("Segoe UI", 9, "bold"))
                lbl.grid(row=0, column=i, pady=5)
            
            # Days
            month_cal = calendar.monthcalendar(y, m)
            for row_idx, week in enumerate(month_cal):
                for col_idx, day in enumerate(week):
                    if day != 0:
                        date_str = f"{y}-{m:02d}-{day:02d}"
                        is_available = date_str in available_dates
                        
                        btn = tk.Button(calendar_frame, text=str(day), width=4, 
                                        relief="flat", activebackground="#0056b3",
                                        activeforeground="white")
                        
                        # Style based on availability
                        if is_available:
                            btn.config(bg="white", fg="black", font=("Segoe UI", 9, "bold"))
                            btn.config(command=lambda d=date_str: select_date(d))
                        else:
                            btn.config(bg="#f0f0f0", fg="#cccccc", state=tk.DISABLED)
                        
                        # Highlight current selection
                        if date_str == self.filter_date_var.get():
                             btn.config(bg="#0056b3", fg="white")
                        
                        btn.grid(row=row_idx+1, column=col_idx, padx=2, pady=2)

        def select_date(d):
            self.filter_date_var.set(d)
            top.destroy()
            self.load_history_logs()

        def prev_month():
            m = month_var.get() - 1
            if m < 1:
                month_var.set(12)
                year_var.set(year_var.get() - 1)
            else:
                month_var.set(m)
            refresh_calendar()

        def next_month():
            m = month_var.get() + 1
            if m > 12:
                month_var.set(1)
                year_var.set(year_var.get() + 1)
            else:
                month_var.set(m)
            refresh_calendar()

        # UI Layout
        nav_frame = ttk.Frame(top, padding=10)
        nav_frame.pack(fill=tk.X)
        
        ttk.Button(nav_frame, text="<", width=3, command=prev_month).pack(side=tk.LEFT)
        header_label = ttk.Label(nav_frame, text="", font=("Segoe UI", 10, "bold"))
        header_label.pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text=">", width=3, command=next_month).pack(side=tk.LEFT)
        
        calendar_frame = ttk.Frame(top, padding=10)
        calendar_frame.pack(fill=tk.BOTH, expand=True)
        
        refresh_calendar()

    def load_history_logs(self):
        """Fetch logs from DB based on filters with AM/PM and specific column order"""
        try:
            # Clear tree
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            v_type = self.filter_type_var.get()
            date_filter = self.filter_date_var.get()
            plate_filter = self.filter_plate_var.get()
            
            # Select columns in specific order: track_id, vehicle_type, plate_number, direction, timestamp, confidence
            query = "SELECT track_id, vehicle_type, plate_number, direction, timestamp, confidence FROM vehicle_logs WHERE timestamp LIKE ?"
            params = [f"{date_filter}%"]
            
            if v_type != "All":
                query += " AND vehicle_type = ?"
                params.append(v_type.lower())
            
            if plate_filter:
                query += " AND plate_number LIKE ?"
                params.append(f"%{plate_filter.upper()}%")
                
            query += " ORDER BY timestamp DESC LIMIT 200"
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            for row in rows:
                display_row = list(row)
                
                # Format Timestamp (Original is at index 4)
                try:
                    dt = datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")
                    display_row[4] = dt.strftime("%Y-%m-%d %I:%M:%S %p")
                except:
                    pass
                
                # Format Confidence (Original is at index 5)
                display_row[5] = f"{row[5]*100:.1f}%"
                
                self.tree.insert("", tk.END, values=display_row)
        except Exception as e:
            logger.error(f"Error loading history: {e}")

    def poll_results(self):
        if not self.is_running:
            return

        try:
            # Process all available items in queue to process fast
            # but limit to avoid UI freeze
            for _ in range(5): 
                try:
                    frame, results = self.result_queue.get_nowait()
                    self.update_stats(results, frame)
                except queue.Empty:
                    # Check if processing is complete
                    if hasattr(self, 'processor_thread') and not self.processor_thread.is_alive():
                        self.stop_processing()
                        self.update_status("Processing Complete.")
                        return
                    break
        finally:
            if self.is_running:
                self.root.after(10, self.poll_results)

    def get_plate_number(self, frame, box):
        """Extract and OCR license plate from a vehicle box. Returns (text, relative_bbox)"""
        if self.reader is None:
             return "No OCR", None
             
        try:
            h_frame, w_frame = frame.shape[:2]
            # Get coordinates
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
            
            # Crop vehicle with small padding
            pad = 10
            cx1, cy1 = max(0, vx1-pad), max(0, vy1-pad)
            cx2, cy2 = min(w_frame, vx2+pad), min(h_frame, vy2+pad)
            vehicle_img = frame[cy1:cy2, cx1:cx2]
            
            if vehicle_img.size == 0: return "Unknown", None

            # Image Enhancement for OCR
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Run OCR with detail=1 to get bboxes
            results = self.reader.readtext(enhanced, detail=1, paragraph=False)
            
            best_text = "Unknown"
            best_bbox = None
            max_digits = 0
            
            import re
            digit_pattern = re.compile(r'\d')

            for (bbox, text, prob) in results:
                if prob < 0.2: continue
                # Clean text
                clean = "".join(filter(str.isalnum, text)).upper()
                if len(clean) < 4: continue
                
                # Heuristic: Plate usually has many digits
                digit_count = len(digit_pattern.findall(clean))
                if digit_count >= max_digits:
                    max_digits = digit_count
                    best_text = clean
                    # Convert relative OCR bbox to absolute frame coordinates
                    # bbox is list of 4 points [(x,y), (x,y), (x,y), (x,y)] relative to vehicle_img
                    abs_bbox = []
                    for pt in bbox:
                        abs_bbox.append((int(pt[0] + cx1), int(pt[1] + cy1)))
                    best_bbox = abs_bbox

            return best_text, best_bbox
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return "Error", None

    def update_stats(self, results, frame):
        # 0. DEEP COPY CLEAN FRAME for OCR (Before any UI overlays are drawn)
        clean_frame = frame.copy()
        
        current_counts = defaultdict(int)
        h_frame, w_frame = frame.shape[:2]
        
        # 1. DRAW CROSSING LINE (Visual Only)
        # ... (rest of the visual drawing logic stays on 'frame', not 'clean_frame')
        if self.zone_defined and len(self.zone_points) == 2:
            z_x1_c, z_y1_c = self.zone_points[0]
            z_x2_c, z_y2_c = self.zone_points[1]
            if hasattr(self, 'scale') and self.scale > 0:
                z_x1 = (z_x1_c - self.offset_x) / self.scale
                z_y1 = (z_y1_c - self.offset_y) / self.scale
                z_x2 = (z_x2_c - self.offset_x) / self.scale
                z_y2 = (z_y2_c - self.offset_y) / self.scale
                zone_center_y = (z_y1 + z_y2) / 2
                line_x1, line_x2 = int(z_x1), int(z_x2)
            else:
                zone_center_y = int(h_frame * 0.65) # Lower default line for better tracking
                line_x1, line_x2 = 0, w_frame
        else:
            zone_center_y = int(h_frame * 0.65)
            line_x1, line_x2 = 0, w_frame

        cv2.line(frame, (line_x1, int(zone_center_y)), (line_x2, int(zone_center_y)), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, int(zone_center_y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 1.5. GLOBAL PLATE SCAN (Fallback) - Use CLEAN_FRAME
        found_any_vehicle = results and len(results.boxes) > 0
        should_scan_global = (not found_any_vehicle and getattr(self, 'frame_idx', 0) % 5 == 0) or (getattr(self, 'frame_idx', 0) % 20 == 0)
        self.frame_idx = getattr(self, 'frame_idx', 0) + 1

        if should_scan_global:
            strip_h = 300 # Larger strip for better coverage
            y_start = max(0, int(zone_center_y - strip_h // 2))
            # GLOBAL SCAN DISABLED: It causes duplicate '999' logs and overrides tracked vehicle logic.
            # Only rely on Tracked Vehicle OCR for consistency.
            pass

        if results:
            for box in results.boxes:
                # Class Mapping
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id] if self.model and hasattr(self.model, 'names') else str(cls_id)
                if not label: continue
                
                # FILTER: Only process labels that are in our custom stats UI
                if label not in self.count_labels:
                    continue

                # Coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                
                track_id = int(box.id[0].item()) if box.id is not None else None
                in_zone = self.is_in_zone(cx, cy) if self.zone_defined else True
                
                color = (0, 255, 0) if in_zone else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label_text = f"{label}"
                if track_id is not None:
                    label_text += f" #{track_id}"
                cv2.putText(frame, label_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 3. Counting Logic: Line Crossing with Direction
                if track_id is not None:
                    t_data = self.track_data[track_id]
                    t_data['min_y'] = min(t_data['min_y'], cy)
                    t_data['max_y'] = max(t_data['max_y'], cy)
                    if t_data['start_y'] is None:
                        t_data['start_y'] = cy
                    
                    # Robust Crossing Logic: Check if center point crossed the line
                    # We use the previous position (if available) to detect the exact frame of crossing
                    prev_cx, prev_cy = self.prev_positions.get(track_id, (None, None))
                    
                    if prev_cy is not None:
                        # Crossing Down (IN)
                        if prev_cy < zone_center_y and cy >= zone_center_y:
                            if label not in self.counted_ids: self.counted_ids[label] = { 'in': set(), 'out': set() }
                            
                            if track_id not in self.counted_ids[label]['in']:
                                self.counted_ids[label]['in'].add(track_id)
                                
                                # Queue Log Event
                                pad = 10
                                vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2)
                                cx1, cy1 = max(0, vx1-pad), max(0, vy1-pad)
                                cx2, cy2 = min(w_frame, vx2+pad), min(h_frame, vy2+pad)
                                vehicle_crop = clean_frame[cy1:cy2, cx1:cx2].copy()
                                
                                self.ocr_queue.put({
                                    'type': 'log',
                                    'image': vehicle_crop,
                                    'vehicle_type': label,
                                    'track_id': track_id,
                                    'direction': 'IN',
                                    'conf': float(box.conf[0].item())
                                })
                                logger.info(f"Counted IN: {label} #{track_id}")

                        # Crossing Up (OUT)
                        elif prev_cy > zone_center_y and cy <= zone_center_y:
                             if label not in self.counted_ids: self.counted_ids[label] = { 'in': set(), 'out': set() }
                             
                             if track_id not in self.counted_ids[label]['out']:
                                self.counted_ids[label]['out'].add(track_id)
                                
                                # Queue Log Event
                                pad = 10
                                vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2)
                                cx1, cy1 = max(0, vx1-pad), max(0, vy1-pad)
                                cx2, cy2 = min(w_frame, vx2+pad), min(h_frame, vy2+pad)
                                vehicle_crop = clean_frame[cy1:cy2, cx1:cx2].copy()
                                
                                self.ocr_queue.put({
                                    'type': 'log',
                                    'image': vehicle_crop,
                                    'vehicle_type': label,
                                    'track_id': track_id,
                                    'direction': 'OUT',
                                    'conf': float(box.conf[0].item())
                                })
                                logger.info(f"Counted OUT: {label} #{track_id}")

                    # 4. DRAW PLATE BOX (Visual Only - Non Blocking)
                    # Check cache
                    cached_plate = self.plate_cache.get(track_id)
                    
                    if cached_plate:
                        cv2.putText(frame, f"PLATE: {cached_plate}", (int(x1), int(y2)+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Force UI update if this is the active track (Safety Net)
                        # We use a simple throttle to avoid saturating the UI thread
                        if self.frame_count_fps % 10 == 0:
                             try:
                                self.root.after(0, lambda p=cached_plate, t=label: self.update_last_plate_ui(p, t))
                             except: pass
                    else:
                        # Periodically trigger a scan for display only (Every 30 frames approx)
                        if self.frame_count_fps % 30 == 0:
                             pad = 10
                             vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2)
                             cx1, cy1 = max(0, vx1-pad), max(0, vy1-pad)
                             cx2, cy2 = min(w_frame, vx2+pad), min(h_frame, vy2+pad)
                             vehicle_crop = clean_frame[cy1:cy2, cx1:cx2].copy()
                             
                             # Only queue if queue isn't backed up
                             if self.ocr_queue.qsize() < 3:
                                 self.ocr_queue.put({
                                    'type': 'scan',
                                    'image': vehicle_crop,
                                    'track_id': track_id
                                 })

                    self.prev_positions[track_id] = (cx, cy)

        # Update UI Labels
        for k in sorted(self.count_labels.keys()):
            in_count = len(self.counted_ids[k]['in'])
            out_count = len(self.counted_ids[k]['out'])
            lbl = self.count_labels[k]
            lbl.config(text=f"{in_count + out_count} (In:{in_count} Out:{out_count})")

        # FPS Calculation
        self.frame_count_fps += 1
        curr_time = time.time()
        if curr_time - self.last_fps_update >= 1.0:
            fps = self.frame_count_fps / (curr_time - self.last_fps_update)
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.frame_count_fps = 0
            self.last_fps_update = curr_time
            
        self.display_frame(frame)
        
        # Track position (simple increment) - this assumes 1 frame processed = 1 frame advanced
        # Ideally VideoProcessor would send this, but this is a good approximation for resume
        self.video_position += 1



    def display_frame(self, frame, is_video=True):
        self.current_frame = frame # Store latest frame for zone configuration
        # Resize to fit canvas
        canvas_w = max(self.canvas.winfo_width(), 640)
        canvas_h = max(self.canvas.winfo_height(), 480)
        
        h, w = frame.shape[:2]
        self.scale = min(canvas_w/w, canvas_h/h)
        new_w, new_h = int(w*self.scale), int(h*self.scale)
        
        # Calculate offsets for centering
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Convert to RGB for PIL
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.delete("all")
        # Draw image centered
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=img)
        self.canvas.image = img # Keep ref
        
        # Draw Zone Overlay on Canvas
        if self.zone_points:
            # Zone points are already in Canvas Coordinates
            if len(self.zone_points) >= 2:
                if is_video and len(self.zone_points) == 2:
                   # Draw rectangle with thick border and label
                   self.canvas.create_rectangle(self.zone_points[0][0], self.zone_points[0][1], 
                                              self.zone_points[1][0], self.zone_points[1][1], 
                                              outline="#00FF00", width=4, tags="zone")
                   # Add label
                   label_x = (self.zone_points[0][0] + self.zone_points[1][0]) // 2
                   label_y = self.zone_points[0][1] - 10
                   self.canvas.create_text(label_x, label_y, text="COUNTING ZONE", 
                                         fill="#00FF00", font=("Arial", 12, "bold"), tags="zone")
                else:
                   # Flatten for polygon
                   pts = [p for point in self.zone_points for p in point]
                   self.canvas.create_polygon(pts, outline="#00FF00", fill="", width=4, tags="zone")
                   # Add label at first point
                   self.canvas.create_text(self.zone_points[0][0], self.zone_points[0][1] - 10, 
                                         text="COUNTING ZONE", fill="#00FF00", 
                                         font=("Arial", 12, "bold"), tags="zone")



    def open_zone_dialog(self, event):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        # Default or current dimensions
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # Initial guess 
        init_w = 400
        init_h = 300
        
        # If zone already defined, try to reverse calc dimensions
        if self.zone_defined and len(self.zone_points) == 2:
             x1, y1 = self.zone_points[0]
             x2, y2 = self.zone_points[1]
             init_w = abs(x2 - x1)
             init_h = abs(y2 - y1)

        # Open Custom Dialog
        ZoneConfigDialog(self.root, init_w, init_h, self.update_zone_preview)

    def update_zone_preview(self, width, height):
        if not width or not height: return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        cx, cy = canvas_w // 2, canvas_h // 2
        
        half_w = width // 2
        half_h = height // 2
        
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(canvas_w, cx + half_w)
        y2 = min(canvas_h, cy + half_h)
        
        self.zone_points = [(x1, y1), (x2, y2)]
        self.zone_defined = True
        
        # Redraw
        if self.current_frame is not None:
             self.display_frame(self.current_frame, is_video=True)
             
        # Enable start button immediately as validation happens in dialog
        self.start_btn.config(state=tk.NORMAL)
        self.update_status(f"Zone set to {width}x{height}. Click 'Start Processing' to begin.")

    # Zone check with coordinate scaling
    def is_in_zone(self, cx, cy):
        if not self.zone_defined: return True
        if not self.zone_points or len(self.zone_points) < 2: return True
        
        # cx, cy are in ORIGINAL VIDEO FRAME coordinates
        # self.zone_points are in CANVAS coordinates
        
        # We need to convert cx, cy to CANVAS coordinates to compare
        # or convert zone_points to FRAME coordinates. 
        # Using FRAME coordinates for processing is better.
        
        # Get Frame -> Canvas scaling factors
        # These were calculated in display_frame
        if not hasattr(self, 'scale') or self.scale == 0: return False # Ensure scale is defined and not zero
        
        # Convert Zone (Canvas) -> Frame
        # x_frame = (x_canvas - offset_x) / scale
        
        # Rectangle Check (Video Mode default)
        if len(self.zone_points) == 2:
            z_x1_c, z_y1_c = self.zone_points[0]
            z_x2_c, z_y2_c = self.zone_points[1]
            
            # Convert canvas zone points to original frame coordinates
            z_x1 = (z_x1_c - self.offset_x) / self.scale
            z_y1 = (z_y1_c - self.offset_y) / self.scale
            z_x2 = (z_x2_c - self.offset_x) / self.scale
            z_y2 = (z_y2_c - self.offset_y) / self.scale
            
            # Ensure min/max for correct comparison
            min_z_x, max_z_x = min(z_x1, z_x2), max(z_x1, z_x2)
            min_z_y, max_z_y = min(z_y1, z_y2), max(z_y1, z_y2)
            
            return (min_z_x <= cx <= max_z_x) and (min_z_y <= cy <= max_z_y)
            
        # Polygon Check (Image Mode or Complex Zone)
        if len(self.zone_points) > 2:
            # Convert canvas zone points to original frame coordinates for polygon check
            frame_zone_points = []
            for p_c_x, p_c_y in self.zone_points:
                p_f_x = (p_c_x - self.offset_x) / self.scale
                p_f_y = (p_c_y - self.offset_y) / self.scale
                frame_zone_points.append((p_f_x, p_f_y))

            # Ray casting algorithm
            n = len(frame_zone_points)
            inside = False
            p1x, p1y = frame_zone_points[0]
            for i in range(n + 1):
                p2x, p2y = frame_zone_points[i % n]
                if cy > min(p1y, p2y):
                    if cy <= max(p1y, p2y):
                        if cx <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (cy - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or cx <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
            
        return True

    def update_status(self, msg):
        self.status_var.set(msg)

    def on_close(self):
        self.stop_processing()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

