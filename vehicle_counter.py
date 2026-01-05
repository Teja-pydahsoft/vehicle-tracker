import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import time
import queue
import threading
import torch
from ultralytics import YOLO
from collections import defaultdict
import psutil
import logging
from collections import deque

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

class VideoProcessor:
    """
    Handles video capture and model inference in a separate thread.
    Produces results into a queue for the UI to consume.
    """
    def __init__(self, model, source, result_queue, stop_event, device='cpu'):
        self.model = model
        self.source = source
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = device
        self.cap = None
    
    def run(self):
        try:
            logger.info(f"Starting VideoProcessor with source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise Exception(f"Could not open video source: {self.source}")
            
            # Optimization for RTSP
            if isinstance(self.source, str) and self.source.startswith('rtsp://'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame_count = 0
            
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break

                # Skip frames if queue is full to prevent lag
                if self.result_queue.full():
                    continue

                # Run Inference
                try:
                    start_time = time.time()
                    # Use track() instead of predict() to get consistent IDs for counting
                    # persist=True is important for video tracking
                    # Use custom tracker config with high track_buffer to handle occlusions
                    results = self.model.track(
                        frame,
                        persist=True,
                        tracker='custom_tracker.yaml',
                        device=self.device,
                        verbose=False,
                        imgsz=640,
                        conf=0.25,
                        iou=0.45,
                        half=(self.device == 'cuda'),
                        max_det=100
                    )
                    inference_time = (time.time() - start_time) * 1000
                    logger.info(f"Frame processed in {inference_time:.1f}ms")
                    
                    # Put result in queue
                    self.result_queue.put((frame.copy(), results[0] if results else None))
                    
                    # Small sleep to prevent tight loop if CPU bound
                    time.sleep(0.001)
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    continue

        except Exception as e:
            logger.error(f"VideoProcessor error: {e}")
            # Optionally signal error to UI
        finally:
            if self.cap:
                self.cap.release()
            logger.info("VideoProcessor stopped")

class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vehicle Counter - Robust Threading")
        self.root.geometry("1100x800")
        
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
        
        # UI
        self.create_widgets()
        self.update_status("Initializing... Loading Model...")
        
        # Load Model Async
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Main Layout
        # Left: Video
        # Right: Controls & Stats
        
        self.root.state('zoomed') # Maximize
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left Side - Video
        self.video_frame = ttk.Frame(main_container)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Zone Events
        self.canvas.bind("<Double-Button-1>", self.open_zone_dialog)

        # Right Side - Controls
        right_panel = ttk.Frame(main_container, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Controls Group
        self.controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding=10)
        self.controls_frame.pack(fill=tk.X, pady=5)
        
        # File Selection
        ttk.Label(self.controls_frame, text="Video File:").pack(anchor=tk.W)
        file_Box = ttk.Frame(self.controls_frame)
        file_Box.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_Box, textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_Box, text="Browse", command=self.browse_file).pack(side=tk.RIGHT, padx=5)
        
        # RTSP
        ttk.Label(self.controls_frame, text="RTSP Camera:").pack(anchor=tk.W, pady=(10, 0))
        rtsp_box = ttk.Frame(self.controls_frame)
        rtsp_box.pack(fill=tk.X, pady=5)
        
        self.rtsp_url = tk.StringVar(value="rtsp://")
        ttk.Entry(rtsp_box, textvariable=self.rtsp_url).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rtsp_box, text="Connect", command=self.start_rtsp).pack(side=tk.RIGHT, padx=5)
        
        # Buttons
        btn_box = ttk.Frame(self.controls_frame)
        btn_box.pack(fill=tk.X, pady=15)
        self.start_btn = ttk.Button(btn_box, text="Start Processing", command=self.start_file_processing, state=tk.DISABLED)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_box, text="Stop", command=self.stop_processing).pack(fill=tk.X, pady=2)
        
        # Statistics Group
        self.stats_frame = ttk.LabelFrame(right_panel, text="Live Statistics", padding=10)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create Labels for classes
        self.count_labels = {}
        classes = ['Car', 'Bus', 'Motorcycle', 'Bicycle', 'Truck']
        
        for cls_name in classes:
            row = ttk.Frame(self.stats_frame)
            row.pack(fill=tk.X, pady=5)
            
            # Icon/Name
            ttk.Label(row, text=f"{cls_name}:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
            
            # Count Value
            lbl = ttk.Label(row, text="0 (In:0 Out:0)", font=("Arial", 11), foreground="blue")
            lbl.pack(side=tk.RIGHT)
            self.count_labels[cls_name] = lbl
            
        # FPS Label at bottom of right panel
        self.fps_label = ttk.Label(right_panel, text="FPS: 0.0", font=("Arial", 10))
        self.fps_label.pack(side=tk.BOTTOM, pady=10)
        
        # Status Bar at very bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress.start(10)

    def load_model(self):
        try:
            model_path = 'models/yolov8n.pt'
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Download if needed (simple check)
            if not os.path.exists(model_path):
                self.root.after(0, lambda: self.update_status("Downloading Model..."))
                YOLO('yolov8n.pt').export() # Triggers download
                
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
                
            logger.info(f"Model loaded on {self.device.upper()}")
            
            # Enable buttons on main thread
            self.root.after(0, self.enable_ui)
            self.root.after(0, lambda: self.update_status(f"Model Ready ({self.device.upper()})"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
        finally:
             self.root.after(0, self.progress.stop)
             self.root.after(0, self.progress.destroy)

    def enable_ui(self):
        self.browse_btn.config(state=tk.NORMAL)
        self.connect_btn.config(state=tk.NORMAL)
        if self.file_path.get():
            self.start_btn.config(state=tk.NORMAL)

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
        path = self.file_path.get()
        if not path: return
        self.start_processing(path)

    def start_rtsp(self):
        url = self.rtsp_url.get()
        if len(url) < 8: return
        self.start_processing(url)

    def start_processing(self, source):
        if self.is_running:
            self.stop_processing()
            
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
            target=VideoProcessor(self.model, source, self.result_queue, self.stop_event, self.device).run,
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

    def update_stats(self, results, frame):
        current_counts = defaultdict(int)
        
        # Canvas Dimensions
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if results:
            for box in results.boxes:
                # Class Mapping
                cls_id = int(box.cls[0].item())
                label = self.get_label(cls_id)
                if not label: continue
                
                # Coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                
                # Check if currently in zone
                in_zone = self.is_in_zone(cx, cy)
                
                # Prepare label
                text = f"{label}"
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    text += f" #{track_id}"
                
                # Draw Box
                color = (0, 255, 0) if in_zone else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Counting Logic: Line Crossing with Dynamic Scale Handling
                if box.id is not None and len(self.zone_points) == 2:
                    track_id = int(box.id[0].item())
                    
                    # Ensure we have valid scale/offset
                    if hasattr(self, 'scale') and self.scale > 0:
                         # Dynamic Zone Calculation (Frame Coords)
                         z_x1_c, z_y1_c = self.zone_points[0]
                         z_x2_c, z_y2_c = self.zone_points[1]
                         
                         # Convert Canvas -> Frame
                         z_y1 = (z_y1_c - self.offset_y) / self.scale
                         z_y2 = (z_y2_c - self.offset_y) / self.scale
                         z_x1 = (z_x1_c - self.offset_x) / self.scale
                         z_x2 = (z_x2_c - self.offset_x) / self.scale
                         
                         zone_center_y = (z_y1 + z_y2) / 2
                         
                         # Draw the Scan Line for visual feedback
                         line_y_canvas = (z_y1_c + z_y2_c) // 2
                         cv2.line(frame, (int(z_x1_c), int(line_y_canvas)), (int(z_x2_c), int(line_y_canvas)), (0, 0, 255), 2)
                         
                         min_x = min(z_x1, z_x2)
                         max_x = max(z_x1, z_x2)
                    
                         # Update Track Data
                         # Only consider points roughly within X-bounds (prevent cross-lane counting)
                         if min_x <= cx <= max_x:
                             t_data = self.track_data[track_id]
                             t_data['min_y'] = min(t_data['min_y'], cy)
                             t_data['max_y'] = max(t_data['max_y'], cy)
                             if t_data['start_y'] is None:
                                 t_data['start_y'] = cy
                             
                             # Check Crossing: Has track spanned across the center line?
                             # AND check direction
                             crossed = (t_data['min_y'] < zone_center_y) and (t_data['max_y'] > zone_center_y)
                             
                             if crossed:
                                 # Determine Direction based on net movement
                                 # Current y vs Start y
                                 delta = cy - t_data['start_y']
                                 
                                 # IN: Moving Down (positive delta)
                                 if delta > 0:
                                     if track_id not in self.counted_ids[label]['in']:
                                        self.counted_ids[label]['in'].add(track_id)
                                        logger.info(f"Counted IN: {label} #{track_id}")
                                 
                                 # OUT: Moving Up (negative delta)
                                 # Use a threshold to avoid jitter
                                 elif delta < 0:
                                     if track_id not in self.counted_ids[label]['out']:
                                        self.counted_ids[label]['out'].add(track_id)
                                        logger.info(f"Counted OUT: {label} #{track_id}")

                    # Update position
                    self.prev_positions[track_id] = (cx, cy)

        # Update UI Labels
        text_parts = []
        for k in sorted(self.count_labels.keys()):
            in_count = len(self.counted_ids[k]['in'])
            out_count = len(self.counted_ids[k]['out'])
            
            # Update label text
            # Format: "Car: 5 (In:3 Out:2)"
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

    def get_label(self, cls_id):
        # 2:car, 5:bus, 3:motorcycle, 1:bicycle, 7:truck
        mapping = {2: 'Car', 5: 'Bus', 3: 'Motorcycle', 1: 'Bicycle', 7: 'Truck'}
        return mapping.get(cls_id)

    def display_frame(self, frame, is_video=True):
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

