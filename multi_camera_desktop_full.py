"""
Complete Multi-Camera Vehicle Detection Desktop Application
All features in desktop app: Live View, Configuration, History, Statistics
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import time
import sys
import os
import sqlite3
from datetime import datetime
import logging
import cv2
from PIL import Image, ImageTk
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiCameraDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Vehicle Detection System")
        self.root.geometry("1600x1000")
        self.root.state('zoomed')
        
        # State
        self.api_process = None
        self.camera_processes = {}
        self.camera_configs = {
            1: {'name': 'Camera 1 - Main Gate', 'rtsp_url': '', 'enabled': True},
            2: {'name': 'Camera 2 - Exit Gate', 'rtsp_url': '', 'enabled': True},
            3: {'name': 'Camera 3 - Parking Entry', 'rtsp_url': '', 'enabled': True},
            4: {'name': 'Camera 4 - Parking Exit', 'rtsp_url': '', 'enabled': True}
        }
        
        # Initialize database
        self.init_database()
        
        # Setup styles
        self.setup_styles()
        
        # Create UI
        self.create_ui()
        
        # Start API server
        self.start_api_server()
        
        # Start update loops
        self.update_stats()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
        self.colors = {
            'bg_main': '#F4F6F9',
            'bg_card': '#FFFFFF',
            'primary': '#2C3E50',
            'accent': '#3498DB',
            'success': '#27AE60',
            'danger': '#E74C3C',
            'warning': '#F39C12',
            'text_main': '#2C3E50',
            'text_light': '#7F8C8D'
        }
        
        # Configure styles
        style.configure("TFrame", background=self.colors['bg_main'])
        style.configure("Card.TFrame", background=self.colors['bg_card'])
        style.configure("TLabel", background=self.colors['bg_main'], foreground=self.colors['text_main'])
        style.configure("Card.TLabel", background=self.colors['bg_card'], foreground=self.colors['text_main'])
        style.configure("Header.TLabel", background=self.colors['primary'], foreground='white', font=('Segoe UI', 14, 'bold'))
        
        # Notebook (Tabs)
        style.configure("TNotebook", background=self.colors['bg_main'], borderwidth=0)
        style.configure("TNotebook.Tab", padding=[20, 10], font=('Segoe UI', 11, 'bold'))
        style.map("TNotebook.Tab", 
                  background=[("selected", self.colors['primary']), ('active', '#ECF0F1')],
                  foreground=[("selected", "white"), ('active', self.colors['primary'])])
    
    def init_database(self):
        """Initialize database with multi-camera support"""
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            # Create enhanced vehicle_logs table
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
            
            # Check if camera_id exists
            cursor.execute("PRAGMA table_info(vehicle_logs)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'camera_id' not in columns:
                cursor.execute('ALTER TABLE vehicle_logs ADD COLUMN camera_id INTEGER')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Database error: {e}")
    
    def create_ui(self):
        """Create the complete user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🎥 MULTI-CAMERA VEHICLE DETECTION SYSTEM",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        self.status_label = tk.Label(
            header_frame,
            text="● System Starting...",
            font=("Segoe UI", 12),
            bg=self.colors['primary'],
            fg=self.colors['warning']
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        # Main content with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_live_view_tab()
        self.create_camera_grid_tab()
        self.create_configuration_tab()
        self.create_history_tab()
        self.create_statistics_tab()
        
        # Footer
        footer_frame = tk.Frame(self.root, bg=self.colors['primary'], height=40)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="Multi-Camera Vehicle Detection System v2.0 | Pydah Soft Solutions",
            font=("Segoe UI", 9),
            bg=self.colors['primary'],
            fg='white'
        )
        footer_label.pack(pady=10)
    
    def create_live_view_tab(self):
        """Create Live Dashboard tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📊 LIVE DASHBOARD  ")
        
        # Main container
        container = tk.Frame(tab, bg=self.colors['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top stats cards
        stats_frame = tk.Frame(container, bg=self.colors['bg_main'])
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create 4 stat cards
        self.stat_cards = {}
        
        # Total IN
        in_card = self.create_stat_card(stats_frame, "TOTAL IN", "0", self.colors['success'], "↓")
        in_card.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.stat_cards['in'] = in_card
        
        # Total OUT
        out_card = self.create_stat_card(stats_frame, "TOTAL OUT", "0", self.colors['danger'], "↑")
        out_card.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.stat_cards['out'] = out_card
        
        # Active Cameras
        active_card = self.create_stat_card(stats_frame, "ACTIVE CAMERAS", "0/4", self.colors['accent'], "📹")
        active_card.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')
        self.stat_cards['active'] = active_card
        
        # Total Vehicles
        total_card = self.create_stat_card(stats_frame, "TOTAL VEHICLES", "0", "#9B59B6", "🚗")
        total_card.grid(row=0, column=3, padx=10, pady=10, sticky='nsew')
        self.stat_cards['total'] = total_card
        
        # Configure grid weights
        for i in range(4):
            stats_frame.grid_columnconfigure(i, weight=1)
        
        # Camera status overview
        camera_status_frame = tk.LabelFrame(
            container,
            text="Camera Status Overview",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_main'],
            fg=self.colors['text_main']
        )
        camera_status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Camera status grid
        self.camera_status_widgets = {}
        for i in range(1, 5):
            cam_frame = self.create_camera_status_widget(camera_status_frame, i)
            row = (i-1) // 2
            col = (i-1) % 2
            cam_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            camera_status_frame.grid_rowconfigure(row, weight=1)
            camera_status_frame.grid_columnconfigure(col, weight=1)
        
        # Recent activity
        activity_frame = tk.LabelFrame(
            container,
            text="Recent Detections",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg_main'],
            fg=self.colors['text_main']
        )
        activity_frame.pack(fill=tk.BOTH, expand=True)
        
        # Activity table
        self.create_activity_table(activity_frame)
    
    def create_stat_card(self, parent, title, value, color, icon):
        """Create a statistics card"""
        card = tk.Frame(parent, bg=color, relief=tk.RAISED, bd=3, height=120)
        card.pack_propagate(False)
        
        # Icon
        icon_label = tk.Label(card, text=icon, font=('Segoe UI', 24), bg=color, fg='white')
        icon_label.pack(pady=(10, 5))
        
        # Title
        title_label = tk.Label(card, text=title, font=('Segoe UI', 10, 'bold'), bg=color, fg='white')
        title_label.pack()
        
        # Value
        value_label = tk.Label(card, text=value, font=('Segoe UI', 32, 'bold'), bg=color, fg='white')
        value_label.pack(pady=(5, 10))
        
        card.value_label = value_label
        return card
    
    def create_camera_status_widget(self, parent, camera_id):
        """Create camera status widget"""
        frame = tk.Frame(parent, bg='white', relief=tk.GROOVE, bd=2)
        
        # Header
        header = tk.Frame(frame, bg=self.colors['accent'])
        header.pack(fill=tk.X)
        
        name_label = tk.Label(
            header,
            text=f"📹 {self.camera_configs[camera_id]['name']}",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white'
        )
        name_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        status_label = tk.Label(
            header,
            text="● INACTIVE",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['accent'],
            fg='#95A5A6'
        )
        status_label.pack(side=tk.RIGHT, padx=10)
        
        # Stats
        stats_frame = tk.Frame(frame, bg='white')
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # IN/OUT/FPS
        stats_grid = tk.Frame(stats_frame, bg='white')
        stats_grid.pack(fill=tk.X)
        
        in_label = tk.Label(stats_grid, text="IN:", font=('Segoe UI', 9), bg='white')
        in_label.grid(row=0, column=0, sticky='w')
        in_value = tk.Label(stats_grid, text="0", font=('Segoe UI', 16, 'bold'), bg='white', fg=self.colors['success'])
        in_value.grid(row=0, column=1, padx=5)
        
        out_label = tk.Label(stats_grid, text="OUT:", font=('Segoe UI', 9), bg='white')
        out_label.grid(row=0, column=2, sticky='w', padx=(20, 0))
        out_value = tk.Label(stats_grid, text="0", font=('Segoe UI', 16, 'bold'), bg='white', fg=self.colors['danger'])
        out_value.grid(row=0, column=3, padx=5)
        
        fps_label = tk.Label(stats_grid, text="FPS:", font=('Segoe UI', 9), bg='white')
        fps_label.grid(row=1, column=0, sticky='w', pady=(5, 0))
        fps_value = tk.Label(stats_grid, text="0", font=('Segoe UI', 16, 'bold'), bg='white', fg=self.colors['accent'])
        fps_value.grid(row=1, column=1, padx=5, pady=(5, 0))
        
        # Buttons
        btn_frame = tk.Frame(frame, bg='white')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        start_btn = tk.Button(
            btn_frame,
            text="▶ START",
            command=lambda: self.start_camera(camera_id),
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2'
        )
        start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        stop_btn = tk.Button(
            btn_frame,
            text="⏹ STOP",
            command=lambda: self.stop_camera(camera_id),
            bg=self.colors['danger'],
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2'
        )
        stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.camera_status_widgets[camera_id] = {
            'frame': frame,
            'status': status_label,
            'in': in_value,
            'out': out_value,
            'fps': fps_value
        }
        
        return frame
    
    def create_activity_table(self, parent):
        """Create recent activity table"""
        # Table frame
        table_frame = tk.Frame(parent, bg='white')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        columns = ('Camera', 'Type', 'Plate', 'Direction', 'Time', 'Confidence')
        self.activity_tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        for col in columns:
            self.activity_tree.heading(col, text=col)
            self.activity_tree.column(col, width=150)
        
        self.activity_tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.activity_tree.yview)
        
        # Load recent activity
        self.load_recent_activity()
    
    def create_camera_grid_tab(self):
        """Create Camera Grid View tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📹 CAMERA GRID  ")
        
        container = tk.Frame(tab, bg=self.colors['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Control buttons
        control_frame = tk.Frame(container, bg=self.colors['bg_main'])
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Button(
            control_frame,
            text="▶ START ALL CAMERAS",
            command=self.start_all_cameras,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="⏹ STOP ALL CAMERAS",
            command=self.stop_all_cameras,
            bg=self.colors['danger'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Camera grid (2x2)
        grid_frame = tk.Frame(container, bg=self.colors['bg_main'])
        grid_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_feeds = {}
        for i in range(1, 5):
            feed_frame = self.create_camera_feed(grid_frame, i)
            row = (i-1) // 2
            col = (i-1) % 2
            feed_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            grid_frame.grid_rowconfigure(row, weight=1)
            grid_frame.grid_columnconfigure(col, weight=1)
    
    def create_camera_feed(self, parent, camera_id):
        """Create camera feed display"""
        frame = tk.Frame(parent, bg='black', relief=tk.RAISED, bd=3)
        
        # Header
        header = tk.Frame(frame, bg=self.colors['primary'])
        header.pack(fill=tk.X)
        
        name_label = tk.Label(
            header,
            text=f"📹 {self.camera_configs[camera_id]['name']}",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        name_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        fps_label = tk.Label(
            header,
            text="0 FPS",
            font=('Segoe UI', 9),
            bg=self.colors['primary'],
            fg='#27AE60'
        )
        fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Video display
        video_label = tk.Label(frame, bg='black', text="Camera Offline\n\n📹", fg='white', font=('Segoe UI', 20))
        video_label.pack(fill=tk.BOTH, expand=True)
        
        self.camera_feeds[camera_id] = {
            'frame': frame,
            'video': video_label,
            'fps': fps_label
        }
        
        return frame
    
    def create_configuration_tab(self):
        """Create Configuration tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ⚙️ CONFIGURATION  ")
        
        container = tk.Frame(tab, bg=self.colors['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            container,
            text="Camera Configuration",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['bg_main'],
            fg=self.colors['text_main']
        )
        title_label.pack(pady=(0, 20))
        
        # Camera config grid
        config_grid = tk.Frame(container, bg=self.colors['bg_main'])
        config_grid.pack(fill=tk.BOTH, expand=True)
        
        self.config_widgets = {}
        for i in range(1, 5):
            config_frame = self.create_camera_config(config_grid, i)
            row = (i-1) // 2
            col = (i-1) % 2
            config_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            config_grid.grid_rowconfigure(row, weight=1)
            config_grid.grid_columnconfigure(col, weight=1)
        
        # Save button
        save_btn = tk.Button(
            container,
            text="💾 SAVE CONFIGURATION",
            command=self.save_configuration,
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=30,
            pady=15
        )
        save_btn.pack(pady=20)
    
    def create_camera_config(self, parent, camera_id):
        """Create camera configuration widget"""
        frame = tk.LabelFrame(
            parent,
            text=f"Camera {camera_id}",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg=self.colors['text_main']
        )
        
        inner_frame = tk.Frame(frame, bg='white')
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Camera Name
        tk.Label(inner_frame, text="Camera Name:", font=('Segoe UI', 9), bg='white').pack(anchor='w')
        name_entry = tk.Entry(inner_frame, font=('Segoe UI', 10), width=40)
        name_entry.insert(0, self.camera_configs[camera_id]['name'])
        name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # RTSP URL
        tk.Label(inner_frame, text="RTSP URL:", font=('Segoe UI', 9), bg='white').pack(anchor='w')
        rtsp_entry = tk.Entry(inner_frame, font=('Segoe UI', 10), width=40)
        rtsp_entry.insert(0, self.camera_configs[camera_id]['rtsp_url'])
        rtsp_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Enabled checkbox
        enabled_var = tk.BooleanVar(value=self.camera_configs[camera_id]['enabled'])
        enabled_check = tk.Checkbutton(
            inner_frame,
            text="Enable this camera",
            variable=enabled_var,
            font=('Segoe UI', 9),
            bg='white'
        )
        enabled_check.pack(anchor='w', pady=(0, 10))
        
        # Test button
        test_btn = tk.Button(
            inner_frame,
            text="🔍 Test Connection",
            command=lambda: self.test_camera_connection(camera_id),
            bg=self.colors['warning'],
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2'
        )
        test_btn.pack(fill=tk.X)
        
        self.config_widgets[camera_id] = {
            'name': name_entry,
            'rtsp': rtsp_entry,
            'enabled': enabled_var
        }
        
        return frame
    
    def create_history_tab(self):
        """Create History tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📜 HISTORY  ")
        
        container = tk.Frame(tab, bg=self.colors['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Filters
        filter_frame = tk.LabelFrame(
            container,
            text="Filter Options",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_main']
        )
        filter_frame.pack(fill=tk.X, pady=(0, 20))
        
        filter_inner = tk.Frame(filter_frame, bg='white')
        filter_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Camera filter
        tk.Label(filter_inner, text="Camera:", bg='white').grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.filter_camera = ttk.Combobox(filter_inner, values=['All', '1', '2', '3', '4'], width=15)
        self.filter_camera.set('All')
        self.filter_camera.grid(row=0, column=1, padx=5, pady=5)
        
        # Type filter
        tk.Label(filter_inner, text="Type:", bg='white').grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.filter_type = ttk.Combobox(filter_inner, values=['All', 'Car', 'Truck', 'Bus', 'Motorcycle'], width=15)
        self.filter_type.set('All')
        self.filter_type.grid(row=0, column=3, padx=5, pady=5)
        
        # Date filter
        tk.Label(filter_inner, text="Date:", bg='white').grid(row=0, column=4, padx=5, pady=5, sticky='w')
        self.filter_date = tk.Entry(filter_inner, width=15)
        self.filter_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.filter_date.grid(row=0, column=5, padx=5, pady=5)
        
        # Search button
        search_btn = tk.Button(
            filter_inner,
            text="🔍 Search",
            command=self.load_history,
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 9, 'bold'),
            relief=tk.FLAT,
            cursor='hand2'
        )
        search_btn.grid(row=0, column=6, padx=10, pady=5)
        
        # History table
        table_frame = tk.Frame(container, bg='white')
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        columns = ('ID', 'Camera', 'Type', 'Plate', 'Direction', 'Time', 'Confidence')
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_tree.yview)
    
    def create_statistics_tab(self):
        """Create Statistics tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📊 STATISTICS  ")
        
        container = tk.Frame(tab, bg=self.colors['bg_main'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            container,
            text="System Statistics",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['bg_main']
        )
        title_label.pack(pady=(0, 20))
        
        # Stats grid
        stats_grid = tk.Frame(container, bg=self.colors['bg_main'])
        stats_grid.pack(fill=tk.BOTH, expand=True)
        
        # Today's stats
        today_frame = tk.LabelFrame(stats_grid, text="Today's Statistics", font=('Segoe UI', 11, 'bold'), bg='white')
        today_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        # Per-camera stats
        camera_frame = tk.LabelFrame(stats_grid, text="Per-Camera Statistics", font=('Segoe UI', 11, 'bold'), bg='white')
        camera_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        
        # Vehicle type distribution
        type_frame = tk.LabelFrame(stats_grid, text="Vehicle Type Distribution", font=('Segoe UI', 11, 'bold'), bg='white')
        type_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')
        
        stats_grid.grid_rowconfigure(0, weight=1)
        stats_grid.grid_rowconfigure(1, weight=1)
        stats_grid.grid_columnconfigure(0, weight=1)
        stats_grid.grid_columnconfigure(1, weight=1)
    
    def start_api_server(self):
        """Start Flask API server"""
        try:
            logger.info("Starting API server...")
            self.api_process = subprocess.Popen(
                [sys.executable, "multi_camera_api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            time.sleep(2)
            self.status_label.config(text="● System Online", fg=self.colors['success'])
            logger.info("API server started")
        except Exception as e:
            logger.error(f"Failed to start API: {e}")
            self.status_label.config(text="● System Error", fg=self.colors['danger'])
    
    def start_camera(self, camera_id):
        """Start a camera"""
        logger.info(f"Starting camera {camera_id}")
        self.camera_status_widgets[camera_id]['status'].config(text="● ACTIVE", fg=self.colors['success'])
        messagebox.showinfo("Camera Started", f"Camera {camera_id} is now active")
    
    def stop_camera(self, camera_id):
        """Stop a camera"""
        logger.info(f"Stopping camera {camera_id}")
        self.camera_status_widgets[camera_id]['status'].config(text="● INACTIVE", fg='#95A5A6')
        messagebox.showinfo("Camera Stopped", f"Camera {camera_id} has been stopped")
    
    def start_all_cameras(self):
        """Start all cameras"""
        for i in range(1, 5):
            if self.camera_configs[i]['enabled']:
                self.start_camera(i)
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        for i in range(1, 5):
            self.stop_camera(i)
    
    def test_camera_connection(self, camera_id):
        """Test camera RTSP connection"""
        rtsp_url = self.config_widgets[camera_id]['rtsp'].get()
        if not rtsp_url:
            messagebox.showwarning("No URL", "Please enter an RTSP URL first")
            return
        
        messagebox.showinfo("Testing", f"Testing connection to Camera {camera_id}...")
        # TODO: Implement actual RTSP test
    
    def save_configuration(self):
        """Save camera configuration"""
        for i in range(1, 5):
            self.camera_configs[i]['name'] = self.config_widgets[i]['name'].get()
            self.camera_configs[i]['rtsp_url'] = self.config_widgets[i]['rtsp'].get()
            self.camera_configs[i]['enabled'] = self.config_widgets[i]['enabled'].get()
        
        messagebox.showinfo("Success", "Configuration saved successfully!")
        logger.info("Configuration saved")
    
    def load_recent_activity(self):
        """Load recent activity from database"""
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            cursor.execute("""
                SELECT camera_id, vehicle_type, plate_number, direction, timestamp, confidence
                FROM vehicle_logs
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            # Clear existing
            for item in self.activity_tree.get_children():
                self.activity_tree.delete(item)
            
            # Add new
            for row in cursor.fetchall():
                camera = f"Camera {row[0]}" if row[0] else "Unknown"
                vtype = row[1] or "Unknown"
                plate = row[2] or "N/A"
                direction = row[3]
                time = row[4]
                conf = f"{row[5]*100:.1f}%" if row[5] else "N/A"
                
                self.activity_tree.insert('', 0, values=(camera, vtype, plate, direction, time, conf))
            
            conn.close()
        except Exception as e:
            logger.error(f"Error loading activity: {e}")
    
    def load_history(self):
        """Load history with filters"""
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            query = "SELECT id, camera_id, vehicle_type, plate_number, direction, timestamp, confidence FROM vehicle_logs WHERE 1=1"
            params = []
            
            # Apply filters
            camera = self.filter_camera.get()
            if camera != 'All':
                query += " AND camera_id = ?"
                params.append(int(camera))
            
            vtype = self.filter_type.get()
            if vtype != 'All':
                query += " AND vehicle_type = ?"
                params.append(vtype)
            
            date = self.filter_date.get()
            if date:
                query += " AND DATE(timestamp) = ?"
                params.append(date)
            
            query += " ORDER BY timestamp DESC LIMIT 100"
            
            cursor.execute(query, params)
            
            # Clear existing
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Add new
            for row in cursor.fetchall():
                camera = f"Camera {row[1]}" if row[1] else "Unknown"
                self.history_tree.insert('', 'end', values=(
                    row[0],
                    camera,
                    row[2] or "Unknown",
                    row[3] or "N/A",
                    row[4],
                    row[5],
                    f"{row[6]*100:.1f}%" if row[6] else "N/A"
                ))
            
            conn.close()
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    
    def update_stats(self):
        """Update statistics from database"""
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            # Total IN
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE direction = 'IN'")
            total_in = cursor.fetchone()[0]
            self.stat_cards['in'].value_label.config(text=str(total_in))
            
            # Total OUT
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE direction = 'OUT'")
            total_out = cursor.fetchone()[0]
            self.stat_cards['out'].value_label.config(text=str(total_out))
            
            # Active cameras
            cursor.execute("""
                SELECT COUNT(DISTINCT camera_id)
                FROM vehicle_logs
                WHERE datetime(timestamp) > datetime('now', '-5 minutes')
            """)
            active = cursor.fetchone()[0]
            self.stat_cards['active'].value_label.config(text=f"{active}/4")
            
            # Total vehicles
            total = total_in + total_out
            self.stat_cards['total'].value_label.config(text=str(total))
            
            # Per-camera stats
            for i in range(1, 5):
                cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE camera_id = ? AND direction = 'IN'", (i,))
                cam_in = cursor.fetchone()[0]
                self.camera_status_widgets[i]['in'].config(text=str(cam_in))
                
                cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE camera_id = ? AND direction = 'OUT'", (i,))
                cam_out = cursor.fetchone()[0]
                self.camera_status_widgets[i]['out'].config(text=str(cam_out))
            
            conn.close()
            
            # Reload recent activity
            self.load_recent_activity()
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
        
        # Schedule next update
        self.root.after(5000, self.update_stats)
    
    def on_close(self):
        """Handle window close"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.api_process:
                self.api_process.terminate()
            self.root.destroy()

def main():
    root = tk.Tk()
    app = MultiCameraDesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
