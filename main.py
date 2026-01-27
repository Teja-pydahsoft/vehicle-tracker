import sys
import os
import ctypes
import sqlite3
import subprocess
import time
import logging
import webbrowser
import multiprocessing
import psutil
from datetime import datetime, timedelta
import calendar
import cv2
import numpy as np
import requests
import tempfile
import zipfile

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QLabel, QPushButton, QTabWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QLineEdit, QComboBox, 
    QCheckBox, QGridLayout, QScrollArea, QSizePolicy, QGraphicsDropShadowEffect,
    QCalendarWidget, QDialog, QStackedWidget, QSplashScreen, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, QRect, QPoint, QDate, QThread, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap, QLinearGradient, QPalette, QPainter, QTextCharFormat, QMovie, QImage

# We will lazy-load heavy modules (vehicle_counter, multi_camera_api) inside the main check to speed up startup.

# Application Versioning
CURRENT_VERSION = "v1.0.21" 
GITHUB_REPO = "Teja-pydahsoft/vehicle-tracker"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"=== SYSTEM STARTUP: {CURRENT_VERSION} ===")
logger.info(f"RUNNING FROM: {os.path.abspath(__file__)}")

# Premium Light Theme Palette
COLORS = {
    'primary': '#6366f1',
    'primary_dark': '#4f46e5',
    'primary_light': '#818cf8',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'bg_main': '#f8fafc',
    'bg_secondary': '#ffffff',
    'bg_card': '#ffffff',
    'bg_hover': '#f1f5f9',
    'text_primary': '#0f172a',
    'text_secondary': '#64748b',
    'text_muted': '#94a3b8',
    'border': '#e2e8f0',
}

QSS_THEME = f"""
QMainWindow {{
    background-color: {COLORS['bg_main']};
}}

QWidget {{
    color: {COLORS['text_primary']};
    font-family: 'Outfit', 'Segoe UI', sans-serif;
}}

/* Header Navigation Bar */
#HeaderNav {{
    background-color: {COLORS['bg_secondary']};
    border-bottom: 1px solid {COLORS['border']};
    min-height: 65px;
    max-height: 65px;
}}

#LogoText {{
    color: {COLORS['primary']};
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-right: 30px;
}}

/* Nav Buttons */
QPushButton.NavButton {{
    background-color: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    color: {COLORS['text_secondary']};
    text-align: center;
    padding: 0px 15px;
    height: 62px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton.NavButton:hover {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['primary']};
}}

QPushButton.NavButton[active="true"] {{
    color: {COLORS['primary']};
    border-bottom: 3px solid {COLORS['primary']};
    font-weight: bold;
}}

/* Cards */
QFrame.Card {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
}}

QFrame.StatCard {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 15px;
}}

/* Titles */
QLabel.PageTitle {{
    font-size: 24px;
    font-weight: 800;
    color: {COLORS['text_primary']};
}}

QLabel.SectionTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {COLORS['text_primary']};
    margin-bottom: 8px;
}}

QLabel.StatTitle {{
    font-size: 10px;
    font-weight: 700;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}

QLabel.StatValue {{
    font-size: 30px;
    font-weight: 900;
    color: {COLORS['text_primary']};
}}

/* Tables */
QTableWidget {{
    background-color: transparent;
    alternate-background-color: #f8fafc;
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    color: {COLORS['text_primary']};
    border-radius: 8px;
}}

QHeaderView::section {{
    background-color: #f1f5f9;
    color: {COLORS['text_secondary']};
    padding: 12px;
    border: none;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
}}

QTableWidget::item {{
    padding: 10px;
}}

/* Tables */
QTableWidget {{
    background-color: transparent;
    alternate-background-color: rgba(255, 255, 255, 0.02);
    border: none;
    gridline-color: {COLORS['border']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['text_secondary']};
    padding: 12px;
    border: none;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
}}

/* Inputs */
QLineEdit, QComboBox {{
    background-color: {COLORS['bg_hover']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px;
    color: {COLORS['text_primary']};
}}

QLineEdit::placeholder {{
    color: {COLORS['text_muted']};
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox QAbstractItemView {{
    background-color: white;
    color: {COLORS['text_primary']};
    selection-background-color: {COLORS['primary']};
    selection-color: white;
    border: 1px solid {COLORS['border']};
}}

/* Buttons */
QPushButton.PrimaryBtn {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: bold;
    font-size: 14px;
}}

QPushButton.PrimaryBtn:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton.SuccessBtn {{
    background-color: {COLORS['success']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}}

QPushButton.DangerBtn {{
    background-color: {COLORS['danger']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}}
"""

class CameraStreamThread(QThread):
    image_data = Signal(QImage)

    def __init__(self, camera_id, source):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.is_running = True
        self.model = None

    def run(self):
        # Initial preparation
        src = self.source
        is_rtsp = isinstance(src, str) and src.startswith('rtsp://')
        if not is_rtsp and isinstance(src, str) and src.isdigit():
            src = int(src)
        
        # Lazy load YOLO inside the thread
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt') 
        except Exception as e:
            logger.error(f"Viewer Thread {self.camera_id}: YOLO Load Fail: {e}")

        cap = None
        proc_source = sanitize_rtsp_url(src) if is_rtsp else src

        while self.is_running:
            # Connection phase
            if cap is None or not cap.isOpened():
                if is_rtsp:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;20000000"
                    logger.info(f"Viewer {self.camera_id} connecting to RTSP (TCP): {proc_source}")
                    cap = cv2.VideoCapture(proc_source, cv2.CAP_FFMPEG)
                    
                    if not cap.isOpened():
                        logger.warning(f"Viewer {self.camera_id} sanitized fail, trying raw...")
                        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(src)

                if not cap or not cap.isOpened():
                    logger.error(f"Viewer {self.camera_id} failed to connect. Retrying in 5s...")
                    time.sleep(5)
                    continue
                
                logger.info(f"Viewer {self.camera_id} connected successfully.")

            # Read phase
            ret, frame = cap.read()
            if ret:
                # Log once every 100 frames to avoid spamming
                if not hasattr(self, '_fcount'): self._fcount = 0
                self._fcount += 1
                if self._fcount % 100 == 0:
                    logger.info(f"Viewer {self.camera_id}: Capturing frames (Heartbeat)")
                
                # Convert to UI format
                frame_small = cv2.resize(frame, (800, 450)) # Better for UI than fixed 640x480
                
                # --- RELAY: Save for other processes (Web/Worker) ---
                try:
                    relay_path = os.path.join(os.getcwd(), "shared_frames", f"cam_{self.camera_id}.jpg")
                    cv2.imwrite(relay_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                except: pass

                rgb_image = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                if not qt_image.isNull():
                    self.image_data.emit(qt_image)
            else:
                if is_rtsp:
                    logger.warning(f"Viewer {self.camera_id} stream lost, re-connecting...")
                    cap.release()
                    cap = None
                    time.sleep(2)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop file
                
            time.sleep(0.001)

        if cap:
            cap.release()

    def stop(self):
        self.is_running = False

class StatCard(QFrame):
    def __init__(self, title, value, color_type='primary', icon="📊"):
        super().__init__()
        self.setObjectName("StatCard")
        self.setProperty("class", "StatCard")
        
        # Glow Effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(25)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(10)
        self.shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(self.shadow)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Icon
        color = COLORS[color_type]
        self.icon_label = QLabel(icon)
        self.icon_label.setFixedSize(55, 55)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet(f"""
            font-size: 24px; 
            background-color: rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.08); 
            border-radius: 12px; 
            color: {color};
            border: 1px solid rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.15);
        """)
        layout.addWidget(self.icon_label)
        
        # Text
        text_layout = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setProperty("class", "StatTitle")
        
        self.value_label = QLabel(value)
        self.value_label.setProperty("class", "StatValue")
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.value_label)
        layout.addLayout(text_layout)
        
    def set_value(self, val):
        self.value_label.setText(str(val))

class StatusDot(QLabel):
    def __init__(self, color=COLORS['success']):
        super().__init__()
        self.color = color
        self.setFixedSize(14, 14)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(800)
        self.opacity = 1.0
        self.direction = -1

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            if not painter.isActive(): return
            painter.setRenderHint(QPainter.Antialiasing)
            self.opacity += self.direction * 0.08
            if self.opacity <= 0.3 or self.opacity >= 1.0: self.direction *= -1
            col = QColor(self.color)
            col.setAlphaF(max(0, min(1.0, self.opacity)))
            painter.setBrush(col)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(2, 2, 10, 10)
        except: pass

class CameraControlWidget(QFrame):
    def __init__(self, camera_id, name, start_func, stop_func):
        super().__init__()
        self.setProperty("class", "Card")
        layout = QVBoxLayout(self)
        
        header = QHBoxLayout()
        self.title = QLabel(f"📹 {name}")
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(self.title)
        
        self.status = QLabel("● OFF")
        self.status.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: bold; font-size: 11px;")
        header.addWidget(self.status, 0, Qt.AlignRight)
        layout.addLayout(header)
        
        stats = QHBoxLayout()
        self.in_lbl = QLabel("IN: 0")
        self.in_lbl.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold;")
        self.out_lbl = QLabel("OUT: 0")
        self.out_lbl.setStyleSheet(f"color: {COLORS['danger']}; font-weight: bold;")
        stats.addWidget(self.in_lbl)
        stats.addWidget(self.out_lbl)
        layout.addLayout(stats)
        
        btns = QHBoxLayout()
        start_btn = QPushButton("START")
        start_btn.setProperty("class", "SuccessBtn")
        start_btn.clicked.connect(lambda: start_func(camera_id))
        
        stop_btn = QPushButton("STOP")
        stop_btn.setProperty("class", "DangerBtn")
        stop_btn.clicked.connect(lambda: stop_func(camera_id))
        
        btns.addWidget(start_btn)
        btns.addWidget(stop_btn)
        layout.addLayout(btns)

class CalendarDialog(QDialog):
    def __init__(self, parent=None, initial_date=None, logged_dates=[]):
        super().__init__(parent)
        self.setWindowTitle("Select Date")
        self.setFixedSize(400, 450)
        self.setStyleSheet(f"""
            QDialog {{ background-color: white; }}
            QCalendarWidget QAbstractItemView {{
                selection-background-color: {COLORS['primary']};
                selection-color: white;
            }}
            QCalendarWidget QWidget#qt_calendar_navigationbar {{ background-color: {COLORS['bg_hover']}; }}
        """)
        
        layout = QVBoxLayout(self)
        self.cal = QCalendarWidget()
        self.cal.setGridVisible(True)
        self.cal.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        
        if initial_date:
            self.cal.setSelectedDate(initial_date)
            
        # Highlight logged dates
        fmt = self.cal.dateTextFormat(self.cal.selectedDate())
        logged_fmt = self.cal.dateTextFormat(self.cal.selectedDate())
        logged_fmt.setBackground(QColor(COLORS['primary_light']))
        logged_fmt.setForeground(Qt.white)
        logged_fmt.setFont(QFont("Outfit", 9, QFont.Bold))
        
        for d_str in logged_dates:
            try:
                date = QDate.fromString(d_str, "yyyy-MM-dd")
                self.cal.setDateTextFormat(date, logged_fmt)
            except: pass
            
        layout.addWidget(self.cal)
        
        btn = QPushButton("SELECT DATE")
        btn.setProperty("class", "PrimaryBtn")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        
    def get_date(self):
        return self.cal.selectedDate().toString("yyyy-MM-dd")

class UpdateOverlay(QDialog):
    def __init__(self, parent=None, version=""):
        super().__init__(parent)
        self.setWindowTitle("System Update")
        self.setFixedSize(450, 200)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QVBoxLayout(self)
        self.container = QFrame()
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 15px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(30, 30, 30, 30)
        
        self.title = QLabel(f"INSTALLING VERSION {version}")
        self.title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {COLORS['primary']};")
        self.title.setAlignment(Qt.AlignCenter)
        
        self.status = QLabel("Downloading secure package...")
        self.status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.status.setAlignment(Qt.AlignCenter)
        
        self.progress = QProgressBar()
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_hover']};
                border: none;
                border-radius: 5px;
                height: 10px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 5px;
            }}
        """)
        self.progress.setRange(0, 0) # Pulse
        
        container_layout.addWidget(self.title)
        container_layout.addSpacing(10)
        container_layout.addWidget(self.status)
        container_layout.addSpacing(20)
        container_layout.addWidget(self.progress)
        
        layout.addWidget(self.container)
        
        # Shadow
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(25)
        self.shadow.setColor(QColor(0,0,0,40)) # Softer shadow for light theme
        self.container.setGraphicsEffect(self.shadow)

    def set_status(self, text, val=None):
        self.status.setText(text)
        if val is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(val)

class UpdatePromptDialog(QDialog):
    def __init__(self, parent=None, version=""):
        super().__init__(parent)
        self.setWindowTitle("System Update Available")
        self.setFixedSize(480, 260)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.result_status = False
        
        layout = QVBoxLayout(self)
        self.container = QFrame()
        self.container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 20px;
                border: 2px solid {COLORS['primary']};
            }}
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(35, 35, 35, 35)
        
        # Icon/Header
        self.icon_lbl = QLabel("󰚰") # Update Icon (Bootstrap/Material style)
        self.icon_lbl.setStyleSheet(f"font-size: 48px; color: {COLORS['primary']}; font-family: 'Segoe UI Symbol';")
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        
        self.title = QLabel("SYSTEM UPGRADE AVAILABLE")
        self.title.setStyleSheet(f"font-size: 16px; font-weight: 900; color: {COLORS['text_primary']};")
        self.title.setAlignment(Qt.AlignCenter)
        
        self.desc = QLabel(f"A new version <b>{version}</b> is ready with enhanced performance and features. Would you like to update now?")
        self.desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        self.desc.setWordWrap(True)
        self.desc.setAlignment(Qt.AlignCenter)
        
        btn_layout = QHBoxLayout()
        self.yes_btn = QPushButton("UPGRADE NOW")
        self.yes_btn.setCursor(Qt.PointingHandCursor)
        self.yes_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 800;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
        """)
        
        self.no_btn = QPushButton("LATER")
        self.no_btn.setCursor(Qt.PointingHandCursor)
        self.no_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: 700;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
            }}
        """)
        
        self.yes_btn.clicked.connect(self.do_yes)
        self.no_btn.clicked.connect(self.do_no)
        
        btn_layout.addWidget(self.no_btn)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(self.yes_btn)
        
        container_layout.addWidget(self.icon_lbl)
        container_layout.addWidget(self.title)
        container_layout.addSpacing(10)
        container_layout.addWidget(self.desc)
        container_layout.addSpacing(25)
        container_layout.addLayout(btn_layout)
        
        layout.addWidget(self.container)
        
        # Shadow
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(30)
        self.shadow.setColor(QColor(0,0,0,50))
        self.container.setGraphicsEffect(self.shadow)

    def do_yes(self):
        self.result_status = True
        self.accept()
        
    def do_no(self):
        self.result_status = False
        self.reject()

class UltraModernApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SMART GATE - MONITORING SYSTEM")
        self.resize(1600, 1000)
        self.setStyleSheet(QSS_THEME)
        
        # Set Window Icon
        icon_path = os.path.join(os.getcwd(), "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # State
        self.api_process = None
        self.camera_processes = {}
        self.cam_labels = {}
        self.cam_threads = {}
        self.cam_title_labels = {}
        self.camera_configs = {
            1: {'name': 'Camera 1 - Main Gate', 'rtsp_url': '', 'enabled': True, 'status': 'inactive'},
            2: {'name': 'Camera 2 - Exit Gate', 'rtsp_url': '', 'enabled': True, 'status': 'inactive'},
            3: {'name': 'Camera 3 - Parking Entry', 'rtsp_url': '', 'enabled': True, 'status': 'inactive'},
            4: {'name': 'Camera 4 - Parking Exit', 'rtsp_url': '', 'enabled': True, 'status': 'inactive'}
        }
        
        self.init_database()
        self.init_camera_config()
        self.setup_ui()
        # Load Configuration
        self.load_camera_config_into_ui()
        
        # Start Update Check (Remotely via GitHub)
        QTimer.singleShot(5000, self.check_for_updates)
        self.start_api_server()
        
        # Update Timers
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(2000) # Increased frequency to 2s
        
        # Auto-launch web view after 3s
        QTimer.singleShot(3000, lambda: webbrowser.open("http://localhost:5000"))

    def init_database(self):
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS vehicle_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id INTEGER, timestamp TEXT,
                    vehicle_type TEXT, track_id INTEGER, direction TEXT, confidence REAL, 
                    plate_number TEXT, vehicle_state TEXT)''')
            
            # Ensure columns exist
            cursor.execute("PRAGMA table_info(vehicle_logs)")
            cols = [c[1] for c in cursor.fetchall()]
            if 'camera_id' not in cols: cursor.execute('ALTER TABLE vehicle_logs ADD COLUMN camera_id INTEGER')
            if 'vehicle_state' not in cols: cursor.execute('ALTER TABLE vehicle_logs ADD COLUMN vehicle_state TEXT')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB Error: {e}")

    def setup_ui(self):
        # Main Layout
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Combined Header Nav Bar (Merged with Clock & Status)
        self.setup_header_nav()
        
        # Content
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(30, 10, 30, 20) # Top margin reduced from 30 to 10
        self.main_layout.addWidget(self.content_area)
        
        # Views (Directly below header)
        self.views = QStackedWidget()
        self.content_layout.addWidget(self.views)
        
        self.setup_dashboard_view()
        self.setup_camera_view()
        self.setup_history_view()
        self.setup_analytics_view()
        self.setup_config_view()
        
        # Clock Timer
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def setup_header_nav(self):
        header_nav = QFrame()
        header_nav.setObjectName("HeaderNav")
        layout = QHBoxLayout(header_nav)
        layout.setContentsMargins(30, 0, 30, 0)
        layout.setSpacing(10)
        
        logo_lbl = QLabel("AI SMART VEHICLE MONITORING SYSTEM")
        logo_lbl.setObjectName("LogoText")
        layout.addWidget(logo_lbl)
        
        # Navigation Tabs
        self.nav_btns = []
        navs = [
            ("Dashboard", 0),
            ("Live Cameras", 1),
            ("History", 2),
            ("Analytics", 3),
            ("Settings", 4)
        ]
        
        for text, idx in navs:
            btn = QPushButton(text)
            btn.setProperty("class", "NavButton")
            btn.setProperty("active", "true" if idx == 0 else "false")
            btn.clicked.connect(lambda checked=False, i=idx, b=btn, t=text: self.switch_view(i, b, t))
            layout.addWidget(btn)
            self.nav_btns.append(btn)
            
        layout.addStretch()
        
        # Clock & Status (Moved here to save space)
        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; margin-right: 15px; font-weight: 500;")
        layout.addWidget(self.clock_lbl)
        
        status_box = QHBoxLayout()
        status_box.setSpacing(5)
        status_box.addWidget(StatusDot())
        status_box.addWidget(QLabel("System Online"))
        layout.addLayout(status_box)
        
        # Exit App Button
        exit_btn = QPushButton("EXIT APP")
        exit_btn.setFixedWidth(90)
        exit_btn.setFixedHeight(32)
        exit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['danger']};
                border: 1px solid {COLORS['danger']};
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                margin-left: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger']};
                color: white;
            }}
        """)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
        
        self.main_layout.addWidget(header_nav)

    def switch_view(self, idx, btn, title):
        logger.info(f"Switching to view {idx}: {title}")
        self.views.setCurrentIndex(idx)
        
        # Reset nav buttons
        for nb in self.nav_btns:
            nb.setProperty("active", "false")
            nb.style().unpolish(nb)
            nb.style().polish(nb)
        
        # Set active
        btn.setProperty("active", "true")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def setup_dashboard_view(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 10, 0, 0) # Reduced from 30 to 10
        layout.setSpacing(20) # Reduced from 30
        
        # Stats
        stats = QHBoxLayout()
        self.total_in = StatCard("Vehicles IN", "0", "success", "↓")
        self.total_out = StatCard("Vehicles OUT", "0", "danger", "↑")
        self.active_cams = StatCard("Active Cameras", "0/4", "info", "🌐")
        self.total_count = StatCard("Total Detected", "0", "primary", "🚗")
        for s in [self.total_in, self.total_out, self.active_cams, self.total_count]: stats.addWidget(s)
        layout.addLayout(stats)
        
        # Main Body
        mid = QHBoxLayout()
        mid.setSpacing(30)
        
        # Left: Live Activity
        activity_box = QFrame()
        activity_box.setProperty("class", "Card")
        a_lay = QVBoxLayout(activity_box)
        a_lay.setContentsMargins(20, 20, 20, 20)
        a_lay.addWidget(QLabel("Recent Activity (Database Logs)"))
        
        # All columns from DB: ID, Timestamp, Camera, Type, Plate, State, Direction, Confidence
        self.live_table = QTableWidget(0, 8)
        self.live_table.setHorizontalHeaderLabels(["ID", "Timestamp", "Camera", "Type", "Plate", "State", "Direction", "Confidence"])
        self.live_table.verticalHeader().setDefaultSectionSize(45) 
        self.live_table.verticalHeader().setVisible(False)
        self.live_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.live_table.setStyleSheet("QTableWidget { border: none; background: white; }")
        a_lay.addWidget(self.live_table)
        mid.addWidget(activity_box, 7)
        
        # Right: Quick Monitor
        mon_box = QFrame()
        mon_box.setProperty("class", "Card")
        m_lay = QVBoxLayout(mon_box)
        m_lay.setContentsMargins(25, 25, 25, 25)
        m_lay.addWidget(QLabel("Camera Monitoring"))
        
        self.cam_controls = {}
        for i in range(1, 5):
            cw = CameraControlWidget(i, f"Camera {i}", self.start_camera_proc, self.stop_camera_proc)
            self.cam_controls[i] = cw
            m_lay.addWidget(cw)
        
        m_lay.addStretch()
        mid.addWidget(mon_box, 3)
        layout.addLayout(mid)
        
        self.views.addWidget(container)

    def setup_camera_view(self):
        view = QWidget()
        layout = QGridLayout(view)
        layout.setContentsMargins(0, 10, 0, 0) # Reduced from 30
        layout.setSpacing(20)
        
        for i in range(1, 5):
            f = QFrame()
            f.setProperty("class", "Card")
            f.setMinimumHeight(400)
            fl = QVBoxLayout(f)
            fl.setContentsMargins(0, 0, 0, 0)
            
            h = QFrame()
            h.setStyleSheet(f"background: {COLORS['bg_hover']}; border-top-left-radius: 16px; border-top-right-radius: 16px;")
            hl = QHBoxLayout(h)
            title_lbl = QLabel(f"Camera #{i} Live Feed")
            title_lbl.setStyleSheet("font-weight: bold; color: #1e293b;")
            hl.addWidget(title_lbl)
            self.cam_title_labels[i] = title_lbl
            fl.addWidget(h)
            
            v = QLabel("CAMERA OFFLINE")
            v.setAlignment(Qt.AlignCenter)
            v.setStyleSheet("background: black; font-size: 18px; color: #475569; font-weight: 800;")
            v.setScaledContents(True)
            fl.addWidget(v)
            self.cam_labels[i] = v
            layout.addWidget(f, (i-1)//2, (i-1)%2)
        
        self.views.addWidget(view)

    def setup_history_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 10, 0, 0) # Reduced from 30
        
        # Filters
        f_card = QFrame()
        f_card.setProperty("class", "Card")
        fl = QHBoxLayout(f_card)
        fl.setContentsMargins(20, 20, 20, 20)
        fl.setSpacing(15)
        
        # Date Pickers
        fl.addWidget(QLabel("From:"))
        self.h_start_date = QPushButton((datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        self.h_start_date.setFixedWidth(110)
        self.h_start_date.setStyleSheet("text-align: left; padding: 10px; background: white; border: 1px solid #e2e8f0; border-radius: 8px;")
        self.h_start_date.clicked.connect(lambda: self.show_cal_dialog(self.h_start_date))
        fl.addWidget(self.h_start_date)
        
        fl.addWidget(QLabel("To:"))
        self.h_end_date = QPushButton(datetime.now().strftime("%Y-%m-%d"))
        self.h_end_date.setFixedWidth(110)
        self.h_end_date.setStyleSheet("text-align: left; padding: 10px; background: white; border: 1px solid #e2e8f0; border-radius: 8px;")
        self.h_end_date.clicked.connect(lambda: self.show_cal_dialog(self.h_end_date))
        fl.addWidget(self.h_end_date)
        
        self.h_state = QComboBox(); self.h_state.addItems(['All States', 'Andhra Pradesh', 'Telangana', 'Karnataka', 'Tamil Nadu', 'Maharashtra'])
        self.h_type = QComboBox(); self.h_type.addItems(['All Types', 'Car', 'Truck', 'Bus', 'Motorcycle'])
        self.h_dir = QComboBox(); self.h_dir.addItems(['All Directions', 'IN', 'OUT'])
        
        fl.addWidget(self.h_state); fl.addWidget(self.h_type); fl.addWidget(self.h_dir)
        
        search = QPushButton("SEARCH LOGS")
        search.setProperty("class", "PrimaryBtn")
        search.clicked.connect(self.load_history)
        fl.addWidget(search)
        
        fl.addStretch() # Pack all controls to the left
        layout.addWidget(f_card)
        
        # Table
        t_card = QFrame(); t_card.setProperty("class", "Card")
        tl = QVBoxLayout(t_card)
        tl.setContentsMargins(10, 10, 10, 10)
        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(["ID", "Timestamp", "Camera", "Type", "Plate", "State", "Dir", "Conf"])
        self.history_table.verticalHeader().setDefaultSectionSize(45)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tl.addWidget(self.history_table)
        layout.addWidget(t_card)
        self.views.addWidget(view)

    def setup_analytics_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 10, 0, 0) # Reduced from 30
        
        # Filters
        f_card = QFrame()
        f_card.setProperty("class", "Card")
        fl = QHBoxLayout(f_card)
        fl.setContentsMargins(20, 15, 20, 15)
        fl.setSpacing(15)
        
        fl.addWidget(QLabel("From:"))
        self.a_start_date = QPushButton((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        self.a_start_date.setFixedWidth(110)
        self.a_start_date.setStyleSheet("text-align: left; padding: 10px; background: white; border: 1px solid #e2e8f0; border-radius: 8px;")
        self.a_start_date.clicked.connect(lambda: self.show_cal_dialog(self.a_start_date))
        fl.addWidget(self.a_start_date)
        
        fl.addWidget(QLabel("To:"))
        self.a_end_date = QPushButton(datetime.now().strftime("%Y-%m-%d"))
        self.a_end_date.setFixedWidth(110)
        self.a_end_date.setStyleSheet("text-align: left; padding: 10px; background: white; border: 1px solid #e2e8f0; border-radius: 8px;")
        self.a_end_date.clicked.connect(lambda: self.show_cal_dialog(self.a_end_date))
        fl.addWidget(self.a_end_date)
        
        self.a_cam = QComboBox(); self.a_cam.addItems(['All Cameras', '1', '2', '3', '4'])
        self.a_type = QComboBox(); self.a_type.addItems(['All Types', 'Car', 'Truck', 'Bus', 'Motorcycle'])
        fl.addWidget(self.a_cam); fl.addWidget(self.a_type)
        
        refresh = QPushButton("REFRESH ANALYTICS")
        refresh.setProperty("class", "PrimaryBtn")
        refresh.clicked.connect(self.update_stats) # update_stats will call specific analytics logic
        fl.addWidget(refresh)
        
        fl.addStretch()
        layout.addWidget(f_card)
        
        # Summary Row 1
        grid = QGridLayout()
        self.a_cards = {}
        for i, t in enumerate(["Cars", "Trucks", "Buses", "Motorcycles"]):
            card = StatCard(t, "0", "info", "📈")
            self.a_cards[t.lower()] = card
            grid.addWidget(card, 0, i)
        layout.addLayout(grid)
        
        # Summary Row 2
        mid = QHBoxLayout()
        self.peak_card = StatCard("Peak Hour (Filtered)", "--:00", "danger", "🔥")
        self.low_card = StatCard("Lowest Hour (Filtered)", "--:00", "success", "❄️")
        mid.addWidget(self.peak_card); mid.addWidget(self.low_card)
        layout.addLayout(mid)
        
        # Daily Trends Table
        trend_box = QFrame()
        trend_box.setProperty("class", "Card")
        trend_lay = QVBoxLayout(trend_box)
        trend_lay.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("DAILY TRAFFIC BREAKDOWN")
        title.setProperty("class", "SectionTitle")
        trend_lay.addWidget(title)
        
        self.trend_table = QTableWidget(0, 4)
        self.trend_table.setHorizontalHeaderLabels(["Date", "Peak Traffic Hour", "Lowest Traffic Hour", "Total Volume"])
        self.trend_table.verticalHeader().setDefaultSectionSize(45)
        self.trend_table.verticalHeader().setVisible(False)
        self.trend_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trend_lay.addWidget(self.trend_table)
        
        layout.addWidget(trend_box)
        self.views.addWidget(view)

    def setup_config_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 10, 0, 0) # Reduced from 30
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        clay = QVBoxLayout(content)
        clay.setSpacing(15)
        
        self.config_inputs = {}
        for i in range(1, 5):
            f = QFrame()
            f.setProperty("class", "Card")
            fl = QVBoxLayout(f)
            fl.setContentsMargins(20, 20, 20, 20)
            
            title = QLabel(f"CAMERA #{i} SETUP")
            title.setProperty("class", "SectionTitle")
            title.setStyleSheet(f"color: {COLORS['primary']}; font-size: 14px;")
            fl.addWidget(title)
            
            grid = QGridLayout()
            grid.setSpacing(10)
            
            grid.addWidget(QLabel("Display Name:"), 0, 0)
            name = QLineEdit()
            name.setPlaceholderText("e.g. South Gate Entrance")
            grid.addWidget(name, 0, 1)
            
            grid.addWidget(QLabel("Camera RTSP URL:"), 1, 0)
            rtsp = QLineEdit()
            rtsp.setPlaceholderText("rtsp://admin:pass@192.168.1.100:554/ch1")
            grid.addWidget(rtsp, 1, 1)
            
            fl.addLayout(grid)
            clay.addWidget(f)
            self.config_inputs[i] = {'name': name, 'rtsp': rtsp}
            
        clay.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        save = QPushButton("SAVE ALL CAMERA CONFIGURATIONS")
        save.setProperty("class", "PrimaryBtn")
        save.setFixedHeight(50)
        save.clicked.connect(self.save_camera_config_to_db)
        layout.addWidget(save)
        self.views.addWidget(view)

    # --- Actions ---
    def update_clock(self):
        self.clock_lbl.setText(datetime.now().strftime("%d %b %Y | %H:%M:%S"))

    def start_api_server(self):
        try:
            # Determine how to run the background script (EXE vs Python script)
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, "--api"]
            else:
                cmd = [sys.executable, __file__, "--api"]

            # Revert to inheriting the parent console (single cmd window for all logs)
            self.api_process = subprocess.Popen(cmd, cwd=os.getcwd())
            logger.info("API Server started successfully.")
        except Exception as e:
            logger.error(f"Failed to start API Server: {e}")

    def start_camera_proc(self, cid):
        # AUTO-SAVE: Capture whatever is in the UI text boxes immediately
        self.save_camera_config_to_db_silent()
        
        rtsp = self.config_inputs[cid]['rtsp'].text()
        if not rtsp: 
            QMessageBox.warning(self, "Config Missing", f"Please provide an RTSP URL for Camera {cid} in Settings.")
            return
        
        # Pass raw URL to worker, worker will sanitize internally
        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--worker", "--headless", "--camera-id", str(cid), "--source", rtsp]
        else:
            cmd = [sys.executable, __file__, "--worker", "--headless", "--camera-id", str(cid), "--source", rtsp]
            
        # Consolidate logs into a single console
        proc = subprocess.Popen(cmd, cwd=os.getcwd())
        self.camera_processes[cid] = proc
        
        # PERSIST STATUS: Sync with Web
        self.update_camera_db_status(cid, 'active')
        
        # Start the internal PySide viewer for this camera
        self.start_camera_viewer(cid, rtsp)
        
        self.cam_controls[cid].status.setText("● ACTIVE")
        self.cam_controls[cid].status.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold;")

    def update_camera_db_status(self, cid, status):
        """Helper to sync camera active state to DB for web dashboard visibility."""
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE camera_config SET status=? WHERE id=?", (status, cid))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update camera status in DB: {e}")

    def stop_camera_proc(self, cid):
        """Terminates background worker and UI thread for a specific camera."""
        # 1. Stop background worker
        if cid in self.camera_processes:
            try:
                proc = self.camera_processes[cid]
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    try: child.terminate()
                    except: pass
                parent.terminate()
                logger.info(f"Worker for Camera {cid} terminated.")
            except Exception as e:
                logger.error(f"Stop worker error: {e}")
            del self.camera_processes[cid]
            # PERSIST STATUS: Sync with Web
            self.update_camera_db_status(cid, 'inactive')
        
        # 2. Stop UI viewer
        if cid in self.cam_threads:
            self.cam_threads[cid].stop()
            self.cam_threads[cid].wait()
            del self.cam_threads[cid]
            logger.info(f"Viewer thread for Camera {cid} stopped.")
            
        # 3. Reset UI
        if cid in self.cam_labels:
            self.cam_labels[cid].setText("CAMERA OFFLINE")
            self.cam_labels[cid].setStyleSheet("background: black; font-size: 18px; color: #475569; font-weight: 800;")
            self.cam_labels[cid].setPixmap(QPixmap())
            
        if cid in self.cam_controls:
            self.cam_controls[cid].status.setText("● OFF")
            self.cam_controls[cid].status.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: bold;")

    def start_camera_viewer(self, cid, source):
        # UI Feedback: Show "CONNECTING..." immediately
        if cid in self.cam_labels:
            self.cam_labels[cid].setText("CONNECTING...")
            self.cam_labels[cid].setStyleSheet("background: black; font-size: 18px; color: #fbbf24; font-weight: 800;")
            self.cam_labels[cid].setPixmap(QPixmap()) # Clear any old frame

        # Stop existing thread if any
        if cid in self.cam_threads:
            self.cam_threads[cid].stop()
            self.cam_threads[cid].wait()
            
        thread = CameraStreamThread(cid, source)
        thread.image_data.connect(lambda img, c=cid: self.update_cam_label(c, img))
        thread.start()
        self.cam_threads[cid] = thread

    def update_cam_label(self, cid, image):
        if cid in self.cam_labels:
            # Clear text and specialized style on first real frame
            if self.cam_labels[cid].text():
                self.cam_labels[cid].setText("")
                self.cam_labels[cid].setStyleSheet("background: black;") # Reset to plain black
            
            pixmap = QPixmap.fromImage(image)
            if not pixmap.isNull():
                self.cam_labels[cid].setPixmap(pixmap)

    # --- Config Persistence ---
    def init_camera_config(self):
        try:
            db_path = os.path.abspath('gate_log.db')
            logger.info(f"Initializing camera config DB at: {db_path}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS camera_config 
                            (id INTEGER PRIMARY KEY, name TEXT, rtsp_url TEXT, 
                             enabled INTEGER DEFAULT 1, position TEXT, status TEXT DEFAULT 'inactive')''')
            
            # Migration check for existing tables
            cursor.execute("PRAGMA table_info(camera_config)")
            cols = [c[1] for c in cursor.fetchall()]
            if 'status' not in cols: cursor.execute('ALTER TABLE camera_config ADD COLUMN status TEXT DEFAULT "inactive"')
            if 'enabled' not in cols: cursor.execute('ALTER TABLE camera_config ADD COLUMN enabled INTEGER DEFAULT 1')
            if 'position' not in cols: cursor.execute('ALTER TABLE camera_config ADD COLUMN position TEXT')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error init camera config: {e}")

    def save_camera_config_to_db(self):
        if self.save_camera_config_to_db_silent():
            QMessageBox.information(self, "Success", "Camera configurations saved successfully.")
        else:
             QMessageBox.critical(self, "Save Failed", "Could not save configurations to database.")

    def save_camera_config_to_db_silent(self):
        """Silently save all UI inputs to DB. Returns True if successful."""
        try:
            db_path = os.path.abspath('gate_log.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA synchronous = EXTRA")
            
            for i in range(1, 4+1):
                name = self.config_inputs[i]['name'].text()
                rtsp = self.config_inputs[i]['rtsp'].text()
                pos = f"cam_{i}"
                cursor.execute("INSERT OR REPLACE INTO camera_config (id, name, rtsp_url, enabled, position) VALUES (?, ?, ?, ?, ?)", 
                               (i, name, rtsp, 1, pos))
            conn.commit()
            conn.close()
            # Update internal model and other UI elements
            self.load_camera_config_into_ui()
            return True
        except Exception as e:
            logger.error(f"Silent Save Error: {e}")
            return False

    def load_camera_config_into_ui(self):
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, rtsp_url FROM camera_config")
            rows = cursor.fetchall()
            for r in rows:
                cid, name, rtsp = r
                if cid in self.config_inputs:
                    self.config_inputs[cid]['name'].setText(name)
                    self.config_inputs[cid]['rtsp'].setText(rtsp)
                
                # Update Dashboard and Camera View Titles
                if name:
                    if hasattr(self, 'cam_controls') and cid in self.cam_controls:
                        self.cam_controls[cid].title.setText(f"📹 {name}")
                    if hasattr(self, 'cam_title_labels') and cid in self.cam_title_labels:
                        self.cam_title_labels[cid].setText(f"{name} - Live Feed")

            conn.close()
        except Exception as e:
            logger.error(f"Error loading UI config: {e}")

    def stop_camera_proc(self, cid):
        if cid in self.camera_processes:
            self.camera_processes[cid].terminate()
            del self.camera_processes[cid]
        self.cam_controls[cid].status.setText("● OFF")
        self.cam_controls[cid].status.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: bold;")

    def update_stats(self):
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            # Global counts
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE direction='IN'")
            cin = cursor.fetchone()[0]
            self.total_in.set_value(cin)
            
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE direction='OUT'")
            cout = cursor.fetchone()[0]
            self.total_out.set_value(cout)
            self.total_count.set_value(cin + cout)
            
            # Active Cameras
            active = len(self.camera_processes)
            self.active_cams.set_value(f"{active}/4")
            
            # Camera specific counts
            for i in range(1, 5):
                cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE camera_id=? AND direction='IN'", (i,))
                n_in = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE camera_id=? AND direction='OUT'", (i,))
                n_out = cursor.fetchone()[0]
                self.cam_controls[i].in_lbl.setText(f"IN: {n_in}")
                self.cam_controls[i].out_lbl.setText(f"OUT: {n_out}")
            
            # Activity Table
            cursor.execute("SELECT id, timestamp, camera_id, vehicle_type, plate_number, vehicle_state, direction, confidence FROM vehicle_logs ORDER BY id DESC LIMIT 15")
            rows = cursor.fetchall()
            self.live_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    display_val = str(val) if val is not None else "-"
                    if j == 7: # Confidence
                        try: display_val = f"{float(val)*100:.1f}%"
                        except: pass
                    item = QTableWidgetItem(display_val)
                    if j == 6: # Direction
                        item.setForeground(QColor(COLORS['success'] if val == 'IN' else COLORS['danger']))
                    self.live_table.setItem(i, j, item)
            
            # Analytics Breakdown
            base_query = " WHERE 1=1"
            a_params = []
            
            # Analytics Filters
            a_start = self.a_start_date.text()
            a_end = self.a_end_date.text()
            if a_start:
                base_query += " AND date(timestamp) >= date(?)"
                a_params.append(a_start)
            if a_end:
                base_query += " AND date(timestamp) <= date(?)"
                a_params.append(a_end)
            
            vcam = self.a_cam.currentText()
            if vcam != 'All Cameras':
                base_query += " AND camera_id=?"
                a_params.append(int(vcam))
            
            # Vehicle type filter for summary cards
            selected_type = self.a_type.currentText()

            for v in ["car", "truck", "bus", "motorcycle"]:
                q_type = f"{base_query} AND lower(vehicle_type)=?"
                cursor.execute(f"SELECT COUNT(*) FROM vehicle_logs {q_type}", a_params + [v])
                count = cursor.fetchone()[0]
                if v in self.a_cards: self.a_cards[v].set_value(count)
                elif v + 's' in self.a_cards: self.a_cards[v+'s'].set_value(count)
            
            # Global Peak/Low for selected period
            # Filter by type if not 'All'
            p_query = base_query
            if selected_type != 'All Types':
                p_query += f" AND lower(vehicle_type)='{selected_type.lower()}'"

            cursor.execute(f"SELECT strftime('%H', timestamp) as h, COUNT(*) FROM vehicle_logs {p_query} GROUP BY h", a_params)
            hr_stats = cursor.fetchall()
            if hr_stats:
                peak = max(hr_stats, key=lambda x: x[1])[0]
                low = min(hr_stats, key=lambda x: x[1])[0]
                self.peak_card.set_value(f"{peak}:00")
                self.low_card.set_value(f"{low}:00")
            else:
                self.peak_card.set_value("--:00")
                self.low_card.set_value("--:00")

            # Daily Trends Table Logic
            cursor.execute(f"SELECT DISTINCT date(timestamp) FROM vehicle_logs {base_query} ORDER BY date(timestamp) DESC", a_params)
            dates = [r[0] for r in cursor.fetchall()]
            
            self.trend_table.setRowCount(len(dates))
            for i, d in enumerate(dates):
                # Peak for this day
                cursor.execute(f"SELECT strftime('%H', timestamp) as h, COUNT(*) FROM vehicle_logs WHERE date(timestamp)=? GROUP BY h ORDER BY COUNT(*) DESC LIMIT 1", (d,))
                p_res = cursor.fetchone()
                p_h = f"{p_res[0]}:00" if p_res else "--"
                
                # Low for this day
                cursor.execute(f"SELECT strftime('%H', timestamp) as h, COUNT(*) FROM vehicle_logs WHERE date(timestamp)=? GROUP BY h ORDER BY COUNT(*) ASC LIMIT 1", (d,))
                l_res = cursor.fetchone()
                l_h = f"{l_res[0]}:00" if l_res else "--"
                
                # Total for this day
                cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE date(timestamp)=?", (d,))
                total_day = cursor.fetchone()[0]
                
                self.trend_table.setItem(i, 0, QTableWidgetItem(str(d)))
                self.trend_table.setItem(i, 1, QTableWidgetItem(p_h))
                self.trend_table.setItem(i, 2, QTableWidgetItem(l_h))
                self.trend_table.setItem(i, 3, QTableWidgetItem(str(total_day)))

            conn.close()
        except: pass

    def load_history(self):
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            query = "SELECT id, timestamp, camera_id, vehicle_type, plate_number, vehicle_state, direction, confidence FROM vehicle_logs WHERE 1=1"
            params = []

            # Date Range
            start = self.h_start_date.text()
            end = self.h_end_date.text()
            if start:
                query += " AND date(timestamp) >= date(?)"
                params.append(start)
            if end:
                query += " AND date(timestamp) <= date(?)"
                params.append(end)
            
            vtype = self.h_type.currentText()
            if vtype != 'All Types': 
                query += " AND lower(vehicle_type)=?"
                params.append(vtype.lower())
            
            state = self.h_state.currentText()
            if state != 'All States': 
                query += " AND vehicle_state=?"
                params.append(state)

            direction = self.h_dir.currentText()
            if direction != 'All Directions':
                query += " AND direction=?"
                params.append(direction)
            
            query += " ORDER BY id DESC LIMIT 200"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            self.history_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    display_val = str(val) if val is not None else "-"
                    if j == 7: # Confidence
                        try: display_val = f"{float(val)*100:.1f}%"
                        except: pass
                    item = QTableWidgetItem(display_val)
                    if j == 6: # Direction
                        item.setForeground(QColor(COLORS['success'] if val == 'IN' else COLORS['danger']))
                    self.history_table.setItem(i, j, item)
            conn.close()
        except Exception as e:
            logger.error(f"Load history error: {e}")

    def show_cal_dialog(self, btn):
        logged_dates = []
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT date(timestamp) FROM vehicle_logs")
            logged_dates = [r[0] for r in cursor.fetchall()]
            conn.close()
        except: pass
        
        initial = QDate.fromString(btn.text(), "yyyy-MM-dd")
        dlg = CalendarDialog(self, initial, logged_dates)
        if dlg.exec():
            btn.setText(dlg.get_date())

    def generate_management_report(self):
        try:
            import generate_report
            generate_report.create_report()
            # Show shell open
            report_file = os.path.join(os.getcwd(), 'Smart_Gate_Project_Report.docx')
            if os.path.exists(report_file):
                os.startfile(report_file)
            logger.info("Management report generated and opened.")
        except Exception as e:
            logger.error(f"Report generation error: {e}")

    def check_for_updates(self):
        """Check GitHub for new releases"""
        try:
            logger.info(f"Checking for updates... (Current: {CURRENT_VERSION})")
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get('tag_name', '')
                
                if latest_version and latest_version != CURRENT_VERSION:
                    logger.info(f"Update available: {latest_version}")
                    
                    dialog = UpdatePromptDialog(self, latest_version)
                    if dialog.exec() == QDialog.Accepted:
                        assets = data.get('assets', [])
                        update_url = None
                        for asset in assets:
                            if 'VehicleCounter' in asset['name'] and asset['name'].endswith('.zip'):
                                update_url = asset['browser_download_url']
                                break
                        
                        if update_url:
                            self.perform_update(update_url, latest_version)
                        else:
                            QMessageBox.information(self, "Manual Update", "Direct package not found. Opening update page...")
                            import webbrowser
                            webbrowser.open(data.get('html_url'))
            
        except Exception as e:
            logger.error(f"Remote update check failed: {e}")

    def perform_update(self, url, version):
        """Downloads and installs the update with a UI overlay."""
        overlay = UpdateOverlay(self, version)
        overlay.show()
        QApplication.processEvents()

        def run_task():
            try:
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "update.zip")
                
                # Download
                r = requests.get(url, stream=True)
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = int((downloaded / total_size) * 100)
                                overlay.set_status(f"Downloading: {percent}%", percent)
                                QApplication.processEvents()
                
                overlay.set_status("Verifying & Extracting...", 100)
                overlay.progress.setRange(0, 0)
                QApplication.processEvents()
                
                # Extract
                extract_path = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_path, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                overlay.set_status("Applying changes... App will restart.")
                QApplication.processEvents()
                time.sleep(1)
                
                # Final Batch Script (Robust Fix)
                install_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                updater_bat = os.path.join(temp_dir, "updater.bat")
                
                with open(updater_bat, "w") as f:
                    f.write(f"@echo off\n")
                    f.write(f"title SYSTEM UPDATE SERVICE\n")
                    f.write(f"echo ------------------------------------------\n")
                    f.write(f"echo  AI SMART GATE - INSTALLING UPDATE v{version}\n")
                    f.write(f"echo ------------------------------------------\n")
                    f.write(f"echo Terminatiing old processes...\n")
                    f.write(f"taskkill /F /IM python.exe /T > nul 2>&1\n")
                    f.write(f"timeout /t 5 /nobreak > nul\n")
                    f.write(f"echo Replacing files in: {install_dir}\n")
                    # Robust retry logic for Robocopy
                    f.write(f"robocopy \"{extract_path}\" \"{install_dir}\" /E /IS /IT /NP /R:3 /W:3\n")
                    f.write(f"echo.\n")
                    f.write(f"echo Application has been updated. Relaunching...\n")
                    
                    if getattr(sys, 'frozen', False):
                        exe_path = os.path.join(install_dir, os.path.basename(sys.executable))
                        f.write(f"start \"\" \"{exe_path}\"\n")
                    else:
                        python_exe = sys.executable
                        entry_point = os.path.join(install_dir, 'main.py')
                        f.write(f"start \"\" \"{python_exe}\" \"{entry_point}\"\n")
                    
                    f.write(f"echo Success. Close this window.\n")
                    f.write(f"timeout /t 2 > nul\n")
                    f.write(f"exit\n")
                
                # Use subprocess to launch cmd with proper priority
                subprocess.Popen(['cmd.exe', '/c', 'start', '/high', 'cmd.exe', '/c', updater_bat], shell=True)
                self.close()
                sys.exit(0)
                
            except Exception as e:
                overlay.close()
                QMessageBox.critical(self, "Update Failed", f"Update error: {e}")

        QTimer.singleShot(500, run_task)

    def closeEvent(self, event):
        logger.info("Application closing. Terminating all background processes...")
        
        # Stop UI viewer threads
        for cid, thread in list(self.cam_threads.items()):
            try:
                thread.stop()
                thread.wait(500)
                self.update_camera_db_status(cid, 'inactive')
            except: pass

        try:
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                try:
                    child.terminate()
                except: pass
            
            # Wait for shutdown
            gone, alive = psutil.wait_procs(parent.children(), timeout=2)
            for p in alive:
                try:
                    p.kill()
                except: pass
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")

        event.accept()

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
        
        # Check if already percent encoded (contains %)
        if '%' in password:
            return url
            
        import urllib.parse
        encoded_pass = urllib.parse.quote(password)
        return f"{prefix}{user}:{encoded_pass}@{host_part}"
    except:
        return url

def get_state_from_plate(plate):
    if not plate or len(plate) < 2: return None
    STATES_MAP = {
        "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar", 
        "CG": "Chhattisgarh", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", 
        "HP": "Himachal Pradesh", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala", 
        "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya", 
        "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab", 
        "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TS": "Telangana", 
        "TR": "Tripura", "UP": "Uttar Pradesh", "UK": "Uttarakhand", "WB": "West Bengal",
        "AN": "Andaman and Nicobar", "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli",
        "DD": "Daman and Diu", "DL": "Delhi", "JK": "Jammu and Kashmir", "LA": "Ladakh",
        "LD": "Lakshadweep", "PY": "Puducherry"
    }
    code = plate[:2].upper()
    corrections = {"GP": "AP", "6P": "AP", "8P": "AP", "T5": "TS", "7S": "TS", "IS": "TS", "0L": "DL"}
    if code in corrections: code = corrections[code]
    return STATES_MAP.get(code, None)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if "--worker" in sys.argv:
        import vehicle_counter
        vehicle_counter.run()
    elif "--api" in sys.argv:
        import multi_camera_api
        multi_camera_api.run()
    else:
        # Fix taskbar icon on Windows
        if os.name == 'nt':
            myappid = 'Teja.VehicleTracker.V2.5.5'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            
        app = QApplication(sys.argv)
        
        # --- Splash Screen ---
        splash_px = QPixmap(600, 420)
        splash_px.fill(QColor("#f8fafc")) # Clean Light Theme BG
        
        splash = QSplashScreen(splash_px)
        splash.show()
        app.processEvents()

        def update_splash(msg, progress):
            painter = QPainter(splash_px)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Background refresh
            painter.fillRect(splash_px.rect(), QColor("#f8fafc"))
            
            # Title (Indigo)
            painter.setPen(QColor("#6366f1"))
            painter.setFont(QFont("Outfit", 28, QFont.Bold))
            painter.drawText(QRect(0, 100, 600, 60), Qt.AlignCenter, "AI SMART GATE")
            
            # Subtitle (Muted Slate)
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Outfit", 12, QFont.Bold))
            painter.drawText(QRect(0, 160, 600, 30), Qt.AlignCenter, "MONITORING SYSTEM")
            
            # Progress Message
            painter.setPen(QColor("#0f172a")) # Dark text for visibility
            painter.setFont(QFont("Outfit", 10))
            painter.drawText(QRect(0, 280, 600, 30), Qt.AlignCenter, msg)
            
            # Progress Bar Track
            bar_w = 400
            bar_h = 6
            bar_x = 100
            bar_y = 320
            
            painter.setBrush(QColor("#e2e8f0")) # Lighter track for light theme
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)
            
            # Progress Bar Fill
            painter.setBrush(QColor("#6366f1"))
            painter.drawRoundedRect(bar_x, bar_y, int(bar_w * (progress/100)), bar_h, 3, 3)
            
            # Percentage
            painter.setPen(QColor("#6366f1"))
            painter.setFont(QFont("Outfit", 9, QFont.Bold))
            painter.drawText(QRect(0, 340, 600, 20), Qt.AlignCenter, f"{progress}%")
            
            painter.end()
            splash.setPixmap(splash_px)
            app.processEvents()

        update_splash("Loading system components...", 10)
        time.sleep(0.1)
        
        update_splash("Initializing Neural Engine (Torch)...", 25)
        # We import here to show progress
        import vehicle_counter
        update_splash("Optimizing Computer Vision core...", 45)
        import multi_camera_api
        
        update_splash("Sanitizing Database...", 60)
        update_splash("Loading Premium UI Theme...", 75)
        
        # Set Window Icon
        icon_path = os.path.join(os.getcwd(), "app_icon.ico")
        if getattr(sys, 'frozen', False):
             icon_path = os.path.join(getattr(sys, '_MEIPASS', os.getcwd()), "app_icon.ico")
             
        if os.name == 'nt' and os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        
        font = QFont("Outfit", 10)
        app.setFont(font)
        
        update_splash("Building Professional Dashboard...", 90)
        window = UltraModernApp()
        
        update_splash("Ready!", 100)
        time.sleep(0.5)
        
        window.showMaximized()
        splash.finish(window)
        
        sys.exit(app.exec())
