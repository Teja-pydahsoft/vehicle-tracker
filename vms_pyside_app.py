
import sys
import os
import sqlite3
import subprocess
import time
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QLabel, QPushButton, QTabWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QLineEdit, QComboBox, 
    QCheckBox, QGridLayout, QScrollArea, QSizePolicy, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, QRect, QPoint
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap, QLinearGradient, QPalette, QPainter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color Palette (Matches multi_camera.css)
COLORS = {
    'primary': '#6366f1',
    'primary_dark': '#4f46e5',
    'primary_light': '#818cf8',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'bg_main': '#0f172a',
    'bg_secondary': '#1e293b',
    'bg_card': '#1e293b',
    'bg_hover': '#334155',
    'text_primary': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'text_muted': '#64748b',
    'border': '#334155',
}

QSS_THEME = f"""
QMainWindow {{
    background-color: {COLORS['bg_main']};
}}

QWidget {{
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', sans-serif;
}}

/* Sidebar */
#Sidebar {{
    background-color: {COLORS['bg_secondary']};
    border-right: 1px solid {COLORS['border']};
    min-width: 240px;
    max-width: 240px;
}}

#SidebarLogo {{
    padding: 20px;
    border-bottom: 1px solid {COLORS['border']};
}}

#LogoText {{
    color: {COLORS['primary_light']};
    font-size: 18px;
    font-weight: bold;
}}

/* Sidebar Navigation */
QPushButton.NavButton {{
    background-color: transparent;
    border: none;
    border-left: 4px solid transparent;
    color: {COLORS['text_secondary']};
    text-align: left;
    padding: 15px 25px;
    font-size: 14px;
    font-weight: 500;
}}

QPushButton.NavButton:hover {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['text_primary']};
}}

QPushButton.NavButton[active="true"] {{
    background-color: rgba(99, 102, 241, 0.1);
    color: {COLORS['primary_light']};
    border-left: 4px solid {COLORS['primary']};
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
    border-radius: 16px;
    min-height: 120px;
}}

/* Titles */
QLabel.SectionTitle {{
    font-size: 20px;
    font-weight: bold;
    color: {COLORS['text_primary']};
    margin-bottom: 15px;
}}

QLabel.StatTitle {{
    font-size: 12px;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
}}

QLabel.StatValue {{
    font-size: 32px;
    font-weight: 900;
    color: {COLORS['text_primary']};
}}

/* Buttons */
QPushButton.ActionBtn {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
}}

QPushButton.ActionBtn:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton.SuccessBtn {{
    background-color: {COLORS['success']};
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 6px;
    font-weight: bold;
}}

QPushButton.DangerBtn {{
    background-color: {COLORS['danger']};
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 6px;
    font-weight: bold;
}}

/* Tables */
QTableWidget {{
    background-color: {COLORS['bg_card']};
    alternate-background-color: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    border-radius: 8px;
}}

QHeaderView::section {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['text_secondary']};
    padding: 10px;
    border: none;
    font-weight: bold;
}}

/* Tabs/StackedWidget */
QTabWidget::pane {{
    border: none;
    background: transparent;
}}

QTabWidget::tab-bar {{
    alignment: center;
}}

/* Input Fields */
QLineEdit {{
    background-color: {COLORS['bg_hover']};
    border: 1px solid {COLORS['border']};
    padding: 8px;
    border-radius: 6px;
    color: white;
}}

QComboBox {{
    background-color: {COLORS['bg_hover']};
    border: 1px solid {COLORS['border']};
    padding: 8px;
    border-radius: 6px;
    color: white;
}}
"""

class StatCard(QFrame):
    def __init__(self, title, value, color_type='primary', icon="📊"):
        super().__init__()
        self.setObjectName("StatCard")
        self.setProperty("class", "StatCard")
        
        # Add glow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(self.shadow)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Icon box
        self.icon_label = QLabel(icon)
        self.icon_label.setFixedSize(60, 60)
        self.icon_label.setAlignment(Qt.AlignCenter)
        color = COLORS[color_type]
        self.icon_label.setStyleSheet(f"""
            font-size: 28px; 
            background-color: rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1); 
            border-radius: 12px; 
            color: {color};
            border: 1px solid rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.2);
        """)
        layout.addWidget(self.icon_label)
        
        # Text box
        text_layout = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setProperty("class", "StatTitle")
        
        self.value_label = QLabel(value)
        self.value_label.setProperty("class", "StatValue")
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.value_label)
        layout.addLayout(text_layout)
        
    def set_value(self, value):
        self.value_label.setText(str(value))

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=COLORS['bg_card'])
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(COLORS['bg_card'])
        self.axes.tick_params(colors=COLORS['text_muted'], labelsize=8)
        for spine in self.axes.spines.values():
            spine.set_color(COLORS['border'])
        super().__init__(self.fig)

class StatusDot(QLabel):
    def __init__(self, color=COLORS['success']):
        super().__init__()
        self.color = color
        self.setFixedSize(12, 12)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)
        self.opacity = 1.0
        self.direction = -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Animate opacity
        self.opacity += self.direction * 0.05
        if self.opacity <= 0.4 or self.opacity >= 1.0:
            self.direction *= -1
            
        color = QColor(self.color)
        color.setAlphaF(self.opacity)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 10, 10)

class CameraWidget(QFrame):
    def __init__(self, camera_id, name):
        super().__init__()
        self.setProperty("class", "Card")
        self.setMinimumHeight(250)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QFrame()
        header.setStyleSheet(f"background-color: {COLORS['bg_hover']}; border-top-left-radius: 12px; border-top-right-radius: 12px;")
        header_layout = QHBoxLayout(header)
        
        self.name_label = QLabel(f"📹 {name}")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.status_label = QLabel("● INACTIVE")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: bold; font-size: 11px;")
        
        header_layout.addWidget(self.name_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        layout.addWidget(header)
        
        # Feed Placeholder
        self.feed_placeholder = QLabel("Camera Offline")
        self.feed_placeholder.setAlignment(Qt.AlignCenter)
        self.feed_placeholder.setStyleSheet("font-size: 16px; color: #4b5563; background-color: #000;")
        layout.addWidget(self.feed_placeholder)
        
        # Controls
        ctrl = QFrame()
        ctrl_layout = QHBoxLayout(ctrl)
        
        self.start_btn = QPushButton("START")
        self.start_btn.setProperty("class", "SuccessBtn")
        
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setProperty("class", "DangerBtn")
        
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        layout.addWidget(ctrl)

class MultiCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Smart Vehicle Monitoring System")
        self.resize(1600, 1000)
        self.setStyleSheet(QSS_THEME)
        
        # Set Window Icon
        icon_path = os.path.join(os.getcwd(), "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.api_process = None
        self.camera_configs = {
            1: {'name': 'Camera 1 - Main Gate', 'rtsp_url': '', 'enabled': True},
            2: {'name': 'Camera 2 - Exit Gate', 'rtsp_url': '', 'enabled': True},
            3: {'name': 'Camera 3 - Parking Entry', 'rtsp_url': '', 'enabled': True},
            4: {'name': 'Camera 4 - Parking Exit', 'rtsp_url': '', 'enabled': True}
        }
        
        self.setup_ui()
        self.start_api_server()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_stats)
        self.timer.start(5000)
        
    def setup_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Sidebar
        self.setup_sidebar()
        
        # Content Area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.addWidget(self.content_area)
        
        # Header / Status bar simulation
        header = QHBoxLayout()
        self.page_title = QLabel("Dashboard")
        self.page_title.setStyleSheet("font-size: 28px; font-weight: 800; color: white;")
        header.addWidget(self.page_title)
        header.addStretch()
        
        # Clock
        self.clock_label = QLabel()
        self.clock_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 16px; margin-right: 20px;")
        header.addWidget(self.clock_label)
        
        # System Status
        status_box = QHBoxLayout()
        self.status_dot = StatusDot()
        status_box.addWidget(self.status_dot)
        self.sys_status = QLabel("System Online")
        self.sys_status.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold; margin-left: 5px;")
        status_box.addWidget(self.sys_status)
        header.addLayout(status_box)
        
        self.content_layout.addLayout(header)
        
        # Update clock every second
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()
        
        # Stacked views (mimicked by tabs for now, or just showing/hiding)
        self.views = QTabWidget()
        self.views.tabBar().hide() # Hide tab bar, use sidebar for nav
        self.content_layout.addWidget(self.views)
        
        # Initialize tabs
        self.dashboard_view = QWidget()
        self.setup_dashboard_view()
        self.views.addTab(self.dashboard_view, "Dashboard")
        
        self.camera_grid_view = QWidget()
        self.setup_camera_grid_view()
        self.views.addTab(self.camera_grid_view, "Cameras")
        
        self.history_view = QWidget()
        self.setup_history_view()
        self.views.addTab(self.history_view, "History")
        
        self.settings_view = QWidget()
        self.setup_settings_view()
        self.views.addTab(self.settings_view, "Settings")

    def setup_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(5)
        
        # Logo
        logo_box = QFrame()
        logo_box.setObjectName("SidebarLogo")
        logo_layout = QHBoxLayout(logo_box)
        logo_text = QLabel("AI SMART VEHICLE MONITORING SYSTEM")
        logo_text.setObjectName("LogoText")
        logo_layout.addWidget(logo_text)
        sidebar_layout.addWidget(logo_box)
        
        # Nav Buttons
        self.nav_btns = []
        
        db_btn = QPushButton("📊  Dashboard")
        db_btn.setProperty("class", "NavButton")
        db_btn.setProperty("active", "true")
        db_btn.clicked.connect(lambda: self.switch_view(0, "Dashboard", db_btn))
        sidebar_layout.addWidget(db_btn)
        self.nav_btns.append(db_btn)
        
        cam_btn = QPushButton("📹  Live Cameras")
        cam_btn.setProperty("class", "NavButton")
        cam_btn.clicked.connect(lambda: self.switch_view(1, "Live Grid", cam_btn))
        sidebar_layout.addWidget(cam_btn)
        self.nav_btns.append(cam_btn)
        
        hist_btn = QPushButton("📜  Detection History")
        hist_btn.setProperty("class", "NavButton")
        hist_btn.clicked.connect(lambda: self.switch_view(2, "History", hist_btn))
        sidebar_layout.addWidget(hist_btn)
        self.nav_btns.append(hist_btn)
        
        conf_btn = QPushButton("⚙️  Configuration")
        conf_btn.setProperty("class", "NavButton")
        conf_btn.clicked.connect(lambda: self.switch_view(3, "Settings", conf_btn))
        sidebar_layout.addWidget(conf_btn)
        self.nav_btns.append(conf_btn)
        
        sidebar_layout.addStretch()
        
        # Footer
        footer = QLabel("v2.2 | Pydah Soft")
        footer.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 20px; font-size: 10px;")
        footer.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(footer)
        
        self.main_layout.addWidget(sidebar)

    def switch_view(self, index, title, btn):
        self.views.setCurrentIndex(index)
        self.page_title.setText(title)
        
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
        layout = QVBoxLayout(self.dashboard_view)
        layout.setContentsMargins(0, 20, 0, 0)
        
        # Stats Row
        stats_layout = QHBoxLayout()
        self.total_in = StatCard("Total Vehicles In", "0", "success", "↓")
        self.total_out = StatCard("Total Vehicles Out", "0", "danger", "↑")
        self.active_cams = StatCard("Active Cameras", "0/4", "info", "📹")
        self.total_count = StatCard("Total Detected", "0", "primary", "🚗")
        
        stats_layout.addWidget(self.total_in)
        stats_layout.addWidget(self.total_out)
        stats_layout.addWidget(self.active_cams)
        stats_layout.addWidget(self.total_count)
        layout.addLayout(stats_layout)
        
        # Split view: Activity & Chart-like area
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(20)
        
        # Recent Activity Table
        activity_box = QFrame()
        activity_box.setProperty("class", "Card")
        activity_layout = QVBoxLayout(activity_box)
        
        title = QLabel("Recent Detections")
        title.setProperty("class", "SectionTitle")
        activity_layout.addWidget(title)
        
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Camera', 'Type', 'Plate', 'Direction', 'Time', 'Confidence'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        activity_layout.addWidget(self.table)
        
        mid_layout.addWidget(activity_box, 2)
        
        # Chart and Status Panel
        right_panel = QVBoxLayout()
        
        # Chart Card
        chart_card = QFrame()
        chart_card.setProperty("class", "Card")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.addWidget(QLabel("Traffic Trends (Last 24h)"))
        
        self.canvas = MplCanvas(self, width=5, height=3, dpi=80)
        chart_layout.addWidget(self.canvas)
        right_panel.addWidget(chart_card, 2)
        
        # Info Panel
        info_panel = QFrame()
        info_panel.setProperty("class", "Card")
        info_layout = QVBoxLayout(info_panel)
        info_layout.addWidget(QLabel("Active Camera Monitor"))
        
        self.cam_stats_list = QVBoxLayout()
        info_layout.addLayout(self.cam_stats_list)
        for i in range(1, 5):
            lbl = QLabel(f"Camera {i}: Inactive")
            lbl.setStyleSheet(f"padding: 10px; background: {COLORS['bg_secondary']}; border-radius: 5px; font-size: 11px;")
            self.cam_stats_list.addWidget(lbl)
            
        right_panel.addWidget(info_panel, 1)
        
        mid_layout.addLayout(right_panel, 1)
        
        layout.addLayout(mid_layout)

    def update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%d %b %Y | %H:%M:%S"))

    def update_chart(self, data):
        self.canvas.axes.cla()
        self.canvas.axes.set_facecolor(COLORS['bg_card'])
        hours = [d[0] for d in data]
        counts = [d[1] for d in data]
        
        self.canvas.axes.plot(hours, counts, color=COLORS['primary'], marker='o', linewidth=2, markersize=4)
        self.canvas.axes.fill_between(hours, counts, color=COLORS['primary'], alpha=0.1)
        self.canvas.axes.grid(True, color=COLORS['border'], linestyle='--', alpha=0.3)
        self.canvas.draw()

    def setup_camera_grid_view(self):
        layout = QVBoxLayout(self.camera_grid_view)
        
        grid = QGridLayout()
        self.cam_widgets = {}
        for i in range(4):
            cam = CameraWidget(i+1, f"Camera {i+1}")
            self.cam_widgets[i+1] = cam
            grid.addWidget(cam, i // 2, i % 2)
            
        layout.addLayout(grid)

    def setup_history_view(self):
        layout = QVBoxLayout(self.history_view)
        
        filters = QHBoxLayout()
        self.h_cam_filter = QComboBox()
        self.h_cam_filter.addItems(['All Cameras', '1', '2', '3', '4'])
        filters.addWidget(QLabel("Camera:"))
        filters.addWidget(self.h_cam_filter)
        
        self.h_type_filter = QComboBox()
        self.h_type_filter.addItems(['All Types', 'Car', 'Truck', 'Bus', 'Motorcycle'])
        filters.addWidget(QLabel("Type:"))
        filters.addWidget(self.h_type_filter)
        
        search_btn = QPushButton("Search")
        search_btn.setProperty("class", "ActionBtn")
        search_btn.clicked.connect(self.load_history)
        filters.addWidget(search_btn)
        filters.addStretch()
        
        layout.addLayout(filters)
        
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(['ID', 'Camera', 'Type', 'Plate', 'Direction', 'Time', 'Conf'])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.history_table)

    def setup_settings_view(self):
        layout = QVBoxLayout(self.settings_view)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout(scroll_content)
        
        for i in range(1, 5):
            box = QFrame()
            box.setProperty("class", "Card")
            box_lat = QVBoxLayout(box)
            box_lat.addWidget(QLabel(f"Camera {i} Configuration"))
            
            name_in = QLineEdit()
            name_in.setPlaceholderText("Camera Name")
            box_lat.addWidget(name_in)
            
            rtsp_in = QLineEdit()
            rtsp_in.setPlaceholderText("RTSP URL")
            box_lat.addWidget(rtsp_in)
            
            scroll_layout.addWidget(box, (i-1)//2, (i-1)%2)
            
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        save_btn = QPushButton("SAVE ALL CONFIGURATIONS")
        save_btn.setProperty("class", "ActionBtn")
        save_btn.setFixedHeight(50)
        layout.addWidget(save_btn)

    def start_api_server(self):
        try:
            self.api_process = subprocess.Popen(
                [sys.executable, "multi_camera_api.py"],
                cwd=os.getcwd()
            )
            logger.info("API server started")
        except Exception as e:
            logger.error(f"Failed to start API: {e}")

    def refresh_stats(self):
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            # Counts
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE direction = 'IN'")
            cin = cursor.fetchone()[0]
            self.total_in.set_value(cin)
            
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE direction = 'OUT'")
            cout = cursor.fetchone()[0]
            self.total_out.set_value(cout)
            
            self.total_count.set_value(cin + cout)
            
            # Active cameras (last 5 mins)
            cursor.execute("SELECT COUNT(DISTINCT camera_id) FROM vehicle_logs WHERE timestamp > datetime('now', '-5 minutes')")
            active = cursor.fetchone()[0]
            self.active_cams.set_value(f"{active}/4")
            
            # Activity Table
            cursor.execute("""
                SELECT camera_id, vehicle_type, plate_number, direction, timestamp, confidence 
                FROM vehicle_logs ORDER BY timestamp DESC LIMIT 20
            """)
            rows = cursor.fetchall()
            self.table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    if j == 3: # Direction
                        # item.setForeground(QColor(COLORS['success'] if val == 'IN' else COLORS['danger']))
                        # Use a more modern look for status
                        pass
                    self.table.setItem(i, j, item)
            
            # Chart Data (Hourly for today)
            cursor.execute("""
                SELECT strftime('%H', timestamp) as hour, COUNT(*) 
                FROM vehicle_logs 
                WHERE date(timestamp) = date('now')
                GROUP BY hour
                ORDER BY hour
            """)
            chart_data = cursor.fetchall()
            if chart_data:
                self.update_chart(chart_data)
            else:
                # Mock data if empty
                mock = [(f"{i:02d}", 0) for i in range(24)]
                self.update_chart(mock)
                    
            conn.close()
        except Exception as e:
            logger.error(f"Refresh error: {e}")

    def load_history(self):
        try:
            conn = sqlite3.connect('gate_log.db')
            cursor = conn.cursor()
            
            cam_filter = self.h_cam_filter.currentText()
            type_filter = self.h_type_filter.currentText()
            
            query = "SELECT id, camera_id, vehicle_type, plate_number, direction, timestamp, confidence FROM vehicle_logs WHERE 1=1"
            params = []
            
            if cam_filter != 'All Cameras':
                query += " AND camera_id = ?"
                params.append(int(cam_filter))
            
            if type_filter != 'All Types':
                query += " AND vehicle_type = ?"
                params.append(type_filter)
                
            query += " ORDER BY timestamp DESC LIMIT 100"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            self.history_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    self.history_table.setItem(i, j, QTableWidgetItem(str(val)))
            
            conn.close()
        except Exception as e:
            logger.error(f"History load error: {e}")

    def closeEvent(self, event):
        if self.api_process:
            self.api_process.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Modern font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MultiCameraApp()
    window.show()
    sys.exit(app.exec())
