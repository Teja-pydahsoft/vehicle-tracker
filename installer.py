import os
import sys
import subprocess
import shutil
import multiprocessing
import time
import requests
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox, QTextEdit, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QFont, QColor

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def find_system_python():
    # 1. Try 'python'
    try:
        res = subprocess.run(["python", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except: pass
    # 2. Try 'py'
    try:
        res = subprocess.run(["py", "-3", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except: pass
    return None

def download_python(log_signal, percent_signal):
    """Downloads and installs Python 3.10 silently if missing"""
    import tempfile
    
    log_signal.emit("CRITICAL > Python runtime not found. Bootstrapping...", "#f59e0b")
    py_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
    installer_path = os.path.join(tempfile.gettempdir(), "python_installer.exe")
    
    log_signal.emit("Step 0/5: Downloading Python Runtime (64-bit)...", "#38bdf8")
    
    response = requests.get(py_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(installer_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    prog = int((downloaded / total_size) * 100)
                    percent_signal.emit(int(prog * 0.05)) # Map to first 5% of total installer

    log_signal.emit("Step 0/5: Installing Python (Silent Mode)...", "#38bdf8")
    # Install silently for current user, add to path
    subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "Include_launcher=1", "PrependPath=1"], check=True)
    log_signal.emit("SUCCESS > Python Runtime initialized.", "#10b981")
    
    # Wait a bit for path refresh
    time.sleep(2)
    return find_system_python()

class InstallWorker(QThread):
    progress = Signal(str)
    log = Signal(str, str) # text, color
    percent = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = target_dir

    def run(self):
        try:
            python_path = find_system_python()
            if not python_path:
                python_path = download_python(self.log, self.percent)
            
            if not python_path:
                raise Exception("Automated Python installation failed. Please install Python 3.10+ manually.")

            if not os.path.exists(self.target_dir):
                os.makedirs(self.target_dir)
            
            # Step 1: Venv
            self.percent.emit(5)
            self.progress.emit("Step 1/5: Initializing isolated environment...")
            venv_path = os.path.join(self.target_dir, "env")
            self.log.emit(f"BUILDING > {venv_path}", "#10b981")
            
            subprocess.run([python_path, "-m", "venv", venv_path], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.percent.emit(15)
            
            pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")

            # Step 2: AI Core
            self.progress.emit("Step 2/5: Downloading and Optimizing AI Core...")
            full_libs = ["PySide6", "ultralytics", "easyocr", "opencv-python", "flask", "flask-cors", "psutil", "lapx", "gitpython"]
            start_p, end_p = 15, 75

            import re
            progress_re = re.compile(r"(\d+\.?\d*)/(\d+\.?\d*)\s+(kB|MB|GB)")

            self.progress.emit("Step 2/5: Installing AI bundles (this may take time)...")
            self.percent.emit(start_p)
            
            # Grouped install is MUCH faster on Windows
            proc = subprocess.Popen(
                [pip_exe, "install", "--prefer-binary"] + full_libs,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            if proc.stdout:
                buffer = ""
                while True:
                    char = proc.stdout.read(1)
                    if not char:
                        break
                    if char == '\r' or char == '\n':
                        clean_line = buffer.strip()
                        if clean_line:
                            self.log.emit(clean_line, "#94a3b8")
                            
                            match = progress_re.search(clean_line)
                            if match:
                                current = float(match.group(1))
                                total = float(match.group(2))
                                if total > 0:
                                    p_ratio = current / total
                                    p_current = start_p + (p_ratio * (end_p - start_p))
                                    self.percent.emit(int(p_current))
                            
                        buffer = ""
                    else:
                        buffer += char
            
            proc.wait()
            if proc.returncode != 0:
                raise Exception("Failed to install AI components. Check internet connection.")
                
            self.log.emit("SUCCESS > AI Core initialized.", "#10b981")
            self.percent.emit(end_p)

            # Step 3: Assets
            self.progress.emit("Step 4/5: Deploying Application Assets...")
            os.makedirs(os.path.join(self.target_dir, "shared_frames"), exist_ok=True)
            files = ["main.py", "vehicle_counter.py", "multi_camera_api.py", "yolov8n.pt", "custom_tracker.yaml", "app_icon.ico"]
            for i, f in enumerate(files):
                src = get_resource_path(f)
                dest = os.path.join(self.target_dir, f)
                if os.path.exists(src):
                    shutil.copy2(src, dest)
                    self.log.emit(f"READY > {f}", "#e2e8f0")
                p_inc = 75 + int(((i+1)/len(files)) * 10)
                self.percent.emit(p_inc)
            
            # Step 4: Dashboard
            self.progress.emit("Step 4/5: Building Visual Dashboard...")
            dash_src = get_resource_path("dashboard")
            dash_dest = os.path.join(self.target_dir, "dashboard")
            if os.path.exists(dash_src):
                if os.path.exists(dash_dest): shutil.rmtree(dash_dest)
                shutil.copytree(dash_src, dash_dest)
                self.log.emit("READY > Web Dashboard Assets", "#e2e8f0")
            self.percent.emit(90)

            # Step 5: Access
            self.progress.emit("Step 5/5: Configuring System Access...")
            shortcut_path = os.path.join(os.environ["USERPROFILE"], "Desktop", "AI Smart Vehicle Monitoring System.lnk")
            launcher_path = os.path.join(self.target_dir, "LAUNCHER.bat")
            with open(launcher_path, "w") as f:
                # Use python.exe (not pythonw.exe) to ensure terminal stays visible for logs
                f.write(f"@echo off\ncd /d \"%~dp0\"\n\"env\\Scripts\\python.exe\" main.py\npause\n")

            icon_p = os.path.join(self.target_dir, 'app_icon.ico')
            ps_cmd = f"$s=(New-Object -COM WScript.Shell).CreateShortcut('{shortcut_path}');$s.TargetPath='{launcher_path}';$s.WorkingDirectory='{self.target_dir}';$s.IconLocation='{icon_p}';$s.Save()"
            subprocess.run(["powershell", "-Command", ps_cmd], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.log.emit("LINK > Desktop Shortcut Created", "#10b981")
            
            self.percent.emit(100)
            self.finished.emit(True, "Success")
        except Exception as e:
            self.finished.emit(False, str(e))

class SetupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Smart Vehicle Monitoring System - Setup")
        self.setFixedSize(650, 520)
        self.setStyleSheet("QMainWindow { background-color: white; }")
        
        self.current_display_val = 0
        self.target_val = 0
        self.smooth_timer = QTimer()
        self.smooth_timer.timeout.connect(self._smooth_tick)
        self.smooth_timer.start(30) # 33fps
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)

        header = QLabel("AI SMART VEHICLE MONITORING SYSTEM")
        header.setStyleSheet("font-size: 22px; font-weight: 900; color: #1e293b; font-family: 'Segoe UI';")
        layout.addWidget(header, 0, Qt.AlignCenter)

        self.status_lbl = QLabel("Ready to configure professional surveillance system.")
        self.status_lbl.setStyleSheet("color: #64748b; font-size: 14px; font-family: 'Segoe UI';")
        layout.addWidget(self.status_lbl, 0, Qt.AlignCenter)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                border-radius: 8px;
                color: #e2e8f0;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                padding: 15px;
            }
        """)
        layout.addWidget(self.log_area)

        # Path Selection Section
        path_group = QWidget()
        path_layout = QVBoxLayout(path_group)
        path_layout.setContentsMargins(0, 5, 0, 5)
        path_layout.setSpacing(5)

        path_label = QLabel("Installation Location:")
        path_label.setStyleSheet("font-weight: bold; color: #475569; font-size: 13px;")
        path_layout.addWidget(path_label)

        path_input_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        default_path = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "AI Smart Surveillance")
        self.path_edit.setText(default_path)
        self.path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px;
                color: #1e293b;
                font-size: 13px;
            }
        """)
        path_input_row.addWidget(self.path_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #e2e8f0; }
        """)
        self.browse_btn.clicked.connect(self.browse_path)
        path_input_row.addWidget(self.browse_btn)
        path_layout.addLayout(path_input_row)

        layout.addWidget(path_group)

        # Progress Box
        prog_lay = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none; border-radius: 4px; background: #f1f5f9; text-align: center; height: 8px; font-size: 0px;
            }
            QProgressBar::chunk { background-color: #6366f1; border-radius: 4px; }
        """)
        prog_lay.addWidget(self.progress_bar)
        
        self.perc_lbl = QLabel("0%")
        self.perc_lbl.setStyleSheet("font-weight: 800; color: #6366f1; font-size: 12px; width: 40px;")
        prog_lay.addWidget(self.perc_lbl)
        layout.addLayout(prog_lay)

        self.install_btn = QPushButton("BEGIN INSTALLATION")
        self.install_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1; color: white; border-radius: 10px;
                padding: 14px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #4f46e5; }
        """)
        self.install_btn.clicked.connect(self.start_installation)
        layout.addWidget(self.install_btn)

    def browse_path(self):
        existing = self.path_edit.text()
        start_dir = existing if os.path.exists(existing) else "C:\\"
        path = QFileDialog.getExistingDirectory(self, "Select Installation Folder", start_dir)
        if path:
            # Append project name if not already there
            if not path.endswith("AI Smart Surveillance"):
                 path = os.path.join(path, "AI Smart Surveillance")
            self.path_edit.setText(path)

    def start_installation(self):
        path = self.path_edit.text()
        if not path:
             QMessageBox.warning(self, "Invalid Path", "Please select a valid installation directory.")
             return
        
        # Check for Program Files permissions
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except PermissionError:
            # Fallback to User AppData if Program Files is protected and not running as admin
            user_path = os.path.join(os.environ["LOCALAPPDATA"], "AI_Smart_Surveillance")
            res = QMessageBox.question(self, "Permission Required", 
                                     f"Standard installation directory requires Administrator privileges.\n\nWould you like to install to your user folder instead?\n{user_path}",
                                     QMessageBox.Yes | QMessageBox.No)
            if res == QMessageBox.Yes:
                path = user_path
                self.path_edit.setText(path)
                os.makedirs(path, exist_ok=True)
            else:
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create directory: {e}")
            return

        self.install_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.path_edit.setReadOnly(True)
        self.install_btn.setText("INSTALLATION IN PROGRESS...")
        self.install_btn.setStyleSheet("""
            QPushButton {
                background-color: #94a3b8; color: white; border-radius: 10px;
                padding: 14px; font-weight: bold; font-size: 14px;
            }
        """)
        self.log_area.append(f"<font color='#38bdf8'><b>[SYSTEM] Starting installation v2.5.5</b></font>")
        
        self.worker = InstallWorker(path)
        self.worker.progress.connect(self.status_lbl.setText)
        self.worker.log.connect(lambda t, c: self.log_area.append(f"<font color='{c}'>{t}</font>"))
        self.worker.percent.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, val):
        self.target_val = val

    def _smooth_tick(self):
        if self.current_display_val < self.target_val:
            self.current_display_val += 1
            self.progress_bar.setValue(self.current_display_val)
            self.perc_lbl.setText(f"{self.current_display_val}%")
        elif self.current_display_val > self.target_val:
            # For sudden resets (if any)
            self.current_display_val = self.target_val
            self.progress_bar.setValue(self.current_display_val)
            self.perc_lbl.setText(f"{self.current_display_val}%")

    def on_finished(self, success, message):
        if success:
            self.log_area.append("<br><font color='#10b981'><b>COMPLETED SUCCESSFULLY</b></font>")
            QMessageBox.information(self, "Success", "Installation Complete! Launch the app using the shortcut on your desktop.")
            sys.exit(0)
        else:
            QMessageBox.critical(self, "Error", f"Installation Failed:\n\n{message}")
            self.install_btn.setEnabled(True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = SetupWindow()
    window.show()
    sys.exit(app.exec())
