import os
import sys
import subprocess
import shutil
import multiprocessing
import time
import requests
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox, QTextEdit, QLineEdit, QDialog)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QFont, QColor, QPixmap

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class PermissionRequestDialog(QDialog):
    """Dialog to request PowerShell execution policy permissions"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Permission Required")
        self.setFixedSize(500, 380)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.result_granted = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Icon/Header
        header_layout = QHBoxLayout()
        icon_label = QLabel("🔒")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_layout = QVBoxLayout()
        title = QLabel("PowerShell Execution Policy")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e293b;")
        subtitle = QLabel("Required for Desktop Shortcut Creation")
        subtitle.setStyleSheet("font-size: 14px; color: #64748b;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Explanation
        explanation = QLabel(
            "To create a desktop shortcut, this installer needs to run a PowerShell command.\n\n"
            "Some Windows systems restrict PowerShell execution for security. We need your permission to temporarily allow this.\n\n"
            "This is safe and only affects this installation process."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("""
            QLabel {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                color: #475569;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(explanation)
        
        # Info box
        info_box = QLabel("💡 This will set: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned")
        info_box.setWordWrap(True)
        info_box.setStyleSheet("""
            QLabel {
                background-color: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 6px;
                padding: 10px;
                color: #1e40af;
                font-size: 11px;
            }
        """)
        layout.addWidget(info_box)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        deny_btn = QPushButton("Skip Shortcut")
        deny_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        deny_btn.clicked.connect(self.deny_permission)
        button_layout.addWidget(deny_btn)
        
        grant_btn = QPushButton("Grant Permission")
        grant_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        grant_btn.clicked.connect(self.grant_permission)
        button_layout.addWidget(grant_btn)
        
        layout.addLayout(button_layout)
        
    def grant_permission(self):
        """Attempt to grant PowerShell execution policy permission"""
        try:
            # Try to set execution policy for current user (doesn't require admin)
            cmd = 'powershell -Command "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result.returncode == 0:
                self.result_granted = True
                QMessageBox.information(self, "Permission Granted", 
                    "PowerShell execution policy has been updated.\n\nYou can now create desktop shortcuts.")
                self.accept()
            else:
                # Try alternative method (Bypass for this session only)
                QMessageBox.warning(self, "Permission Update", 
                    "Could not permanently change execution policy.\n\n"
                    "The installer will try to use alternative methods to create the shortcut.")
                self.accept()
                
        except Exception as e:
            QMessageBox.warning(self, "Permission Update", 
                f"Could not update execution policy: {str(e)}\n\n"
                "The installer will try alternative methods to create the shortcut.")
            self.accept()
    
    def deny_permission(self):
        """User chose to skip shortcut creation"""
        self.result_granted = False
        self.accept()
    
    def was_granted(self):
        return self.result_granted

class PermissionRequestDialog(QDialog):
    """Dialog to request PowerShell execution policy permissions"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Permission Required")
        self.setFixedSize(500, 350)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.result_granted = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Icon/Header
        header_layout = QHBoxLayout()
        icon_label = QLabel("🔒")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_layout = QVBoxLayout()
        title = QLabel("PowerShell Execution Policy")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e293b;")
        subtitle = QLabel("Required for Desktop Shortcut Creation")
        subtitle.setStyleSheet("font-size: 14px; color: #64748b;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Explanation
        explanation = QLabel(
            "To create a desktop shortcut, this installer needs to run a PowerShell command.\n\n"
            "Some Windows systems restrict PowerShell execution for security. We need your permission to temporarily allow this.\n\n"
            "This is safe and only affects this installation process."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("""
            QLabel {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                color: #475569;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(explanation)
        
        # Info box
        info_box = QLabel("💡 This will set: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned")
        info_box.setWordWrap(True)
        info_box.setStyleSheet("""
            QLabel {
                background-color: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 6px;
                padding: 10px;
                color: #1e40af;
                font-size: 11px;
            }
        """)
        layout.addWidget(info_box)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        deny_btn = QPushButton("Skip Shortcut")
        deny_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        deny_btn.clicked.connect(self.deny_permission)
        button_layout.addWidget(deny_btn)
        
        grant_btn = QPushButton("Grant Permission")
        grant_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        grant_btn.clicked.connect(self.grant_permission)
        button_layout.addWidget(grant_btn)
        
        layout.addLayout(button_layout)
        
    def grant_permission(self):
        """Attempt to grant PowerShell execution policy permission"""
        try:
            # Try to set execution policy for current user (doesn't require admin)
            cmd = 'powershell -Command "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result.returncode == 0:
                self.result_granted = True
                QMessageBox.information(self, "Permission Granted", 
                    "PowerShell execution policy has been updated.\n\nYou can now create desktop shortcuts.")
                self.accept()
            else:
                # Try alternative method (Bypass for this session only)
                QMessageBox.warning(self, "Permission Update", 
                    "Could not permanently change execution policy.\n\n"
                    "The installer will try to use alternative methods to create the shortcut.")
                self.accept()
                
        except Exception as e:
            QMessageBox.warning(self, "Permission Update", 
                f"Could not update execution policy: {str(e)}\n\n"
                "The installer will try alternative methods to create the shortcut.")
            self.accept()
    
    def deny_permission(self):
        """User chose to skip shortcut creation"""
        self.result_granted = False
        self.accept()
    
    def was_granted(self):
        return self.result_granted

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

            # --- FORCE CLEAN INSTALL ---
            self.progress.emit("Cleaning existing installation...")
            try:
                import psutil
                # Force kill any running instances that might block file copying
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline')
                        if cmdline and ("main.py" in " ".join(cmdline) or "vehicle_counter.py" in " ".join(cmdline)):
                            self.log.emit(f"Terminating running process: {proc.info['pid']}", "#f59e0b")
                            proc.terminate()
                            proc.wait(timeout=3)
                    except: pass
            except: pass

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
            
            # Create launcher batch file
            launcher_path = os.path.join(self.target_dir, "LAUNCHER.bat")
            with open(launcher_path, "w") as f:
                # Use python.exe (not pythonw.exe) to ensure terminal stays visible for logs
                f.write(f"@echo off\ncd /d \"%~dp0\"\nstart \"\" \"env\\Scripts\\python.exe\" main.py\nexit\n")
            self.log.emit("READY > Launcher Script Created", "#10b981")
            
            # Try to create desktop shortcut (non-blocking - don't fail installation if this fails)
            shortcut_created = False
            desktop_path = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
            
            # Check if Desktop exists (some systems might have different paths)
            if not os.path.exists(desktop_path):
                # Try Public Desktop as fallback
                public_desktop = os.path.join(os.environ.get("PUBLIC", ""), "Desktop")
                if os.path.exists(public_desktop):
                    desktop_path = public_desktop
                else:
                    # Try to find Desktop via shell folder
                    try:
                        import winreg
                        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
                        desktop_path = winreg.QueryValueEx(key, "Desktop")[0]
                        winreg.CloseKey(key)
                    except:
                        pass
            
            if os.path.exists(desktop_path):
                shortcut_path = os.path.join(desktop_path, "AI Smart Vehicle Monitoring System.lnk")
                icon_p = os.path.join(self.target_dir, 'app_icon.ico')
                
                # Method 1: Try PowerShell (most reliable)
                try:
                    # First, check if we need to request permission
                    # Test PowerShell execution policy
                    test_cmd = 'powershell -Command "Get-ExecutionPolicy -Scope CurrentUser"'
                    test_result = subprocess.run(
                        test_cmd,
                        shell=True,
                        capture_output=True,
                        timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    execution_policy_ok = False
                    if test_result.returncode == 0:
                        policy = test_result.stdout.decode().strip().lower()
                        # RemoteSigned, Unrestricted, or Bypass are OK
                        if any(p in policy for p in ['remotesigned', 'unrestricted', 'bypass']):
                            execution_policy_ok = True
                    
                    # If policy is restricted, show permission dialog (but only once)
                    if not execution_policy_ok and not hasattr(self, '_permission_asked'):
                        self._permission_asked = True
                        # We can't show dialog from worker thread, so log and try anyway
                        self.log.emit("INFO > Attempting PowerShell with Bypass flag...", "#94a3b8")
                    
                    # Escape paths properly for PowerShell
                    shortcut_escaped = shortcut_path.replace("'", "''").replace("$", "`$")
                    launcher_escaped = launcher_path.replace("'", "''").replace("$", "`$")
                    target_dir_escaped = self.target_dir.replace("'", "''").replace("$", "`$")
                    icon_escaped = icon_p.replace("'", "''").replace("$", "`$")
                    
                    ps_cmd = f"$s=(New-Object -COM WScript.Shell).CreateShortcut('{shortcut_escaped}');$s.TargetPath='{launcher_escaped}';$s.WorkingDirectory='{target_dir_escaped}';if(Test-Path '{icon_escaped}'){{$s.IconLocation='{icon_escaped}'}};$s.Save()"
                    
                    result = subprocess.run(
                        ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                        capture_output=True,
                        timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    if result.returncode == 0:
                        shortcut_created = True
                        self.log.emit("LINK > Desktop Shortcut Created (PowerShell)", "#10b981")
                    else:
                        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                        if "execution policy" in error_msg.lower() or "script execution" in error_msg.lower():
                            raise Exception("Execution policy restricted")
                        else:
                            raise Exception(f"PowerShell returned {result.returncode}: {error_msg[:100]}")
                        
                except Exception as e:
                    # Method 2: Try VBScript (fallback)
                    try:
                        vbscript = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{launcher_path}"
oLink.WorkingDirectory = "{self.target_dir}"
oLink.IconLocation = "{icon_p}"
oLink.Save
"""
                        vbscript_path = os.path.join(self.target_dir, "create_shortcut.vbs")
                        with open(vbscript_path, "w") as vbs:
                            vbs.write(vbscript)
                        
                        result = subprocess.run(
                            ["cscript", "//nologo", vbscript_path],
                            cwd=self.target_dir,
                            capture_output=True,
                            timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW
                        )
                        
                        if result.returncode == 0:
                            shortcut_created = True
                            self.log.emit("LINK > Desktop Shortcut Created (VBScript)", "#10b981")
                        else:
                            raise Exception(f"VBScript returned {result.returncode}")
                        
                        # Clean up temp VBScript
                        try:
                            os.remove(vbscript_path)
                        except:
                            pass
                            
                    except Exception as e2:
                        # Method 3: Try direct COM object via Python (if available)
                        try:
                            import win32com.client
                            shell = win32com.client.Dispatch("WScript.Shell")
                            shortcut = shell.CreateShortCut(shortcut_path)
                            shortcut.Targetpath = launcher_path
                            shortcut.WorkingDirectory = self.target_dir
                            if os.path.exists(icon_p):
                                shortcut.IconLocation = icon_p
                            shortcut.save()
                            shortcut_created = True
                            self.log.emit("LINK > Desktop Shortcut Created (COM)", "#10b981")
                        except:
                            pass
                
                if not shortcut_created:
                    # Installation succeeded, but shortcut creation failed - log warning but don't fail
                    self.log.emit("WARNING > Could not create desktop shortcut (installation still successful)", "#f59e0b")
                    self.log.emit(f"INFO > You can launch from: {launcher_path}", "#94a3b8")
            else:
                self.log.emit("WARNING > Desktop folder not found, skipping shortcut creation", "#f59e0b")
                self.log.emit(f"INFO > You can launch from: {launcher_path}", "#94a3b8")
            
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

        # Check PowerShell execution policy and request permission if needed
        try:
            test_cmd = 'powershell -Command "Get-ExecutionPolicy -Scope CurrentUser"'
            test_result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            execution_policy_ok = False
            if test_result.returncode == 0:
                policy = test_result.stdout.decode().strip().lower()
                # RemoteSigned, Unrestricted, or Bypass are OK
                if any(p in policy for p in ['remotesigned', 'unrestricted', 'bypass']):
                    execution_policy_ok = True
            
            # If policy is restricted, show permission dialog
            if not execution_policy_ok:
                perm_dialog = PermissionRequestDialog(self)
                if perm_dialog.exec() == QDialog.Accepted:
                    if perm_dialog.was_granted():
                        # Permission was granted, continue with installation
                        pass
                    else:
                        # User chose to skip shortcut, continue anyway
                        pass
                else:
                    # User closed dialog, continue anyway
                    pass
        except Exception as e:
            # If we can't check policy, just continue - we'll handle it during shortcut creation
            pass

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
