import os
import re
import shutil
import subprocess
import zipfile
import webbrowser
import sys

def get_current_version(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'CURRENT_VERSION\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return None

def update_version(file_path, new_version):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = re.sub(
        r'CURRENT_VERSION\s*=\s*"[^"]+"',
        f'CURRENT_VERSION = "{new_version}"',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def zip_folder(folder_path, output_path):
    print(f"Zipping {folder_path} to {output_path}...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)

def split_file(file_path, chunk_size=1500 * 1024 * 1024): # 1.5 GB chunks
    """Splits a file into multiple parts .001, .002, etc."""
    part_num = 1
    files_created = []
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            part_name = f"{file_path}.{part_num:03d}"
            print(f"Creating split part: {part_name} ({len(chunk)/1024/1024:.1f} MB)")
            
            with open(part_name, 'wb') as part_file:
                part_file.write(chunk)
                
            files_created.append(part_name)
            part_num += 1
            
    return files_created

def main():
    print("=== AUTOMATED BUILD & RELEASE SCRIPT ===")
    
    target_file = 'vehicle_counter.py'
    current_ver = get_current_version(target_file)
    print(f"Current System Version: {current_ver}")
    
    new_ver = input("Enter new version number (e.g., v1.0.1): ").strip()
    if not new_ver:
        print("Build cancelled.")
        return

    # 1. Update Version in Code
    print(f"\n1. Updating code version to {new_ver}...")
    update_version(target_file, new_ver)
    update_version('main.py', new_ver)
    
    # 2. Check for EXE Build
    print("\n2. Checking for EXE build...")
    exe_path = None
    exe_name = "AI_Smart_Vehicle_Monitoring_System.exe"
    
    # Check common EXE locations
    possible_exe_paths = [
        os.path.join('dist', exe_name),
        os.path.join('dist', 'AI_Smart_Vehicle_Monitoring_System', exe_name),
        exe_name
    ]
    
    for path in possible_exe_paths:
        if os.path.exists(path):
            exe_path = path
            print(f"Found EXE: {exe_path}")
            break
    
    if not exe_path:
        print("No EXE found. Creating source-only update (clients must have Python installed).")
        print("To include EXE: Run 'build_single_exe.bat' first, then run this script again.")
    else:
        print(f"EXE will be included in update package.")

    # 3. Create Zip Package
    # 3. Create Release ZIP (Source Based for fast Remote Updates)
    print("\n3. Creating Source-based Release ZIP...")
    zip_name = 'VehicleCounter.zip'
    
    # Core Python files (required for updates)
    source_files = [
        'main.py', 
        'vehicle_counter.py', 
        'multi_camera_api.py',
        'api_server.py',  # Added - might be needed
        'installer.py', 
        'custom_tracker.yaml', 
        'data.yaml',  # Added - YOLO config file
        'app_icon.ico',
        'requirements.txt'
    ]
    
    # Create zip file
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add core source files
        for f in source_files:
            if os.path.exists(f):
                print(f"Adding to ZIP: {f}")
                zipf.write(f, f)  # Store with same name (root level)
            else:
                print(f"Warning: File not found, skipping: {f}")
        
        # Add dashboard folder if it exists (web interface)
        dashboard_folder = 'dashboard'
        if os.path.exists(dashboard_folder):
            print(f"Adding folder to ZIP: {dashboard_folder}/")
            for root, dirs, files in os.walk(dashboard_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Preserve folder structure: dashboard/file.html
                    arcname = file_path
                    zipf.write(file_path, arcname)
        
        # Add EXE if it exists (for EXE-based installations)
        if exe_path and os.path.exists(exe_path):
            print(f"Adding EXE to ZIP: {exe_path}")
            zipf.write(exe_path, os.path.basename(exe_path))
            print("Note: EXE included - this will work for both EXE and source installations")
        
        # Note: We're NOT including model files (yolov8n.pt, yolo11n.pt) 
        # because they're large and can be downloaded on first run
        # If you need to include them, uncomment below:
        # model_files = ['yolov8n.pt', 'yolo11n.pt']
        # for mf in model_files:
        #     if os.path.exists(mf):
        #         print(f"Adding model file: {mf} (this will increase ZIP size significantly)")
        #         zipf.write(mf, mf)
    
    zip_size_mb = os.path.getsize(zip_name) / 1024 / 1024
    print(f"\nZIP created: {zip_name} ({zip_size_mb:.2f} MB)")
    
    if zip_size_mb > 100:
        print("WARNING: ZIP is large (>100MB). Consider excluding EXE for faster updates.")
        print("         Source-only updates are typically < 10MB.")
    
    # CHECK SPLIT LOGIC
    release_files = [zip_name]
    zip_size = os.path.getsize(zip_name)
    
    if zip_size > 2 * 1024 * 1024 * 1024: # > 2GB
        print(f"\nWARNING: Zip file is huge ({zip_size/1024/1024/1024:.2f} GB). Splitting for GitHub...")
        split_parts = split_file(zip_name)
        
        # Remove original huge zip to avoid confusion
        os.remove(zip_name)
        release_files = split_parts
        print(f"Split complete. Created {len(release_files)} parts.")
    
    print("\n" + "="*50)
    print(f"SUCCESS! Build complete for {new_ver}")
    print("Files ready for release:")
    for f in release_files:
        print(f" - {os.path.abspath(f)}")
    print("="*50)
    
    print("\nNext Steps:")
    print("1. Git Commit & Push is starting now...")
    
    # 4. Git Operations
    try:
        subprocess.check_call(['git', 'add', 'main.py', 'vehicle_counter.py', 'installer.py', 'release_update.py'], shell=True)
        subprocess.check_call(['git', 'commit', '-m', f"Release {new_ver}"], shell=True)
        subprocess.check_call(['git', 'push'], shell=True)
        print("Git Push Complete.")
    except Exception as e:
        print(f"Git Error (you may need to push manually): {e}")

    # 5. Open Browser
    print("\nOpening GitHub Releases page...")
    webbrowser.open("https://github.com/Teja-pydahsoft/vehicle-tracker/releases/new")
    
    print(f"\nINSTRUCTION: Drag ALL these files into the GitHub page to publish:")
    for f in release_files:
        print(f" - {f}")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
