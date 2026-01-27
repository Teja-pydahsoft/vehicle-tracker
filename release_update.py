import os
import re
import shutil
import subprocess
import zipfile
import webbrowser
import sys

def get_current_version(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'CURRENT_VERSION\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return None

def update_version(file_path, new_version):
    with open(file_path, 'r') as f:
        content = f.read()
    
    new_content = re.sub(
        r'CURRENT_VERSION\s*=\s*"[^"]+"',
        f'CURRENT_VERSION = "{new_version}"',
        content
    )
    
    with open(file_path, 'w') as f:
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
    
    # 2. Build EXE
    print("\n2. Building Executable (this may take a few minutes)...")
    build_cmd = [
        'pyinstaller',
        '--noconfirm',
        '--onedir',
        '--console',
        '--clean',
        '--name', 'VehicleCounter',
        '--add-data', 'custom_tracker.yaml;.',
        '--add-data', 'models;models',
        '--collect-all', 'ultralytics',
        '--collect-all', 'easyocr',
        'vehicle_counter.py'
    ]
    
    try:
        subprocess.check_call(build_cmd, shell=True)
    except subprocess.CalledProcessError:
        print("Error: Build failed!")
        return

    # 3. Create Zip Package
    print("\n3. Creating Release ZIP...")
    dist_folder = os.path.join('dist', 'VehicleCounter')
    zip_name = 'VehicleCounter.zip'
    
    # Ensure dist folder exists
    if not os.path.exists(dist_folder):
        print(f"Error: Could not find build output at {dist_folder}")
        return
        
    zip_folder(dist_folder, zip_name)
    
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
        subprocess.check_call(['git', 'add', target_file], shell=True)
        subprocess.check_call(['git', 'commit', '-m', f"Release version {new_ver}"], shell=True)
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
