from ultralytics import YOLO
import os

def check_model(path):
    print(f"Checking model: {path}")
    if not os.path.exists(path):
        print("  File not found.")
        return
    try:
        model = YOLO(path)
        print(f"  Classes ({len(model.names)}):")
        print(f"  {model.names}")
    except Exception as e:
        print(f"  Error loading model: {e}")

check_model('models/custom_model.pt')
check_model('yolo11n.pt')
check_model('yolov8n.pt')
