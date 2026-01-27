from ultralytics import YOLO
import os
import torch

def train_custom_model():
    # 1. Initialize YOLOv11 model
    model = YOLO('yolo11n.pt') 

    # 2. Path to your dataset YAML file
    dataset_yaml = 'data.yaml' 

    if not os.path.exists(dataset_yaml):
        print(f"Error: {dataset_yaml} not found! Please make sure your dataset YAML is named 'data.yaml'.")
        return

    print("Starting training on custom dataset...")
    
    # 3. Train the model
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=50,
            imgsz=640,
            batch=8,           # Reduced from 16 to save memory
            workers=0,         # Set to 0 to prevent multiprocessing DLL loading errors
            amp=False,         # Disabled because GTX 1650 can have issues with AMP
            plots=True,
            device=0 if torch.cuda.is_available() else 'cpu' 
        )
        print("Training complete!")
        print(f"Your trained model is saved at: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    train_custom_model()
