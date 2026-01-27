import cv2
import numpy as np
import os

def create_realistic_test_video(output_path, image_path="test_bus.jpg"):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Using dummy instead.")
        # Create a dummy colored box that looks a bit more like a car
        bus_img = np.zeros((100, 200, 3), dtype=np.uint8)
        bus_img[:,:] = (128, 128, 128) # Grey
        cv2.rectangle(bus_img, (20, 70), (60, 95), (0,0,0), -1) # Wheel
        cv2.rectangle(bus_img, (140, 70), (180, 95), (0,0,0), -1) # Wheel
    else:
        bus_img = cv2.imread(image_path)
        bus_img = cv2.resize(bus_img, (300, 150))

    width, height = 1280, 720
    fps = 30
    duration = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    bh, bw = bus_img.shape[:2]
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background
        cv2.rectangle(frame, (width//2 - 200, 0), (width//2 + 200, height), (40, 40, 40), -1)
        
        # Move the bus from top to bottom
        y = -bh + int((i / (duration * fps)) * (height + bh))
        x = width // 2 - bw // 2
        
        # Overlay
        if 0 <= y < height - bh:
            frame[y:y+bh, x:x+bw] = bus_img
            
        out.write(frame)
    
    out.release()
    print(f"Realistic test video created: {output_path}")

if __name__ == "__main__":
    create_realistic_test_video("realistic_test.mp4")
