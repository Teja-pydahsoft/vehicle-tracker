import cv2
import os
import numpy as np
# removed tqdm for compatibility

def create_dataset_video(output_path, images_dir="train/images", max_images=300):
    """
    Creates a video from training images. 
    Sorted to keep sequences (like adit_mp4-100, 101...) together.
    """
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} not found.")
        return

    # Get all jpg files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sort naturally (so -100 comes before -101)
    image_files.sort()
    
    # Limit count for a reasonable test video length
    image_files = image_files[:max_images]
    
    if not image_files:
        print("No images found in directory.")
        return

    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
    h, w, _ = first_img.shape
    
    # Define video writer (10 FPS for a natural look of these sequences)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))
    
    print(f"Generating video from {len(image_files)} images...")
    
    for idx, img_name in enumerate(image_files):
        if idx % 50 == 0: print(f"Processing frame {idx}...")
        img_path = os.path.join(images_dir, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
            
        # Resize if dimensions differ (unlikely in standardized dataset)
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
            
        out.write(frame)
        
    out.release()
    print(f"\nSuccess! Training test video saved as: {output_path}")

if __name__ == "__main__":
    create_dataset_video("training_test_video.mp4")
