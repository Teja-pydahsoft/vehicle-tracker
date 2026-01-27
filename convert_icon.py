
from PIL import Image
import os
import glob

def convert_png_to_ico():
    # Find the most recent generated icon
    png_files = glob.glob(os.path.join(os.getcwd(), "*.png"))
    # Or just use the one I know I generated if I can find it in the current dir.
    # Since generate_image saves to a specific path, I'll use the path from the tool response if possible.
    # Actually, I'll just look for 'ai_smart_vehicle_icon' in common dirs or ask the user.
    # Better: I'll assume it's in the current dir if it was saved as an artifact that I can access?
    # No, it's in the gemini dir.
    
    # I will look for the image in the path provided in the previous turn.
    src = r"C:/Users/Ashok Kumar/.gemini/antigravity/brain/049ae141-f81a-4583-b6a0-ad986a8ce8c7/ai_smart_vehicle_icon_1769415601774.png"
    dest = r"c:\Users\Ashok Kumar\Desktop\Vehical Detection\app_icon.ico"
    
    if os.path.exists(src):
        img = Image.open(src)
        # Create different sizes for the ICO
        icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(dest, format='ICO', sizes=icon_sizes)
        print(f"Icon saved to {dest}")
    else:
        print(f"Source image not found at {src}")

if __name__ == "__main__":
    convert_png_to_ico()
