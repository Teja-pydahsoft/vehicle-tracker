import cv2
import numpy as np
import random

def generate_test_video(output_path, duration=10, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Dummy vehicle colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    # List of fake plates to display
    fake_plates = ["HR26-AB-1234", "DL-3C-9876", "MH-12-GH-5555", "KA-01-MJ-1212", "UP-16-BD-0007"]
    
    # Vehicle state: [x, y, color, plate_text]
    vehicles = []
    
    for frame_idx in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add some background noise/road lines
        cv2.line(frame, (width//2 - 100, 0), (width//2 - 100, height), (255, 255, 255), 5)
        cv2.line(frame, (width//2 + 100, 0), (width//2 + 100, height), (255, 255, 255), 5)

        # Spawn new vehicle every few seconds
        if frame_idx % 60 == 0:
            direction = random.choice(["IN", "OUT"])
            y_start = -100 if direction == "IN" else height + 100
            speed = 10 if direction == "IN" else -10
            vehicles.append({
                'x': width // 2 - 50,
                'y': y_start,
                'speed': speed,
                'color': random.choice(colors),
                'plate': random.choice(fake_plates)
            })

        for v in vehicles:
            v['y'] += v['speed']
            # Draw vehicle body (Top-down view box)
            cv2.rectangle(frame, (v['x'], v['y']), (v['x'] + 100, v['y'] + 180), v['color'], -1)
            # Add a white "License Plate" area
            plate_y = v['y'] + 140 if v['speed'] > 0 else v['y'] + 20
            cv2.rectangle(frame, (v['x'] + 20, plate_y), (v['x'] + 80, plate_y + 30), (255, 255, 255), -1)
            # Write plate text (Very clear for OCR)
            cv2.putText(frame, v['plate'], (v['x'] + 25, plate_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        out.write(frame)
        
    out.release()
    print(f"Test video generated: {output_path}")

if __name__ == "__main__":
    generate_test_video("anpr_test_video.mp4")
