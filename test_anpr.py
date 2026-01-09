import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import torch
import easyocr
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_CODES = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CG": "Chhattisgarh",
    "DN": "Dadra & Nagar Haveli", "DD": "Daman & Diu", "DL": "Delhi",
    "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh",
    "JK": "Jammu & Kashmir", "JH": "Jharkhand", "KA": "Karnataka",
    "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MP": "Madhya Pradesh",
    "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya", "MZ": "Mizoram",
    "NL": "Nagaland", "OD": "Odisha", "PY": "Puducherry", "PB": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TS": "Telangana",
    "TR": "Tripura", "UP": "Uttar Pradesh", "UK": "Uttarakhand", "WB": "West Bengal"
}

IGNORE_WORDS = [
    "YELLOW", "NUMBER", "PLATE", "BLACK", "WHITE", "IND", "CITY", "ZX", "HONDA", "BMW", "TOYOTA", 
    "NEWS", "V6", "LIVE", "TV", "HD", "MOVIES", "INDIA", "PLANET", "FORD", "MARUTI", "SUZUKI", 
    "HYUNDAI", "TEAM-BHP.COM", "WWW", "SHOP", "FORC", "FORO", "PLNET", "PLANT", "MOTORS", "CARS"
]

class StandaloneANPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Gate - Standalone ANPR Test Tool")
        self.root.geometry("1100x800")
        
        # App State
        self.model = None
        self.reader = None
        self.current_image = None
        self.processed_image = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.create_widgets()
        
        # Load models in background
        threading.Thread(target=self.init_models, daemon=True).start()

    def init_models(self):
        self.update_status("Initializing Models... Please wait.")
        
        # 1. Load YOLO
        try:
            model_path = 'models/custom_model.pt' if os.path.exists('models/custom_model.pt') else 'yolov8n.pt'
            logger.info(f"Loading YOLO from {model_path} on {self.device}")
            self.model = YOLO(model_path)
            if self.device == 'cuda':
                try:
                    self.model.to('cuda')
                except Exception as e:
                    logger.warning(f"Failed to move YOLO to CUDA: {e}. Falling back to CPU.")
                    self.device = 'cpu'
                    self.model.to('cpu')
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load YOLO model: {e}"))

        # 2. Load EasyOCR
        try:
            logger.info(f"Initializing EasyOCR on {self.device}")
            self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        except Exception as e:
            logger.warning(f"Failed to load EasyOCR on {self.device}: {e}. Retrying on CPU.")
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e2:
                logger.error(f"Failed to load EasyOCR even on CPU: {e2}")

        self.update_status(f"Ready. Device: {self.device.upper()}")
        self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))

    def create_widgets(self):
        # Layout
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top Controls
        ctrl_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        self.load_btn = ttk.Button(ctrl_frame, text="Select Image", command=self.load_image, state=tk.DISABLED)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(ctrl_frame, text="Detect Vehicle & Plate", command=self.run_detection, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.cpu_var = tk.BooleanVar(value=(self.device == 'cpu'))
        self.cpu_check = ttk.Checkbutton(ctrl_frame, text="Force CPU Mode (Safer)", variable=self.cpu_var, command=self.toggle_device)
        self.cpu_check.pack(side=tk.RIGHT, padx=5)

        # Middle Content
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Image Display
        self.canvas_frame = ttk.LabelFrame(content_frame, text="Visualization", padding=5)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right: Results
        res_frame = ttk.LabelFrame(content_frame, text="Results", padding=10, width=400)
        res_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        res_frame.pack_propagate(False)
        
        self.res_tree = ttk.Treeview(res_frame, columns=("Type", "Plate", "State", "Conf"), show='headings')
        self.res_tree.heading("Type", text="Vehicle")
        self.res_tree.heading("Plate", text="Plate Number")
        self.res_tree.heading("State", text="State")
        self.res_tree.heading("Conf", text="Conf %")
        self.res_tree.column("Type", width=70)
        self.res_tree.column("Plate", width=110)
        self.res_tree.column("State", width=120)
        self.res_tree.column("Conf", width=60)
        self.res_tree.pack(fill=tk.BOTH, expand=True)

        # Status
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def toggle_device(self):
        new_device = 'cpu' if self.cpu_var.get() else ('cuda' if torch.cuda.is_available() else 'cpu')
        if new_device != self.device:
            self.device = new_device
            threading.Thread(target=self.init_models, daemon=True).start()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.current_image = img
                self.display_image(img)
                self.detect_btn.config(state=tk.NORMAL)
                self.res_tree.delete(*self.res_tree.get_children())
                self.update_status(f"Loaded: {os.path.basename(path)}")
            else:
                messagebox.showerror("Error", "Could not load image.")

    def display_image(self, img):
        # Resize for canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        
        h, w = img.shape[:2]
        ratio = min(cw/w, ch/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        
        resized = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=photo)
        self.canvas.image = photo 

    def get_plate(self, frame, box=None):
        if self.reader is None: return []
        try:
            if box is not None:
                # Local Scan within box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_img, w_img = frame.shape[:2]
                box_h = y2 - y1
                box_w = x2 - x1
                
                # Precise box expansion (20% for license plate focus only)
                pad_y = int(box_h * 0.20)
                pad_x = int(box_w * 0.20)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w_img, x2 + pad_x)
                y2 = min(h_img, y2 + pad_y)
                roi = frame[y1:y2, x1:x2]
            else:
                # Global Scan (Entire Image)
                roi = frame

            if roi.size == 0: return []
            
            # Preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR (detail=1 returns bbox, text, prob)
            # OCR (detail=1 returns bbox, text, prob)
            results = self.reader.readtext(enhanced, detail=1, paragraph=False) 
            
            import re
            digit_pattern = re.compile(r'\d.*\d.*\d') 
            # Fuzzy state search: look for these anywhere in the first 4 chars
            STATE_KEYS = list(STATE_CODES.keys())
            
            # 1. First pass: Aggressive Cleaning & Basic Filtering
            raw_candidates = []
            for (bbox, text, prob) in results:
                if prob < 0.2: continue
                clean_text = text.upper().replace(" ", "")
                for char in "/|()_.-,;:'\"":
                    clean_text = clean_text.replace(char, "")
                for word in IGNORE_WORDS:
                    clean_text = clean_text.replace(word, "")
                
                if len(clean_text) >= 2:
                    raw_candidates.append({'bbox': bbox, 'text': clean_text, 'prob': prob})
            
            # 2. Stage A: Horizontal Merging (Glue neighbors on the same line)
            # PRECISE: Tight dx tolerance
            raw_candidates.sort(key=lambda x: x['bbox'][0][0])
            horiz_merged = []
            used_h = set()
            
            for i, c1 in enumerate(raw_candidates):
                if i in used_h: continue
                curr_box = c1['bbox']
                curr_text = c1['text']
                curr_prob = c1['prob']
                
                while True:
                    found_next = False
                    c1_x2 = max(pt[0] for pt in curr_box)
                    c1_y_mid = (min(pt[1] for pt in curr_box) + max(pt[1] for pt in curr_box)) / 2
                    c1_h = max(pt[1] for pt in curr_box) - min(pt[1] for pt in curr_box)
                    
                    for j, c2 in enumerate(raw_candidates):
                        if i == j or j in used_h: continue
                        c2_x1 = min(pt[0] for pt in c2['bbox'])
                        c2_y_mid = (min(pt[1] for pt in c2['bbox']) + max(pt[1] for pt in c2['bbox'])) / 2
                        
                        dx = c2_x1 - c1_x2
                        dy = abs(c2_y_mid - c1_y_mid)
                        
                        # Precise dx tolerance to 1.2x height (tight)
                        if -c1_h < dx < c1_h * 1.2 and dy < c1_h * 0.3:
                            curr_text += c2['text']
                            all_pts = curr_box + c2['bbox']
                            curr_box = [
                                [min(p[0] for p in all_pts), min(p[1] for p in all_pts)],
                                [max(p[0] for p in all_pts), min(p[1] for p in all_pts)],
                                [max(p[0] for p in all_pts), max(p[1] for p in all_pts)],
                                [min(p[0] for p in all_pts), max(p[1] for p in all_pts)]
                            ]
                            curr_prob = (curr_prob + c2['prob']) / 2
                            used_h.add(j)
                            found_next = True
                            break
                    if not found_next: break
                
                horiz_merged.append({'bbox': curr_box, 'text': curr_text, 'prob': curr_prob})
                used_h.add(i)

            # 3. Stage B: Vertical Grouping (Glue stacked snippets)
            # PRECISE: Tight v_gap tolerance
            horiz_merged.sort(key=lambda x: x['bbox'][0][1])
            grouped_plates = []
            used_v = set()
            
            for i, cand1 in enumerate(horiz_merged):
                if i in used_v: continue
                box1 = cand1['bbox']
                y1_bottom = max(pt[1] for pt in box1)
                x1_center = (min(pt[0] for pt in box1) + max(pt[0] for pt in box1)) / 2
                box1_h = max(pt[1] for pt in box1) - min(pt[1] for pt in box1)
                
                found_v = False
                for j, cand2 in enumerate(horiz_merged):
                    if i == j or j in used_v: continue
                    box2 = cand2['bbox']
                    y2_top = min(pt[1] for pt in box2)
                    x2_center = (min(pt[0] for pt in box2) + max(pt[0] for pt in box2)) / 2
                    
                    v_gap = y2_top - y1_bottom
                    h_gap = abs(x2_center - x1_center)
                    
                    # Precise v_gap tolerance to 0.6x height (prevents catching stickers)
                    if -box1_h * 0.2 < v_gap < box1_h * 0.6 and h_gap < (max(pt[0] for pt in box1)-min(pt[0] for pt in box1)) * 0.3:
                        combined_text = cand1['text'] + cand2['text']
                        all_pts = cand1['bbox'] + cand2['bbox']
                        combined_bbox = [
                            [min(p[0] for p in all_pts), min(p[1] for p in all_pts)],
                            [max(p[0] for p in all_pts), min(p[1] for p in all_pts)],
                            [max(p[0] for p in all_pts), max(p[1] for p in all_pts)],
                            [min(p[0] for p in all_pts), max(p[1] for p in all_pts)]
                        ]
                        grouped_plates.append({'text': combined_text, 'bbox': combined_bbox, 'prob': (cand1['prob']+cand2['prob'])/2})
                        used_v.add(i); used_v.add(j)
                        found_v = True
                        break
                if not found_v:
                    grouped_plates.append(cand1)
                    used_v.add(i)

            # 4. Validation & Fuzzy State Search
            final_found = []
            for item in grouped_plates:
                text = item['text']
                # CLEAN AGAIN: Remove noise that might have been glued
                for word in IGNORE_WORDS:
                    text = text.replace(word, "")
                
                # LIMIT LENGTH: 12 is the sweet spot for Indian Plates
                if len(text) > 12: continue
                
                if not digit_pattern.search(text): continue
                if len(text) < 4: continue
                
                state_name = "International"
                # Check first 5 chars for state code
                prefix_to_check = text[:5]
                for code in STATE_KEYS:
                    if code in prefix_to_check:
                        state_name = STATE_CODES[code]
                        break
                
                offset_x, offset_y = 0, 0
                if box is not None:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    offset_x = max(0, bx1 - int((bx2-bx1)*0.20))
                    offset_y = max(0, by1 - int((by2-by1)*0.20))
                
                abs_bbox = []
                for pt in item['bbox']:
                    abs_bbox.append([int(pt[0] + offset_x), int(pt[1] + offset_y)])
                
                final_found.append((text, state_name, item['prob'], abs_bbox))
            
            return final_found
            
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return []

    def calculate_iou(self, boxA, boxB):
        # Convert bbox [[x,y]...] to [x1,y1,x2,y2]
        xA1 = min(pt[0] for pt in boxA); yA1 = min(pt[1] for pt in boxA)
        xA2 = max(pt[0] for pt in boxA); yA2 = max(pt[1] for pt in boxA)
        xB1 = min(pt[0] for pt in boxB); yB1 = min(pt[1] for pt in boxB)
        xB2 = max(pt[0] for pt in boxB); yB2 = max(pt[1] for pt in boxB)
        inter = max(0, min(xA2, xB2) - max(xA1, xB1)) * max(0, min(yA2, yB2) - max(yA1, yB1))
        areaA = (xA2-xA1)*(yA2-yA1); areaB = (xB2-xB1)*(yB2-yB1)
        return inter / float(areaA + areaB - inter + 1e-6)

    def run_detection(self):
        if self.current_image is None or self.model is None: return
        self.detect_btn.config(state=tk.DISABLED)
        self.update_status("Processing... scanning for plates and sorting...")
        
        def task():
            try:
                img_copy = self.current_image.copy()
                results = self.model(img_copy, device=self.device, verbose=False)[0]
                candidates = []
                
                # 1. Vehicle Scans
                for box in results.boxes:
                    conf = float(box.conf[0].item())
                    if conf < 0.25: continue
                    label = self.model.names[int(box.cls[0].item())]
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                    cv2.rectangle(img_copy, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                    
                    plates = self.get_plate(self.current_image, box)
                    for p_text, p_state, p_prob, p_bbox in plates:
                        candidates.append({'label': label, 'text': p_text, 'state': p_state, 
                                          'conf': f"{conf*100:.1f}%", 'bbox': p_bbox, 'prob': p_prob})

                # 2. Global Scan
                global_plates = self.get_plate(self.current_image, None)
                for p_text, p_state, p_prob, p_bbox in global_plates:
                    candidates.append({'label': "GLOBAL SCAN", 'text': p_text, 'state': p_state, 
                                      'conf': f"{p_prob*100:.1f}%", 'bbox': p_bbox, 'prob': p_prob})

                # 3. Deduplication & Sorting
                candidates.sort(key=lambda x: x['prob'], reverse=True)
                deduped = []
                for cand in candidates:
                    if any(cand['text'] == d['text'] for d in deduped): continue
                    if any(self.calculate_iou(cand['bbox'], d['bbox']) > 0.4 for d in deduped): continue
                    deduped.append(cand)

                # Spatial Sorting: Top-to-Bottom, then Left-to-Right
                # Use a tolerance for the same "row" (e.g., 60px)
                def spatial_sort(item):
                    y_top = min(pt[1] for pt in item['bbox'])
                    x_left = min(pt[0] for pt in item['bbox'])
                    row = y_top // 60 # Group into rows of 60px
                    return (row, x_left)
                
                deduped.sort(key=spatial_sort)

                # 4. Final Output
                y_off = 35
                overlay = img_copy.copy()
                cv2.rectangle(overlay, (10, 10), (380, 20 + len(deduped)*30), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
                
                ui_data = []
                for i, res in enumerate(deduped):
                    # Label with Index on Image
                    pts = np.array(res['bbox'], np.int32)
                    cv2.polylines(img_copy, [pts], True, (0, 255, 255), 2)
                    cv2.putText(img_copy, f"#{i+1}", (res['bbox'][0][0], res['bbox'][0][1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Overlay Summary: Include Vehicle Type (res['label'])
                    vehicle_type = res['label'].capitalize() if res['label'] != "GLOBAL SCAN" else "Plate"
                    summary_text = f"#{i+1} {vehicle_type}: {res['text']} ({res['state']})"
                    
                    cv2.putText(img_copy, summary_text, (20, y_off), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_off += 28
                    ui_data.append((res['label'], res['text'], res['state'], res['conf']))

                self.root.after(0, lambda: self.finish_detection(img_copy, ui_data))
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                self.update_status(f"Error: {str(e)}")
                self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))

        threading.Thread(target=task, daemon=True).start()

    def finish_detection(self, img, results):
        self.display_image(img)
        self.res_tree.delete(*self.res_tree.get_children())
        for r in results:
            self.res_tree.insert("", tk.END, values=r)
        
        self.update_status(f"Processed {len(results)} objects.")
        self.detect_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = StandaloneANPRApp(root)
    root.mainloop()
