import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import mysql.connector
import numpy as np
from skimage.morphology import skeletonize
import io
import json
import math
import logging
import time
from contextlib import closing

# ---------------- Configuration ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",        # <-- Change
    "password": "db_password",  # <-- Change
    "database": "fingerprints_db"
}

# Tuning parameters
SPATIAL_TOLERANCE = 12   # pixels tolerance for matching minutiae
ANGLE_TOLERANCE = 25.0   # degrees tolerance for orientation match
MATCH_THRESHOLD = 0.55   # fraction threshold to declare a match

# Setup logging
logging.basicConfig(
    filename='fingerprint_app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ----------------- Database Setup -----------------
def connect_db():
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        return cnx
    except mysql.connector.Error as err:
        logging.exception("DB connection failed")
        messagebox.showerror("Database Error", f"Error connecting to DB:\n{err}")
        return None

def ensure_tables():
    cnx = connect_db()
    if cnx is None:
        return False
    try:
        with closing(cnx.cursor()) as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    minutiae JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        cnx.commit()
    except Exception as e:
        logging.exception("Failed to ensure tables")
        messagebox.showerror("DB Error", f"Failed to create tables:\n{e}")
        return False
    finally:
        cnx.close()
    return True

# ----------------- Preprocessing Helpers -----------------
def gabor_filter(img, ksize=21, sigma=5.0, theta=0, lambd=10.0, gamma=0.5):
    """Apply a single Gabor kernel and return filtered image."""
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return filtered

def enhance_ridges(gray):
    """Enhance ridge structure using histogram equalization + multi-orientation Gabor filtering."""
    # CLAHE equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)

    # Combine multiple Gabor responses (several orientations)
    accum = np.zeros_like(eq, dtype=np.float32)
    for theta in np.linspace(0, np.pi, 6, endpoint=False):
        resp = gabor_filter(eq, ksize=21, sigma=4.0, theta=theta, lambd=12.0, gamma=0.5)
        # Convert to float and accumulate the absolute response
        accum += np.abs(resp.astype(np.float32) - np.mean(resp))

    # Normalize accumulation to 0..255
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Smooth and threshold adaptively
    blurred = cv2.GaussianBlur(accum, (3,3), 0)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th

# ----------------- Minutiae Extraction -----------------
def compute_orientation_at(img_float, y, x, window=7):
    """Estimate ridge orientation (in degrees) at a pixel using gradients around a small window."""
    h, w = img_float.shape
    r = window // 2
    ys = max(0, y-r); ye = min(h, y+r+1)
    xs = max(0, x-r); xe = min(w, x+r+1)
    patch = img_float[ys:ye, xs:xe]
    if patch.size == 0:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    # compute average gradient direction
    gxx = np.sum(gx * gx)
    gyy = np.sum(gy * gy)
    gxy = np.sum(gx * gy)
    # orientation in radians: 0.5 * atan2(2*gxy, gxx - gyy)
    denom = (gxx - gyy)
    angle_rad = 0.5 * math.atan2(2.0 * gxy, denom + 1e-8)
    angle_deg = math.degrees(angle_rad)
    # normalize to 0..180
    angle_deg = (angle_deg + 180.0) % 180.0
    return float(angle_deg)

def extract_minutiae_with_orientation(processed_img):
    """
    processed_img: binary image (0 ridge, 255 background) expected.
    returns list of minutiae dicts: {'x':int,'y':int,'type':'ending'|'bifurcation','angle':float}
    """
    binary = (processed_img == 0)
    skeleton = skeletonize(binary).astype(np.uint8)  # 0/1
    rows, cols = skeleton.shape

    minutiae = []
    # Precompute float grayscale version for orientation estimate
    # we want intensity where ridges are dark, so invert processed_img to float
    img_float = processed_img.astype(np.float32)
    img_float = 255.0 - img_float  # ridges brighter in this float image

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                neighbors = [
                    skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                    skeleton[i, j-1],                   skeleton[i, j+1],
                    skeleton[i+1, j-1], skeleton[i+1, j], skeleton[i+1, j+1]
                ]
                neighbor_count = int(np.sum(neighbors))
                if neighbor_count == 1:
                    typ = 'ending'
                elif neighbor_count > 2:
                    typ = 'bifurcation'
                else:
                    continue
                angle = compute_orientation_at(img_float, i, j, window=9)
                minutiae.append({'x': int(j), 'y': int(i), 'type': typ, 'angle': angle})
    return minutiae, skeleton

def draw_minutiae_on_image(processed_img, minutiae):
    img_color = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    for m in minutiae:
        x, y = m['x'], m['y']
        if m['type'] == 'ending':
            cv2.circle(img_color, (x, y), 3, (0, 0, 255), -1)
        else:
            cv2.circle(img_color, (x, y), 3, (0, 255, 0), -1)
            # draw orientation line
            ang = math.radians(m['angle'])
            x2 = int(round(x + 8 * math.cos(ang)))
            y2 = int(round(y + 8 * math.sin(ang)))
            cv2.line(img_color, (x, y), (x2, y2), (255, 0, 0), 1)
    return img_color

# ----------------- Matching Logic -----------------
def match_minutiae_sets(set_in, set_db, spatial_tol=SPATIAL_TOLERANCE, angle_tol=ANGLE_TOLERANCE):
    """
    set_in and set_db: lists of dicts with x,y,type,angle.
    returns fractional match score (0..1)
    """
    matched = 0
    db_used = [False] * len(set_db)

    for mi in set_in:
        xi, yi, ti, ai = mi['x'], mi['y'], mi['type'], mi['angle']
        for idx, mj in enumerate(set_db):
            if db_used[idx]:
                continue
            xj, yj, tj, aj = mj['x'], mj['y'], mj['type'], mj['angle']
            if ti != tj:
                continue
            if abs(xi - xj) <= spatial_tol and abs(yi - yj) <= spatial_tol:
                # angle difference (circular)
                diff = abs(ai - aj) % 180.0
                diff = min(diff, 180.0 - diff)
                if diff <= angle_tol:
                    matched += 1
                    db_used[idx] = True
                    break
    total = max(len(set_in), len(set_db), 1)
    return matched / total

# ----------------- GUI App Class -----------------
class FingerprintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint System — Enroll & Verify (v2)")
        self.root.geometry("820x620")
        self.root.resizable(False, False)

        self.image_path = None
        self.tk_img = None  # keep reference to avoid GC
        self.tk_img_matched = None

        # Ensure DB tables exist
        if not ensure_tables():
            logging.warning("ensure_tables failed or DB unreachable")

        # ttk style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#e9e9e9")

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main, text="Fingerprint Enrollment & Verification (v2)", style="Header.TLabel", anchor="center")
        header.pack(fill=tk.X, pady=(0,8))

        # Input row
        row1 = ttk.Frame(main)
        row1.pack(fill=tk.X, pady=6)
        ttk.Label(row1, text="Student Name:").pack(side=tk.LEFT, padx=(0,8))
        self.name_entry = ttk.Entry(row1, width=42)
        self.name_entry.pack(side=tk.LEFT)

        # Top area: Left preview (selected / processed) and right preview (matched)
        preview_frame = ttk.Frame(main)
        preview_frame.pack(pady=8, fill=tk.X)

        left_frame = ttk.Frame(preview_frame)
        left_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(left_frame, text="Selected / Processed").pack()
        self.canvas_left = tk.Canvas(left_frame, width=360, height=300, bg="#dcdcdc", highlightthickness=0)
        self.canvas_left.pack()
        self.canvas_left.create_text(180,150, text="No Image Selected", fill="#333", font=("Segoe UI", 12))

        right_frame = ttk.Frame(preview_frame)
        right_frame.pack(side=tk.LEFT, padx=12)
        ttk.Label(right_frame, text="Matched (Verification Result)").pack()
        self.canvas_right = tk.Canvas(right_frame, width=360, height=300, bg="#dcdcdc", highlightthickness=0)
        self.canvas_right.pack()
        self.canvas_right.create_text(180,150, text="No Match Yet", fill="#333", font=("Segoe UI", 12))

        # Buttons and progress
        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=8)

        self.btn_select = ttk.Button(btn_frame, text="Select Image", command=self.select_image, width=20)
        self.btn_select.grid(row=0, column=0, padx=6, pady=4)

        self.btn_enroll = ttk.Button(btn_frame, text="Enroll Fingerprint", command=self.process_and_save, width=20)
        self.btn_enroll.grid(row=1, column=0, padx=6, pady=4)

        self.btn_verify = ttk.Button(btn_frame, text="Verify Fingerprint", command=self.verify_fingerprint, width=20)
        self.btn_verify.grid(row=2, column=0, padx=6, pady=4)

        self.btn_exit = ttk.Button(btn_frame, text="Exit", command=self.root.quit, width=20)
        self.btn_exit.grid(row=3, column=0, padx=6, pady=4)

        self.progress = ttk.Progressbar(main, orient=tk.HORIZONTAL, length=600, mode='determinate')
        self.progress.pack(pady=(10,0))

        # status
        self.status_label = ttk.Label(main, text="Ready", anchor="w")
        self.status_label.pack(fill=tk.X, pady=(6,0))

    # Utility to enable/disable main buttons during background ops
    def set_busy(self, busy=True):
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_select.config(state=state)
        self.btn_enroll.config(state=state)
        self.btn_verify.config(state=state)
        if busy:
            self.set_status("Processing...")
        else:
            self.set_status("Ready")

    # -------- Select fingerprint image --------
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image_on_canvas(file_path, self.canvas_left)
            self.set_status("Image Loaded")
            logging.info(f"Loaded image: {file_path}")

    # -------- Display image on canvas helper --------
    def display_image_on_canvas(self, path_or_pil, canvas):
        try:
            if isinstance(path_or_pil, str):
                img = Image.open(path_or_pil).convert("RGB")
            else:
                img = path_or_pil.convert("RGB")
            img.thumbnail((360, 300))
            tk_img = ImageTk.PhotoImage(img)
            # assign to appropriate attribute so it won't be GC'd
            if canvas is self.canvas_left:
                self.tk_img = tk_img
            else:
                self.tk_img_matched = tk_img
            canvas.delete("all")
            canvas.create_image(180, 150, image=tk_img)
        except Exception as e:
            logging.exception("display_image_on_canvas failed")
            messagebox.showerror("Image Error", f"Could not open image:\n{e}")
            canvas.delete("all")
            canvas.create_text(180, 150, text="No Image Selected", fill="#333", font=("Segoe UI", 12))

    # -------- Fingerprint preprocessing (improved) --------
    def process_fingerprint(self, img_path):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError("Unable to read image file")
        # Basic smoothing & crop/resize if too big
        h, w = img_gray.shape
        scale = 1.0
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_gray = cv2.resize(img_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Ridge enhancement pipeline
        processed = enhance_ridges(img_gray)

        return processed

    # -------- Save to DB (features) --------
    def save_to_db_features(self, name, minutiae_list):
        cnx = connect_db()
        if cnx is None:
            return False
        try:
            with closing(cnx.cursor()) as cursor:
                # Insert minutiae JSON
                j = json.dumps(minutiae_list)
                cursor.execute("INSERT INTO students (name, minutiae) VALUES (%s, %s)", (name, j))
            cnx.commit()
            logging.info(f"Saved features for {name} (points: {len(minutiae_list)})")
            return True
        except Exception as e:
            logging.exception("save_to_db_features failed")
            messagebox.showerror("DB Error", f"Error: {e}")
            return False
        finally:
            cnx.close()

    # -------- Enroll Fingerprint --------
    def process_and_save(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter student name")
            return
        if not self.image_path:
            messagebox.showwarning("Input Error", "Please select a fingerprint image")
            return

        # UI busy
        self.set_busy(True)
        self.progress['value'] = 5
        self.root.update_idletasks()
        try:
            processed = self.process_fingerprint(self.image_path)
            self.progress['value'] = 30
            self.root.update_idletasks()
        except Exception as e:
            logging.exception("process_fingerprint failed")
            messagebox.showerror("Processing Error", str(e))
            self.set_busy(False)
            self.progress['value'] = 0
            return

        minutiae, skeleton = extract_minutiae_with_orientation(processed)
        self.progress['value'] = 70
        self.root.update_idletasks()

        img_with_minutiae = draw_minutiae_on_image(processed, minutiae)

        # Show processed image with minutiae in GUI
        pil_img = Image.fromarray(cv2.cvtColor(img_with_minutiae, cv2.COLOR_BGR2RGB))
        self.display_image_on_canvas(pil_img, self.canvas_left)
        msg = f"Ridge Endings/Bifs: {sum(1 for m in minutiae if m['type']=='ending')}/{sum(1 for m in minutiae if m['type']=='bifurcation')}"
        self.set_status(f"Processed — {msg}")
        messagebox.showinfo("Minutiae Points", msg)

        # Save features (JSON) to DB
        saved = self.save_to_db_features(name, minutiae)
        if saved:
            self.set_status(f"Saved: {name} | {msg}")
            messagebox.showinfo("Success", f"Fingerprint features saved for '{name}' with {msg}")
            logging.info(f"Enrollment success for {name}")
        else:
            self.set_status("Failed to save to DB")
            logging.error("Failed to save features to DB")
        self.progress['value'] = 0
        self.set_busy(False)

    # -------- Verify Fingerprint --------
    def verify_fingerprint(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            return

        # busy UI
        self.set_busy(True)
        self.progress['value'] = 5
        self.root.update_idletasks()
        try:
            processed_input = self.process_fingerprint(file_path)
            self.progress['value'] = 30
            self.root.update_idletasks()
        except Exception as e:
            logging.exception("process_fingerprint failed during verify")
            messagebox.showerror("Processing Error", f"Could not process input image:\n{e}")
            self.set_busy(False)
            self.progress['value'] = 0
            return

        minutiae_input, _ = extract_minutiae_with_orientation(processed_input)
        self.progress['value'] = 50
        self.root.update_idletasks()

        # Display the processed input on left canvas
        img_left = draw_minutiae_on_image(processed_input, minutiae_input)
        pil_left = Image.fromarray(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
        self.display_image_on_canvas(pil_left, self.canvas_left)

        # query DB for all stored features
        cnx = connect_db()
        if cnx is None:
            self.set_busy(False)
            self.progress['value'] = 0
            return

        best_match = None
        best_score = 0.0
        try:
            with closing(cnx.cursor()) as cursor:
                cursor.execute("SELECT name, minutiae FROM students")
                rows = cursor.fetchall()
        except Exception as e:
            logging.exception("DB fetch failed")
            messagebox.showerror("DB Error", f"Error fetching records:\n{e}")
            cnx.close()
            self.set_busy(False)
            self.progress['value'] = 0
            return
        finally:
            cnx.close()

        self.progress['value'] = 60
        self.root.update_idletasks()

        for idx, (name, minutiae_json) in enumerate(rows):
            try:
                # Parse JSON to list of dicts
                db_minutiae = json.loads(minutiae_json)
            except Exception:
                logging.exception("Failed to decode minutiae JSON")
                continue

            score = match_minutiae_sets(minutiae_input, db_minutiae)
            logging.info(f"Comparing with {name} -> score: {score:.3f}")
            if score > best_score:
                best_score = score
                best_match = (name, db_minutiae)
            # update progress nicely
            self.progress['value'] = 60 + int(30 * (idx+1) / max(len(rows),1))
            self.root.update_idletasks()

        # Decision
        if best_match and best_score >= MATCH_THRESHOLD:
            name, db_minutiae = best_match
            # Reconstruct a small visualization of matched template (approx)
            # For demonstration: render a blank image and draw db_minutiae
            h, w = processed_input.shape
            vis = np.full((h, w), 255, dtype=np.uint8)
            vis_img = draw_minutiae_on_image(vis, db_minutiae)
            pil_right = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            self.display_image_on_canvas(pil_right, self.canvas_right)
            msg = f"Match Found: {name} (Score: {best_score:.2f})"
            self.set_status(msg)
            messagebox.showinfo("Verification Result", msg)
            logging.info(f"Verification matched {name} score={best_score:.3f}")
        else:
            self.canvas_right.delete("all")
            self.canvas_right.create_text(180,150, text="No Match Found", fill="#900", font=("Segoe UI", 12))
            self.set_status("No Match Found")
            messagebox.showwarning("Verification Failed", "No matching fingerprint found.")
            logging.info("Verification failed - no match")

        self.progress['value'] = 0
        self.set_busy(False)

    # Utility
    def set_status(self, text):
        self.status_label.config(text=text)


# ----------------- Run App -----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()

