"""
Fingerprint verfication app with:
 - improved preprocessing & minutiae extraction
 - RANSAC-based alignment for rotation/translation invariance
 - Export / Import DB functionality (JSON + base64 encoded images)
 - Simple admin login (username/password stored hashed in DB) 
 -Username- admin  Password- admin123
 
 Update DB_CONFIG below before running.
"""

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
import base64
import hashlib
import time
from contextlib import closing

# ---------------- Configuration ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",        # <-- Change
    "password": "db_password",  # <-- Change
    "database": "fingerprints_db"
}

# Admin default credentials (created if users table empty)
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"

# Matching / RANSAC parameters
SPATIAL_TOLERANCE = 20   # initial candidate distance to consider pairing (px)
ANGLE_TOLERANCE = 30.0   # degrees allowed difference for candidate pair filtering
RANSAC_REPROJ_THRESH = 6.0
MATCH_THRESHOLD = 0.5    # fraction to accept a match (inliers / max points)

# Logging
logging.basicConfig(filename='fingerprint_app.log', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# ----------------- Database Helpers -----------------
def connect_db():
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        return cnx
    except mysql.connector.Error as err:
        logging.exception("DB connect failed")
        messagebox.showerror("Database Error", f"Error connecting to DB:\n{err}")
        return None

def ensure_schema():
    """Create students and users tables if missing; ensure 'name' unique."""
    cnx = connect_db()
    if cnx is None:
        return False
    try:
        with closing(cnx.cursor()) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    fingerprint LONGBLOB NULL,
                    minutiae LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_name (name)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) NOT NULL UNIQUE,
                    password_hash VARCHAR(256) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Ensure a default admin user exists (only insert if not present)
            cur.execute("SELECT COUNT(*) FROM users")
            (count,) = cur.fetchone()
            if count == 0:
                ph = hash_password(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASS)
                cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                            (DEFAULT_ADMIN_USER, ph))
        cnx.commit()
    except Exception as e:
        logging.exception("Schema ensure failed")
        messagebox.showerror("DB Error", f"Failed to create/ensure tables:\n{e}")
        return False
    finally:
        cnx.close()
    return True

# ----------------- Password Hashing -----------------
def hash_password(username, password):
    """Hash password using SHA-256 with username as salt (simple approach)."""
    salt = username.encode('utf-8')
    return hashlib.sha256(salt + password.encode('utf-8')).hexdigest()

def verify_password(username, password, stored_hash):
    return hash_password(username, password) == stored_hash

# ----------------- Image / Preprocessing Helpers -----------------
def gabor_filter(img, ksize=21, sigma=5.0, theta=0, lambd=10.0, gamma=0.5):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    return filtered

def enhance_ridges(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    accum = np.zeros_like(eq, dtype=np.float32)
    for theta in np.linspace(0, np.pi, 6, endpoint=False):
        resp = gabor_filter(eq, ksize=21, sigma=4.0, theta=theta, lambd=12.0, gamma=0.5)
        accum += np.abs(resp.astype(np.float32) - np.mean(resp))
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(accum, (3,3), 0)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th

def compute_orientation_at(img_float, y, x, window=7):
    h, w = img_float.shape
    r = window // 2
    ys = max(0, y-r); ye = min(h, y+r+1)
    xs = max(0, x-r); xe = min(w, x+r+1)
    patch = img_float[ys:ye, xs:xe]
    if patch.size == 0:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    gxx = np.sum(gx * gx)
    gyy = np.sum(gy * gy)
    gxy = np.sum(gx * gy)
    denom = (gxx - gyy)
    angle_rad = 0.5 * math.atan2(2.0 * gxy, denom + 1e-8)
    angle_deg = math.degrees(angle_rad)
    angle_deg = (angle_deg + 180.0) % 180.0
    return float(angle_deg)

def extract_minutiae_with_orientation(processed_img):
    binary = (processed_img == 0)
    skeleton = skeletonize(binary).astype(np.uint8)
    rows, cols = skeleton.shape
    minutiae = []
    img_float = processed_img.astype(np.float32)
    img_float = 255.0 - img_float
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
        ang = math.radians(m['angle'])
        x2 = int(round(x + 8 * math.cos(ang)))
        y2 = int(round(y + 8 * math.sin(ang)))
        cv2.line(img_color, (x, y), (x2, y2), (255, 0, 0), 1)
    return img_color

# ----------------- RANSAC-based Matching -----------------
def build_candidate_pairs(min_in, min_db, max_dist=SPATIAL_TOLERANCE, angle_tol=ANGLE_TOLERANCE):
    """Return list of (src_pt, dst_pt) pairs where types match and within max_dist and angle_tol."""
    pairs = []
    for a in min_in:
        for b in min_db:
            if a['type'] != b['type']:
                continue
            dx = a['x'] - b['x']
            dy = a['y'] - b['y']
            d = math.hypot(dx, dy)
            if d <= max_dist:
                # angle diff circular
                diff = abs(a['angle'] - b['angle']) % 180.0
                diff = min(diff, 180.0 - diff)
                if diff <= angle_tol:
                    pairs.append(((a['x'], a['y']), (b['x'], b['y'])))
    return pairs

def ransac_match(min_in, min_db):
    """
    Attempt RANSAC alignment. Returns (score, inlier_count, transform_matrix or None)
    score = inliers / max(len(min_in), len(min_db))
    """
    # Build candidate pairs with a larger initial tolerance to allow RANSAC to refine
    candidates = build_candidate_pairs(min_in, min_db, max_dist=SPATIAL_TOLERANCE * 2, angle_tol=ANGLE_TOLERANCE * 1.2)
    if len(candidates) < 3:
        # Not enough correspondences for affine estimation
        return 0.0, 0, None

    src_pts = np.array([p[0] for p in candidates], dtype=np.float32)
    dst_pts = np.array([p[1] for p in candidates], dtype=np.float32)

    # Use OpenCV estimateAffinePartial2D with RANSAC
    try:
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                 ransacReprojThreshold=RANSAC_REPROJ_THRESH,
                                                 maxIters=2000, confidence=0.99)
    except Exception as e:
        logging.exception("estimateAffinePartial2D failed")
        return 0.0, 0, None

    if M is None or inliers is None:
        return 0.0, 0, None

    inlier_count = int(np.sum(inliers))
    score = inlier_count / max(len(min_in), len(min_db), 1)
    return float(score), int(inlier_count), M

# ----------------- GUI App -----------------
class FingerprintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Verification System")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        self.image_path = None
        self.tk_img = None
        self.tk_img_matched = None

        # admin state
        self.admin_user = None  # username when logged in
        self.admin_authed = False

        if not ensure_schema():
            logging.warning("DB schema ensure failed or DB unreachable")

        # ttk style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main, text="Fingerprint Verification System", style="Header.TLabel")
        header.pack(fill=tk.X, pady=(0, 8))

        # row: name + buttons
        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=6)
        ttk.Label(controls, text="Student Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(controls, width=40)
        self.name_entry.pack(side=tk.LEFT, padx=(6, 12))

        self.btn_select = ttk.Button(controls, text="Select Image", command=self.select_image)
        self.btn_select.pack(side=tk.LEFT, padx=4)
        self.btn_enroll = ttk.Button(controls, text="Enroll", command=self.process_and_save)
        self.btn_enroll.pack(side=tk.LEFT, padx=4)
        self.btn_verify = ttk.Button(controls, text="Verify", command=self.verify_fingerprint)
        self.btn_verify.pack(side=tk.LEFT, padx=4)

        # admin controls
        admin_frame = ttk.Frame(main)
        admin_frame.pack(fill=tk.X, pady=(8, 6))
        self.lbl_admin = ttk.Label(admin_frame, text="Not logged in")
        self.lbl_admin.pack(side=tk.LEFT)
        ttk.Button(admin_frame, text="Admin Login", command=self.open_admin_login).pack(side=tk.LEFT, padx=6)
        ttk.Button(admin_frame, text="Export DB", command=self.export_db).pack(side=tk.LEFT, padx=4)
        ttk.Button(admin_frame, text="Import DB", command=self.import_db).pack(side=tk.LEFT, padx=4)

        # preview area
        preview_frame = ttk.Frame(main)
        preview_frame.pack(pady=8, fill=tk.X)

        left = ttk.Frame(preview_frame)
        left.pack(side=tk.LEFT, padx=6)
        ttk.Label(left, text="Selected / Processed").pack()
        self.canvas_left = tk.Canvas(left, width=420, height=360, bg="#dcdcdc")
        self.canvas_left.pack()
        self.canvas_left.create_text(210, 180, text="No Image Selected", fill="#333")

        right = ttk.Frame(preview_frame)
        right.pack(side=tk.LEFT, padx=12)
        ttk.Label(right, text="Matched (Verification Result)").pack()
        self.canvas_right = tk.Canvas(right, width=420, height=360, bg="#dcdcdc")
        self.canvas_right.pack()
        self.canvas_right.create_text(210, 180, text="No Match Yet", fill="#333")

        # status + progress
        self.progress = ttk.Progressbar(main, orient=tk.HORIZONTAL, length=820, mode='determinate')
        self.progress.pack(pady=(10,0))
        self.status_label = ttk.Label(main, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(6,0))

    # ---------- UI helpers ----------
    def set_status(self, text):
        self.status_label.config(text=text)

    def set_busy(self, busy=True):
        state = tk.DISABLED if busy else tk.NORMAL
        for b in (self.btn_select, self.btn_enroll, self.btn_verify):
            b.config(state=state)
        if busy:
            self.set_status("Processing...")
        else:
            self.set_status("Ready")

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files","*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        self.image_path = file_path
        self.display_image_on_canvas(file_path, self.canvas_left)
        self.set_status("Image loaded")

    def display_image_on_canvas(self, path_or_pil, canvas):
        try:
            if isinstance(path_or_pil, str):
                img = Image.open(path_or_pil).convert("RGB")
            else:
                img = path_or_pil.convert("RGB")
            img.thumbnail((420, 360))
            tk_img = ImageTk.PhotoImage(img)
            if canvas is self.canvas_left:
                self.tk_img = tk_img
            else:
                self.tk_img_matched = tk_img
            canvas.delete("all")
            canvas.create_image(210, 180, image=tk_img)
        except Exception as e:
            logging.exception("display image failed")
            messagebox.showerror("Image Error", str(e))

    # ---------- Processing & Enrollment ----------
    def process_fingerprint(self, img_path):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError("Unable to read image file")
        h, w = img_gray.shape
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_gray = cv2.resize(img_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        processed = enhance_ridges(img_gray)
        return processed

    def save_to_db_features(self, name, minutiae_list, fingerprint_bytes=None):
        cnx = connect_db()
        if cnx is None:
            return False
        try:
            with closing(cnx.cursor()) as cur:
                # upsert by name
                minutiae_json = json.dumps(minutiae_list)
                if fingerprint_bytes is None:
                    cur.execute("""
                        INSERT INTO students (name, minutiae, fingerprint)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE minutiae = VALUES(minutiae), fingerprint = COALESCE(VALUES(fingerprint), fingerprint)
                    """, (name, minutiae_json, None))
                else:
                    cur.execute("""
                        INSERT INTO students (name, minutiae, fingerprint)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE minutiae = VALUES(minutiae), fingerprint = VALUES(fingerprint)
                    """, (name, minutiae_json, fingerprint_bytes))
            cnx.commit()
            logging.info(f"Saved/updated student {name}")
            return True
        except Exception as e:
            logging.exception("save_to_db_features failed")
            messagebox.showerror("DB Error", f"Error saving to DB: {e}")
            return False
        finally:
            cnx.close()

    def process_and_save(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Enter student name")
            return
        if not self.image_path:
            messagebox.showwarning("Input Error", "Select fingerprint image")
            return

        self.set_busy(True)
        self.progress['value'] = 5
        self.root_update()

        try:
            processed = self.process_fingerprint(self.image_path)
            self.progress['value'] = 30
            self.root_update()
        except Exception as e:
            logging.exception("processing failed")
            messagebox.showerror("Processing Error", str(e))
            self.set_busy(False)
            self.progress['value'] = 0
            return

        minutiae, skeleton = extract_minutiae_with_orientation(processed)
        self.progress['value'] = 70
        self.root_update()

        # show visualization
        vis = draw_minutiae_on_image(processed, minutiae)
        pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        self.display_image_on_canvas(pil, self.canvas_left)

        # encode fingerprint image bytes (optional)
        success, enc = cv2.imencode('.png', processed)
        fingerprint_bytes = enc.tobytes() if success else None

        saved = self.save_to_db_features(name, minutiae, fingerprint_bytes=fingerprint_bytes)
        if saved:
            messagebox.showinfo("Saved", f"Enrollment saved for {name} (points: {len(minutiae)})")
            logging.info(f"Enrollment: {name} saved")
            self.set_status(f"Saved {name}")
        else:
            self.set_status("Save failed")
        self.progress['value'] = 0
        self.set_busy(False)

    # ---------- Verification with RANSAC ----------
    def verify_fingerprint(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files","*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        self.set_busy(True)
        self.progress['value'] = 5
        self.root_update()

        try:
            processed_input = self.process_fingerprint(file_path)
            self.progress['value'] = 20
            self.root_update()
        except Exception as e:
            logging.exception("processing input failed")
            messagebox.showerror("Processing Error", str(e))
            self.set_busy(False)
            self.progress['value'] = 0
            return

        minutiae_input, _ = extract_minutiae_with_orientation(processed_input)
        vis_input = draw_minutiae_on_image(processed_input, minutiae_input)
        pil_left = Image.fromarray(cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB))
        self.display_image_on_canvas(pil_left, self.canvas_left)

        # fetch DB entries
        cnx = connect_db()
        if cnx is None:
            self.set_busy(False)
            return
        try:
            with closing(cnx.cursor()) as cur:
                cur.execute("SELECT name, minutiae, fingerprint FROM students")
                rows = cur.fetchall()
        except Exception as e:
            logging.exception("fetch failed")
            messagebox.showerror("DB Error", f"Error fetching DB: {e}")
            cnx.close()
            self.set_busy(False)
            return
        finally:
            cnx.close()

        best_score = 0.0
        best_match = None

        for idx, (name, minutiae_json, fingerprint_blob) in enumerate(rows):
            try:
                db_minutiae = json.loads(minutiae_json) if minutiae_json else []
            except Exception:
                logging.exception("bad minutiae JSON")
                db_minutiae = []

            # First, try RANSAC matching
            score, inliers, M = ransac_match(minutiae_input, db_minutiae)
            logging.info(f"Compared to {name}: ransac score={score:.3f}, inliers={inliers}")
            # If RANSAC fails or score low, fallback to simple pairwise matching
            if score < MATCH_THRESHOLD:
                # fallback: pairwise proportion matching (without alignment)
                score_fallback = simple_pairwise_score(minutiae_input, db_minutiae)
                logging.info(f"  fallback score={score_fallback:.3f}")
                if score_fallback > score:
                    score = score_fallback

            if score > best_score:
                best_score = score
                best_match = (name, db_minutiae, fingerprint_blob, M)

            # update progress visually
            self.progress['value'] = 20 + int(70 * (idx+1)/max(len(rows),1))
            self.root_update()

        # Decision & visualization
        if best_match and best_score >= MATCH_THRESHOLD:
            name, db_minutiae, fingerprint_blob, M = best_match
            # Visualize db_minutiae on right canvas, and if fingerprint blob available show it
            if fingerprint_blob:
                try:
                    nparr = np.frombuffer(fingerprint_blob, np.uint8)
                    db_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if db_img is not None:
                        vis_db = draw_minutiae_on_image(db_img, db_minutiae)
                        pil_right = Image.fromarray(cv2.cvtColor(vis_db, cv2.COLOR_BGR2RGB))
                        self.display_image_on_canvas(pil_right, self.canvas_right)
                    else:
                        # fallback: draw on blank canvas
                        self.draw_minutiae_on_blank(db_minutiae)
                except Exception:
                    logging.exception("failed to render fingerprint blob")
                    self.draw_minutiae_on_blank(db_minutiae)
            else:
                self.draw_minutiae_on_blank(db_minutiae)
            msg = f"Match: {name} (score={best_score:.2f})"
            self.set_status(msg)
            messagebox.showinfo("Verification", msg)
            logging.info(f"Verification match: {name} score={best_score:.3f}")
        else:
            self.canvas_right.delete("all")
            self.canvas_right.create_text(210,180, text="No Match Found", fill="#900")
            self.set_status("No Match Found")
            messagebox.showwarning("Verification", "No matching fingerprint found.")
            logging.info("Verification: no match")

        self.progress['value'] = 0
        self.set_busy(False)

    def draw_minutiae_on_blank(self, minutiae):
        h, w = 360, 420
        blank = np.full((h, w), 255, dtype=np.uint8)
        vis = draw_minutiae_on_image(blank, minutiae)
        pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        self.display_image_on_canvas(pil, self.canvas_right)

    # ---------- Admin Login ----------
    def open_admin_login(self):
        win = tk.Toplevel(self.root)
        win.title("Admin Login")
        win.geometry("320x160")
        ttk.Label(win, text="Username:").pack(pady=(12,0))
        ent_user = ttk.Entry(win)
        ent_user.pack()
        ttk.Label(win, text="Password:").pack(pady=(8,0))
        ent_pass = ttk.Entry(win, show="*")
        ent_pass.pack()
        def do_login():
            user = ent_user.get().strip()
            pw = ent_pass.get()
            if not user or not pw:
                messagebox.showwarning("Input", "Enter credentials")
                return
            cnx = connect_db()
            if cnx is None:
                return
            try:
                with closing(cnx.cursor()) as cur:
                    cur.execute("SELECT password_hash FROM users WHERE username=%s", (user,))
                    row = cur.fetchone()
                    if not row:
                        messagebox.showerror("Login Failed", "User not found")
                        return
                    stored = row[0]
                    if verify_password(user, pw, stored):
                        self.admin_authed = True
                        self.admin_user = user
                        self.lbl_admin.config(text=f"Logged in: {user}")
                        messagebox.showinfo("Login", "Admin authenticated")
                        win.destroy()
                    else:
                        messagebox.showerror("Login Failed", "Wrong password")
            except Exception as e:
                logging.exception("admin login error")
                messagebox.showerror("DB Error", str(e))
            finally:
                cnx.close()
        ttk.Button(win, text="Login", command=do_login).pack(pady=12)

    # ---------- Export / Import DB ----------
    def export_db(self):
        # require admin login
        if not self.admin_authed:
            messagebox.showwarning("Permission", "Admin login required to export DB")
            return

        cnx = connect_db()
        if cnx is None:
            return

        try:
            with closing(cnx.cursor()) as cur:
                cur.execute("SELECT name, fingerprint, minutiae, created_at FROM students")
                rows = cur.fetchall()
        except Exception as e:
            logging.exception("export fetch failed")
            messagebox.showerror("DB Error", str(e))
            cnx.close()
            return
        finally:
            cnx.close()

        export_list = []
        for name, fingerprint_blob, minutiae_json, created_at in rows:
            fingerprint_b64 = None
            if fingerprint_blob:
                try:
                    fingerprint_b64 = base64.b64encode(fingerprint_blob).decode('ascii')
                except Exception:
                    logging.exception(f"Failed to base64 encode fingerprint for {name}")
                    fingerprint_b64 = None

            if isinstance(minutiae_json, (bytes, bytearray)):
                try:
                    minutiae_json = minutiae_json.decode('utf-8')
                except Exception:
                    logging.exception(f"Failed to decode minutiae JSON for {name}")
                    minutiae_json = "[]"
            elif minutiae_json is None:
                minutiae_json = "[]"

            created_str = str(created_at) if created_at else ""

            export_list.append({
                "name": name,
                "fingerprint_b64": fingerprint_b64,
                "minutiae_json": minutiae_json,
                "created_at": created_str
            })

        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_list, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Export", f"DB exported {len(export_list)} records")
        except Exception as e:
            logging.exception("export save failed")
            messagebox.showerror("File Error", str(e))

    def import_db(self):
        # require admin login
        if not self.admin_authed:
            messagebox.showwarning("Permission", "Admin login required to import DB")
            return

        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to read JSON: {e}")
            return

        cnx = connect_db()
        if cnx is None:
            return

        inserted = 0
        try:
            with closing(cnx.cursor()) as cur:
                for rec in records:
                    name = rec.get("name")
                    minutiae_json = rec.get("minutiae_json", "[]")
                    fingerprint_b64 = rec.get("fingerprint_b64")
                    fingerprint_bytes = None
                    if fingerprint_b64:
                        try:
                            fingerprint_bytes = base64.b64decode(fingerprint_b64)
                        except Exception:
                            logging.warning(f"Failed to decode fingerprint for {name}")
                            fingerprint_bytes = None
                    try:
                        cur.execute("""
                            INSERT INTO students (name, minutiae, fingerprint)
                            VALUES (%s,%s,%s)
                            ON DUPLICATE KEY UPDATE minutiae=VALUES(minutiae), fingerprint=COALESCE(VALUES(fingerprint), fingerprint)
                        """, (name, minutiae_json, fingerprint_bytes))
                        inserted += 1
                    except Exception:
                        logging.exception(f"Failed to insert {name}")
                cnx.commit()
        finally:
            cnx.close()
        messagebox.showinfo("Import", f"Imported / updated {inserted} records")

    # ---------- utility ----------
    def root_update(self):
        self.root.update_idletasks()

# Simple pairwise fallback
def simple_pairwise_score(min_in, min_db, max_dist=SPATIAL_TOLERANCE, angle_tol=ANGLE_TOLERANCE):
    count = 0
    for a in min_in:
        for b in min_db:
            if a['type'] != b['type']:
                continue
            d = math.hypot(a['x']-b['x'], a['y']-b['y'])
            if d > max_dist:
                continue
            diff = abs(a['angle'] - b['angle']) % 180.0
            diff = min(diff, 180.0-diff)
            if diff <= angle_tol:
                count += 1
                break
    return count / max(len(min_in), len(min_db), 1)

# ------------------ Run App ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
