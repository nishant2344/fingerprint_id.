import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import mysql.connector
import numpy as np
from skimage.morphology import skeletonize
import io

# ----------------- Database Setup -----------------
def connect_db():
    try:
        cnx = mysql.connector.connect(
            host="localhost",
            user="root",        # <-- Change to your MySQL username
            password="your_mysql_password_here",  # <-- Change to your MySQL password
            database="fingerprints_db"  # <-- Change to your DB name
        )
        return cnx
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error connecting to DB:\n{err}")
        return None


# ----------------- Minutiae Extraction -----------------
def extract_minutiae(processed_img):
    # processed_img is expected to be a binary image (0 and 255)
    # Convert to boolean where ridge pixels are True
    binary = (processed_img == 0)
    skeleton = skeletonize(binary).astype(np.uint8)  # result is 0/1

    minutiae_end = []
    minutiae_bif = []

    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                neighbors = [
                    skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                    skeleton[i, j-1],                   skeleton[i, j+1],
                    skeleton[i+1, j-1], skeleton[i+1, j], skeleton[i+1, j+1]
                ]
                crossing_number = np.sum(neighbors)

                if crossing_number == 1:   # Ridge ending
                    minutiae_end.append((i, j))
                elif crossing_number > 2:  # Bifurcation
                    minutiae_bif.append((i, j))

    return minutiae_end, minutiae_bif, skeleton


def draw_minutiae(processed_img, minutiae_end, minutiae_bif):
    # processed_img expected as uint8 grayscale (0..255)
    img_color = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    # Ridge endings (red)
    for (y, x) in minutiae_end:
        cv2.circle(img_color, (x, y), 3, (0, 0, 255), -1)

    # Bifurcations (green)
    for (y, x) in minutiae_bif:
        cv2.circle(img_color, (x, y), 3, (0, 255, 0), -1)

    return img_color


# ----------------- GUI App Class -----------------
class FingerprintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint System — Enroll & Verify")
        self.root.geometry("720x580")
        self.root.resizable(False, False)

        # ttk style (Gray Professional)
        self.style = ttk.Style()
        # Use default theme then tweak
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#e9e9e9")
        self.style.configure("TLabel", background="#f0f0f0")
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("Status.TLabel", font=("Segoe UI", 10), background="#f0f0f0")
        self.style.map("TButton",
                       foreground=[('pressed', 'black'), ('active', 'black')],
                       background=[('active', '#d9d9d9')])

        self.image_path = None
        self.tk_img = None  # keep reference to avoid GC

        # Main frame
        main = ttk.Frame(root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main, text="Fingerprint Enrollment & Verification", style="Header.TLabel", anchor="center")
        header.pack(fill=tk.X, pady=(0, 8))

        # Input frame
        input_frame = ttk.Frame(main)
        input_frame.pack(fill=tk.X, pady=6)

        name_lbl = ttk.Label(input_frame, text="Student Name:")
        name_lbl.grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.name_entry = ttk.Entry(input_frame, width=42)
        self.name_entry.grid(row=0, column=1, sticky=tk.W)

        # Canvas for image preview (centered)
        preview_frame = ttk.Frame(main, relief=tk.SOLID)
        preview_frame.pack(pady=12)
        self.canvas = tk.Canvas(preview_frame, width=420, height=320, bg="#dcdcdc", highlightthickness=0)
        self.canvas.pack()
        self.canvas_text_id = self.canvas.create_text(210, 160, text="No Image Selected", fill="#333", font=("Segoe UI", 12))

        # Button frame (vertical layout under canvas)
        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=8)

        self.btn_select = ttk.Button(btn_frame, text="Select Image", command=self.select_image, width=18)
        self.btn_select.grid(row=0, column=0, padx=6, pady=4)

        self.btn_enroll = ttk.Button(btn_frame, text="Enroll Fingerprint", command=self.process_and_save, width=18)
        self.btn_enroll.grid(row=1, column=0, padx=6, pady=4)

        self.btn_verify = ttk.Button(btn_frame, text="Verify Fingerprint", command=self.verify_fingerprint, width=18)
        self.btn_verify.grid(row=2, column=0, padx=6, pady=4)

        self.btn_exit = ttk.Button(btn_frame, text="Exit", command=self.root.quit, width=18)
        self.btn_exit.grid(row=3, column=0, padx=6, pady=4)

        # Status bar
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_label = ttk.Label(status_frame, text="Ready", style="Status.TLabel", anchor="w")
        self.status_label.pack(fill=tk.X)

    # -------- Select fingerprint image --------
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.set_status("Image Loaded", "blue")

    # -------- Display image on canvas --------
    def display_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((420, 320))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(210, 160, image=self.tk_img)
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open image:\n{e}")
            self.canvas.delete("all")
            self.canvas.create_text(210, 160, text="No Image Selected", fill="#333", font=("Segoe UI", 12))

    # -------- Fingerprint preprocessing --------
    def process_fingerprint(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to read image file")
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return thresh

    # -------- Save to DB --------
    def save_to_db(self, name, image_bytes):
        cnx = connect_db()
        if cnx is None:
            return False

        try:
            cursor = cnx.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    fingerprint LONGBLOB
                )
            """)
            cursor.execute("INSERT INTO students (name, fingerprint) VALUES (%s, %s)", (name, image_bytes))
            cnx.commit()
            cursor.close()
            cnx.close()
            return True
        except mysql.connector.Error as err:
            messagebox.showerror("DB Error", f"Error: {err}")
            return False

    # -------- Enroll Fingerprint --------
    def process_and_save(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter student name")
            return
        if not self.image_path:
            messagebox.showwarning("Input Error", "Please select a fingerprint image")
            return

        try:
            processed = self.process_fingerprint(self.image_path)
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            return

        minutiae_end, minutiae_bif, skeleton = extract_minutiae(processed)
        img_with_minutiae = draw_minutiae(processed, minutiae_end, minutiae_bif)

        # Show processed image with minutiae in GUI
        img_rgb = cv2.cvtColor(img_with_minutiae, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((420, 320))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(210, 160, image=self.tk_img)

        msg = f"Ridge Endings: {len(minutiae_end)}, Bifurcations: {len(minutiae_bif)}"
        self.set_status(f"Processed — {msg}", "purple")
        messagebox.showinfo("Minutiae Points", msg)

        # Save binary processed image to DB (as PNG bytes)
        success, encoded_img = cv2.imencode('.png', processed)
        if success:
            img_bytes = encoded_img.tobytes()
            saved = self.save_to_db(name, img_bytes)
            if saved:
                self.set_status(f"Saved: {name} | {msg}", "green")
                messagebox.showinfo("Success", f"Fingerprint saved for '{name}' with {msg}")
            else:
                self.set_status("Failed to save to DB", "red")
        else:
            messagebox.showerror("Encode Error", "Failed to encode processed fingerprint image")

    # -------- Compare Minutiae Sets --------
    def compare_minutiae(self, set1, set2, tolerance=10):
        # set1 and set2 are iterables of (y,x) tuples
        matches = 0
        # convert to list for nested iteration
        s1 = list(set1)
        s2 = list(set2)
        for (y1, x1) in s1:
            for (y2, x2) in s2:
                if abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance:
                    matches += 1
                    break
        total_points = max(len(s1), len(s2))
        return matches / total_points if total_points > 0 else 0

    # -------- Verify Fingerprint --------
    def verify_fingerprint(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            return

        try:
            processed_input = self.process_fingerprint(file_path)
        except Exception as e:
            messagebox.showerror("Processing Error", f"Could not process input image:\n{e}")
            return

        minutiae_end_in, minutiae_bif_in, _ = extract_minutiae(processed_input)
        input_points = set(minutiae_end_in + minutiae_bif_in)

        cnx = connect_db()
        if cnx is None:
            return
        cursor = cnx.cursor()
        try:
            cursor.execute("SELECT name, fingerprint FROM students")
            records = cursor.fetchall()
        except mysql.connector.Error as err:
            messagebox.showerror("DB Error", f"Error fetching records:\n{err}")
            cursor.close()
            cnx.close()
            return

        cursor.close()
        cnx.close()

        best_match = None
        best_score = 0

        for name, blob in records:
            # Decode DB image
            nparr = np.frombuffer(blob, np.uint8)
            db_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if db_img is None:
                continue

            minutiae_end_db, minutiae_bif_db, _ = extract_minutiae(db_img)
            db_points = set(minutiae_end_db + minutiae_bif_db)

            score = self.compare_minutiae(input_points, db_points)
            if score > best_score:
                best_score = score
                best_match = (name, db_img)

        # Threshold check
        if best_match and best_score > 0.6:  # 60% threshold
            name, matched_img = best_match

            img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((420, 320))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.canvas.delete("all")
            self.canvas.create_image(210, 160, image=self.tk_img)

            msg = f"Match Found: {name} (Score: {best_score:.2f})"
            self.set_status(msg, "green")
            messagebox.showinfo("Verification Result", msg)
        else:
            self.set_status("No Match Found", "red")
            messagebox.showwarning("Verification Failed", "No matching fingerprint found.")

    # -------- Utility: update status ----------
    def set_status(self, text, color="black"):
        # color argument is illustrative; ttk doesn't directly support foreground per-label via style here,
        # so we set the text and keep it simple.
        self.status_label.config(text=text)


# ----------------- Run App -----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
