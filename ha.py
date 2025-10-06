import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mysql.connector
import os

# ----------------- Database Setup -----------------
def connect_db():
    try:
        cnx = mysql.connector.connect(
            host="localhost",
            user="root",       # <-- Change to your MySQL username
            password="*******",       # <-- Change to your MySQL password
            database="fingerprint_system"  # <-- Change to your DB name
        )
        return cnx
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")
        return None


# ----------------- GUI App Class -----------------
class FingerprintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Enrollment (Tkinter + MySQL)")
        self.root.geometry("600x500")

        self.image_path = None
        self.processed_image = None

        # Student Name
        tk.Label(root, text="Student Name:", font=("Arial", 12)).pack(pady=5)
        self.name_entry = tk.Entry(root, font=("Arial", 12), width=40)
        self.name_entry.pack(pady=5)

        # Canvas for image preview
        self.canvas = tk.Canvas(root, width=400, height=300, bg="lightgray")
        self.canvas.pack(pady=10)
        self.canvas.create_text(200, 150, text="No Image Selected", fill="black", font=("Arial", 14))

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Process & Save", command=self.process_and_save).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Exit", command=root.quit).grid(row=0, column=2, padx=5)

        # Status Label
        self.status_label = tk.Label(root, text="", fg="green", font=("Arial", 11))
        self.status_label.pack(pady=5)

    # Select fingerprint image
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.status_label.config(text="Image Loaded", fg="blue")

    # Display image
    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 300))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(200, 150, image=self.tk_img)
    
    # Fingerprint processing using OpenCV
    def process_fingerprint(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return thresh

    # Save data to database
    def save_to_db(self, name, image_bytes):
        cnx = connect_db()
        if cnx is None:
            return

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

    # Process and save image
    def process_and_save(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter student name")
            return
        if not self.image_path:
            messagebox.showwarning("Input Error", "Please select a fingerprint image")
            return

        # Process fingerprint
        processed = self.process_fingerprint(self.image_path)
        success, encoded_img = cv2.imencode('.png', processed)
        if success:
            img_bytes = encoded_img.tobytes()
            if self.save_to_db(name, img_bytes):
                self.status_label.config(text=f"Saved: {name}", fg="green")
                messagebox.showinfo("Success", "Fingerprint saved to database")


# ----------------- Run App -----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintApp(root)
    root.mainloop()
