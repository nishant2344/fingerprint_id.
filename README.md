# fingerprint_id.
A forensic fingerprint identification system with image processing, minutiae extraction, and secure template matching. Includes GUI, database storage, encryption, and performance metrics (FAR, FRR, EER).

# Forensic Fingerprint Identification System

This project provides a reliable and automated fingerprint-based personal identification system for forensic science and security applications. It captures or uploads fingerprint images, processes them using advanced image processing techniques, extracts unique fingerprint features (minutiae points), and matches them against a secure database for verification and identification.

## ✅ Features
- Fingerprint acquisition (upload or real-time capture)
- Image preprocessing (normalization, enhancement, binarization, thinning)
- Minutiae-based feature extraction
- One-to-one (verification) and one-to-many (identification) matching
- Secure template storage with AES encryption
- Performance evaluation (FAR, FRR, EER)
- User-friendly GUI and optional REST API

## 🏗️ Project Structure
fingerprint_id/
├── app/ # GUI, API, CLI
<br>
├── core/ # Image processing & matching
<br>
├── security/ # Encryption & authentication
<br>
├── storage/ # Database handling
<br>
├── utils/ # Helpers & configs
<br>
├── tests/ # Unit tests
<br>
└── notebooks/ # Experiments & evaluations
<br>

## ⚙️ Tech Stack
- **Python 3.10+**
- OpenCV, NumPy, Matplotlib
- PyQt5 or Tkinter for GUI
- SQLite/MySQL for database
- Cryptography (AES-GCM), Argon2 for security

## 🚀 Getting Started
1. Clone this repo:
   ```bash
   git clone https://github.com/nishant2344/fingerprint_id.git
