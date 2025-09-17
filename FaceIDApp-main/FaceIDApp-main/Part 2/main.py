# main_pyqt.py
# Single-file PyQt5 port of your merged Kivy app.
# Preserves DeepFace/ArcFace logic and model usage exactly.
# Applies robustness fixes from the earlier review.

import os
import sys
import platform
import tempfile
import sqlite3
import logging
import threading
import subprocess
import json
from datetime import datetime
import configparser
import pyzbar.pyzbar as pyzbar

import cv2
import numpy as np
from deepface import DeepFace

from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------------------
# Paths and DB initialization
# ---------------------------
TMP = os.path.join("C:\\")
DATA_DIR = os.path.join(TMP, "data")
VERIF_DIR = os.path.join(DATA_DIR, "verification_image")
INPUT_DIR = os.path.join(DATA_DIR, "input_image")
THRESHOLD_DIR = os.path.join(DATA_DIR, "threshold")
os.makedirs(VERIF_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(THRESHOLD_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "profiles.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            name TEXT,
            id2 TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON profiles (name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_id2 ON profiles (id2)")
    conn.commit()
    conn.close()

def profile_exists(id_val: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM profiles WHERE id = ? LIMIT 1", (id_val,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def insert_profile(id_val: str, name: str, id2_val: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO profiles (id, name, id2) VALUES (?, ?, ?)", (id_val, name, id2_val))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception:
        return False

# ---------------------------
# Logging
# ---------------------------
log_dir = os.path.join(TMP, "data", "log")
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime("face_recognition_%Y%m%d_%H%M%S.log")
log_path = os.path.join(log_dir, log_filename)

logger = logging.getLogger("FaceAppLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
logger.info(f"Logger initialized — writing to {log_path}")

# ---------------------------
# Image utilities (single copy)
# ---------------------------
def enhance_image(img):
    try:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        hsv = cv2.cvtColor(img_output, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 25)
        final_hsv = cv2.merge((h, s, v))
        img_output = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img_output
    except Exception as e:
        logger.warning(f"enhance_image failed: {e}")
        return img

def is_sharp(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def open_camera(index=0):
    cap = None
    try:
        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                # attempt v4l2 backend
                try:
                    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"open_camera exception: {e}")
    if cap is None or not cap.isOpened():
        logger.error("Camera open failed.")
        return None
    return cap

def crop_center(frame, rel_y1=0.2, rel_y2=0.8, rel_x1=0.3, rel_x2=0.7):
    h, w = frame.shape[:2]
    y1 = int(h * rel_y1); y2 = int(h * rel_y2)
    x1 = int(w * rel_x1); x2 = int(w * rel_x2)
    y1, x1 = max(y1, 0), max(x1, 0)
    y2, x2 = min(y2, h), min(x2, w)
    roi = frame[y1:y2, x1:x2, :]
    return roi if roi.size else frame

# ---------------------------
# Embedding cache builder (inline)
# ---------------------------
def build_embedding_cache(verify_root, model_name="ArcFace"):
    cache = {}
    if not os.path.isdir(verify_root):
        return cache
    for person in sorted(os.listdir(verify_root)):
        person_dir = os.path.join(verify_root, person)
        if not os.path.isdir(person_dir):
            continue

        # Extract ID from folder name
        if "_" not in person:
            logger.warning(f"Skipping folder with unexpected name: {person}")
            continue
        id_val = person.split("_")[0]

        # Retrieve ID and Name from profiles.db
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name,id2 FROM profiles WHERE id = ?", (id_val,))
        row = cur.fetchone()
        conn.close()
        if not row:
            logger.warning(f"No DB record found for ID {id_val}")
            continue
        name_val = row[0]
        id2_val = row[1]

        embeddings = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(person_dir, fname)
            try:
                rep = DeepFace.represent(
                    img_path=fpath,
                    model_name=model_name,
                    enforce_detection=False
                )
                if not rep:
                    continue
                rep_emb = rep[0].get('embedding', None)
                if rep_emb is None:
                    continue
                emb = np.array(rep_emb, dtype=np.float32)
                n = np.linalg.norm(emb) + 1e-9
                emb = emb / n
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Embed failed for {fpath}: {e}")
        if embeddings:
            cache[person] = {
                "id": id_val,
                "name": name_val,
                "id_2": id2_val,
                "embeddings": np.vstack(embeddings)
            }
    return cache

# ---------------------------
# Worker threads using Qt signals
# ---------------------------
class GenerateCacheWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(dict)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, verify_root, model_name="ArcFace", parent=None):
        super().__init__(parent)
        self.verify_root = verify_root
        self.model_name = model_name

    def run(self):
        try:
            cache = build_embedding_cache(self.verify_root, model_name=self.model_name)
            # Save to disk
            out_path = os.path.join(THRESHOLD_DIR, "embedding_cache.npy")
            np.save(out_path, cache, allow_pickle=True)
            logger.info(f"Embedding cache built: {len(cache)} people, saved to {out_path}")
            self.finished_signal.emit(cache)
        except Exception as e:
            logger.error(f"GenerateCacheWorker failed: {e}")
            self.error_signal.emit(str(e))

class VerifyWorker(QtCore.QThread):
    result_signal = QtCore.pyqtSignal(object, float)  # best_match (or None), min_dist
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, input_image_path, reference_embeddings, threshold, model_name="ArcFace", parent=None):
        super().__init__(parent)
        self.input_image_path = input_image_path
        self.reference_embeddings = reference_embeddings
        self.threshold = threshold
        self.model_name = model_name

    def run(self):
        try:
            rep = DeepFace.represent(
                img_path=self.input_image_path,
                model_name=self.model_name,
                enforce_detection=False
            )
            if not rep:
                raise RuntimeError("DeepFace.represent returned empty.")
            input_embedding = np.array(rep[0]['embedding'], dtype=np.float32)
            input_embedding = input_embedding / (np.linalg.norm(input_embedding) + 1e-9)
        except Exception as e:
            logger.error(f"Failed to get input embedding: {e}")
            self.error_signal.emit(f"embed_error:{e}")
            return

        best_match, min_dist = None, float('inf')
        logger.info(f"Using detection_threshold = {self.threshold}")
        try:
            for person_key, data in self.reference_embeddings.items():
                embeddings = data["embeddings"]
                dists = np.linalg.norm(embeddings - input_embedding, axis=1)
                avg_dist = float(np.mean(dists))
                logger.info(f"Compared with: {data['name']} (ID: {data['id']}), avg_dist: {avg_dist:.4f}")
                if avg_dist < self.threshold and avg_dist < min_dist:
                    min_dist = avg_dist
                    best_match = data  # or store full data if needed
            self.result_signal.emit(best_match, min_dist)
        except Exception as e:
            logger.error(f"Verification compare failed: {e}")
            self.error_signal.emit(str(e))

class BarcodeWorker(QtCore.QThread):
    # Signal to emit the processed camera frame
    frame_updated = QtCore.pyqtSignal(np.ndarray)
    # Signal to emit when the correct barcode is found
    barcode_matched = QtCore.pyqtSignal(str)
    # Signal to emit when the process finishes (matched or timeout)
    finished_scanning = QtCore.pyqtSignal()

    def __init__(self, capture, expected_id2, timeout=180, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.expected_id2 = expected_id2
        self.timeout = timeout
        self._is_running = True

    def run(self):
        start_time = datetime.now()
        matched = False

        while self._is_running and (datetime.now() - start_time).seconds < self.timeout:
            if not self.capture or not self.capture.isOpened():
                logger.warning("Camera not available in BarcodeWorker.")
                break

            ret, frame = self.capture.read()
            if not ret or frame is None or frame.size == 0:
                continue

            # Convert to grayscale for better barcode detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Decode barcodes
            barcodes = pyzbar.decode(gray_frame)
            logger.debug(f"Barcodes detected: {len(barcodes)}")  # debug log

            for barcode in barcodes:
                barcode_data = barcode.data.decode("utf-8")
                x, y, w, h = barcode.rect

                # Draw green rectangle and barcode text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, barcode_data, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                logger.info(f"Detected barcode: {barcode_data}")

                if barcode_data == self.expected_id2:
                    logger.info(f"Barcode matched expected: {barcode_data}")
                    self.barcode_matched.emit(barcode_data)
                    matched = True
                    break

            # Emit the frame to the UI (green boxes if detected)
            self.frame_updated.emit(frame)

            if matched:
                break

        self.stop()
        self.finished_scanning.emit()
        logger.info("Barcode scanning thread finished.")


    def stop(self):
        self._is_running = False

# ---------------------------
# PyQt GUI
# ---------------------------
class HomeWidget(QtWidgets.QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        layout = QtWidgets.QVBoxLayout()
        layout.addStretch(1)
        btn_faceid = QtWidgets.QPushButton("Face Recognition")
        btn_capture = QtWidgets.QPushButton("New Profile Creation")
        btn_exit = QtWidgets.QPushButton("Exit App")
        for btn in (btn_faceid, btn_capture, btn_exit):
            btn.setFixedHeight(72)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setStyleSheet(self.button_style())

        btn_faceid.clicked.connect(self.goto_faceid)
        btn_capture.clicked.connect(self.goto_capture)
        btn_exit.clicked.connect(lambda: QtWidgets.QApplication.quit())

        layout.addWidget(btn_faceid)
        layout.addWidget(btn_capture)
        layout.addStretch(2)
        layout.addWidget(btn_exit)
        self.setLayout(layout)

    @staticmethod
    def button_style():
        return """
        QPushButton {
            border-radius: 12px;
            background-color: #1f4b70;
            color: white;
            font-size: 16px;
        }
        QPushButton:hover { background-color: #245c89; }
        QPushButton:pressed { background-color: #163a55; }
        """

    def goto_faceid(self):
        self.parent_app.show_faceid()

    def goto_capture(self):
        self.parent_app.show_capture()

class CaptureWidget(QtWidgets.QWidget):
    # Signal to tell main to rebuild cache when capture done
    capture_finished = QtCore.pyqtSignal()

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.capture = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.saving = False
        self.save_count = 0
        self.max_photos = 20
        self.user_folder = ""
        self.file_prefix = ""

        # UI
        layout = QtWidgets.QVBoxLayout()
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background: #222; border-radius:8px;")
        layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)

        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")
        self.id_input = QtWidgets.QLineEdit()
        self.id_input.setPlaceholderText("Enter ID")
        self.id2_input = QtWidgets.QLineEdit()
        self.id2_input.setPlaceholderText("Enter ID2")
        form = QtWidgets.QFormLayout()
        form.addRow("Name:", self.name_input)
        form.addRow("ID:", self.id_input)
        form.addRow("ID2:", self.id2_input)
        layout.addLayout(form)

        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)

        self.start_btn = QtWidgets.QPushButton("Start Capture")
        self.start_btn.clicked.connect(self.start_capture)
        layout.addWidget(self.start_btn)

        self.back_btn = QtWidgets.QPushButton("← Back to Home")
        self.back_btn.clicked.connect(self.go_home)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

        # Timer for video frames
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_app(self):
        # init capture
        self.capture = open_camera(0)
        if self.capture is None:
            self.status_label.setText("Camera open failed.")
            return
        self.timer.start(30)  # ~33fps

    def stop_app(self):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if self.capture and self.capture.isOpened():
                self.capture.release()
        except Exception:
            pass

    def update_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self._set_label_image(self.image_label, frame_display)
        if self.saving and self.save_count < self.max_photos:
            self.save_face(frame, faces)

    def _set_label_image(self, label, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pix)

    def save_face(self, frame, faces):
        if len(faces) == 0:
            self.status_label.setText("No face detected")
            return
        # pick largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_img = frame[y:y+h, x:x+w]
        try:
            face_img = cv2.resize(face_img, (250, 250))
        except Exception:
            self.status_label.setText("Resize failed")
            return
        face_img = enhance_image(face_img)
        if not is_sharp(face_img):
            self.status_label.setText("⏳ Waiting for sharp image...")
            return
        save_path = os.path.join(self.user_folder, f"{self.file_prefix}_{self.save_count+1:02d}.jpg")
        cv2.imwrite(save_path, face_img)
        self.save_count += 1
        self.status_label.setText(f"Saved {self.save_count}/{self.max_photos}")
        if self.save_count >= self.max_photos:
            self.status_label.setText(f"✅ Done! Saved to:\n{self.user_folder}")
            self.saving = False
            # after capture, rebuild embedding cache
            threading.Thread(target=self._rebuild_cache_background, daemon=True).start()
            # emit finished to parent if needed
            self.capture_finished.emit()

    def _rebuild_cache_background(self):
        try:
            logger.info("Rebuilding embedding cache after capture...")
            cache = build_embedding_cache(VERIF_DIR, model_name="ArcFace")
            out_path = os.path.join(THRESHOLD_DIR, "embedding_cache.npy")
            np.save(out_path, cache, allow_pickle=True)
            logger.info("Rebuild complete.")
        except Exception as e:
            logger.error(f"Rebuild cache failed: {e}")

    def start_capture(self):
        name_val = self.name_input.text().strip().replace(" ", "_").upper()
        id_val = self.id_input.text().strip().upper()
        id2_val = self.id2_input.text().strip().upper()
        if not name_val or not id_val or not id2_val:
            self.status_label.setText("Please enter Name, ID, and ID2")
            return
        if profile_exists(id_val):
            self.status_label.setText(f"❌ Error: ID '{id_val}' already exists in database. Use a different ID.")
            return
        ok = insert_profile(id_val, name_val, id2_val)
        if not ok:
            self.status_label.setText(f"❌ Error: failed to insert profile for ID '{id_val}'.")
            return
        base_dir = VERIF_DIR
        # Folder name = ID_NAME
        folder_name = f"{id_val}_{name_val}"
        logger.info(f"Folder name is: {folder_name}")
        self.user_folder = os.path.join(base_dir, folder_name)
        os.makedirs(self.user_folder, exist_ok=True)
        self.file_prefix = f"{folder_name}"
        self.saving = True
        self.save_count = 0
        self.status_label.setText(f"📷 Capturing photos for {name_val}...")

    def go_home(self):
        self.stop_app()
        self.parent_app.show_home()

class FaceIDWidget(QtWidgets.QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.capture = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.reference_embeddings = {}
        self.SAVE_PATH = os.path.join(INPUT_DIR, 'input_image.jpg')
        self.threshold = 0.85 # default value, will be overridden by config
        self.freeze_frame = None
        self.detection_enabled = True
        self.barcode_worker = None # Add this line

        # Load threshold from config
        self.load_threshold_config()

        # UI
        layout = QtWidgets.QVBoxLayout()
        self.label_status = QtWidgets.QLabel("Initializing")
        self.label_status.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status.setFixedHeight(80)
        self.label_status.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.label_status)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background: #222; border-radius:8px;")
        layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)

        self.back_btn = QtWidgets.QPushButton("← Back to Home")
        self.back_btn.clicked.connect(self.go_home)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

        # Timers
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_frame)

        self.check_timer = QtCore.QTimer()
        self.check_timer.timeout.connect(self.check_face_detected)

        # load config / embeddings in background
        self._init_sequence_index = 0
        self._init_steps = [
            "Removing old embedding cache...",
            "Copying threshold config...",
            "Running generate_embedding_cache...",
            "Loading embeddings...",
            "Initializing camera...",
            "Initialization Complete."
        ]
        # start initialization
        QtCore.QTimer.singleShot(100, self.initialize_sequence)

    def load_threshold_config(self):
        cfg_path = os.path.join(THRESHOLD_DIR, "config.ini")
        # Create default config if missing
        if not os.path.exists(cfg_path):
            os.makedirs(THRESHOLD_DIR, exist_ok=True)
            with open(cfg_path, "w") as f:
                f.write("[threshold]\n")
                f.write("detection_threshold = 0.85\n")
            logger.info(f"Default config.ini created at {cfg_path}")

        # Read threshold value
        config = configparser.ConfigParser()
        config.read(cfg_path)
        try:
            self.threshold = float(config["threshold"].get("detection_threshold", "0.85"))
            logger.info(f"Detection threshold set to {self.threshold} from config")
        except Exception as e:
            logger.warning(f"Failed to read threshold from config, using default 0.85: {e}")
            self.threshold = 0.85

    def initialize_sequence(self):
        # run through steps
        if self._init_sequence_index >= len(self._init_steps):
            self._init_sequence_index = 0
        step = self._init_steps[self._init_sequence_index]
        self.label_status.setText(step)
        logger.info(step)
        if step == "Removing old embedding cache...":
            cache_file = os.path.join(THRESHOLD_DIR, "embedding_cache.npy")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    logger.info("Old embedding cache removed.")
                except Exception as e:
                    logger.warning(f"Could not remove cache: {e}")
            QtCore.QTimer.singleShot(300, self.initialize_sequence)
        elif step == "Copying threshold config...":
            # Ensure config exists; if not, create simple config.ini
            cfg_path = os.path.join(THRESHOLD_DIR, "config.ini")
            if not os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "w") as f:
                        f.write("[threshold]\n")
                        f.write("detection_threshold = 0.85\n")
                    logger.info("Default config.ini created.")
                except Exception as e:
                    logger.warning(f"Could not create config: {e}")
            QtCore.QTimer.singleShot(300, self.initialize_sequence)
        elif step == "Running generate_embedding_cache...":
            # run generate cache worker
            self.gen_worker = GenerateCacheWorker(VERIF_DIR, model_name="ArcFace")
            self.gen_worker.finished_signal.connect(self.on_cache_built)
            self.gen_worker.error_signal.connect(lambda e: logger.error(f"Cache gen err: {e}"))
            self.gen_worker.start()
            # move to next step once worker finishes (handled in on_cache_built)
        elif step == "Loading embeddings...":
            # load cached embeddings if present
            path = os.path.join(THRESHOLD_DIR, "embedding_cache.npy")
            if os.path.exists(path):
                try:
                    self.reference_embeddings = np.load(path, allow_pickle=True).item()
                    logger.info(f"Embedding cache loaded with {len(self.reference_embeddings)} people.")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
                    self.reference_embeddings = {}
            else:
                logger.warning("No embedding cache found.")
                self.reference_embeddings = {}
            QtCore.QTimer.singleShot(300, self.initialize_sequence)
        elif step == "Initializing camera...":
            self.capture = open_camera(0)
            if not self.capture:
                self.label_status.setText("Camera open failed.")
            else:
                self.update_timer.start(30)
                self.check_timer.start(1000)
            QtCore.QTimer.singleShot(300, self.initialize_sequence)
        elif step == "Initialization Complete.":
            self.label_status.setText("Ready for Detection")
            self._init_sequence_index = 0
            return
        self._init_sequence_index += 1

    def on_cache_built(self, cache):
        self.reference_embeddings = cache
        logger.info("Cache build finished signal received.")
        # continue to next init step (Loading embeddings)
        self._init_sequence_index += 1
        QtCore.QTimer.singleShot(50, self.initialize_sequence)

    def update_frame(self):
        '''if self.freeze_frame is not None:
            frame = self.freeze_frame.copy()
        else:
            if not self.capture:
                return
            ret, frame = self.capture.read()
            if not ret:
                return
            frame = crop_center(frame)
        if self.detection_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self._set_label_image(self.image_label, frame)'''
        if not self.capture or not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        # Remove crop_center, just resize to maintain proper aspect
        frame_display = cv2.resize(frame, (self.image_label.width(), self.image_label.height()))

        if self.detection_enabled:
            gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self._set_label_image(self.image_label, frame_display)

    def _set_label_image(self, label, frame):
        '''h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pix)'''
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix)

    def check_face_detected(self):
        if not self.detection_enabled or not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = crop_center(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            enhanced = enhance_image(frame)
            try:
                cv2.imwrite(self.SAVE_PATH, enhanced)
            except Exception as e:
                logger.error(f"Failed to save input image: {e}")
                return
            logger.info(f"Face detected — input image saved to: {self.SAVE_PATH}")
            self.freeze_frame = frame.copy()
            self.detection_enabled = False
            self.label_status.setText("Checking...")
            # start verify worker
            self.verify_worker = VerifyWorker(self.SAVE_PATH, self.reference_embeddings, self.threshold, model_name="ArcFace")
            self.verify_worker.result_signal.connect(self.on_verify_result)
            self.verify_worker.error_signal.connect(self.on_verify_error)
            self.verify_worker.start()
            # pause check timer until resumed
            self.check_timer.stop()
            # schedule resumption after 10 seconds
            QtCore.QTimer.singleShot(10000, self.resume_check)

    def resume_check(self):
        self.label_status.setText("Ready for Detection")
        self.freeze_frame = None
        self.detection_enabled = True
        self.check_timer.start(1000)

    def on_verify_result(self, best_match, min_dist):
        if best_match:
            id_val = best_match['id']
            name = best_match['name'].replace("_", " ").upper()
            self.label_status.setText(f"Welcome\n{name} ({id_val})\nNow show the barcode")
            self.label_status.setStyleSheet("color: #ff0; font-size: 16px;")
            logger.info(f"Face matched: {name} (distance={min_dist:.4f}). Awaiting barcode.")

            # Stop the main timers to let the barcode worker control the camera
            self.update_timer.stop()
            self.check_timer.stop()
            self.detection_enabled = False

            # --- Start the Barcode Worker ---
            self.barcode_worker = BarcodeWorker(self.capture, expected_id2=best_match['id_2'])
            
            # Connect signals to slots
            self.barcode_worker.frame_updated.connect(self.update_barcode_frame)
            self.barcode_worker.barcode_matched.connect(self.on_barcode_matched)
            self.barcode_worker.finished_scanning.connect(self.on_barcode_scan_finished)
            
            # Start the thread
            self.barcode_worker.start()

        else:
            self.show_denied()
            logger.info("Access denied — no match within threshold.")
            threading.Thread(target=self.play_beep, daemon=True).start()
            # Reset after 10 seconds
            QtCore.QTimer.singleShot(10000, self.resume_check)
    
    def update_barcode_frame(self, frame):
        """This slot receives a frame from the worker and updates the image label."""
        self._set_label_image(self.image_label, frame)

    def on_barcode_matched(self, barcode_data):
        """
        This slot is called when the expected barcode is detected.
        It updates the label, plays a beep, and logs access granted.
        """
        # Safely extract the user's name from the label
        current_text = self.label_status.text()
        lines = current_text.split('\n')

        # Try to find the line containing the name (skip "Welcome" and empty lines)
        name_text = ""
        for line in lines:
            line = line.strip()
            if line and line.lower() != "welcome" and "barcode" not in line.lower():
                name_text = line
                break

        # Update label with barcode match
        self.label_status.setText(f"Welcome\n{name_text}\nBARCODE MATCHED")
        self.label_status.setStyleSheet("color: #0f0; font-size: 16px;")

        # Log access granted
        logger.info(f"Access granted with barcode: {barcode_data}")

        # Play beep in background
        threading.Thread(target=self.play_beep, daemon=True).start()
      
    def on_barcode_scan_finished(self):
        """This slot cleans up after the barcode worker is done."""
        logger.info("Barcode scanning finished.")
        # Re-enable face detection after a delay
        self.detection_enabled = True
        self.freeze_frame = None
        self.update_timer.start(30) # Restart the main camera loop
        self.check_timer.start(1000)
        QtCore.QTimer.singleShot(5000, lambda: self.label_status.setText("Ready for Detection"))
        self.barcode_worker = None

    def on_verify_error(self, err):
        logger.error(f"Verify error: {err}")
        self.show_denied()
        threading.Thread(target=self.play_beep, daemon=True).start()
        QtCore.QTimer.singleShot(10000, lambda: self.label_status.setText("Ready for Detection"))

    def show_denied(self):
        self.label_status.setText("ACCESS DENIED")
        self.label_status.setStyleSheet("color: #f00; font-size: 18px;")

    def play_beep(self):
        if platform.system() == 'Windows':
            try:
                import winsound
                winsound.Beep(800, 300)
            except Exception:
                pass
        else:
            # try system bell
            os.system('printf "\\a"')

    def go_home(self):
        try:
            self.update_timer.stop()
            self.check_timer.stop()
            if self.capture and self.capture.isOpened():
                self.capture.release()
        except Exception:
            pass
        self.parent_app.show_home()

# ---------------------------
# Main Window and navigation
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        init_db()

        self.setWindowTitle("AI Face Recognition")
        screen = QtWidgets.QApplication.primaryScreen()
        size = screen.size()  # returns QSize(width, height)
        self.resize(size.width(), size.height())

        # central stacked widget
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home = HomeWidget(self)
        self.capture = CaptureWidget(self)
        self.faceid = FaceIDWidget(self)

        # connect capture finished to faceid cache reload
        self.capture.capture_finished.connect(self.on_capture_finished)

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.faceid)
        self.stack.addWidget(self.capture)

        self.show_home()

        # Apply a modern dark-ish style
        self.apply_styles()

    def apply_styles(self):
        # Minimal dark theme
        style = """
        QWidget { background: #0f1720; color: #ddd; font-family: Segoe UI, Arial; }
        QLabel { color: #ddd; }
        QLineEdit { background: #111316; color: #ddd; border: 1px solid #2b2f33; padding:6px; border-radius:6px; }
        QPushButton { background: #1f4b70; color: #fff; border-radius:8px; padding:8px 12px; }
        """
        self.setStyleSheet(style)

    def show_home(self):
        self.stack.setCurrentWidget(self.home)

    def show_capture(self):
        self.stack.setCurrentWidget(self.capture)
        QtCore.QTimer.singleShot(50, self.capture.start_app)

    def show_faceid(self):
        self.stack.setCurrentWidget(self.faceid)
        # faceid initializes itself on creation; ensure timers running
        # If camera was closed earlier, reinitialize
        if not self.faceid.capture or not getattr(self.faceid.capture, 'isOpened', lambda: False)():
            QtCore.QTimer.singleShot(50, self.faceid.initialize_sequence)

    def closeEvent(self, event):
        # Clean up cameras and threads
        try:
            self.capture.stop_app()
        except Exception:
            pass
        try:
            self.faceid.update_timer.stop()
            self.faceid.check_timer.stop()
            if self.faceid.capture and self.faceid.capture.isOpened():
                self.faceid.capture.release()
        except Exception:
            pass
        event.accept()

    def on_capture_finished(self):
        # After capture, rebuild embedding cache and notify faceid widget to reload
        logger.info("MainWindow caught capture_finished: rebuilding cache & reloading embeddings.")
        try:
            path = os.path.join(THRESHOLD_DIR, "embedding_cache.npy")
            if os.path.exists(path):
                self.faceid.reference_embeddings = np.load(path, allow_pickle=True).item()
                logger.info("FaceID widget reloaded embeddings after capture.")
        except Exception as e:
            logger.warning(f"Failed to reload embeddings after capture: {e}")

# ---------------------------
# Run
# ---------------------------
def main():
    if hasattr(QtWidgets.QApplication, 'setAttribute'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
