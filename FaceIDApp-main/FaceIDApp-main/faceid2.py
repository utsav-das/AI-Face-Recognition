# faceid.py — Updated with full logger and debug info

import os
import sys
import cv2
import imghdr
import shutil
import tempfile
import platform
import numpy as np
import configparser
import threading
import logging
import subprocess
from datetime import datetime

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.animation import Animation

from deepface import DeepFace

# ---------------------- Logging Setup ----------------------
temp_dir = tempfile.gettempdir()
log_dir = os.path.join(temp_dir, "data", "log")
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
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info(f"Logger initialized — writing to {log_path}")

# ---------------------- Folder Structure ----------------------
class FolderStruct:
    def __init__(self):
        self.VERIF_IMG = os.path.join(temp_dir, 'data', 'verification_image')
        self.INPUT_IMG = os.path.join(temp_dir, 'data', 'input_image')
        self.THRESHOLD = os.path.join(temp_dir, 'data', 'threshold')
        for path in [self.VERIF_IMG, self.INPUT_IMG, self.THRESHOLD]:
            os.makedirs(path, exist_ok=True)

# ---------------------- Utility Functions ----------------------
def play_beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(800, 300)
    else:
        os.system('printf "\\a"')

def enhance_image(img):
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

# ---------------------- Main App ----------------------
class CamApp(App):
    def build(self):
        self.title = "AI Face Recognition (Optimized)"
        Window.size = (300, 450)

        self.folder = FolderStruct()
        self.SAVE_PATH = os.path.join(self.folder.INPUT_IMG, 'input_image.jpg')

        # --- Loading screen UI ---
        self.loading_main_label = Label(
            text="[b]Initializing[/b]",
            markup=True,
            font_size='26sp',
            halign='center',
            valign='middle',
            color=[1, 1, 1, 1],
            size_hint=(1, 0.5)
        )
        self.loading_main_label.bind(size=lambda inst, val: setattr(inst, 'text_size', inst.size))

        self.loading_detail_label = Label(
            text="Starting...",
            font_size='18sp',
            halign='center',
            valign='middle',
            color=[1, 1, 1, 1],
            size_hint=(1, 0.2)
        )
        self.loading_detail_label.bind(size=lambda inst, val: setattr(inst, 'text_size', inst.size))

        self.dots_label = Label(
            text="",
            font_size='28sp',
            halign='center',
            valign='middle',
            color=[1, 1, 1, 1],
            size_hint=(1, 0.3)
        )
        self.dots_label.bind(size=lambda inst, val: setattr(inst, 'text_size', inst.size))

        loading_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        loading_layout.add_widget(self.loading_main_label)
        loading_layout.add_widget(self.loading_detail_label)
        loading_layout.add_widget(self.dots_label)

        self.init_steps = [
            "Removing old embedding cache...",
            "Copying threshold config...",
            "Running generate_embedding_cache.py...",
            "Loading embeddings...",
            "Initializing camera...",
            "Initialization Complete."
        ]
        self.current_step = 0
        self.is_init_complete = False

        Clock.schedule_interval(self.update_loading_screen, 1.2)

        self.root = loading_layout
        return self.root

    def update_loading_screen(self, dt):
        if self.is_init_complete:
            Clock.unschedule(self.update_loading_screen)
            self.show_main_ui()
            return

        step_text = self.init_steps[self.current_step]
        self.loading_detail_label.text = step_text
        self.dots_label.text = "." * ((len(self.dots_label.text) % 3) + 1)

        if step_text == "Removing old embedding cache...":
            self.remove_embedding_cache()
        elif step_text == "Running generate_embedding_cache.py...":
            self.run_generate_embedding_cache()
        elif step_text == "Loading embeddings...":
            self.load_reference_embeddings()
        elif step_text == "Initializing camera...":
            self.initialize_camera()
        elif step_text == "Initialization Complete.":
            self.is_init_complete = True

        self.current_step += 1
        if self.current_step >= len(self.init_steps):
            self.current_step = 0

    def remove_embedding_cache(self):
        cache_file = os.path.join(self.folder.THRESHOLD, "embedding_cache.npy")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info("Old embedding cache removed.")

    def run_generate_embedding_cache(self):
        script_path = os.path.join(os.path.dirname(__file__), "generate_embedding_cache.py")
        if not os.path.exists(script_path):
            logger.error(f"generate_embedding_cache.py not found at {script_path}")
            return
        try:
            subprocess.run([sys.executable, script_path], check=True)
            logger.info("Embedding cache generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate embedding cache: {e}")

    def load_reference_embeddings(self):
        path = os.path.join(self.folder.THRESHOLD, "embedding_cache.npy")
        if os.path.exists(path):
            self.reference_embeddings = np.load(path, allow_pickle=True).item()
            logger.info(f"Embedding cache loaded with {len(self.reference_embeddings)} people.")
            for k in self.reference_embeddings:
                logger.info(f"Loaded: {k} — shape: {self.reference_embeddings[k].shape}")
        else:
            logger.warning("No embedding cache found.")
            self.reference_embeddings = {}

    def initialize_camera(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        logger.info("Camera initialized.")

    def show_main_ui(self):
        self.web_cam = Image(size_hint=(1, .7))
        self.verification_label = Label(
            text="[size=18]Ready for Detection[/size]",
            size_hint=(1, .3),
            font_size='24sp',
            markup=True,
            halign='center',
            valign='middle',
            color=[1, 1, 1, 1]
        )
        self.verification_label.bind(size=lambda inst, val: setattr(inst, 'text_size', inst.size))

        main_layout = BoxLayout(orientation='vertical')
        main_layout.add_widget(self.verification_label)
        main_layout.add_widget(self.web_cam)

        self.root.clear_widgets()
        self.root.add_widget(main_layout)

        config = self.read_config()
        self.threshold = config.getfloat("threshold", "detection_threshold", fallback=0.85)
        logger.info(f"Detection threshold set to: {self.threshold}")
        self.freeze_frame = None
        self.detection_enabled = True

        Clock.schedule_interval(self.update, 1.0 / 33.0)
        Clock.schedule_interval(self.check_face_detected, 1.0)

    def read_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.folder.THRESHOLD, "config.ini"))
        logger.info("Config loaded")
        return config

    def resource_path(self, relative):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative)
        return os.path.abspath(relative)

    def update(self, *args):
        if self.freeze_frame is not None:
            frame = self.freeze_frame.copy()
        else:
            ret, frame = self.capture.read()
            if not ret:
                return
            frame = frame[120:370, 200:450, :]  # Adjust if needed

        if self.detection_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        buf = cv2.flip(frame, 0).tobytes()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = tex

    def check_face_detected(self, dt):
        if not self.detection_enabled:
            return

        ret, frame = self.capture.read()
        if not ret:
            return
        frame = frame[120:370, 200:450, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            enhanced = enhance_image(frame)
            cv2.imwrite(self.SAVE_PATH, enhanced)
            logger.info(f"Face detected — input image saved to: {self.SAVE_PATH}")
            self.freeze_frame = frame.copy()
            self.detection_enabled = False
            self.verification_label.text = "[size=18]Checking...[/size]"
            self.verification_label.color = [1, 1, 0, 1]
            Clock.unschedule(self.check_face_detected)
            threading.Thread(target=self.verify, daemon=True).start()
            Clock.schedule_once(self.resume_check, 10)

    def resume_check(self, dt):
        self.verification_label.text = "[size=18]Ready for Detection[/size]"
        self.verification_label.color = [1, 1, 1, 1]
        self.freeze_frame = None
        self.detection_enabled = True
        Clock.schedule_interval(self.check_face_detected, 1.0)

    def verify(self):
        try:
            input_embedding = DeepFace.represent(
                img_path=self.SAVE_PATH,
                model_name="ArcFace",
                enforce_detection=False
            )[0]['embedding']
            input_embedding = np.array(input_embedding)
            input_embedding = input_embedding / np.linalg.norm(input_embedding)
        except Exception as e:
            logger.error(f"Failed to get input embedding: {e}")
            self.show_denied()
            return

        best_match, min_dist = None, float('inf')
        for person, embeddings in self.reference_embeddings.items():
            dists = np.linalg.norm(embeddings - input_embedding, axis=1)
            avg_dist = np.mean(dists)
            logger.info(f"Compared with: {person}, avg_dist: {avg_dist:.4f}")
            if avg_dist < self.threshold and avg_dist < min_dist:
                min_dist = avg_dist
                best_match = person

        if best_match:
            name = best_match.replace("_", " ").upper()
            self.verification_label.text = f"[size=18]Welcome[/size]\n[size=24][b]{name}[/b][/size]\n[size=18]Access granted[/size]"
            self.verification_label.color = [0, 1, 0, 1]
            logger.info(f"Access granted: {name} (distance={min_dist:.4f})")
        else:
            self.show_denied()
            logger.info("Access denied — no match within threshold.")

        threading.Thread(target=play_beep, daemon=True).start()
        Clock.schedule_once(lambda dt: self.fade_message(), 10)

    def show_denied(self):
        self.verification_label.text = '[b]ACCESS DENIED[/b]'
        self.verification_label.color = [1, 0, 0, 1]

    def fade_message(self):
        anim = Animation(opacity=0, duration=0.3)
        anim.bind(on_complete=lambda *a: setattr(self.verification_label, 'opacity', 1))
        anim.start(self.verification_label)


if __name__ == '__main__':
    logger.info("Launching app...")
    CamApp().run()
    logger.info("App closed.")