# ===============================
# main.py — Unified Application
# ===============================

# ---------- Common Imports ----------
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
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

from deepface import DeepFace

# ===============================
# Section 1 — capture_face.py
# ===============================

DB_PATH = os.path.join(tempfile.gettempdir(), "data", "profiles.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    import sqlite3
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
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM profiles WHERE id = ? LIMIT 1", (id_val,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def insert_profile(id_val: str, name: str, id2_val: str) -> bool:
    import sqlite3
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


class FaceCaptureApp(App):
    def build(self):
        init_db()

        self.capture = None
        self.image_widget = Image()

        self.name_input = TextInput(hint_text="Enter Name", size_hint=(1, 0.08), multiline=False)
        self.id_input = TextInput(hint_text="Enter ID", size_hint=(1, 0.08), multiline=False)
        self.id2_input = TextInput(hint_text="Enter ID2", size_hint=(1, 0.08), multiline=False)

        self.status_label = Label(text="Ready", size_hint=(1, 0.1))
        self.capture_btn = Button(text="Start Capture", size_hint=(1, 0.1))
        self.capture_btn.bind(on_press=self.start_capture)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image_widget)
        layout.add_widget(self.name_input)
        layout.add_widget(self.id_input)
        layout.add_widget(self.id2_input)
        layout.add_widget(self.status_label)
        layout.add_widget(self.capture_btn)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        self.saving = False
        self.save_count = 0
        self.max_photos = 20
        self.user_folder = ""
        self.file_prefix = ""

        return layout

    def update_frame(self, dt):
        if not self.capture or not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.frame = frame.copy()
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture
        if self.saving and self.save_count < self.max_photos:
            self.save_face(self.frame, faces)

    def enhance_image(self, img):
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

    def is_sharp(self, image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > threshold

    def save_face(self, frame, faces):
        if len(faces) == 0:
            self.status_label.text = "No face detected"
            return
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (250, 250))
        face_img = self.enhance_image(face_img)
        if not self.is_sharp(face_img):
            self.status_label.text = "⏳ Waiting for sharp image..."
            return
        save_path = os.path.join(self.user_folder, f"{self.file_prefix}_{self.save_count+1:02d}.jpg")
        cv2.imwrite(save_path, face_img)
        self.save_count += 1
        self.status_label.text = f"Saved {self.save_count}/{self.max_photos}"
        if self.save_count >= self.max_photos:
            self.status_label.text = f"✅ Done! Saved to:\n{self.user_folder}"
            self.saving = False

    def start_capture(self, instance):
        name_val = self.name_input.text.strip().replace(" ", "_").upper()
        id_val = self.id_input.text.strip().upper()
        id2_val = self.id2_input.text.strip().upper()
        if not name_val or not id_val or not id2_val:
            self.status_label.text = "Please enter Name, ID, and ID2"
            return
        if profile_exists(id_val):
            self.status_label.text = f"❌ Error: ID '{id_val}' already exists in database. Use a different ID."
            return
        ok = insert_profile(id_val, name_val, id2_val)
        if not ok:
            self.status_label.text = f"❌ Error: failed to insert profile for ID '{id_val}'."
            return
        temp_dir = tempfile.gettempdir()
        base_dir = os.path.join(temp_dir, "data", "verification_image")
        self.user_folder = os.path.join(base_dir, name_val)
        os.makedirs(self.user_folder, exist_ok=True)
        self.file_prefix = f"{name_val}_({id_val})"
        self.saving = True
        self.save_count = 0
        self.status_label.text = f"📷 Capturing photos for {name_val}..."

    def on_stop(self):
        if self.capture:
            self.capture.release()


# ===============================
# Section 2 — faceid2.py
# ===============================

# ----- Logging Setup -----
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


class FolderStruct:
    def __init__(self):
        self.VERIF_IMG = os.path.join(temp_dir, 'data', 'verification_image')
        self.INPUT_IMG = os.path.join(temp_dir, 'data', 'input_image')
        self.THRESHOLD = os.path.join(temp_dir, 'data', 'threshold')
        for path in [self.VERIF_IMG, self.INPUT_IMG, self.THRESHOLD]:
            os.makedirs(path, exist_ok=True)


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
            frame = frame[120:370, 200:450, :]
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


# ===============================
# Section 3 — main_app.py
# ===============================

Window.size = (360, 640)
Window.clearcolor = (1, 1, 1, 1)

DARK_BLUE = (0/255, 51/255, 102/255, 1)
WHITE = (1, 1, 1, 1)


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        layout.add_widget(Widget(size_hint_y=None, height=100))
        btn_faceid = self.create_button('Face Recognition', self.goto_faceid)
        btn_capture = self.create_button('New Profile Creation', self.goto_capture)
        btn_exit = self.create_button('Exit App', lambda x: App.get_running_app().stop())
        layout.add_widget(btn_faceid)
        layout.add_widget(btn_capture)
        layout.add_widget(btn_exit)
        layout.add_widget(Widget(size_hint_y=None, height=100))
        self.add_widget(layout)

    def create_button(self, text, callback):
        btn = Button(
            text=text,
            size_hint=(1, None),
            height=80,
            background_normal='',
            background_color=DARK_BLUE,
            color=WHITE,
            font_name='Calibri',
            font_size='20sp'
        )
        btn.bind(on_press=callback)
        return btn

    def goto_faceid(self, instance):
        self.manager.current = 'faceid'
        Clock.schedule_once(lambda dt: self.manager.get_screen('faceid').start_app(), 0)

    def goto_capture(self, instance):
        self.manager.current = 'capture'
        Clock.schedule_once(lambda dt: self.manager.get_screen('capture').start_app(), 0)


class FaceIDScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded = False

    def start_app(self):
        if not self.loaded:
            self.faceid_app = CamApp()
            self.faceid_app.root = self.faceid_app.build()
            back_btn = Button(
                text="← Back to Home",
                size_hint=(1, 0.1),
                background_normal='',
                background_color=DARK_BLUE,
                color=WHITE,
                font_name='Calibri',
                font_size='18sp'
            )
            back_btn.bind(on_press=self.go_home)
            layout = BoxLayout(orientation='vertical')
            layout.add_widget(back_btn)
            layout.add_widget(self.faceid_app.root)
            self.add_widget(layout)
            self.loaded = True

    def go_home(self, instance):
        self.clear_widgets()
        self.loaded = False
        self.manager.current = 'home'


class CaptureScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded = False

    def start_app(self):
        if not self.loaded:
            self.capture_app = FaceCaptureApp()
            self.capture_app.root = self.capture_app.build()
            back_btn = Button(
                text="← Back to Home",
                size_hint=(1, 0.1),
                background_normal='',
                background_color=DARK_BLUE,
                color=WHITE,
                font_name='Calibri',
                font_size='18sp'
            )
            back_btn.bind(on_press=self.go_home)
            layout = BoxLayout(orientation='vertical')
            layout.add_widget(back_btn)
            layout.add_widget(self.capture_app.root)
            self.add_widget(layout)
            self.loaded = True

    def go_home(self, instance):
        if hasattr(self.capture_app, 'capture'):
            self.capture_app.capture.release()
        self.clear_widgets()
        self.loaded = False
        self.manager.current = 'home'


class MainApp(App):
    def build(self):
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(FaceIDScreen(name='faceid'))
        sm.add_widget(CaptureScreen(name='capture'))
        return sm


if __name__ == '__main__':
    MainApp().run()
