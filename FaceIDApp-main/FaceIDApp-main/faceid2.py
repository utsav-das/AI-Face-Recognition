import cv2
import numpy as np
import os
import sys
import threading
import platform
import imghdr
import tempfile
import shutil
import configparser
from datetime import datetime
import logging

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.clock import Clock
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
class folder_struct:
    def __init__(self):
        self.VERIFICATION_IMAGE = os.path.join(temp_dir, 'data', 'verification_image')
        self.INPUT_IMAGE = os.path.join(temp_dir, 'data', 'input_image')
        self.THRESHOLD = os.path.join(temp_dir, 'data', 'threshold')

        os.makedirs(self.VERIFICATION_IMAGE, exist_ok=True)
        os.makedirs(self.INPUT_IMAGE, exist_ok=True)
        os.makedirs(self.THRESHOLD, exist_ok=True)


# ---------------------- Utility Functions ----------------------
def play_beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(800, 300)
    else:
        os.system('printf "\a"')


def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# ---------------------- Main App ----------------------
class CamApp(App):
    def build(self):
        self.title = "AI Face Recognition (DeepFace + Logging)"
        logger.info("Building UI...")

        self.web_cam = Image(size_hint=(1, .7))
        self.verification_label = Label(
            text="[size=18]Ready for Detection[/size]",
            size_hint=(1, .3), font_size='24sp', markup=True,
            halign='center', valign='middle', color=[1, 1, 1, 1]
        )
        self.verification_label.bind(size=self.update_label_text_size)

        Window.size = (300, 450)
        try:
            Window.set_icon(self.resource_path("icon_highres.ico"))
        except:
            logger.warning("icon_highres.ico not found, skipping icon set")

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.verification_label)
        layout.add_widget(self.web_cam)

        self.folder = folder_struct()
        self.copy_defaults_to_temp()

        self.config = self.read_config(os.path.join(self.folder.THRESHOLD, 'config.ini'))

        self.cascade_path = self.resource_path("cv2/data/haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.SAVE_PATH = os.path.join(self.folder.INPUT_IMAGE, 'input_image.jpg')

        Clock.schedule_interval(self.update, 1.0 / 33.0)
        Clock.schedule_interval(self.check_face_detected, 1.0)

        logger.info("Application started.")
        return layout

    def update_label_text_size(self, instance, value):
        instance.text_size = instance.size

    def copy_defaults_to_temp(self):
        config_src = self.resource_path("config.ini")
        config_dest = os.path.join(self.folder.THRESHOLD, "config.ini")
        if not os.path.exists(config_dest):
            shutil.copy(config_src, config_dest)
            logger.info(f"Copied config.ini to {config_dest}")

    def resource_path(self, relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        elif "haarcascade_frontalface_default.xml" in relative_path:
            return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        else:
            return os.path.abspath(relative_path)

    def read_config(self, file_path):
        config = configparser.ConfigParser()
        config.read(file_path)
        logger.info(f"Loaded config from {file_path}")
        return config

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            logger.warning("Camera frame not received")
            return

        frame = frame[120:370, 200:450, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def check_face_detected(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            logger.warning("No webcam frame")
            return

        frame = frame[120:370, 200:450, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            enhanced_frame = enhance_image(frame)
            cv2.imwrite(self.SAVE_PATH, enhanced_frame)

            self.verification_label.text = "[size=18]Checking...[/size]"
            self.verification_label.color = [1, 1, 0, 1]

            Clock.unschedule(self.check_face_detected)
            Clock.schedule_once(lambda dt: self.verify(), 0.2)
            Clock.schedule_once(self.resume_face_detection, 15)

    def resume_face_detection(self, dt):
        self.verification_label.text = "[size=18]Ready for Detection[/size]"
        self.verification_label.color = [1, 1, 1, 1]
        Clock.schedule_interval(self.check_face_detected, 1.0)

    def verify(self):
        threshold = self.config.getfloat("threshold", "detection_threshold", fallback=0.65)

        verified = False
        matched_person = ""
        input_img_path = self.SAVE_PATH

        try:
            input_embedding = DeepFace.represent(img_path=input_img_path, model_name='ArcFace', enforce_detection=False)[0]["embedding"]
            input_embedding = np.array(input_embedding)
            input_embedding = input_embedding / np.linalg.norm(input_embedding)
        except Exception as e:
            logger.error(f"Failed to extract input embedding: {e}")
            self.show_access_denied()
            return

        for sub_folder in os.listdir(self.folder.VERIFICATION_IMAGE):
            sub_path = os.path.join(self.folder.VERIFICATION_IMAGE, sub_folder)
            if not os.path.isdir(sub_path): continue

            distances = []
            for img_file in os.listdir(sub_path):
                img_path = os.path.join(sub_path, img_file)
                if not imghdr.what(img_path): continue

                try:
                    ref_embedding = DeepFace.represent(img_path=img_path, model_name='ArcFace', enforce_detection=False)[0]["embedding"]
                    ref_embedding = np.array(ref_embedding)
                    ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue

                dist = np.linalg.norm(input_embedding - ref_embedding)
                distances.append(dist)

            if not distances:
                continue

            avg_dist = np.mean(distances)
            logger.info(f"Avg distance for {sub_folder}: {avg_dist:.4f}")
            if avg_dist < threshold:
                verified = True
                matched_person = sub_folder.replace("_", " ").upper()
                break

        if verified:
            self.verification_label.text = f"[size=18]Welcome[/size]\n[size=24][b]{matched_person}[/b][/size]\n[size=18]Access granted[/size]"
            self.verification_label.color = [0, 1, 0, 1]
            logger.info(f"✅ Access granted: {matched_person}")
        else:
            self.show_access_denied()
            logger.info("❌ Access denied")

        threading.Thread(target=play_beep, daemon=True).start()

        Clock.schedule_once(lambda dt: self.fade_message(), 15)

    def fade_message(self):
        anim = Animation(opacity=0, duration=0.3)
        anim.bind(on_complete=lambda *a: setattr(self.verification_label, 'opacity', 1))
        anim.start(self.verification_label)

    def show_access_denied(self):
        self.verification_label.text = '[b]ACCESS DENIED[/b]'
        self.verification_label.color = [1, 0, 0, 1]


if __name__ == '__main__':
    logger.info("Launching app...")
    CamApp().run()
    logger.info("App closed.")
