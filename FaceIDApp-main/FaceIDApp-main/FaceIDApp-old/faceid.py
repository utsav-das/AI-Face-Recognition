from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.animation import Animation

import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import sys
import numpy as np
import configparser
import threading
import platform
import imghdr
import tempfile
import shutil


# Custom L1 Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# Folder structure
class folder_struct:
    def __init__(self):
        temp_dir = tempfile.gettempdir()
        self.VERIFICATION_IMAGE = os.path.join(temp_dir, 'data', 'verification_image')
        self.INPUT_IMAGE = os.path.join(temp_dir, 'data', 'input_image')
        self.SAVE_MODEL = os.path.join(temp_dir, 'data', 'save_model')
        self.THRESHOLD = os.path.join(temp_dir, 'data', 'threshold')

        os.makedirs(self.VERIFICATION_IMAGE, exist_ok=True)
        os.makedirs(self.INPUT_IMAGE, exist_ok=True)
        os.makedirs(self.SAVE_MODEL, exist_ok=True)
        os.makedirs(self.THRESHOLD, exist_ok=True)


# Beep function
def play_beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(800, 300)
    else:
        os.system('printf "\a"')


# Kivy App
class CamApp(App):
    def build(self):
        self.title = "AI Face Recognition"

        self.web_cam = Image(size_hint=(1, .7))
        self.verification_label = Label(
            text="[size=18]Ready for Detection[/size]",
            size_hint=(1, .3), font_size='24sp', markup=True,
            halign='center', valign='middle', color=[1, 1, 1, 1]
        )
        self.verification_label.bind(size=self.update_label_text_size)

        Window.size = (250, 400)
        Window.set_icon(self.resource_path("icon_highres.ico"))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.verification_label)
        layout.add_widget(self.web_cam)

        self.folder = folder_struct()
        self.copy_defaults_to_temp()

        self.config = self.read_config(os.path.join(self.folder.THRESHOLD, 'config.ini'))
        self.model_name = self.config.get("save_model", "model_name").strip('"')
        self.model = tf.keras.models.load_model(
            os.path.join(self.folder.SAVE_MODEL, self.model_name),
            custom_objects={'L1Dist': L1Dist}
        )

        self.cascade_path = self.resource_path("cv2/data/haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        if self.face_cascade.empty():
            raise IOError(f"Failed to load Haar cascade from {self.cascade_path}")

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.SAVE_PATH = os.path.join(self.folder.INPUT_IMAGE, 'input_image.jpg')

        Clock.schedule_interval(self.update, 1.0 / 33.0)
        Clock.schedule_interval(self.check_face_detected, 1.0)

        return layout

    def update_label_text_size(self, instance, value):
        instance.text_size = instance.size

    def copy_defaults_to_temp(self):
        model_src = self.resource_path("siamesemodel_latest.h5")
        config_src = self.resource_path("config.ini")

        model_dest = os.path.join(self.folder.SAVE_MODEL, "siamesemodel_latest.h5")
        config_dest = os.path.join(self.folder.THRESHOLD, "config.ini")

        if not os.path.exists(model_dest):
            shutil.copy(model_src, model_dest)
        if not os.path.exists(config_dest):
            shutil.copy(config_src, config_dest)

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
        return config

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
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
            return

        frame = frame[120:370, 200:450, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            cv2.imwrite(self.SAVE_PATH, frame)

            self.verification_label.text = "[size=18]Checking...[/size]"
            self.verification_label.color = [1, 1, 0, 1]

            Clock.unschedule(self.check_face_detected)
            Clock.schedule_once(lambda dt: self.verify(), 0.1)
            Clock.schedule_once(self.resume_face_detection, 10)

    def resume_face_detection(self, dt):
        self.verification_label.text = "[size=18]Ready for Detection[/size]"
        self.verification_label.color = [1, 1, 1, 1]
        Clock.schedule_interval(self.check_face_detected, 1.0)

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        return img / 255.0

    def verify(self):
        config = self.read_config(os.path.join(self.folder.THRESHOLD, 'config.ini'))
        detection_threshold = config.getfloat("threshold", "detection_threshold")
        verification_threshold = config.getfloat("threshold", "verification_threshold")

        verified = False
        matched_person = ""

        for sub_folder in os.listdir(self.folder.VERIFICATION_IMAGE):
            sub_path = os.path.join(self.folder.VERIFICATION_IMAGE, sub_folder)
            if not os.path.isdir(sub_path):
                continue

            results = []
            for img in os.listdir(sub_path):
                img_path = os.path.join(sub_path, img)
                if not os.path.isfile(img_path) or imghdr.what(img_path) is None:
                    continue

                input_img = self.preprocess(self.SAVE_PATH)
                validation_img = self.preprocess(img_path)
                result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
                results.append(result)

            if not results:
                continue

            detection = np.sum(np.array(results) > detection_threshold)
            verification = detection / len(results)
            verified = verification > verification_threshold

            if verified:
                matched_person = sub_folder.replace("_", " ").upper()
                break

        if verified:
            self.verification_label.text = (
                f"[size=18]Welcome[/size]\n"
                f"[size=24][b]{matched_person}[/b][/size]\n"
                f"[size=18]Access granted[/size]"
            )
            self.verification_label.color = [0, 1, 0, 1]
        else:
            self.verification_label.text = '[b]ACCESS DENIED[/b]'
            self.verification_label.color = [1, 0, 0, 1]

        threading.Thread(target=play_beep, daemon=True).start()

        anim = Animation(opacity=0, duration=0.3)
        anim.bind(on_complete=lambda *a: setattr(self.verification_label, 'opacity', 1))
        Clock.schedule_once(lambda dt: anim.start(self.verification_label), 10)

        Logger.info(f"Verified: {verified}")


if __name__ == '__main__':
    CamApp().run()