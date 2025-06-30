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
from layers import L1Dist
from folder_struct import folder_struct
import os
import sys
import numpy as np
import configparser
import threading
import platform
import imghdr


def play_beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(800, 300)
    else:
        os.system('printf "\a"')


class CamApp(App):
    def build(self):
        self.title = "AI Face Recognition"

        # Layout
        self.web_cam = Image(size_hint=(1, .7))
        self.verification_label = Label(text="", size_hint=(1, .1), font_size='24sp', markup=True)
        Window.size = (250, 400)
        Window.set_icon(self.resource_path("icon.png"))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.verification_label)
        layout.add_widget(self.web_cam)

        # Folder structure and model
        self.folder = folder_struct()
        self.model = tf.keras.models.load_model(
            os.path.join(self.folder.SAVE_MODEL, 'siamesemodel_latest.h5'),
            custom_objects={'L1Dist': L1Dist}
        )

        # Load face cascade
        self.cascade_path = self.resource_path("cv2/data/haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        if self.face_cascade.empty():
            raise IOError(f"Failed to load Haar cascade from {self.cascade_path}")

        # Open webcam (Windows-safe)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.SAVE_PATH = os.path.join(self.folder.INPUT_IMAGE, 'input_image.jpg')
        self.detected_faces = []

        Clock.schedule_interval(self.update, 1.0 / 33.0)
        Clock.schedule_interval(self.check_face_detected, 1.0)

        return layout

    def resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        else:
            # Special case for OpenCV cascade in dev mode
            if "haarcascade_frontalface_default.xml" in relative_path:
                return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            return os.path.join(os.path.abspath("."), relative_path)

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = frame[120:120 + 250, 200:200 + 250, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        self.detected_faces = faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

        self.detected_faces = []

    def check_face_detected(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = frame[120:120 + 250, 200:200 + 250, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            cv2.imwrite(self.SAVE_PATH, frame)
            self.verify()

            Clock.unschedule(self.check_face_detected)
            Clock.schedule_once(self.resume_face_detection, 10)

    def resume_face_detection(self, dt):
        self.verification_label.text = ""
        self.detected_faces = []
        Clock.schedule_interval(self.check_face_detected, 1.0)

    def read_config(self, file_path):
        config = configparser.ConfigParser()
        config.read(file_path)
        return config

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self, *args):
        config = self.read_config(os.path.join(self.folder.THRESHOLD, 'config.ini'))
        detection_threshold = config.getfloat("threshold", "detection_threshold")
        verification_threshold = config.getfloat("threshold", "verification_threshold")

        verified = False
        matched_person = ""

        for sub_folder in os.listdir(self.folder.VERIFICATION_IMAGE):
            sub_folder_path = os.path.join(self.folder.VERIFICATION_IMAGE, sub_folder)

            if not os.path.isdir(sub_folder_path):
                continue

            results = []

            for image in os.listdir(sub_folder_path):
                image_path = os.path.join(sub_folder_path, image)
                if not os.path.isfile(image_path) or imghdr.what(image_path) is None:
                    continue

                input_img = self.preprocess(self.SAVE_PATH)
                validation_img = self.preprocess(image_path)
                result = self.model.predict(
                    list(np.expand_dims([input_img, validation_img], axis=1)),
                    verbose=0
                )
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
            self.verification_label.text = f"[b]{matched_person}[/b]"
            self.verification_label.color = [0, 1, 0, 1]
        else:
            self.verification_label.text = '[b]UNVERIFIED[/b]'
            self.verification_label.color = [1, 0, 0, 1]

        threading.Thread(target=play_beep, daemon=True).start()

        anim = Animation(opacity=0, duration=0.3)
        anim.bind(on_complete=lambda *a: setattr(self.verification_label, 'opacity', 1))
        Clock.schedule_once(lambda dt: anim.start(self.verification_label), 10)

        Logger.info(f"Detection: {detection}")
        Logger.info(f"Verification: {verification}")
        Logger.info(f"Verified: {verified}")

        return results, verified


if __name__ == '__main__':
    CamApp().run()