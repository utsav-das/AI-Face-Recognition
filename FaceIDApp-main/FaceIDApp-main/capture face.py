from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

import cv2
import os

class FaceCaptureApp(App):
    def build(self):
        self.capture = None
        self.image_widget = Image()
        self.name_input = TextInput(hint_text="Enter your name", size_hint=(1, 0.1), multiline=False)
        self.status_label = Label(text="Ready", size_hint=(1, 0.1))
        self.capture_btn = Button(text="Start Capture", size_hint=(1, 0.1))
        self.capture_btn.bind(on_press=self.start_capture)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image_widget)
        layout.add_widget(self.name_input)
        layout.add_widget(self.status_label)
        layout.add_widget(self.capture_btn)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        self.saving = False
        self.save_count = 0
        self.max_photos = 20
        self.user_folder = ""

        return layout

    def update_frame(self, dt):
        if not self.capture or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        self.frame = frame.copy()

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

        if self.saving and self.save_count < self.max_photos:
            self.save_face(frame)

    def enhance_image(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def save_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            size = max(w, h, 600)
            x1, y1 = max(0, cx - size // 2), max(0, cy - size // 2)
            x2, y2 = x1 + size, y1 + size

            face_img = frame[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (600, 600))
            face_img = self.enhance_image(face_img)

            save_path = os.path.join(self.user_folder, f"{self.user_name}_{self.save_count+1:02d}.jpg")
            cv2.imwrite(save_path, face_img)
            self.save_count += 1

            self.status_label.text = f"Saved {self.save_count}/{self.max_photos}"
            break  # Save only one face per frame

        if self.save_count >= self.max_photos:
            self.status_label.text = f"✅ Done! Saved to:\n{self.user_folder}"
            self.saving = False

    def start_capture(self, instance):
        self.user_name = self.name_input.text.strip().replace(" ", "_").upper()
        if not self.user_name:
            self.status_label.text = "Please enter a name"
            return

        base_dir = os.path.join(os.path.expanduser("~"), "verification_image")
        self.user_folder = os.path.join(base_dir, self.user_name)
        os.makedirs(self.user_folder, exist_ok=True)

        self.saving = True
        self.save_count = 0
        self.status_label.text = f"Capturing photos for {self.user_name}..."

    def on_stop(self):
        if self.capture:
            self.capture.release()

if __name__ == '__main__':
    FaceCaptureApp().run()
