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
import tempfile
import numpy as np
import sqlite3

DB_PATH = os.path.join(tempfile.gettempdir(), "data", "profiles.db")


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
    """Return True if a profile with this ID already exists."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM profiles WHERE id = ? LIMIT 1", (id_val,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def insert_profile(id_val: str, name: str, id2_val: str) -> bool:
    """Insert a new profile. Returns True on success, False on failure."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO profiles (id, name, id2) VALUES (?, ?, ?)", (id_val, name, id2_val))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Shouldn't happen if profile_exists used, but handle defensively
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

        # NEW BEHAVIOR: check DB first — if ID exists, show error and DO NOT insert or capture
        if profile_exists(id_val):
            self.status_label.text = f"❌ Error: ID '{id_val}' already exists in database. Use a different ID."
            return

        # Insert profile (will fail if race condition creates same ID concurrently)
        ok = insert_profile(id_val, name_val, id2_val)
        if not ok:
            self.status_label.text = f"❌ Error: failed to insert profile for ID '{id_val}'."
            return

        temp_dir = tempfile.gettempdir()
        base_dir = os.path.join(temp_dir, "data", "verification_image")
        self.user_folder = os.path.join(base_dir, name_val)
        os.makedirs(self.user_folder, exist_ok=True)

        # Filename: NAME_(ID)_XX.jpg
        self.file_prefix = f"{name_val}_({id_val})"

        self.saving = True
        self.save_count = 0
        self.status_label.text = f"📷 Capturing photos for {name_val}..."

    def on_stop(self):
        if self.capture:
            self.capture.release()


if __name__ == '__main__':
    FaceCaptureApp().run()
