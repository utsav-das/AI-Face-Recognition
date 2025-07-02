import os
import cv2
import numpy as np
import imghdr
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.animation import Animation
import onnxruntime as ort

# FaceNet ONNX embedding extractor
class FaceNetONNX:
    def __init__(self, model_path="models/facenet.onnx"):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, face):
        img = cv2.resize(face, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, :, :, :]
        return img

    def embed(self, face):
        x = self.preprocess(face)
        emb = self.sess.run(None, {self.input_name: x})[0][0]
        return emb / np.linalg.norm(emb)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_face(input_face, model: FaceNetONNX, verification_dir, detection_threshold=0.35):
    input_emb = model.embed(input_face)
    best_match = None
    best_score = 0

    for person in os.listdir(verification_dir):
        person_dir = os.path.join(verification_dir, person)
        if not os.path.isdir(person_dir):
            continue

        sims = []
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            if not imghdr.what(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            emb = model.embed(img)
            sim = cosine_similarity(input_emb, emb)
            sims.append(sim)

        if not sims:
            continue

        avg_sim = np.mean(sims)
        print(f"[INFO   ] [Avg similarity for {person}] {avg_sim:.4f}")

        if avg_sim > best_score and avg_sim > detection_threshold:
            best_score = avg_sim
            best_match = person

    if best_match:
        print(f"[INFO   ] [Verified    ] True")
        return best_match.replace("_", " ").upper()
    else:
        print(f"[INFO   ] [Verified    ] False")
        return None


class FaceIDApp(App):
    def build(self):
        self.face_net = FaceNetONNX("models/facenet.onnx")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.image_widget = Image()
        self.status_label = Label(text="[b]Ready for face recognition[/b]", markup=True, font_size=20, size_hint=(1, 0.2))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image_widget)
        layout.add_widget(self.status_label)

        Clock.schedule_interval(self.update, 1/30)  # 30 FPS

        # Verification folder path
        self.verification_path = r"C:\Users\utsav\AppData\Local\Temp\data\verification_image"

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            # Take the first face only for simplicity
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]

            # Verify face
            match_name = verify_face(face_img, self.face_net, self.verification_path)

            if match_name:
                self.status_label.text = f"[b]Welcome {match_name}[/b]"
                self.status_label.color = (0, 1, 0, 1)
            else:
                self.status_label.text = "[b]ACCESS DENIED[/b]"
                self.status_label.color = (1, 0, 0, 1)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if match_name else (255, 0, 0), 3)
        else:
            self.status_label.text = "[b]No face detected[/b]"
            self.status_label.color = (1, 1, 1, 1)

        # Display webcam image in Kivy Image widget
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def on_stop(self):
        self.capture.release()


if __name__ == "__main__":
    FaceIDApp().run()
