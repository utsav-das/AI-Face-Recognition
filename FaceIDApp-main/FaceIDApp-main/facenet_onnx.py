import onnxruntime as ort
import numpy as np
import cv2

class FaceNetONNX:
    def __init__(self, model_path="models/facenet.onnx"):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def preprocess(self, face):
        img = cv2.resize(face, (112,112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None,:,:,:]
        return img

    def embed(self, face):
        x = self.preprocess(face)
        emb = self.sess.run(None, {self.inp: x})[0][0]
        return emb / np.linalg.norm(emb)
