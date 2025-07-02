import os, urllib.request

os.makedirs("models", exist_ok=True)
url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/face-recognition-resnet100-arcface-onnx/face-recognition-resnet100-arcface-onnx.onnx"
urllib.request.urlretrieve(url, "models/facenet.onnx")
print("Downloaded FaceNet ONNX model to models/facenet.onnx")