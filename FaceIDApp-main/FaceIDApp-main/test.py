import os
import cv2
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
print("Cascade path exists:", os.path.exists(cascade_path))
print(cascade_path)