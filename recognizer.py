import os
import cv2
import numpy as np
import onnxruntime as ort

class BukvaRecognizer:
    def __init__(self, model_path="MobileNetV2_TSM.onnx", class_map_path="classes.txt"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        with open(class_map_path, encoding="utf-8") as f:
            self.class_map = {i: line.strip().split()[1] for i, line in enumerate(f)}

    def preprocess(self, frame):
        resized = cv2.resize(frame, (224, 224))
        input_data = resized.astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)  # CHW
        input_data = np.expand_dims(input_data, axis=0)  # NCHW
        return input_data

    def predict_from_frame(self, frame):
        input_tensor = self.preprocess(frame)
        input_tensor = np.expand_dims(input_tensor, axis=2)  # [1, 3, 1, 224, 224]
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        pred_class = int(np.argmax(output))
        return self.class_map.get(pred_class, "неизвестно")

def load_knn():
    return BukvaRecognizer()

def recognize_gesture(frame, model):
    return model.predict_from_frame(frame)
