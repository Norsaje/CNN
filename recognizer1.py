import cv2
import numpy as np
import onnxruntime

class DynamicRecognizer:
    def __init__(self, model_path='S3D.onnx'):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.frames = []
        self.frame_skip = 1  # Пропускаем каждый второй кадр
        self.skip_counter = 0

        # Только нужные классы(пока что)
        self.class_map = {
            318: "досвидания",
            392: "з",
            434: "здравствуйте",
            470: "инвалид",
            492: "й",
            493: "к",
            1093: "привет",
            1351: "спасибо",
            1538: "ц",
            1580: "щ",
            1581: "ъ",
            1597: "ё"
        }

    def preprocess(self, frame):
        frame = cv2.resize(frame, (224, 224))
        return frame.astype(np.float32) / 255.0

    def predict_dynamic(self, frame):
        if self.skip_counter == 0:
            self.frames.append(self.preprocess(frame))
        self.skip_counter = (self.skip_counter + 1) % self.frame_skip

        if len(self.frames) < 32:
            return None

        self.frames = self.frames[-32:]
        clip = np.stack(self.frames, axis=0)                # (32, 224, 224, 3)
        clip = np.transpose(clip, (3, 0, 1, 2))              # (3, 32, 224, 224)
        clip = np.expand_dims(clip, axis=0)                 # (1, 3, 32, 224, 224)

        preds = self.session.run(None, {self.input_name: clip})[0][0]

        # Оставляем только нужные классы
        filtered_preds = {k: preds[k] for k in self.class_map.keys()}
        pred_class = max(filtered_preds, key=filtered_preds.get)
        confidence = filtered_preds[pred_class]

        if confidence >= 0.6:
            self.frames = []  # Очистка после успешного распознавания
            return self.class_map[pred_class]

        return None





