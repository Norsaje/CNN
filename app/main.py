#!/usr/bin/env python3
import sys
from queue import Queue, Full, Empty
from threading import Thread, Event

import cv2
import mediapipe as mp
from PyQt6 import QtCore, QtGui, QtWidgets

# Импорт простого ResNet50-классификатора
from model_pytorch import ResNet50Classifier

# Параметры очереди и задержек
MAX_QUEUE_SIZE = 3
CAPTURE_FPS = 30


class CaptureThread(Thread):
    """Захват кадров из камеры в отдельном потоке."""
    def __init__(self, queue: Queue, stop_event: Event, device_index: int = 0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self.queue = queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            try:
                self.queue.put(frame, timeout=0.01)
            except Full:
                # если очередь полна — дропаем самый старый
                try:
                    _ = self.queue.get_nowait()
                    self.queue.put(frame, timeout=0.01)
                except Empty:
                    pass
        self.cap.release()


class DetectorThread(Thread):
    """Обработка кадров, отрисовка и классификация жестов."""
    def __init__(self, in_queue: Queue, out_queue: Queue, stop_event: Event):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event

        # Инициализация MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Инициализация ResNet50-классификатора
        import torchvision.transforms as T
        self.classifier = ResNet50Classifier(
            weights_path="best_resnet50.pth",
            device="cuda"
        )
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.in_queue.get(timeout=0.01)
            except Empty:
                continue

            # детектируем руку
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            label = None

            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    # отрисовываем скелет
                    self.mp_drawing.draw_landmarks(
                        frame, lm, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(thickness=2),
                    )

                    # вычисляем bounding box по ключевым точкам
                    h, w, _ = frame.shape
                    xs = [p.x for p in lm.landmark]
                    ys = [p.y for p in lm.landmark]
                    x1, x2 = int(min(xs) * w), int(max(xs) * w)
                    y1, y2 = int(min(ys) * h), int(max(ys) * h)

                    # обрезаем область руки
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # предобрабатываем и предсказываем
                    img_tensor = self.classifier.preprocess_frame(crop, self.transform)
                    label = self.classifier.predict(img_tensor)

            # выводим текст предсказания
            if label:
                cv2.putText(
                    frame, label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA
                )

            # отправляем кадр в UI
            try:
                self.out_queue.put(frame, timeout=0.01)
            except Full:
                pass

        self.hands.close()


class VideoWidget(QtWidgets.QLabel):
    """Qt-виджет для отображения OpenCV-кадров."""
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)

    @QtCore.pyqtSlot(object)
    def update_frame(self, frame):
        h, w, _ = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix)


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения."""
    def __init__(self, camera_idx: int = 0):
        super().__init__()
        self.setWindowTitle("Live Sign Detector")
        self.resize(700, 550)

        # очереди и событие остановки
        self.capture_queue = Queue(MAX_QUEUE_SIZE)
        self.process_queue = Queue(MAX_QUEUE_SIZE)
        self.stop_event = Event()

        # запускаем потоки
        self.capture_thread = CaptureThread(self.capture_queue, self.stop_event, camera_idx)
        self.detector_thread = DetectorThread(self.capture_queue, self.process_queue, self.stop_event)
        self.capture_thread.start()
        self.detector_thread.start()

        # видео-виджет
        self.video_widget = VideoWidget()
        self.setCentralWidget(self.video_widget)

        # таймер для обновления
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.pull_frame)
        self.timer.start(int(1000 / CAPTURE_FPS))

    def pull_frame(self):
        try:
            frame = self.process_queue.get_nowait()
        except Empty:
            return
        self.video_widget.update_frame(frame)

    def closeEvent(self, event):
        self.stop_event.set()
        self.capture_thread.join()
        self.detector_thread.join()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    window = MainWindow(camera_idx=cam_idx)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
