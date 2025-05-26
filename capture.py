import cv2

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру!")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
