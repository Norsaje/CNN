import os
import cv2

def play_gesture(gesture_name):
    folder = "templates"
    supported = [".mp4", ".avi", ".gif"]
    file_path = None
    for ext in supported:
        candidate = os.path.join(folder, f"{gesture_name}{ext}")
        if os.path.exists(candidate):
            file_path = candidate
            break
    if not file_path:
        print(f"Файл жеста не найден: {gesture_name}")
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Ошибка при открытии файла {file_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f'Демонстрация жеста "{gesture_name}" (нажмите Q для выхода)', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
