import cv2
import mediapipe as mp
import csv

# Настройка детектора рук от MediaPipe
hands_module = mp.solutions.hands
hands_detector = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=2
)

# Запуск камеры
camera = cv2.VideoCapture(0)

# Запрашиваем у пользователя название жеста
gesture_label = input("Введите название жеста: ")
gesture_samples = []

while True:
    success, frame = camera.read()
    if not success:
        break

    # Зеркалим изображение для эффекта зеркала
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка кадра с помощью MediaPipe
    detection_results = hands_detector.process(rgb_frame)
    sample_row = []

    if detection_results.multi_hand_landmarks:
        hands_detected = detection_results.multi_hand_landmarks

        for hand_index in range(2):  # Максимум 2 руки
            if hand_index < len(hands_detected):
                landmarks = hands_detected[hand_index]
                base_x = landmarks.landmark[0].x
                base_y = landmarks.landmark[0].y
                base_z = landmarks.landmark[0].z

                # Сохраняем относительные координаты всех 21 точки
                for landmark in landmarks.landmark:
                    sample_row.append(landmark.x - base_x)
                    sample_row.append(landmark.y - base_y)
                    sample_row.append(landmark.z - base_z)
            else:
                # Если рук меньше двух, заполняем нулями
                sample_row.extend([0.0] * 63)

        sample_row.append(gesture_label)
        gesture_samples.append(sample_row)

    # Отображаем количество сохранённых жестов на экране
    cv2.putText(
        frame,
        f"Сохранил: {len(gesture_samples)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Сбор жестов", frame)

    # Выход по нажатию клавиши 'q' или при закрытии окна
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Сбор жестов", cv2.WND_PROP_VISIBLE) < 1:
        break

# Сохраняем данные в CSV с поддержкой русских символов
with open("dataset.csv", "a", newline="", encoding="utf-8") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(gesture_samples)

print(f"Сохранили {len(gesture_samples)} записей в dataset.csv")

# Освобождаем ресурсы
camera.release()
cv2.destroyAllWindows()
