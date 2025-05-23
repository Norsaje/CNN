from capture import video

import cv2
import mediapipe as mp

# Инициализация распознавания рук
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = video.read()
    if not success:
        break

    # Перевод в RGB
    cameraRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Обработка изображения
    results = hands.process(cameraRGB)

    # Отрисовка
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Показываем видео
    cv2.imshow('Video', img)

    # Выход по клавише 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break