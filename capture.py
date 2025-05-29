import cv2
import mediapipe as mp
from recognizer import extract_landmark_vector as extract_static_vector, predict_gesture
from recognizer1 import DynamicRecognizer
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from collections import deque, Counter

def draw_cyrillic_text(image, text, position, font_path='arial.ttf', font_size=32, color=(0, 255, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

#Настройки
STATIC_DISPLAY_TIME = 30

#Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

#Камера
camera = cv2.VideoCapture(0)

#История и переменные
history_static = deque(maxlen=10)
history_dynamic = deque(maxlen=5)

frame_count = 0
mode = 'static'
static_display_counter = 0
dynamic_display_counter = 0
last_static_prediction = None
last_dynamic_prediction = None

#Динамический распознаватель
recognizer_dynamic = DynamicRecognizer()

while True:
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    display_frame = frame.copy()
    frame_count += 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #Статическое распознавание
        if mode == 'static' and frame_count % 3 == 0:
            vector = extract_static_vector(results.multi_hand_landmarks)
            prediction = predict_gesture(vector)
            history_static.append(prediction)

            most_common = Counter(history_static).most_common(1)[0][0]
            if most_common != last_static_prediction:
                last_static_prediction = most_common
                static_display_counter = STATIC_DISPLAY_TIME

        if static_display_counter > 0 and last_static_prediction:
            display_frame = draw_cyrillic_text(display_frame, f'Жест: {last_static_prediction}', (10, 40), font_size=40)
            static_display_counter -= 1

    #Динамический
    if mode == 'dynamic':
        prediction = recognizer_dynamic.predict_dynamic(frame)
        if prediction:
            history_dynamic.append(prediction)
            common_dynamic = Counter(history_dynamic).most_common(1)[0]
            if common_dynamic[1] >= 2 and common_dynamic[0] != last_dynamic_prediction:
                last_dynamic_prediction = common_dynamic[0]
                dynamic_display_counter = 30

        display_frame = draw_cyrillic_text(display_frame, "Считывание кадров...", (10, 80), font_size=28, color=(255, 255, 0))

    if dynamic_display_counter > 0 and last_dynamic_prediction:
        display_frame = draw_cyrillic_text(display_frame, f'Динамический: {last_dynamic_prediction}', (10, 120), font_size=40)
        dynamic_display_counter -= 1


    #Текущий режим
    mode_label = "СТАТИЧЕСКИЙ" if mode == "static" else "ДИНАМИЧЕСКИЙ"
    color = (0, 255, 0) if mode == "static" else (0, 128, 255)
    display_frame = draw_cyrillic_text(display_frame, f"Режим: {mode_label}", (10, 10), font_size=32, color=color)
    display_frame = draw_cyrillic_text(display_frame, 'Нажмите M (или Ь) для смены режима, Q — выход', (10, 440), font_size=24, color=(255, 255, 255))
    display_frame = cv2.copyMakeBorder(display_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)

    cv2.imshow('Распознавание жестов', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('m'), 0xFC, 0x044C]:  # 'm' или 'ь'
        mode = 'dynamic' if mode == 'static' else 'static'
        last_static_prediction = None
        last_dynamic_prediction = None
        static_display_counter = 0
        dynamic_display_counter = 0
        history_dynamic.clear()
        history_static.clear()

    if cv2.getWindowProperty('Распознавание жестов', cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
