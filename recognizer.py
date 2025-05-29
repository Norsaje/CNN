import numpy as np
import joblib

# Попытка загрузить модель из файла
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Внимание: файл model.pkl не найден. Классификация не будет работать.")
    model = None

def extract_landmark_vector(multi_hand_landmarks):
    def normalize(landmarks):
        # Нормализация координат относительно первой точки
        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z
        return [(lm.x - base_x, lm.y - base_y, lm.z - base_z) for lm in landmarks]

    # Инициализируем вектор признаков
    vector = []

    # Если нет данных о руках, возвращаем вектор нулей
    if multi_hand_landmarks is None:
        return np.zeros(126)

    # Ограничиваем количество обрабатываемых рук до двух
    hands = list(multi_hand_landmarks)[:2]

    for hand in hands:
        # Нормализуем координаты для каждой руки
        normalized_landmarks = normalize(hand.landmark)
        for x, y, z in normalized_landmarks:
            vector.extend([x, y, z])  # Добавляем координаты в вектор

    # Если обнаружена только одна рука, добавляем нули для второй
    if len(hands) == 1:
        vector.extend([0.0] * 63)

    return np.array(vector)  # Возвращаем вектор признаков как массив NumPy

def predict_gesture(feature_vector, threshold=0.7):
    # Проверяем, загружена ли модель
    if model is None:
        return "Модель не обучена"  # Возвращаем сообщение, если модель не загружена

    # Получаем вероятности для каждого класса
    probabilities = model.predict_proba([feature_vector])[0]
    max_probability = max(probabilities)  # Находим максимальную вероятность
    predicted_class = model.classes_[probabilities.argmax()]  # Определяем предсказанный класс

    # Проверяем, превышает ли максимальная вероятность заданный порог
    if max_probability < threshold:
        return None  # Если вероятность ниже порога, возвращаем None

    return predicted_class  # Возвращаем предсказанный класс
