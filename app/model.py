import numpy as np
import mediapipe as mp
import pickle
from pathlib import Path
from typing import List

mp_hands = mp.solutions.hands

def extract_hand_features(landmarks: mp_hands.NormalizedLandmarkList) -> np.ndarray: # type: ignore
    """
    Преобразует 21 точки кисти в вектор признаков: 5 углов + 42 нормализованных координаты.
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float32)
    # нормализуем относительно запястья
    wrist = pts[0].copy()
    pts -= wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist

    # углы суставов пяти пальцев
    angles: List[float] = []
    finger_joints = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
    for a,b,c in finger_joints:
        v1 = pts[b] - pts[a]
        v2 = pts[c] - pts[b]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        angles.append(np.arccos(np.clip(cosang, -1.0, 1.0)))

    return np.concatenate([np.array(angles, dtype=np.float32), pts.flatten()])

def save_classifier(clf, path: str = "gesture_classifier.pkl") -> None:
    """Сохранить обученный классификатор в файл."""
    with open(path, "wb") as f:
        pickle.dump(clf, f)

def load_classifier(path: str = "gesture_classifier.pkl"):
    """Загрузить классификатор из файла."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Classifier not found at {path}. Train it first.")
    with open(path, "rb") as f:
        return pickle.load(f)
