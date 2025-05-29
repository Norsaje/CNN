import customtkinter as ctk
import cv2
import mediapipe as mp
import os
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import deque, Counter
from recognizer import extract_landmark_vector as extract_static_vector, predict_gesture
from recognizer1 import DynamicRecognizer
from customtkinter import CTkImage

# Загрузка словаря жестов
try:
    with open("dictionary.json", "r", encoding="utf-8") as f:
        gesture_dict = json.load(f)
except FileNotFoundError:
    gesture_dict = {}
    print("dictionary.json не найден — используется прямое имя файла.")


def draw_text(img, text, pos, font_path='arial.ttf', size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Жестовый Переводчик")
        self.running = True

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root.geometry("900x750")
        self.root.resizable(False, False)

        #Расположение
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(pady=(10, 5))
        self.video_label.configure(width=880, height=500)

        self.text_var = ctk.StringVar()
        ctk.CTkLabel(self.main_frame, text="Распознанный текст:", text_color="white").pack(pady=(5, 2))
        self.text_label = ctk.CTkLabel(self.main_frame, textvariable=self.text_var, font=("Segoe UI", 18))
        self.text_label.pack(pady=(0, 5))

        ctk.CTkLabel(self.main_frame, text="Введите слово:", text_color="white").pack(pady=(0, 2))
        self.input_entry = ctk.CTkEntry(self.main_frame, width=300)
        self.input_entry.pack(pady=(0, 10))

        #Кнопки
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(pady=(5, 10))

        ctk.CTkButton(self.button_frame, text="Показать жесты", command=self.show_gesture).grid(row=0, column=0, padx=10)
        ctk.CTkButton(self.button_frame, text="Сброс", command=self.clear_text).grid(row=0, column=1, padx=10)
        ctk.CTkButton(self.button_frame, text="Переключить режим", command=self.toggle_mode).grid(row=0, column=2, padx=10)
        ctk.CTkButton(self.button_frame, text="Стоп", command=self.stop).grid(row=0, column=3, padx=10)

        #Обработка камеры
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.dynamic_recognizer = DynamicRecognizer()

        self.mode = 'static'
        self.frame_count = 0
        self.history = deque(maxlen=10)
        self.last_static = None
        self.static_timer = 0
        self.last_dynamic = None
        self.dynamic_timer = 0

        self.update()


    def update(self):
        if not self.running:
            return

        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        out = frame.copy()
        self.frame_count += 1

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(out, hand, mp.solutions.hands.HAND_CONNECTIONS)

            if self.mode == 'static' and self.frame_count % 3 == 0:
                vec = extract_static_vector(results.multi_hand_landmarks)
                pred = predict_gesture(vec)
                self.history.append(pred)
                if self.history:
                    common = Counter(self.history).most_common(1)[0][0]
                    if common != self.last_static:
                        self.last_static = common
                        self.static_timer = 30

            elif self.mode == 'dynamic' and self.frame_count % 2 == 0:
                pred = self.dynamic_recognizer.predict_dynamic(out)
                if pred:
                    self.last_dynamic = pred
                    self.dynamic_timer = 30

        if self.mode == 'static' and self.static_timer > 0 and self.last_static:
            self.text_var.set(self.last_static)
            self.static_timer -= 1

        if self.mode == 'dynamic' and self.dynamic_timer > 0 and self.last_dynamic:
            self.text_var.set(self.last_dynamic)
            self.dynamic_timer -= 1

        mode_text = "СТАТИЧЕСКИЙ" if self.mode == 'static' else "ДИНАМИЧЕСКИЙ"
        out = draw_text(out, f"Режим: {mode_text}", (10, 10), size=28, color=(0, 200, 255))

        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        img = img.resize((880, 500))
        ctk_img = CTkImage(light_image=img, size=(880, 500))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img


        self.root.after(30, self.update)

    def show_gesture(self):
        word = self.input_entry.get().strip().lower()
        if not word:
            self.text_var.set("Введите слово")
            return

        filename = gesture_dict.get(word, f"{word}.mp4")
        path = os.path.join("templates", filename)

        if not os.path.exists(path):
            self.text_var.set(f"Файл не найден: {filename}")
            return

        self.text_var.set(f"Показ: {word}")
        self.running = False
        cap = cv2.VideoCapture(path)

        def play():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.running = True
                self.update()
                return

            frame = cv2.resize(frame, (880, 500))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = CTkImage(light_image=img, size=(880, 500))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

            self.root.after(30, play)

        play()

    def toggle_mode(self):
        self.mode = 'dynamic' if self.mode == 'static' else 'static'
        self.last_static = None
        self.static_timer = 0
        self.last_dynamic = None
        self.dynamic_timer = 0
        self.dynamic_recognizer.frames.clear()

    def clear_text(self):
        self.input_entry.delete(0, "end")
        self.text_var.set("")

    def stop(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app = ctk.CTk()
    GestureApp(app)
    app.mainloop()

