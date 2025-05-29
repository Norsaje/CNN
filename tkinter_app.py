import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import os
import json
import numpy as np
from collections import deque, Counter
from recognizer import extract_landmark_vector as extract_static_vector, predict_gesture
from recognizer1 import DynamicRecognizer

# Новая цветовая палитра
BG_COLOR = "#232946"         # Темно-синий
FG_COLOR = "#121629"         # Почти черный
TEXT_COLOR = "#eebbc3"       # Светло-розовый
ACCENT_COLOR = "#b8c1ec"     # Светло-фиолетовый
BUTTON_COLOR = "#eebbc3"     # Светло-розовый
BUTTON_TEXT = "#232946"      # Темно-синий
MODE_BG = "#393e46"          # Серый для режима

# Загрузка словаря жестов
try:
    with open("dictionary.json", "r", encoding="utf-8") as f:
        gesture_dict = json.load(f)
except FileNotFoundError:
    gesture_dict = {}

def draw_text(img, text, pos, font_path='arial.ttf', size=32, color=(0, 255, 0)):
    from PIL import ImageDraw, ImageFont
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Жестовый Переводчик (Tkinter)")
        self.running = True
        self.root.geometry("950x800")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_COLOR)

        # Main frame
        self.main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Верхняя панель с режимом и кнопкой выхода
        self.top_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.top_frame.pack(fill="x", pady=(0, 10))
        self.mode_var = tk.StringVar(value="СТАТИЧЕСКИЙ")
        self.mode_btn = tk.Button(self.top_frame, textvariable=self.mode_var, bg=MODE_BG, fg=ACCENT_COLOR, font=("Segoe UI", 14, "bold"), padx=16, pady=6, anchor="e", relief=tk.FLAT, bd=0, activebackground=ACCENT_COLOR, activeforeground=FG_COLOR, command=self.toggle_mode)
        self.mode_btn.pack(side="right", anchor="ne", padx=10)
        self.exit_btn = tk.Button(self.top_frame, text="Выйти", command=self.stop, bg="#ff2e63", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, bd=0, padx=12, pady=4, activebackground="#b3003c", activeforeground="white")
        self.exit_btn.pack(side="left", anchor="nw", padx=10)

        # Video label
        self.video_label = tk.Label(self.main_frame, bg=FG_COLOR, bd=0, relief=tk.FLAT)
        self.video_label.pack(pady=(10, 5), ipadx=2, ipady=2)
        self.video_label.config(width=880, height=500)

        # Распознанный текст
        tk.Label(self.main_frame, text="Распознанный текст:", fg=ACCENT_COLOR, bg=BG_COLOR, font=("Segoe UI", 13)).pack(pady=(10, 2))
        self.text_var = tk.StringVar()
        self.text_label = tk.Label(self.main_frame, textvariable=self.text_var, fg=TEXT_COLOR, bg=BG_COLOR, font=("Segoe UI", 22, "bold"))
        self.text_label.pack(pady=(0, 10))

        # Ввод слова и кнопка справа с надписью Uganda слева и разделителями
        separator1 = tk.Frame(self.main_frame, bg=ACCENT_COLOR, height=2)
        separator1.pack(fill="x", pady=(8, 4))
        input_row = tk.Frame(self.main_frame, bg=BG_COLOR)
        input_row.pack(pady=(0, 0))
        self.uganda_label = tk.Label(input_row, text="Uganda", fg=ACCENT_COLOR, bg=BG_COLOR, font=("Segoe UI", 13, "bold"))
        self.uganda_label.pack(side="left", padx=(0, 10))
        self.input_entry = tk.Entry(input_row, width=30, font=("Segoe UI", 16), bg=FG_COLOR, fg=TEXT_COLOR, insertbackground=ACCENT_COLOR, relief=tk.FLAT, highlightthickness=2, highlightcolor=ACCENT_COLOR)
        self.input_entry.pack(side="left", pady=0, padx=(0, 8))
        self.play_input_btn = tk.Button(input_row, text="Показать жест по слову", command=self.play_input, bg=BUTTON_COLOR, fg=BUTTON_TEXT, font=("Segoe UI", 13, "bold"), relief=tk.FLAT, bd=0, padx=18, pady=8, activebackground=ACCENT_COLOR, activeforeground=FG_COLOR, highlightthickness=0)
        self.play_input_btn.pack(side="left", pady=0)
        separator2 = tk.Frame(self.main_frame, bg=ACCENT_COLOR, height=2)
        separator2.pack(fill="x", pady=(4, 16))

        # Кнопки
        self.button_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.button_frame.pack(pady=(5, 20))
        self.show_gesture_btn = tk.Button(self.button_frame, text="Показать жесты", command=self.show_gesture, bg=BUTTON_COLOR, fg=BUTTON_TEXT, font=("Segoe UI", 13, "bold"), relief=tk.FLAT, bd=0, padx=18, pady=8, activebackground=ACCENT_COLOR, activeforeground=FG_COLOR, highlightthickness=0)
        self.show_gesture_btn.grid(row=0, column=0, padx=12)
        self.clear_btn = tk.Button(self.button_frame, text="Сброс", command=self.clear_text, bg=BUTTON_COLOR, fg=BUTTON_TEXT, font=("Segoe UI", 13, "bold"), relief=tk.FLAT, bd=0, padx=18, pady=8, activebackground=ACCENT_COLOR, activeforeground=FG_COLOR, highlightthickness=0)
        self.clear_btn.grid(row=0, column=1, padx=12)

        # Camera and recognizers
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

        # В самом низу — подпись о команде
        self.footer = tk.Label(self.main_frame, text="Проект команды Uganda", fg=ACCENT_COLOR, bg=BG_COLOR, font=("Segoe UI", 10, "italic"))
        self.footer.pack(side="bottom", pady=(16, 0))

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
        if self.mode == 'static':
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(out, hand, mp.solutions.hands.HAND_CONNECTIONS)
                if self.frame_count % 3 == 0:
                    vec = extract_static_vector(results.multi_hand_landmarks)
                    pred = predict_gesture(vec)
                    self.history.append(pred)
                    if self.history:
                        common = Counter(self.history).most_common(1)[0][0]
                        if common != self.last_static:
                            self.last_static = common
                            self.static_timer = 30
            if self.static_timer > 0 and self.last_static:
                self.text_var.set(self.last_static)
                self.static_timer -= 1
        elif self.mode == 'dynamic':
            # В динамическом режиме не рисуем скелет
            if self.frame_count % 2 == 0:
                pred = self.dynamic_recognizer.predict_dynamic(out)
                if pred:
                    self.last_dynamic = pred
                    self.dynamic_timer = 30
            if self.dynamic_timer > 0 and self.last_dynamic:
                self.text_var.set(self.last_dynamic)
                self.dynamic_timer -= 1
        # Обновление видео
        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        img = img.resize((880, 500))
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
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
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.root.after(30, play)
        play()

    def toggle_mode(self):
        if self.mode == 'static':
            self.mode = 'dynamic'
            self.mode_var.set("ДИНАМИЧЕСКИЙ")
        else:
            self.mode = 'static'
            self.mode_var.set("СТАТИЧЕСКИЙ")
        self.last_static = None
        self.static_timer = 0
        self.last_dynamic = None
        self.dynamic_timer = 0
        self.dynamic_recognizer.frames.clear()
        self.text_var.set("")

    def clear_text(self):
        self.input_entry.delete(0, "end")
        self.text_var.set("")

    def stop(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def play_input(self):
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
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.root.after(30, play)
        play()

if __name__ == "__main__":
    root = tk.Tk()
    GestureApp(root)
    root.mainloop() 