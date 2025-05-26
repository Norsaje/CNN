import PySimpleGUI as sg
import threading
import json
import cv2
import os
import capture
import recognizer
import generator
import traceback

if not os.path.exists("dictionary.json"):
    sg.popup_error("dictionary.json не найден!")
    exit(1)

with open("dictionary.json", "r", encoding="utf-8") as f:
    gesture_dict = json.load(f)
reverse_dict = {v.lower(): k for k, v in gesture_dict.items()}

sg.theme('LightGreen')

layout = [
    [sg.Text('Видеопоток:'), sg.Image(filename='', key='-IMAGE-')],
    [sg.Text('Распознанный жест:'), sg.Text('', size=(30,1), key='-GESTURE-')],
    [sg.Text('Набранный текст:'), sg.Multiline(size=(40,2), key='-TEXT-')],
    [sg.Text('Ввести слово:'), sg.Input(key='-INPUT-', size=(30,1)), sg.Button('Показать жесты')],
    [sg.Button('Сброс'), sg.Button('Стоп'), sg.Button('Выход')]
]

window = sg.Window('Жесты: распознавание и генерация', layout, return_keyboard_events=True, finalize=True)

try:
    cap = capture.open_camera()
except Exception as e:
    sg.popup_error(f"Ошибка камеры: {e}")
    exit(1)

try:
    onnx_model = recognizer.BukvaRecognizer(
    model_path="MobileNetV2_TSM.onnx",
    class_map_path="classes.txt"
)

except Exception as e:
    sg.popup_error(str(e))
    cap.release()
    exit(1)

recognize_enabled = True
accumulated_text = ""
last_pred = None

def show_gesture_video(gesture_key):
    generator.play_gesture(gesture_key)

while True:
    event, values = window.read(timeout=10)
    if event in (sg.WIN_CLOSED, 'Выход'):
        break
    frame = capture.get_frame(cap)
    if frame is not None:
        try:
            pred = recognizer.recognize_gesture(frame, onnx_model)
            if pred:
                gesture_text = gesture_dict.get(pred, "")
                last_pred = pred
                window['-GESTURE-'].update(gesture_text)
            else:
                window['-GESTURE-'].update("Жест не распознан")
        except Exception as ex:
            window['-GESTURE-'].update("Ошибка распознавания")
            print(traceback.format_exc())
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)
    if (event == ' ' or event == 'Return') and last_pred:
        accumulated_text += gesture_dict.get(last_pred, "") + " "
        window['-TEXT-'].update(accumulated_text)
    if event == 'Сброс':
        accumulated_text = ""
        window['-TEXT-'].update(accumulated_text)
    if event == 'Показать жесты':
        text = values['-INPUT-'].strip().lower()
        key = reverse_dict.get(text)
        if key:
            threading.Thread(target=show_gesture_video, args=(key,), daemon=True).start()
        else:
            sg.popup("Нет такого жеста в словаре!")
    if event == 'Стоп':
        cv2.destroyAllWindows()

cap.release()
window.close()
