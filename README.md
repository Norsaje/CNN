# Uganda Sign Language Translator

Uganda — современное приложение для распознавания жестов и обучения Русскому жестовому языку (РЖЯ) с помощью камеры и интерактивного интерфейса на Python (Tkinter).

## О проекте

- Распознаёт статические и динамические жесты с помощью камеры.
- Позволяет проигрывать видео жеста по введённому слову или букве.
- Современный, удобный и красивый интерфейс.
- Поддержка более 1500 жестов РЖЯ.
- Работает на CPU, не требует мощной видеокарты.

**Проект команды Uganda.**

---

## Установка

```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
# или
source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

## Запуск

```bash
python tkinter_app.py
```

## Возможности

- Захват видео с камеры и распознавание жестов в реальном времени.
- Переключение между статическим и динамическим режимами.
- Ввод слова/буквы и просмотр жеста в формате mp4.
- Современный интерфейс с поддержкой тёмной темы.
- Быстрый выход, сброс, удобное управление.

## Структура

- `tkinter_app.py` — основной интерфейс Uganda.
- `recognizer.py` — статическое распознавание.
- `recognizer1.py` — динамическое распознавание.
- `templates/` — видеофайлы жестов.
- `dictionary.json` — словарь для сопоставления слов и видео.

## Скриншот

![Uganda GUI](screenshot.png)

---

## Лицензия

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.