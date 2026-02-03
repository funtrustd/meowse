import cv2
import time
import math
import threading

import mediapipe as mp
import pyautogui

import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter.messagebox as tkmsg

from PIL import Image, ImageTk

# ================= BASIC CONFIG =================
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTHING = 0.85
PINCH_THRESHOLD = 0.35
FIST_THRESHOLD = 0.45
SCROLL_THRESHOLD = 0.22
SCROLL_SPEED = 40

running = False
show_video = False
sensitivity = 20

# ================= LICENSE =================
LICENSE_TEXT = """Hand Mouse Controller

Proprietary License

© 2026 Funtrustd (aka Cutie<3). All rights reserved.

This software is a proprietary hobby project.
Unauthorized copying, redistribution, or modification is prohibited.

Developer:
I Funtrustd (aka Cutie<3)

Discord:
https://discord.com/users/Funtrustd
"""

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def ui_safe(widget, func, **kwargs):
    widget.after(0, lambda: func(**kwargs))


def ui_image_safe(label, image):
    def update():
        label.configure(image=image)
        label.image = image
    label.after(0, update)


# ================= HAND TRACKING THREAD =================
def hand_tracking_loop(status_label, video_label):
    global running, sensitivity, show_video

    cap = cv2.VideoCapture(0)
    prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2
    mouse_down = False
    last_double = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark

                wrist = lm[0]
                index = lm[8]
                thumb = lm[4]
                middle = lm[12]

                hand_size = dist(wrist, lm[9]) + 1e-6

                target_x = index.x * SCREEN_W
                target_y = index.y * SCREEN_H

                dx = (target_x - prev_x) * (sensitivity / 10)
                dy = (target_y - prev_y) * (sensitivity / 10)

                cur_x = prev_x + dx * (1 - SMOOTHING)
                cur_y = prev_y + dy * (1 - SMOOTHING)

                pyautogui.moveTo(cur_x, cur_y)
                prev_x, prev_y = cur_x, cur_y

                pinch = dist(index, thumb) / hand_size
                if pinch < PINCH_THRESHOLD and not mouse_down:
                    pyautogui.mouseDown()
                    mouse_down = True
                elif pinch >= PINCH_THRESHOLD and mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False

                fist = dist(index, wrist) / hand_size
                if fist < FIST_THRESHOLD and time.time() - last_double > 1:
                    pyautogui.doubleClick()
                    last_double = time.time()

                scroll = dist(index, middle) / hand_size
                if scroll < SCROLL_THRESHOLD:
                    pyautogui.scroll(
                        SCROLL_SPEED if index.y < wrist.y else -SCROLL_SPEED
                    )

                ui_safe(status_label, status_label.config, text="Tracking")

                if show_video:
                    mp_draw.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS
                    )

                    img = ImageTk.PhotoImage(
                        Image.fromarray(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ).resize((320, 240))
                    )
                    ui_image_safe(video_label, img)

            else:
                ui_safe(status_label, status_label.config, text="No Hand")

            time.sleep(0.01)

    cap.release()
    ui_safe(status_label, status_label.config, text="Stopped")
    ui_image_safe(video_label, "")


# ================= GUI CALLBACKS =================
def start_tracking():
    global running
    if not running:
        running = True
        threading.Thread(
            target=hand_tracking_loop,
            args=(status_label, video_label),
            daemon=True
        ).start()


def stop_tracking():
    global running
    running = False


def toggle_camera():
    global show_video
    show_video = not show_video
    if not show_video:
        ui_image_safe(video_label, "")


def set_sensitivity(val):
    global sensitivity
    sensitivity = int(float(val))


def show_about():
    tkmsg.showinfo("About / License", LICENSE_TEXT)


def change_theme(theme):
    app.style.theme_use(theme)


# ================= GUI =================
app = tb.Window(themename="darkly")
app.title("Hand Mouse Controller")
app.geometry("520x640")
app.resizable(False, False)

tb.Label(
    app,
    text="Hand Mouse Controller",
    font=("Segoe UI", 18, "bold")
).pack(pady=12)

status_label = tb.Label(app, text="Idle", bootstyle=INFO)
status_label.pack()

video_label = tb.Label(app)
video_label.pack(pady=6)

controls = tb.Frame(app)
controls.pack(pady=10)

tb.Button(
    controls, text="Start",
    bootstyle=SUCCESS, padding=12,
    command=start_tracking
).grid(row=0, column=0, padx=6)

tb.Button(
    controls, text="Stop",
    bootstyle=DANGER, padding=12,
    command=stop_tracking
).grid(row=0, column=1, padx=6)

tb.Button(
    app,
    text="Toggle Camera Preview",
    bootstyle=SECONDARY,
    padding=10,
    command=toggle_camera
).pack(pady=6)

tb.Label(app, text="Mouse Sensitivity").pack(pady=(16, 0))
sens_slider = tb.Scale(
    app,
    from_=5,
    to=50,
    orient=HORIZONTAL,
    length=300,
    command=set_sensitivity
)
sens_slider.set(sensitivity)
sens_slider.pack()

theme_var = tb.StringVar(value="darkly")
tb.OptionMenu(
    app,
    theme_var,
    "darkly",
    "darkly",
    "flatly",
    "litera",
    "superhero",
    command=change_theme
).pack(pady=12)

tb.Button(
    app,
    text="About / License",
    bootstyle=LIGHT,
    padding=8,
    command=show_about
).pack(pady=12)

app.mainloop()
