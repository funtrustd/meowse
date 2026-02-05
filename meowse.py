import cv2
import time
import math
import threading
import urllib.request
import io

import mediapipe as mp
import pyautogui

import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter.messagebox as tkmsg
from tkinter import PhotoImage

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

# ================= ASSETS & THEME CONFIG =================
# Hello Kitty Pink Palette
PINK_DARK = "#FF69B4"   # Hot Pink
PINK_LIGHT = "#FFB7C5"  # Sakura Pink
PINK_BG = "#FFF0F5"     # Lavender Blush
TEXT_COLOR = "#8B008B"  # Dark Magenta

# Asset URLs from your README
URL_ICON = "https://cdn3.emoji.gg/emojis/2696-hellokitty-sparkle.png"
URL_BYE = "https://cdn3.emoji.gg/emojis/5349-hellokittybyebye.png"

# ================= LICENSE (OPEN SOURCE) =================
LICENSE_TEXT = """🎀 Meowse Controller v1.1.0 🎀

MIT License

© 2026 Funtrustd (aka Cutie<3).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

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
    try:
        widget.after(0, lambda: func(**kwargs))
    except:
        pass

def ui_image_safe(label, image):
    def update():
        try:
            label.configure(image=image)
            label.image = image
        except:
            pass
    try:
        label.after(0, update)
    except:
        pass

# ================= UTILS =================
def load_online_image(url, size=None):
    """Downloads an image from a URL and converts it to ImageTk format."""
    try:
        with urllib.request.urlopen(url) as u:
            raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        
        # Handle transparency for icons
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        if size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)
    except Exception as e:
        print(f"Failed to load image {url}: {e}")
        return None

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

                ui_safe(status_label, status_label.config, text="✨ Tracking Active ✨", bootstyle="danger")

                if show_video:
                    # Draw pink landmarks
                    mp_draw.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(255, 105, 180), thickness=2, circle_radius=2), # Pink dots
                        mp_draw.DrawingSpec(color=(255, 182, 193), thickness=2, circle_radius=2)  # Light pink lines
                    )

                    img = ImageTk.PhotoImage(
                        Image.fromarray(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ).resize((320, 240))
                    )
                    ui_image_safe(video_label, img)

            else:
                ui_safe(status_label, status_label.config, text="Waiting for Paws... 🐾", bootstyle="secondary")

            time.sleep(0.01)

    cap.release()
    ui_safe(status_label, status_label.config, text="Stopped 💤", bootstyle="secondary")
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
    tkmsg.showinfo("About Meowse 🎀", LICENSE_TEXT)

def change_theme(theme):
    app.style.theme_use(theme)
    # Re-apply pink styling hacks if needed after theme change
    style_widgets()

# ================= GUI SETUP =================
# Use 'cosmo' as base for a light, clean look, then we override with pink
app = tb.Window(themename="cosmo") 
app.title("🎀 Meowse Controller v1.1.0")
app.geometry("520x700")
app.resizable(False, False)

# --- Load Assets ---
print("Downloading Hello Kitty Assets... Please wait...")
icon_img = load_online_image(URL_ICON, (64, 64))
bye_img = load_online_image(URL_BYE, (100, 100))

# Set App Icon
if icon_img:
    app.iconphoto(False, icon_img)

# --- Styling Helper ---
def style_widgets():
    style = tb.Style()
    # Configure generic background (Note: ttkbootstrap themes handle most of this, 
    # but we force some colors where possible)
    
    # Customizing fonts
    style.configure('.', font=('Segoe UI', 10))
    style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground=PINK_DARK)
    style.configure('Subtitle.TLabel', font=('Segoe UI', 12, 'italic'), foreground=TEXT_COLOR)

style_widgets()

# --- Header Section ---
header_frame = tb.Frame(app)
header_frame.pack(pady=15)

# Try to put icon next to title
if icon_img:
    icon_label = tb.Label(header_frame, image=icon_img)
    icon_label.image = icon_img # Keep reference
    icon_label.pack(side=LEFT, padx=10)

title_label = tb.Label(
    header_frame,
    text="🎀 Meowse v1.1.0",
    style="Title.TLabel"
)
title_label.pack(side=LEFT)

# --- Main Content ---
status_label = tb.Label(app, text="Idle 💤", bootstyle="secondary", font=("Segoe UI", 12))
status_label.pack(pady=5)

video_label = tb.Label(app)
video_label.pack(pady=6)

# --- Controls ---
controls = tb.Frame(app)
controls.pack(pady=10)

# We use 'danger' bootstyle because it's usually Red/Pink in themes
btn_start = tb.Button(
    controls, text="✨ Start Magic",
    bootstyle="danger-outline", padding=12,
    command=start_tracking
)
btn_start.grid(row=0, column=0, padx=10)

btn_stop = tb.Button(
    controls, text="💤 Sleep",
    bootstyle="secondary-outline", padding=12,
    command=stop_tracking
)
btn_stop.grid(row=0, column=1, padx=10)

tb.Button(
    app,
    text="📷 Toggle Camera Mirror",
    bootstyle="info-link",
    command=toggle_camera
).pack(pady=5)

# --- Slider ---
tb.Label(app, text="✨ Magic Sensitivity ✨", foreground=PINK_DARK).pack(pady=(15, 0))
sens_slider = tb.Scale(
    app,
    from_=5,
    to=50,
    orient=HORIZONTAL,
    length=300,
    bootstyle="danger", # Pink slider
    command=set_sensitivity
)
sens_slider.set(sensitivity)
sens_slider.pack()

# --- Theme Selector (Keep functionality, but minimal) ---
theme_frame = tb.Frame(app)
theme_frame.pack(pady=15)
tb.Label(theme_frame, text="Theme:", foreground="gray").pack(side=LEFT, padx=5)

theme_var = tb.StringVar(value="cosmo")
menu = tb.OptionMenu(
    theme_frame,
    theme_var,
    "cosmo",
    "cosmo",
    "flatly",
    "minty",
    "pulse", # Purple/Pinkish
    command=change_theme
)
menu.pack(side=LEFT)

# --- Footer ---
footer_frame = tb.Frame(app)
footer_frame.pack(side=BOTTOM, pady=20)

if bye_img:
    bye_lbl = tb.Label(footer_frame, image=bye_img)
    bye_lbl.image = bye_img
    bye_lbl.pack()

tb.Button(
    footer_frame,
    text="🌸 About / License 🌸",
    bootstyle="link",
    command=show_about
).pack()

app.mainloop()