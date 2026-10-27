import cv2
import numpy as np
import tkinter as tk
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import math
from PIL import Image, ImageTk
import customtkinter as ctk
import sys

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

def landmark_to_pixel(landmark, image_width, image_height):
    return int(landmark.x * image_width), int(landmark.y * image_height)

start_timer = None
pushup_count = 0
pushup_already_added = False
Pushups_Needed = None
start_vid = False
target_pushups = 10     # Default target
target_minutes = None       # Default timer interval
countdown_seconds = 0
timer_running = False
lock_reps = False
vid = cv2.VideoCapture(0)
time.sleep(1)
width, height = 800, 500

vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bg_color = "#313532"
panel_color = "#353a35"
panel_border = "#798777"
accent_color = "#9ca290"
text_color = "#d9d9c6"
button_color = "#748070"
entry_color = "#2f3430"

button_hover = "#95a18a"


def calculate_angle(a, b, c):
    """
    Calculates the inner angle at point B given three points A, B, and C.
    Points are MediaPipe landmarks (e.g., a = shoulder, b = elbow, c = wrist)
    """
    # 1. Get vectors from Elbow (b) to Shoulder (a) and Wrist (c)
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    
    # 2. Make sure we always get the inner angle (0 to 180)
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def hide_all_frames():
    input_frame.pack_forget()
    reps_frame.pack_forget()
    continue_frame.pack_forget()
    


def show_input_frame():
    hide_all_frames()
    input_frame.pack(anchor="n", pady=(60, 20), expand=True)


def show_reps_frame():
    hide_all_frames()
    global vid
    vid = cv2.VideoCapture(0)
    reps_frame.pack(anchor="n", pady=(20, 10), expand=True)
    pushup_text.pack(padx=20, pady=10)
    if not start_vid:
        StartRepsButton.pack(padx=20, pady=10)
    cam_widget.pack(padx=20, pady=10)


def show_continue_frame():
    hide_all_frames()
    continue_frame.pack(anchor="n", pady=(20, 10), expand=True)


def on_unmap(event):
    if lock_reps:
        app.after(50, app.deiconify)


def start_countdown():
    global start_timer, target_minutes, Pushups_Needed, countdown_seconds, timer_running, lock_reps, pushup_already_added

    start_timer = time.perf_counter()
    target_minutes = User_Stopwatch_input.get().strip()
    pushup_value = User_Pushup_input.get().strip()

    if not target_minutes or not pushup_value:
        show_input_frame()
        return

    
    target_minutes = int(target_minutes)
    Pushups_Needed = int(pushup_value)
    

    countdown_seconds = int(target_minutes * 60)
    timer_running = True
    lock_reps = False
    pushup_already_added = False
    cam_widget.configure(image=None)
    cam_widget.photo_image = None
    show_input_frame()
    timer_label.configure(text=f"Time Remaining: {target_minutes:02}:00")
    app.iconify()
    update_countdown()


def update_countdown():
    global countdown_seconds, timer_running
    if not timer_running:
        return

    if countdown_seconds <= 0:
        timer_label.configure(text="Time Remaining: 00:00")
        timer_running = False
        wake_up_screen()
        return

    mins, secs = divmod(countdown_seconds, 60)
    timer_label.configure(text=f"Time Remaining: {mins:02}:{secs:02}")
    countdown_seconds -= 1
    app.after(1000, update_countdown)


def wake_up_screen():
    global lock_reps
    
    lock_reps = True
    pushup_text.configure(text=f"Reps Done: 0/{Pushups_Needed}")
    show_reps_frame()

    app.deiconify()
    app.lift()
    app.attributes("-topmost", True)
    app.after(100, lambda: app.attributes("-topmost", False))


def open_camera():
    global start_vid
    pushup_text.configure(text=f"Reps Done: {pushup_count}/{Pushups_Needed}")
    show_reps_frame()
    StartRepsButton.pack_forget()
    if not start_vid:
        start_vid = True
        
        update_frame()


def on_close():
    if lock_reps:
        return
    app.quit()


def update_frame():
    global pushup_count, pushup_already_added, start_vid

    if not start_vid:
        return
    success, image = vid.read()
    if not success:
        
        cam_widget.after(10, update_frame)
        return
        
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    
    imlist = []
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h, w, _ = image.shape

    # 2. Pad the frame to a perfect square center
    max_dim = max(h, w)
    pad_vert = max_dim - h
    pad_horiz = max_dim - w
    top = pad_vert // 2
    bottom = pad_vert - top
    left = pad_horiz // 2
    right = pad_horiz - left

    square_image = cv2.copyMakeBorder(
        image,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    
    square_image = cv2.flip(square_image, 1)
    
  
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=square_image)
    timestamp_ms = int(time.time() * 1000)

    # Run detector on the full square image
    assert square_image.shape[0] == square_image.shape[1]
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    output_image = cv2.cvtColor(square_image, cv2.COLOR_RGB2BGR)

    if detection_result.pose_landmarks:
        first_person_landmarks = detection_result.pose_landmarks[0]
        h_s, w_s, _ = square_image.shape
        coords = lambda idx: landmark_to_pixel(first_person_landmarks[idx], w_s, h_s)

        left_shoulder = coords(11)
        right_shoulder = coords(12)
        left_elbow = coords(13)
        right_elbow = coords(14)
        left_wrist = coords(15)
        right_wrist = coords(16)
        left_hip = coords(23)
        right_hip = coords(24)

        #For drawing lines
        # Arms
        
        

        #calcuate angles
        left_ang = calculate_angle(first_person_landmarks[11], first_person_landmarks[13], first_person_landmarks[15])
        right_ang = calculate_angle(first_person_landmarks[12], first_person_landmarks[14], first_person_landmarks[16])
    

        

        if (left_ang < 90) and (right_ang < 90) and not pushup_already_added:
            pushup_already_added = True
        
        if (left_ang > 160) and (right_ang > 160):
            if pushup_already_added: 
                    pushup_count += 1
                    
                    pushup_text.configure(text=f"Reps Done: {pushup_count}/{Pushups_Needed}")
                    pushup_already_added = False
        if pushup_count >= Pushups_Needed:
            
            app.attributes("-topmost", False)
            pushup_text.configure(text="Goal reached! Press Continue to Restart Timer")
            pushup_text.pack_forget()
            
            start_vid = False
            cam_widget.pack_forget()
            cam_widget.configure(image=None)
            vid.release()
            pushup_count = 0
            
            
            show_continue_frame()
            
            return
    
    # Zoom the rendered output slightly so the camera fills the widget and the black edge crop disappears.
    display_width = cam_widget.winfo_width() or 500
    display_height = cam_widget.winfo_height() or 340
    src_h, src_w, _ = output_image.shape
    scale = max(display_width / src_w, display_height / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized_image = cv2.resize(output_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset = max((new_w - display_width) // 2, 0)
    y_offset = max((new_h - display_height) // 2, 0)
    cropped_image = resized_image[y_offset:y_offset + display_height, x_offset:x_offset + display_width]

    final_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(final_rgb)
    photo_image = ImageTk.PhotoImage(image=captured_image)

    cam_widget.photo_image = photo_image
    cam_widget.configure(image=photo_image)

    # Schedules the next frame update in 10ms without blocking Tkinter
    cam_widget.after(10, update_frame)











app = ctk.CTk()
app.bind('<Escape>', lambda e: on_close())
app.protocol("WM_DELETE_WINDOW", on_close)
app.bind('<Unmap>', on_unmap)
app.title("RepLock")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)
app.configure(fg_color=bg_color)
app.update_idletasks()
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
app.geometry(f"{screen_width}x{screen_height}+0+0")


content_frame = ctk.CTkFrame(app, fg_color=bg_color)
content_frame.grid(row=0, column=0, sticky="nsew")
content_frame.pack_propagate(False)

input_frame = ctk.CTkFrame(content_frame, corner_radius=12, fg_color=panel_color, border_width=2, border_color=panel_border)
reps_frame = ctk.CTkFrame(content_frame, corner_radius=12, fg_color=panel_color, border_width=2, border_color=panel_border)
continue_frame = ctk.CTkFrame(content_frame, corner_radius=12, fg_color=panel_color, border_width=2, border_color=panel_border)

title_label = ctk.CTkLabel(content_frame, text="RepLock", font=("Courier", 28, "bold"), text_color=accent_color, fg_color="transparent")
subtitle_label = ctk.CTkLabel(content_frame, text="Hey Couch Potato, Get Moving.", font=("Courier", 12), text_color=text_color, fg_color="transparent")
esc_hint = ctk.CTkLabel(content_frame, text="Press Esc to exit when idle.", font=("Courier", 10), text_color=text_color, fg_color="transparent")

pushup_text = ctk.CTkLabel(reps_frame, text=f"Reps Done: 0/{Pushups_Needed}", font=("Courier", 16), text_color=text_color, fg_color="transparent")

User_Stopwatch_input = ctk.CTkEntry(input_frame, placeholder_text="set clock in minutes", width=150, fg_color=entry_color, text_color=text_color, border_color=panel_border)
User_Pushup_input = ctk.CTkEntry(input_frame, placeholder_text="set reps required ie. 5", width=150, fg_color=entry_color, text_color=text_color, border_color=panel_border)
timer_button = ctk.CTkButton(input_frame, text="Start Timer", command=start_countdown, fg_color=button_color, hover_color=button_hover, text_color=text_color)
timer_label = ctk.CTkLabel(input_frame, text="Time Remaining: 00:00", font=("Courier", 14), text_color=text_color, fg_color="transparent")

StartRepsButton = ctk.CTkButton(reps_frame, text="Start Pushups", command=open_camera, fg_color=button_color, hover_color=button_hover, text_color=text_color)
cam_widget = ctk.CTkLabel(reps_frame, text="", fg_color=entry_color, width=500, height=340)
continue_message = ctk.CTkLabel(continue_frame, text="Goal reached! Ready for another round?", font=("Courier", 16), text_color=text_color, fg_color="transparent")
continue_button = ctk.CTkButton(continue_frame, text="CONTINUE", command=start_countdown, fg_color=button_color, hover_color=button_hover, text_color=text_color)

User_Stopwatch_input.grid(row=0, column=0, padx=10, pady=10)
User_Pushup_input.grid(row=0, column=1, padx=10, pady=10)
timer_button.grid(row=1, column=0, columnspan=2, pady=10)
timer_label.grid(row=2, column=0, columnspan=2, pady=(5, 15))

pushup_text.pack(padx=20, pady=10)
StartRepsButton.pack(padx=20, pady=10)
cam_widget.pack(padx=20, pady=10)

continue_message.pack(padx=20, pady=(20, 5))
continue_button.pack(padx=20, pady=(0, 20))

title_label.pack(pady=(30, 5))
subtitle_label.pack(pady=(0, 5))
esc_hint.pack(pady=(0, 20))

show_input_frame()

app.mainloop()
vid.release()
cv2.destroyAllWindows()

