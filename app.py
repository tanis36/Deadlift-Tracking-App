import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks

window = tk.Tk()
window.geometry("480x700")
window.title("Deadlift Tracker")
ck.set_appearance_mode("dark")

class_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
class_label.place(x=10, y=1)
class_label.configure(text='STAGE')

counter_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counter_label.place(x=180, y=1)
counter_label.configure(text='REPS')

prob_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
prob_label.place(x=350, y=1)
prob_label.configure(text='PROB')

class_box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
class_box.place(x=10, y=41)
class_box.configure(text='0')

counter_box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counter_box.place(x=180, y=41)
counter_box.configure(text='0')

prob_box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
prob_box.place(x=350, y=41)
prob_box.configure(text='0')

window.mainloop()