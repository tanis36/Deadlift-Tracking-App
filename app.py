# Necessary dependencies
import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

from landmarks import landmarks

# Create a window that will display the video capture
window = tk.Tk()
window.geometry("480x700")
window.title("Deadlift Tracker")
ck.set_appearance_mode("dark")

# Label for the class/stage of the deadlift
class_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
class_label.place(x=10, y=1)
class_label.configure(text='STAGE')

# Label for the rep counter
counter_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counter_label.place(x=180, y=1)
counter_label.configure(text='REPS')

# Label for the probability that the stage is correct
prob_label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
prob_label.place(x=350, y=1)
prob_label.configure(text='PROB')

# Box that displays the current class/stage
class_box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
class_box.place(x=10, y=41)
class_box.configure(text='0')

# Box that displays the current rep count
counter_box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counter_box.place(x=180, y=41)
counter_box.configure(text='0')

# Box that displays the probability that the stage being displayed is correct
prob_box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
prob_box.place(x=350, y=41)
prob_box.configure(text='0')

# Function to reset the rep counter
def reset_counter():
    global counter
    counter = 0

# Button that calls reset_counter() to reset the rep count when clicked
button = ck.CTkButton(window, text="RESET", command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

# Frame where the video capture will appear
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Initialize mediapipe variables for drawing poses
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Open the pickle file containing the trained model
with open("deadlift.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize variables for current stage, rep counter, probability of body language, and class of body language
current_stage = ""
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ""

# Function to detect deadlifts
def detect():
    # Global variables
    global current_stage
    global counter
    global bodylang_prob
    global bodylang_class

    # Capture a frame from the webcam
    ret, frame = cap.read()
    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image to detect pose landmarks
    results = pose.process(image)
    # Draw detected pose landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    
    try:
        # Extract landmark coordinates and visibility data then flatten it into a single list
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        # Convert landmarks data into a DataFrame
        X = pd.DataFrame([row], columns = landmarks)
        # Predict the pose class and probability
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        # Check if the detected pose class is "down" with confidence above 0.7
        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"

        # Check if the pose transitions from "down" to "up" with confidence above 0.7 and increment the rep count
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1

    except Exception as e:
        pass

    # Take the first 460 columns of the image and convert to suitable format
    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)

    # Update the Tkinter label with the new image
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # Schedule the detect function to run again after 10 milliseconds, creating a loop
    lmain.after(10, detect)

    # Update the counter, probability, and class boxes with the current information
    counter_box.configure(text=counter)
    prob_box.configure(text=bodylang_prob[bodylang_prob.argmax()])
    class_box.configure(text=current_stage)

# Call the detect function
detect()
# Run the main loop
window.mainloop()