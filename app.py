import os
import cv2
import base64
import threading
import time
import signal
import numpy as np
import pyttsx3
import face_recognition
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)


IMAGE_PATH = "./Images"
CSV_FILE = "Attendance.csv"
COOLDOWN_SECONDS = 10 


if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)


known_encodings = []
known_names = []
last_spoken_times = {}


def load_known_faces():
    print("SYSTEM: Encoding faces... Please wait.")
    for file in os.listdir(IMAGE_PATH):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img = face_recognition.load_image_file(f"{IMAGE_PATH}/{file}")
            encs = face_recognition.face_encodings(img)
            if len(encs) > 0:
                known_encodings.append(encs[0])
                known_names.append(os.path.splitext(file)[0].upper())
    print(f"SYSTEM: Loaded faces for: {known_names}")

load_known_faces()



def speak(text):
    """Voice feedback in a separate thread so it doesn't freeze the app."""
    def run_speech():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")
    
    thread = threading.Thread(target=run_speech, daemon=True)
    thread.start()

def mark_attendance(name):
    """Logs attendance to CSV if not already marked for today."""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write('Name,Date,Time\n')

    
    with open(CSV_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            entry = line.strip().split(',')
            if len(entry) >= 2:
                if entry[0] == name and entry[1] == date_str:
                    return False 
    
    
    with open(CSV_FILE, 'a') as f:
        f.write(f"{name},{date_str},{time_str}\n")
    return True



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives camera frame, performs face recognition, and updates status."""
    data = request.json['image']
    try:
        
        img_bytes = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"status": "Image Error"})

    
    face_locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    status = "Scanning..."
    current_time = time.time()

    for enc in encodings:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        
        if True in matches:
            name = known_names[matches.index(True)]
            last_time = last_spoken_times.get(name, 0)
            should_speak = (current_time - last_time) > COOLDOWN_SECONDS

            if mark_attendance(name):
                status = f"Success: {name}"
                if should_speak:
                    speak(f"Attendance marked for {name}")
                    last_spoken_times[name] = current_time
            else:
                status = f"{name} already marked"
                if should_speak:
                    speak(f"{name}, you are already present")
                    last_spoken_times[name] = current_time
        else:
            status = "Unknown Student"

    return jsonify({"status": status})

@app.route('/download')
def download_file():
    """Route to download the attendance report."""
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True)
    return "No records found.", 404

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Route to safely stop the server via the UI Quit button."""
    print("SYSTEM: Shutting down server...")
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"status": "Server stopped"})

if __name__ == '__main__':
    
    app.run(debug=True, port=5000)