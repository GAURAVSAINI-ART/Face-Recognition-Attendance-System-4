import os
import cv2
import base64
import numpy as np
import face_recognition
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# --- CONFIGURATION ---
IMAGE_PATH = "./Images"
CSV_FILE = "Attendance.csv"
ADMIN_PASSWORD = "admin123" 
TOLERANCE = 0.55 

# Persistent storage for face data
known_encodings = []
known_names = []

def load_known_faces():
    """Initializes the system by encoding images in the folder."""
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
        print(f"SYSTEM: Created {IMAGE_PATH} folder.")
        return

    print("SYSTEM: Encoding facial signatures... please wait.")
    for file in os.listdir(IMAGE_PATH):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img = face_recognition.load_image_file(f"{IMAGE_PATH}/{file}")
                encs = face_recognition.face_encodings(img)
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    # Clean filename (e.g., "John_Doe.jpg" -> "JOHN DOE")
                    name = os.path.splitext(file)[0].replace("_", " ").upper()
                    known_names.append(name)
                    print(f"Success: Encoded {file}")
            except Exception as e:
                print(f"ERROR: {file}: {e}")
    
    print(f"SYSTEM: {len(known_names)} students loaded.")

load_known_faces()

def mark_attendance(name):
    """Writes to CSV only if not already marked today."""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w') as f:
            f.write('Name,Date,Time\n')

    # Read current entries to prevent double marking
    with open(CSV_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            entry = line.strip().split(',')
            if len(entry) >= 2 and entry[0] == name and entry[1] == date_str:
                return False 
    
    with open(CSV_FILE, 'a') as f:
        f.write(f"{name},{date_str},{time_str}\n")
    return True

@app.route('/')
def index():
    return render_template('index.html')

# --- THE FIX: LIVE COUNT ROUTE ---
@app.route('/get_count')
def get_count():
    """Returns the count of unique students marked today."""
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify({"count": 0})
            
        today = datetime.now().strftime('%Y-%m-%d')
        present_today = set() # Set handles uniqueness automatically
        
        with open(CSV_FILE, 'r') as f:
            lines = f.readlines()[1:] # Skip CSV header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1] == today:
                    present_today.add(parts[0]) # Add the name
                    
        return jsonify({"count": len(present_today)})
    except Exception as e:
        print(f"COUNT ERROR: {e}")
        return jsonify({"count": 0})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        img_bytes = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # Performance: Resize to 1/4 size
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        status = "Scanning..."
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    if mark_attendance(name):
                        status = f"Success: {name}"
                    else:
                        status = f"{name} already marked"
                else:
                    status = "Unknown Student"
        
        return jsonify({"status": status})
    except Exception as e:
        return jsonify({"status": "Error"}), 500

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    if request.json.get("password") == ADMIN_PASSWORD:
        with open(CSV_FILE, 'w') as f:
            f.write('Name,Date,Time\n')
        return jsonify({"status": "success"})
    return jsonify({"status": "wrong_password"}), 401

@app.route('/download')
def download_file():
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True)
    return "No records found.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)