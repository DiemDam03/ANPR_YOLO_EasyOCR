import os
import cv2
from ultralytics import YOLO
import easyocr
import sqlite3
from datetime import datetime, timedelta
import re
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'anpr_model.pt')
DB_PATH = os.path.join(BASE_DIR, 'plates.db')

reader = easyocr.Reader(['en'])
model = YOLO(MODEL_PATH)

recent_detections = {}

def init_database(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def is_valid_vietnamese_plate(plate_text):
    if not plate_text:
        return False
    clean_plate = re.sub(r'\s+', '', plate_text.upper())
    
    patterns = [
        r'^[0-9]{2}[A-Z]-?[0-9]{4,5}$',
        r'^[0-9]{2}[A-Z]-?[0-9]{3}\.[0-9]{2}$',
        r'^[0-9]{2}[A-Z]{2}-?[0-9]{4,5}$',
        r'^[0-9]{2}[A-Z]{2}-?[0-9]{3}\.[0-9]{2}$',
        r'^[0-9]{2}-?[A-Z]{1,2}[0-9]{3}\.[0-9]{2}$',
        r'^[0-9]{2}-?[A-Z][0-9][0-9]{3}\.[0-9]{2}$'
    ]
    
    for pattern in patterns:
        if re.match(pattern, clean_plate):
            return True
    
    return False

def normalize_plate_text(plate_text):
    if not plate_text:
        return None
    normalized = re.sub(r'[^\w\.-]', '', plate_text.upper())
    normalized = re.sub(r'\s+', '', normalized)
    
    return normalized

def is_recent_detection(plate_text, time_window_seconds=10):
    current_time = datetime.now()    

    to_remove = []
    for plate, timestamp in recent_detections.items():
        if (current_time - timestamp).seconds > time_window_seconds:
            to_remove.append(plate)
    
    for plate in to_remove:
        del recent_detections[plate]

    if plate_text in recent_detections:
        time_diff = (current_time - recent_detections[plate_text]).seconds
        if time_diff <= time_window_seconds:
            return True
    
    recent_detections[plate_text] = current_time
    return False

def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

def extract_license_plate_text(plate_img):
    processed_img = preprocess_for_ocr(plate_img)
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(processed_img_rgb)

    texts = []
    for result in results:
        text, confidence = result[1], result[2]
        if confidence > 0.5:
            texts.append(text.strip())

    return ' '.join(texts) if texts else None

def save_to_database(cursor, conn, plate_text):
    timestamp = datetime.now().isoformat()
    cursor.execute('INSERT INTO plates (plate_text, timestamp) VALUES (?, ?)', (plate_text, timestamp))
    conn.commit()
    return timestamp

def process_frame(frame, model, cursor, conn):
    results = model(frame, stream=True, conf=0.50, iou=0.70, vid_stride=5, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate_img = frame[y1:y2, x1:x2]

            if cls_id == 0:  
                license_text = extract_license_plate_text(plate_img)
                if license_text:
                    
                    normalized_plate = normalize_plate_text(license_text)
                    
                    for idx, line in enumerate(license_text.split('\n')):
                        cv2.putText(frame, line, (x1, y1 - 30 - idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if is_valid_vietnamese_plate(normalized_plate):

                        if not is_recent_detection(normalized_plate, 10):
                            save_to_database(cursor, conn, normalized_plate)

                            cv2.putText(frame, "SAVED!", (x1, y1 - 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            cv2.putText(frame, "DUPLICATE", (x1, y1 - 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "INVALID FORMAT", (x1, y1 - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

def generate_frames():
    conn, cursor = init_database(DB_PATH)
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_frame(frame, model, cursor, conn)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    conn.close()