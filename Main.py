import os
import cv2
from ultralytics import YOLO
import easyocr
import sqlite3
from datetime import datetime

# Get absolute base directory (where the script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Universal paths
MODEL_PATH = os.path.join(BASE_DIR, 'anpr_model.pt')
DB_PATH = os.path.join(BASE_DIR, 'plates.db')

# Initialize YOLO model and EasyOCR
reader = easyocr.Reader(['en'])
model = YOLO(MODEL_PATH)

# Initialize database connection and ensure table exists
def init_database(db_path):
    conn = sqlite3.connect(db_path)
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

# Preprocess image for OCR
def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

# Run OCR and extract license plate text
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

# Save license plate and timestamp to database
def save_to_database(cursor, conn, plate_text):
    timestamp = datetime.now().isoformat()
    cursor.execute('INSERT INTO plates (plate_text, timestamp) VALUES (?, ?)', (plate_text, timestamp))
    conn.commit()

# Process YOLO detection results on a single frame
def process_frame(frame, model, cursor, conn):
    results = model(frame, stream=True, conf=0.60, iou=0.70, vid_stride=5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate_img = frame[y1:y2, x1:x2]

            if cls_id == 0:  # Class 0 is license plate
                license_text = extract_license_plate_text(plate_img)
                if license_text:
                    for idx, line in enumerate(license_text.split('\n')):
                        cv2.putText(frame, line, (x1, y1 - 30 - idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  
                    save_to_database(cursor, conn, license_text)
    return frame

# Main function to run the video capture loop
def run_detection():
    conn, cursor = init_database(DB_PATH)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_frame(frame, model, cursor, conn)
        cv2.imshow("License Plate Detection & OCR", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

# Entry point
if __name__ == '__main__':
    run_detection()
