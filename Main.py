import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['vi', 'en'])  
# Load the YOLO model
model = YOLO("./ver1/PyAI/First.pt")

# Open the video capture
cap = cv2.VideoCapture(0)


def preprocess_for_ocr(plate_img):
    # Resize 
    plate_img = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

    # Denoise
    plate_img = cv2.fastNlMeansDenoisingColored(plate_img)

    # Enhance 
    plate_img = cv2.detailEnhance(plate_img, sigma_s=10, sigma_r=0.15)

    # Grayscale 
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Threshold
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Blur
    # gray = cv2.medianBlur(gray, 3)

    return gray

def extract_license_plate_text(plate_img):

    # Preprocess the image
    processed_img = preprocess_for_ocr(plate_img)
    
    # Convert to RGB for EasyOCR
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

    results = reader.readtext(processed_img_rgb)
    # coord = results[0] 
    # text = results[1]
    # conf = results[2]

    if results:
        text, confidence = results[0][1], results[0][2]
        if confidence > 0.6:
            return text.strip()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, stream=True, conf=0.60, iou=0.70, vid_stride=2)

        # Visualize the results on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class ID
                cls_id = int(box.cls[0].item())

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract the region of the detected object
                plate_img = frame[y1:y2, x1:x2]
                
                # Attempt to read license plate text if the class matches license plate
                if cls_id == 0:  # 0 is the class ID for license plates
                    license_text = extract_license_plate_text(plate_img)
                    
                    # Display the license plate text
                    if license_text:
                        cv2.putText(frame, f"Plate: {license_text}", (x1, y1 - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("License Plate Detection & OCR", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()