import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO

# Ensure Tesseract is installed and configured
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
# Load the YOLO model
model = YOLO("./ver1/PyAI/First.pt")

# Open the video capture
cap = cv2.VideoCapture(0)

def preprocess_for_ocr(plate_img):
    """
    Preprocess the license plate image for better OCR accuracy
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Optional: Apply some denoising
    gray = cv2.medianBlur(gray, 3)
    
    return gray

def extract_license_plate_text(plate_img):
    """
    Extract text from a license plate image
    """
    # Preprocess the image
    processed_img = preprocess_for_ocr(plate_img)
    
    # Perform OCR
    try:
        # Configure Tesseract to focus on alphanumeric characters
        ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        license_text = pytesseract.image_to_string(processed_img, config=ocr_config).strip()
        return license_text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, stream=True, conf=0.60, iou=0.70, vid_stride=3)

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
                if cls_id == 0:  # Assuming 0 is the class ID for license plates
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
