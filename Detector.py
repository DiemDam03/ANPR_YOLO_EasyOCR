import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:/Users/buiph/Desktop/New PyAI/runs/detect/train/weights/best.pt")
# Open the video file
cap = cv2.VideoCapture(0)

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
                # Optionally, draw the class label
                cv2.putText(frame, str(cls_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

