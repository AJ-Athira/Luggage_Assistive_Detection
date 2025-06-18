import cv2
from ultralytics import YOLO
import pyttsx3
import time

# ------------------- Setup -------------------

# Define luggage-related object labels
LUGGAGE_LABELS = {"suitcase", "backpack", "handbag", "briefcase"}

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Ask user for input type
choice = input("Enter 'live' for webcam or 'file' for video file: ").strip().lower()

if choice == 'file':
    path = input("Enter path to video file (e.g., luggage.mp4): ").strip()
    cap = cv2.VideoCapture(path)
else:
    cap = cv2.VideoCapture(0)

# Create a resizable OpenCV window
cv2.namedWindow("Luggage Detection", cv2.WINDOW_NORMAL)

# Track spoken objects
announced = set()
last_announcement_time = 0

# Set screen size for resizing
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# ------------------- Main Loop -------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in LUGGAGE_LABELS:
                color = (0, 255, 0)
                if label not in announced and time.time() - last_announcement_time > 3:
                    engine.say(f"{label} detected")
                    engine.runAndWait()
                    announced.add(label)
                    last_announcement_time = time.time()
            else:
                color = (0, 0, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------- Dynamic Resizing --------
    frame_height, frame_width = annotated_frame.shape[:2]
    scale_w = SCREEN_WIDTH / frame_width
    scale_h = SCREEN_HEIGHT / frame_height
    scale = min(scale_w, scale_h)

    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

    cv2.imshow("Luggage Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- Cleanup -------------------

cap.release()
cv2.destroyAllWindows()
