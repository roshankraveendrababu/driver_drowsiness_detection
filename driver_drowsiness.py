import cv2
from scipy.spatial import distance as dist
import winsound
import numpy as np

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Improved Eye Aspect Ratio function using bounding box dimensions
def eye_aspect_ratio(x, y, w, h):
    A = h  # Height of the bounding box (simulating eye opening)
    C = w  # Width of the bounding box
    return A / C

# Thresholds and constants
ear_threshold = 0.2  # Adjusted for better accuracy
consecutive_frames = 20
frequency = 1500
duration = 1000

# Initialize variables
count = 0
blinks = 0

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (required for Haar cascade detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        # Focus on face region to reduce false detections
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        roi_color = frame[fy:fy+fh, fx:fx+fw]

        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

        # Calculate EAR for each detected eye
        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_color[ey:ey+eh, ex:ex+ew]
            ear = eye_aspect_ratio(ex, ey, ew, eh)

            # Draw rectangle around detected eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Blink detection based on EAR threshold
            if ear < ear_threshold:
                count += 1
                if count >= consecutive_frames:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    winsound.Beep(frequency, duration)
                    blinks += 1
                    count = 0  # Reset after detecting a blink
            else:
                count = 0

            cv2.putText(frame, f"Blink Counter: {blinks}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display frame
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


