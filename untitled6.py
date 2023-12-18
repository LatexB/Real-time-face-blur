import cv2
import mediapipe as mp
import numpy as np

# Inicjalizacja
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Rozpoczęcie strumienia wideo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Detekcja twarzy
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Wyświetlanie obrazu przed rozmyciem
    cv2.imshow('Before Blur', frame.copy())

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
            # Stworzenie maski
            mask = np.zeros(face.shape, dtype=np.uint8)
            mask = cv2.ellipse(mask, (bbox[2]//2, bbox[3]//2), (bbox[2]//2, bbox[3]//2), 0, 0, 360, (255,255,255), -1)
            
            # Zastosowanie maski do twarzy przed rozmyciem
            face = cv2.bitwise_and(face, mask)
            
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = blurred_face

    # Wyświetlanie wyników
    cv2.imshow('MediaPipe Face Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()