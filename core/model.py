# Loading the YOLOv8 model
from ultralytics import YOLO
model = YOLO("best.pt")

# Detection Function
import cv2
import time

def detect_objects(rtsp_url):                                                       # Enter IP Camera URL
    cap = cv2.VideoCapture(rtsp_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, save_txt=False)

        # Save results to a .txt file every 30 seconds
        if int(time.time()) % 30 == 0:
            with open('detections.txt', 'a') as f:
                for result in results:
                    f.write(str(result) + '\n')

    cap.release()