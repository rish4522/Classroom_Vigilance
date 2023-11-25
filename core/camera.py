import cv2
import os
import pandas as pd
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("C://Users//risha//OneDrive//Desktop//SP//core//best.pt")
# Run inference on a video stream
results = model.predict(source=0, stream=True)
# Filter detections by class name
sleeping_detections = []
for result in results:
    for cls in result.boxes.cls:
        if result.names[int(cls)] == 'sleeping':
            sleeping_detections.append(result)
