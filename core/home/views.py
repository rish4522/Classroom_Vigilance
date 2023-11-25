import cv2
import threading
import numpy as np
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import time

global_cam = None
model = YOLO("best.pt")

def index(request):
   return render(request, 'index.html')

class WebCam(object):
   def __init__(self):
       self.video = cv2.VideoCapture(0)
       self.is_running = True
       threading.Thread(target=self.update, args=()).start()

   def __del__(self):
       self.is_running = False
       self.video.release()

   def get_frame(self):
       _, frame = self.video.read()
       if frame is not None:
           _, jpeg = cv2.imencode('.jpg', frame)
           return jpeg.tobytes()
       else:
           return b''

   def update(self):
       while self.is_running:
           _ = self.get_frame()

def detect_objects(cam):
  print("entering detect_objects")
  while True:
      frame = cam.get_frame()
      if frame:
          image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
          if image is not None:
              results = model.predict(source=image, save=False, save_txt=False)
              print("Number of detections: ", len(results))


def generate(cam):
   while True:
       yield cam.get_frame()

@gzip.gzip_page
def video_feed(request):
   return StreamingHttpResponse(generate(global_cam), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def start_detection(request):
   global global_cam

   if request.method == 'POST':
       # If global_cam is not initialized, create a new WebCam instance
       if not global_cam:
           global_cam = WebCam()

       # Run detect_objects in a separate thread to avoid blocking the server
       threading.Thread(target=detect_objects, args=(global_cam,), daemon=True).start()

       return JsonResponse({'status': 'Detection started successfully'})
   else:
       return JsonResponse({'status': 'Invalid request method'})
