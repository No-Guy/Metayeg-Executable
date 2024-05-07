from ultralytics import YOLO
import sys
import os
import cv2
model = YOLO(model=sys.argv[1],task='detect') 
path = sys.argv[2]
cwd = os.getcwd()
cap = cv2.VideoCapture(path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0
while frame_number < total_frames:
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, conf = float(sys.argv[3]), verbose = False)
        for r in results:
            for box in r.boxes:
                print(f"{int(box.cls.item())} {box.xywhn[0][0].item():.4f} {box.xywhn[0][1].item():.4f} {box.xywhn[0][2].item():.4f} {box.xywhn[0][3].item():.4f}\n")
    print("--")
    frame_number += 1