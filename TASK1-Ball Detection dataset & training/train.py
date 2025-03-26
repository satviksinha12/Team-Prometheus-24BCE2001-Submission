#THis program file is strictly intended for training the model for program and shall not be modified until location change or saving
#models at custom location. Made by Satvik Sinha
from ultralytics import YOLO


model = YOLO("yolov8s.pt")

# Train
model.train(data="/home/satvik/Documents/objectdetection/resources/Robot.v20i.yolov8/data.yaml", epochs=50, imgsz=640)

