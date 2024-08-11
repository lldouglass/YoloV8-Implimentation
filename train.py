from ultralytics import YOLO
import torch

torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.is_initialized() 

model = YOLO('yolov8s.pt')
model.to(device)
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=32

)
