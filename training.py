from ultralytics import YOLO
model = YOLO('yolo11n-seg.pt')
model.train(data='/home/jacktran/project_2/dataset/data.yaml', epochs=100, imgsz=640, batch=16, device='cpu')