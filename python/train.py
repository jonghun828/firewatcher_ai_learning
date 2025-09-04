from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='/home/kjonghun0828/firewatcher_ai_learning/python/train.yaml' , epochs=20)
# yolov8 학습 시작.