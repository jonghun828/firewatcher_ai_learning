# import numpy as np
# import cv2
# from ultralytics import YOLO

# image = cv2.imread("/home/kjonghun0828/firewatcher_ai_learning/test_images_videos/mount_fire_image.jpg", cv2.IMREAD_COLOR)
# model = YOLO('best.pt').to('cuda')

# cv2.imshow("image", image)
# cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

# # 1ms 대기 후 ESC 키로 종료 (키보드 입력 확인)
# if cv2.waitKey(1) & 0xFF == 27: 
#     break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO('/home/kjonghun0828/firewatcher_ai_learning/python/runs/detect/train/weights/best.pt').to('cuda')
image = cv2.imread("/home/kjonghun0828/firewatcher_ai_learning/test_images_videos/mount_fire_image.jpg", cv2.IMREAD_COLOR)

while True:
    results = model(image)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

# 1ms 대기 후 ESC 키로 종료 (키보드 입력 확인)
    if cv2.waitKey(1) & 0xFF == 27: 
        break
        
cv2.destroyAllWindows()

