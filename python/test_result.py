import cv2
from ultralytics import YOLO

model = YOLO('best.pt').to('cuda')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # 1ms 대기 후 ESC 키로 종료 (키보드 입력 확인)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()