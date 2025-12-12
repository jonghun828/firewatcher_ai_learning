import base64
import threading
import time
import cv2
import asyncio
import websockets
from ultralytics import YOLO
import json

model = YOLO('/home/kjonghun0828/firewatcher_ai_learning/python/best.pt').to('cuda')

camera_urls = {
    "cam0": "rtsp://firewatcher_cam:Kjonghun0828@192.168.43.114/stream2"
    "cam1": "rtsp://firewatcher_cam:Kjonghun0828@192.168.43.115/stream2"
}

camera_states = {
    cam_id: {
        "frame": None,
        "fire_start_time": None,
        "fire_alert_sent": False
    } for cam_id in camera_urls
}

locks = {name: threading.Lock() for name in camera_urls}


ALERT_WEBHOOK_URL = "http://localhost:8080/api/incident"
def send_fire_alert(cam_name, b64img):

    """화재 감지 시 HTTP POST 요청 보내기"""
    try:
        data = {
        "zone_id" : cam_name,
        "base64Img" : b64img,
        "isIncidentResolved" : False,
        "incidentType" : "fire"
    }
        response = requests.post(ALERT_WEBHOOK_URL, json=data, timeout=3)
        if response.status_code == 200:
            print(f"[{cam_name}] Fire alert sent successfully via HTTP POST")
        else:
            print(f"[{cam_name}] Failed to send alert (status: {response.status_code})")
    except Exception as e:
        print(f"[{cam_name}] HTTP POST failed: {e}")


def capture_loop(name, rtsp_url):
    cam_id = name
    cap = cv2.VideoCapture(rtsp_url)
    print(f"[{name}] Camera capture started")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{name}] Frame failed, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # YOLO 모델 감지
        if time.time() % 0.1 < 0.05:  # 10fps 정도로 YOLO 수행
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
        else:
            annotated = frame.copy()
            detected_classes = []

        # 화재 상태 업데이트
        with locks[name]:
            state = camera_states[cam_id]
            state["frame"] = annotated.copy()

            if '화염' in detected_classes:
                if state["fire_start_time"] is None:
                    state["fire_start_time"] = time.time()
            else:
                state["fire_start_time"] = None
                state["fire_alert_sent"] = False

        time.sleep(0.05)  # 약 20fps

    cap.release()


async def send_video(websocket, cam_name):
    print(f"[{cam_name}] Client connected")
    try:
        while True:
            with locks[cam_name]:
                state = camera_states[cam_name].copy()

            frame = state["frame"]
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            # 프레임 전송
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(json.dumps({
                "camera": cam_name,
                "type": "frame",
                "data": b64_frame
            }))

            # 화재 알림 조건 확인
            now = time.time()
            if (
                state["fire_start_time"]
                and (now - state["fire_start_time"] >= 5)  # 5초 이상 감지
                and not state["fire_alert_sent"]
            ):
                alert = {
                    "camera": cam_name,
                    "type": "alert",
                    "event": "fire_detected",
                    "duration": int(now - state["fire_start_time"])
                }
                await websocket.send(json.dumps(alert))
                camera_states[cam_name]["fire_alert_sent"] = True

            await asyncio.sleep(0.05)

    except websockets.ConnectionClosed:
        print(f"[{cam_name}] Client disconnected")


def run_websocket_server(cam_name, port):
    async def handler(websocket):
        """websockets 15.x에서는 path 인자를 받지 않음"""
        await send_video(websocket, cam_name)

    async def server():
        async with websockets.serve(handler, "0.0.0.0", port):
            print(f"[{cam_name}] WebSocket server running on port {port}")
            await asyncio.Future()  # keep running forever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server())


def main():
    # 각 카메라 캡처 스레드 실행
    for name, url in camera_urls.items():
        threading.Thread(target=capture_loop, args=(name, url), daemon=True).start()

    # 각 WebSocket 서버 실행
    for idx, name in enumerate(camera_urls.keys()):
        port = 8764 + idx
        threading.Thread(target=run_websocket_server, args=(name, port), daemon=True).start()

    # 메인 스레드에서 영상 출력 (디버깅용)
    try:
        while True:
            for name in camera_urls.keys():
                with locks[name]:
                    frame = camera_states[name]["frame"]
                if frame is not None:
                    cv2.imshow(name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
