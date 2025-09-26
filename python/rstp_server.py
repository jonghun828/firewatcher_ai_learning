import cv2
import time

frame = None
video_capture= None
rtsp_url='rtsp://210.99.70.120:1935/live/cctv001.stream'
#rtsp_url='rtsp://127.0.0.1:8554/live'

def read_frame_with_timeout(video_capture, timeout=5):
    start_time = time.time()
    while True:
        ret, frame = video_capture.read()
        if ret:
            return frame

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Failed to read frame within {timeout} seconds.")

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(rtsp_url)
    while True:
        try:
            frame = read_frame_with_timeout(video_capture, timeout=5)
            cv2.imshow("Live Stream", frame)
        except TimeoutError as e:
            print(e)
            video_capture.release()
            video_capture = cv2.VideoCapture(rtsp_url)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    pass