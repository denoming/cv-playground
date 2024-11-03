import cv2 as cv
from ultralytics import YOLO
from common import CV_MODELS_DIR, CV_DATA_DIR

model = YOLO(CV_MODELS_DIR / "YOLOv11" / "yolo11n.pt")

cap = cv.VideoCapture(CV_DATA_DIR / "cars-on-highway.mp4")
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False)
        frame = results[0].plot()
        cv.imshow("Tracking", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
