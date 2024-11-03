import cv2 as cv
from ultralytics import YOLO
from common import CV_MODELS_DIR

model = YOLO(CV_MODELS_DIR / "YOLOv11" / "yolo11n.pt")

cap = cv.VideoCapture(0, cv.CAP_V4L2)
assert cap.isOpened()

images_size = (800, 448)
if not cap.set(cv.CAP_PROP_FRAME_WIDTH, images_size[0]):
    print("Unable to set video frame width")
if not cap.set(cv.CAP_PROP_FRAME_HEIGHT, images_size[1]):
    print("Unable to set video frame height")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run tracking
        results = model.track(frame, stream=False, persist=True, imgsz=images_size, verbose=False)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display annotated frame
        cv.imshow("YOLOv11 Tracking", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()