import cv2 as cv
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from common import CV_MODELS_DIR

model = YOLO(CV_MODELS_DIR / "YOLOv11" / "yolo11n.pt")

cap = cv.VideoCapture(0, cv.CAP_V4L2)
assert cap.isOpened()

track_history = defaultdict(lambda: [])

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
        if len(results) == 0:
            continue

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display annotated frame
        cv.imshow("YOLOv11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()