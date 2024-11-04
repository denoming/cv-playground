import cv2 as cv
import numpy as np
from ultralytics import YOLO
from common import CV_MODELS_DIR, CV_DATA_DIR
from collections import defaultdict

HISTORY_LENGTH = 30

def ccw(A, B, C) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D) -> bool:
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Open YOLO model
model = YOLO(CV_MODELS_DIR / "YOLOv11" / "yolo11l.pt")

# Setup video capture
cap = cv.VideoCapture(CV_DATA_DIR / "cars-on-highway.mp4")
assert cap.isOpened()

# The main line to check intersection with
p1, p2 = (210, 450), (980, 450)
# The history of object locations
history = defaultdict(lambda: [])
# The set of unique objects that intersects main line
objects = set()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Detect objects on frame (with persisting)
        results = model.track(frame, persist=True, verbose=False)
        if len(results) > 0:
            # Plot objects on image
            frame = results[0].plot()
            # Put the text with intersection counts
            frame = cv.putText(frame, str(len(objects)), (1200, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            # Put the main line
            frame = cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
            # Get object boxes
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                for box, id in zip(boxes, ids):
                    x, y, _, _ = box
                    track = history[id]
                    track.append((int(x), int(y)))
                    if len(track) > HISTORY_LENGTH:
                        track.pop(0)
                    # Draw object locations among 30 frames as polyline
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
                    # If we have at least two points check intersection with main line
                    if len(track) > 1:
                        p3, p4 = (track[-2][0], track[-2][1]), (track[-1][0], track[-1][1])
                        frame = cv.line(frame, p1, p2, (0, 0, 255), 2, cv.LINE_AA)
                        if intersect(p1, p2, p3, p4):
                            objects.add(id)

        cv.imshow("Tracking", frame)
        if cv.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
