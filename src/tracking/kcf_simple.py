import cv2
import cv2 as cv
from common import CV_VIDEO_DIR

video = cv.VideoCapture(str(CV_VIDEO_DIR / 'race.mp4'))
ok, frame = video.read()
assert ok is True, \
    'Unable to read video frame'

bbox = cv.selectROI(frame)
print('Selected bbox:', bbox)

tracker = cv.TrackerKCF.create()
tracker.init(frame, bbox)
print(ok)

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = bbox
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Error', (100, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Tracking KCF', frame)
    if cv.waitKey(30) & 0xff == 27:  # Esc
        break
