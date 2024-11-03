import os
from pathlib import Path
from common import CV_WORKAREA_DIR, CV_DATA_DIR, CV_MODELS_DIR
from ultralytics import YOLO
from ultralytics.models.sam import Predictor
from PIL import Image
import numpy as np

OUTPUT_ROOT_PATH = CV_WORKAREA_DIR / "yolo" / "runs"
os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)

MODEL_FILE = CV_MODELS_DIR / "YOLOv11" / "yolo11n.pt"

model = YOLO(MODEL_FILE)

print("Before: ", model.device)
model.to("cuda")
print("After: ", model.device)

def on_predict_batch_end(predictor: Predictor):
    _, image, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)

# Save images of detected objects of each frames
index = 0
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, image in model.predict(source=CV_DATA_DIR / "video_street01.mp4",
                                   stream=True, predictor=None, save=False):
    prefix = OUTPUT_ROOT_PATH / f"frame_{index}"

    # Save annotated frame
    result.save(filename=f"{prefix}.jpg")

    # Convert BGR to RGB
    image = np.flip(image, axis=2)

    # Save origin frame
    im = Image.fromarray(image, mode="RGB")
    im.save(f"{prefix}_origin.jpg")

    index += 1
