import io
import requests
from PIL import Image
from PIL import ImageOps

def download_image(url, out, mode="RGB"):
    headers = {"User-Agent": "TF/1.0 (denoming@gmail.com)"}
    response = requests.get(url, headers=headers)
    data = io.BytesIO(response.content)
    pil_image = Image.open(data)
    pil_image = pil_image.convert(mode=mode)
    pil_image.save(out, format="JPEG", quality=90)
    return out

def download_image_and_resize(url, out, width, height):
    headers = {"User-Agent": "TF/1.0 (denoming@gmail.com)"}
    response = requests.get(url, headers=headers)
    data = io.BytesIO(response.content)
    pil_image = Image.open(data)
    pil_image = ImageOps.fit(pil_image, (width, height), Image.Resampling.LANCZOS)
    pil_image = pil_image.convert("RGB")
    pil_image.save(out, format="JPEG", quality=90)
    return out
