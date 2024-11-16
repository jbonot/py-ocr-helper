import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab
import re

# Ensure Tesseract path is configured
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def process_image_from_path(image_path, resize_factor=1):
    img = Image.open(image_path)
    return process_image(img, resize_factor)


def process_image(image, resize_factor=1):
    processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    if resize_factor != 1:
        processed_image = cv2.resize(
            processed_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

    _, binary = cv2.threshold(
        processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def save_processed_image(processed_image_path, binary):
    cv2.imwrite(processed_image_path, binary)


def get_hocr_from_image(image, tesseract_params={}):
    hocr_bytes = pytesseract.image_to_pdf_or_hocr(
        image, extension="hocr", **tesseract_params
    )
    return hocr_bytes.decode("utf-8")


def read_text_at_position(bbox, should_process=False, tesseract_params={}):
    img = ImageGrab.grab(bbox)
    if should_process:
        img = process_image(img)
    return pytesseract.image_to_string(img, **tesseract_params)


def locate_text_at_position(
    target_text, bbox, should_process=False, tesseract_params={}
):
    matches = []
    img = ImageGrab.grab(bbox)
    if should_process:
        img = process_image(img)
    hocr_content = get_hocr_from_image(img, tesseract_params)

    # Search for target text in HOCR content using bounding box coordinates
    for match in re.finditer(
        r"bbox\s(\d+)\s(\d+)\s(\d+)\s(\d+).*?" + re.escape(target_text), hocr_content
    ):
        x1, y1, x2, y2 = map(int, match.groups())
        centerX = bbox["x"] + x1 + (x2 - x1) // 2
        centerY = bbox["y"] + y1 + (y2 - y1) // 2
        matches.append((centerX, centerY))

    return matches
