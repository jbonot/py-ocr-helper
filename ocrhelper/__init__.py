import cv2
import numpy as np
import pytesseract
from PIL import Image

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
