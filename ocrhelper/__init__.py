import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab
import re

# Ensure Tesseract path is configured
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def process_image(image, color_mask=None, resize_factor=1):
    """
    Process the image with optional resizing and color masking.

    Parameters:
    - image: The input image to process.
    - color_mask: A tuple of two values (lowerb, upperb) for color masking (default None).
    - resize_factor: The factor by which to resize the image (default 1, no resizing).

    Returns:
    - The processed image.
    """
    processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if color_mask:
        if not (isinstance(color_mask, tuple) and len(color_mask) == 2):
            print(
                "Invalid color mask. It should be a tuple of length 2. Color mask will be ignored."
            )
        else:
            mask = cv2.inRange(
                processed_image, np.array(color_mask[0]), np.array(color_mask[1])
            )
            processed_image = cv2.bitwise_and(
                processed_image, processed_image, mask=mask
            )
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    if resize_factor != 1:
        processed_image = cv2.resize(
            processed_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

    _, binary = cv2.threshold(
        processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def process_image_from_path(image_path, color_mask=None, resize_factor=1):
    """
    Process an image from a file path with optional resizing and color masking.

    Parameters:
    - image: The input image to process.
    - color_mask: A tuple of two values (lowerb, upperb) for color masking (default None).
    - resize_factor: The factor by which to resize the image (default 1, no resizing).

    Returns:
    - The processed image.
    """
    img = Image.open(image_path)
    return process_image(img, resize_factor)


def save_processed_image(processed_image_path: str, binary: np.ndarray):
    """
    Saves the processed image (in binary format) to the specified file path.

    Parameters:
    - processed_image_path (str): The path where the processed image will be saved.
    - binary (np.ndarray): The binary image to be saved (processed image in NumPy array format).

    Returns:
    - None: This function saves the image to the specified location but does not return anything.
    """
    cv2.imwrite(processed_image_path, binary)


def get_hocr_from_image(image, tesseract_params: dict = {}) -> str:
    """
    Converts an image to HOCR (HTML-based OCR) format using Tesseract.

    Parameters:
    - image (Any): The input image to be processed.
    - tesseract_params (dict, optional): Additional parameters to be passed to Tesseract
      (default is an empty dictionary).

    Returns:
    - str: The HOCR output as a string, representing OCR results in HTML format.
    """
    hocr_bytes = pytesseract.image_to_pdf_or_hocr(
        image, extension="hocr", **tesseract_params
    )
    return hocr_bytes.decode("utf-8")


def read_text_in_bbox(
    bbox: tuple[int, int, int, int],
    should_process: bool = False,
    tesseract_params: dict = {},
) -> str:
    """
    Extracts text from an image within a specified bounding box using Tesseract OCR.

    Parameters:
    - bbox (tuple[int, int, int, int]): A tuple representing the bounding box coordinates
      in the form (x1, y1, x2, y2).
    - should_process (bool, optional): Whether to preprocess the image before OCR (default is False).
    - tesseract_params (dict, optional): Additional parameters for Tesseract OCR (default is an empty dictionary).

    Returns:
    - str: The text extracted from the image within the bounding box.
    """
    img = ImageGrab.grab(bbox)
    if should_process:
        img = process_image(img)
    return pytesseract.image_to_string(img, **tesseract_params)


def get_hocr_from_bbox(
    bbox: tuple[int, int, int, int],
    should_process: bool = False,
    tesseract_params: dict = {},
) -> str:
    """
    Extracts the HOCR (HTML-based OCR) from an image within a specified bounding box.

    Parameters:
    - bbox (tuple[int, int, int, int]): A tuple representing the bounding box coordinates
      in the form (x1, y1, x2, y2).
    - should_process (bool, optional): Whether to preprocess the image before OCR (default is False).
    - tesseract_params (dict, optional): Additional parameters for Tesseract OCR (default is an empty dictionary).

    Returns:
    - str: The HOCR output as a string, representing OCR results in HTML format.
    """
    img = ImageGrab.grab(bbox)
    if should_process:
        img = process_image(img)
    return get_hocr_from_image(img, tesseract_params)


def locate_text_within_bbox(
    target_text: str,
    bbox: tuple[int, int, int, int],
    should_process: bool = False,
    tesseract_params: dict = {},
) -> list:
    """
    Locates the positions of a specific target text within a bounding box in an image
    using HOCR output to extract coordinates.

    Parameters:
    - target_text (str): The text to search for within the bounding box.
    - bbox (tuple[int, int, int, int]): A tuple representing the bounding box coordinates
      in the form (x1, y1, x2, y2).
    - should_process (bool, optional): Whether to preprocess the image before OCR (default is False).
    - tesseract_params (dict, optional): Additional parameters for Tesseract OCR (default is an empty dictionary).

    Returns:
    - list: A list of tuples, each containing the (x, y) coordinates of the found text within the bounding box.
    """
    matches = []
    hocr_content = get_hocr_from_bbox(bbox, should_process, tesseract_params)

    # Search for target text in HOCR content using bounding box coordinates
    for match in re.finditer(
        r"bbox\s(\d+)\s(\d+)\s(\d+)\s(\d+).*?" + re.escape(target_text), hocr_content
    ):
        x1, y1, x2, y2 = map(int, match.groups())
        centerX = bbox[0] + x1 + (x2 - x1) // 2
        centerY = bbox[1] + y1 + (y2 - y1) // 2
        matches.append((centerX, centerY))

    return matches
