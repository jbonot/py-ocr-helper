import cv2
import numpy as np
from lxml import etree
from PIL import Image, ImageGrab
import pytesseract

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
            processed_image,
            None,
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_CUBIC,
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


def get_hocr_from_image(image, tesseract_args: dict = {}) -> str:
    """
    Generate HOCR (HTML-based OCR) content from an image using Tesseract.

    Parameters:
    - image: The input image to process.
    - tesseract_args (dict, optional): A dictionary of additional arguments for Tesseract.
      Example: {'lang': 'eng'} (default: {}).

    Returns:
    - str: The HOCR content as a UTF-8 decoded string.
    """
    hocr_bytes = pytesseract.image_to_pdf_or_hocr(
        image, extension="hocr", **tesseract_args
    )
    return hocr_bytes.decode("utf-8")


def read_text_in_bbox(
    bbox: tuple[int, int, int, int],
    processing_args: dict | None = None,
    tesseract_args: dict | None = {},
) -> str:
    """
    Extract text from an image captured within a bounding box.

    Parameters:
    - bbox (tuple[int, int, int, int]): The bounding box (x1, y1, x2, y2) coordinates
      defining the region to capture the image.
    - processing_args (dict | None, optional): A dictionary of arguments for preprocessing
      the image before performing OCR. If None, no preprocessing is applied (default: None).
    - tesseract_args (dict | None, optional): A dictionary of additional arguments for Tesseract.
      Example: {'lang': 'eng'} (default: {}).

    Returns:
    - str: The text extracted from the image within the bounding box.
    """
    img = ImageGrab.grab(bbox)
    if processing_args:
        img = process_image(**processing_args)
    return pytesseract.image_to_string(img, **tesseract_args)


def get_hocr_from_bbox(
    bbox: tuple[int, int, int, int],
    processing_args: dict | None = None,
    tesseract_args: dict | None = {},
) -> str:
    """
    Generate HOCR (HTML-based OCR) content from an image captured within a bounding box.

    Parameters:
    - bbox (tuple[int, int, int, int]): The bounding box (x1, y1, x2, y2) coordinates
      defining the region to capture the image.
    - processing_args (dict | None, optional): A dictionary of arguments for preprocessing
      the image before performing OCR. If None, no preprocessing is applied (default: None).
    - tesseract_args (dict | None, optional): A dictionary of additional arguments for Tesseract.
      Example: {'lang': 'eng'} (default: {}).

    Returns:
    - str: The HOCR content as a UTF-8 decoded string.
    """
    img = ImageGrab.grab(bbox)
    if processing_args:
        img = process_image(**processing_args)
    return get_hocr_from_image(img, tesseract_args)


def locate_text_within_bbox(
    target_text: str,
    bbox: tuple[int, int, int, int],
    processing_args: dict | None = None,
    tesseract_args: dict | None = {},
) -> list[tuple[int, int]]:
    """
    Locate occurrences of a target text within a bounding box, returning the center coordinates of each occurance.

    Parameters:
    - target_text (str): The text to search for in the HOCR output.
    - bbox (tuple[int, int, int, int]): The bounding box (x1, y1, x2, y2) coordinates
      defining the region to capture the image.
    - processing_args (dict | None, optional): A dictionary of arguments for preprocessing
      the image before performing OCR. If None, no preprocessing is applied (default: None).
    - tesseract_args (dict | None, optional): A dictionary of additional arguments for Tesseract.
      Example: {'lang': 'eng'} (default: {}).

    Returns:
    - list[tuple[int, int]]: A list of tuples, each containing the rounded center coordinates
      (center_x, center_y) of the bounding box for each element matching the target text.
    """
    hocr_content = get_hocr_from_bbox(bbox, processing_args, tesseract_args)
    elements = etree.fromstring(hocr_content, etree.HTMLParser()).xpath(
        f'//*[contains(text(), "{target_text}")]'
    )
    results = []
    for element in elements:
        title_attr = element.get("title")
        if title_attr and "bbox" in title_attr:
            # Extract and process the bbox coordinates
            bbox_str = title_attr.split("bbox")[-1].split(";")[0].strip()
            bbox_coords = list(map(int, bbox_str.split()))
            center_x = round((bbox_coords[0] + bbox_coords[2]) / 2)
            center_y = round((bbox_coords[1] + bbox_coords[3]) / 2)
            results.append((center_x, center_y))

    return results
