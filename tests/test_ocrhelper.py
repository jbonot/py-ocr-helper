import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ocrhelper

input_dir = os.path.join(os.path.dirname(__file__), "data", "input")
processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
output_dir = os.path.join(os.path.dirname(__file__), "data", "output")

# Create output directories if they don't exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def test_save_processed_and_hocr():
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, file_name)
            processed_path = os.path.join(processed_dir, f"processed_{file_name}")
            output_hocr_path = os.path.join(
                output_dir, f"{os.path.splitext(file_name)[0]}.hocr"
            )

            binary_image = ocrhelper.process_image_from_path(input_path, processed_path)

            # Save processed image
            ocrhelper.save_processed_image(processed_path, binary_image)

            extracted_hocr = ocrhelper.get_hocr_from_image(
                binary_image,
                tesseract_params={
                    "lang": "nld+fra",
                    "config": r"--psm 6",
                },
            )

            # Save HOCR to a file
            with open(output_hocr_path, "w", encoding="utf-8") as hocr_file:
                hocr_file.write(extracted_hocr)
