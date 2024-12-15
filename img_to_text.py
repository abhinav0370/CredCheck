#img_to_text.py

from PIL import Image
import pytesseract
import os

# Specify the path to the Tesseract executable (needed for Windows users)
# It's recommended to set this via an environment variable for flexibility
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_from_image(image_path):
    try:
        # Open the image file
        img = Image.open(image_path)
        
        # Extract text from image
        extracted_text = pytesseract.image_to_string(img)
        
        return extracted_text.strip()
    except Exception as e:
        return f"Error: {e}"

