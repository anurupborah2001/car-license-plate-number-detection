import pytesseract as pt
import easyocr

pt.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def extract_text_from_image_using_tesseract(region_of_interest):
    return pt.image_to_string(region_of_interest)


def extract_text_from_image_using_easyocr(region_of_interest):
    reader = easyocr.Reader(["en"])
    results = reader.readtext(region_of_interest)
    for bbox, text, prob in results:
        print(f"Detected: {text} (Confidence: {prob:.2f})")
        return text
