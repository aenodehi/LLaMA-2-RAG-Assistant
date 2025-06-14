from PIL import Image
import pytesseract

# Set tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # adjust if it's somewhere else

image_path = "Folder/GS240261624572 - COP1CO.png"
text = pytesseract.image_to_string(Image.open(image_path))

print("\nExtracted OCR text:\n" + "-" * 40)
print(text)

