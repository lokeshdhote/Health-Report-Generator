from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import io

# ✅ Add your Tesseract executable path here
# Make sure this matches your installation location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"[ERROR] Failed to extract text from image: {e}"

def extract_text_from_pdf(pdf_file):
    try:
        if isinstance(pdf_file, bytes):
            pdf_stream = io.BytesIO(pdf_file)
        else:
            pdf_stream = pdf_file

        doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text.strip()
    except Exception as e:
        return f"[ERROR] Failed to extract text from PDF: {e}"
