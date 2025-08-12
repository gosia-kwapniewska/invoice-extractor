from PIL import Image
import fitz  # PyMuPDF for PDF to image/text
import pytesseract


def ocr_extract_text(file_path: str) -> str:
    text = ""
    if file_path.lower().endswith(".pdf"):
        pdf_doc = fitz.open(file_path)
        for page in pdf_doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"
    else:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
    return text.strip()