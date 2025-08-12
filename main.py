from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import fitz  # PyMuPDF for PDF to image/text
from PIL import Image
import pytesseract
import requests
import base64

app = FastAPI(title="Invoice Extraction PoC")

# ------------------------
# Data Model
# ------------------------
class InvoiceData(BaseModel):
    consignor: str | None = None
    consignee: str | None = None
    country_of_origin: str | None = None
    country_of_destination: str | None = None
    hs_code: str | None = None
    description_of_goods: str | None = None
    means_of_transport: str | None = None
    vessel: str | None = None

# ------------------------
# OCR Helper
# ------------------------
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

# ------------------------
# LLM Multimodal Helper (OpenRouter + Gemini)
# ------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def llm_extract(file_path: str, model: str = "google/gemini-2.5-flash") -> dict:
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    file_b64 = base64.b64encode(file_bytes).decode()

    prompt = """
    Extract the following fields from the provided document:
    consignor, consignee, country of origin, country of destination,
    HS Code, description of goods, means of transport, vessel.
    Respond in valid JSON. If info is missing, use null.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Invoice Extractor"
    }

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": "You are an extraction assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_data": file_b64}
                ]
            }
        ]
    }

    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": str(e)}

# ------------------------
# Main Endpoint
# ------------------------
@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...), mode: str = Form("llm")):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if mode == "ocr":
            text = ocr_extract_text(tmp_path)
            return JSONResponse(content={"raw_text": text})
        elif mode == "llm":
            result = llm_extract(tmp_path)
            return JSONResponse(content={"structured_data": result})
        elif mode == "both":
            text = ocr_extract_text(tmp_path)
            result = llm_extract(tmp_path)
            return JSONResponse(content={"ocr_text": text, "structured_data": result})
        else:
            return JSONResponse(content={"error": "Invalid mode"}, status_code=400)
    finally:
        os.remove(tmp_path)
