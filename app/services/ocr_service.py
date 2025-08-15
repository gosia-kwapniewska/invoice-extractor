import os
import re
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests
from json_repair import repair_json
from typing import Dict
from app.prompt.prompt import prompt

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def ocr_extract(file_path: str) -> str:
    """
    Extract text from a PDF or image via OCR.
    """
    text = ""
    if file_path.lower().endswith(".pdf"):
        pdf_doc = fitz.open(file_path)
        for page in pdf_doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img) + "\n"
    else:
        with Image.open(file_path) as img:
            text = pytesseract.image_to_string(img)
    return text.strip()


def llm_extract_text(text: str, model: str = "google/gemini-2.5-flash") -> Dict:
    """
    Send OCR-extracted text to OpenRouter/OpenAI LLM for structured JSON extraction.
    """
    if not text:
        return {"error": "No text to analyze."}

    complete_prompt = prompt + "Document text:" + text


    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": complete_prompt}]}],
    }

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    try:
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"]

        # Clean markdown code fences
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

        # Repair any broken JSON
        repaired = repair_json(cleaned)
        return json.loads(repaired)
    except Exception as e:
        return {"error": str(e)}


def ocr_and_structure(file_path: str, model: str = "google/gemini-2.5-flash") -> Dict:
    """
    High-level orchestrator: OCR a file and extract structured data via LLM.
    """
    text = ocr_extract(file_path)
    return llm_extract_text(text, model=model)
