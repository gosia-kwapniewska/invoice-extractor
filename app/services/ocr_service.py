import os
import re
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests
from json_repair import repair_json
from typing import Dict, Tuple
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


def llm_extract_text(text: str, model: str = "google/gemini-2.5-flash") -> Tuple[Dict, Dict | None]:
    """
    Send OCR-extracted text to OpenRouter/OpenAI LLM for structured JSON extraction.
    Also returns token usage if available.
    """
    if not text:
        return {"error": "No text to analyze."}, None

    complete_prompt = prompt + "Document text:" + text

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": complete_prompt}]}],
    }

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    try:
        resp.raise_for_status()
        resp_json = resp.json()
        raw_content = resp_json["choices"][0]["message"]["content"]

        # Clean markdown code fences
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

        # Repair any broken JSON
        repaired = repair_json(cleaned)
        parsed_data = json.loads(repaired)

        # Include token usage if present
        usage = {}
        if "usage" in resp_json:
            usage = resp_json["usage"]

        return parsed_data, usage if usage else None

    except Exception as e:
        return {"error": str(e)}, None



def ocr_and_structure(file_path: str, model: str = "google/gemini-2.5-flash") -> Tuple[Dict, Dict | None]:
    """
    High-level orchestrator: OCR a file and extract structured data via LLM.
    """
    text = ocr_extract(file_path)
    return llm_extract_text(text, model=model)
