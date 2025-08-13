from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os

from app.services.llm_extraction import llm_extract
from app.services.ocr_service import ocr_extract_text


app = FastAPI(title="Invoice Extraction PoC")


@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...), mode: str = Form("llm"), model: str = "google/gemini-2.5-flash"):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if mode == "ocr":
            text = ocr_extract_text(tmp_path)
            return JSONResponse(content={"raw_text": text})
        elif mode == "llm":
            result = llm_extract(tmp_path, model)
            return JSONResponse(content={"structured_data": result})
        elif mode == "both":
            text = ocr_extract_text(tmp_path)
            result = llm_extract(tmp_path, model)
            return JSONResponse(content={"ocr_text": text, "structured_data": result})
        else:
            return JSONResponse(content={"error": "Invalid mode"}, status_code=400)
    finally:
        os.remove(tmp_path)
