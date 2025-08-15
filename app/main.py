from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os

from app.services.llm_extraction import llm_extract
from app.services.ocr_service import ocr_and_structure
from app.schema.Response import Response


app = FastAPI(title="Invoice Extraction PoC")


@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...), mode: str = Form("llm"), model_ocr: str = Form("google/gemini-2.5-flash"), model_llm: str = Form("google/gemini-2.5-flash")):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if mode == "ocr":
            result, usage = ocr_and_structure(tmp_path, model_ocr)
            return Response(structured_data= result, model=model_ocr, method='ocr', usage=usage)
        elif mode == "llm":
            result, usage = llm_extract(tmp_path, model_llm)
            return Response(structured_data= result, model=model_llm, method='llm', usage=usage)
        elif mode == "both":
            text, usage_ocr = ocr_and_structure(tmp_path, model_ocr)
            result, usage_llm = llm_extract(tmp_path, model_llm)
            return {'OCR': Response(structured_data=text, model=model_ocr, method='ocr', usage=usage_ocr), 'LLM': Response(structured_data=result, model=model_llm, method='llm', usage=usage_llm)}
        else:
            return JSONResponse(content={"error": "Invalid mode"}, status_code=400)
    finally:
        os.remove(tmp_path)
