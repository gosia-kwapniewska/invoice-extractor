import io
import json
import re
from typing import Tuple, Dict

import requests
import base64
import os
import fitz
from json_repair import repair_json
from PIL import Image
from openai import OpenAI
from app.prompt.prompt import prompt


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def pdf_to_data_urls(pdf_path: str, max_pages: int = 5, dpi: int = 200) -> list:
    """Convert up to `max_pages` of a PDF to base64 data URLs for OpenRouter."""
    pdf_doc = fitz.open(pdf_path)
    urls = []

    for i, page in enumerate(pdf_doc):
        if i >= max_pages:
            break

        # Render page to pixmap
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Downscale large pages
        img.thumbnail((2000, 2000))

        # Save to JPEG for compression
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        urls.append(f"data:image/jpeg;base64,{b64}")

    return urls

def image_file_to_data_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL."""
    with Image.open(image_path) as img:
        img.thumbnail((2000, 2000))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

def parse_structured_data(data_str: str) -> dict:
    """Extracts JSON from Markdown code block, repairs it if broken, and returns dict."""
    # Remove markdown fences like ```json ... ```
    cleaned = re.sub(r"^```json\s*|\s*```$", "", data_str.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    # Try to repair JSON first
    repaired = repair_json(cleaned)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Still invalid JSON after repair: {e}\nContent: {repaired}")

def llm_extract(file_path: str, model: str = "google/gemini-2.5-flash") -> Tuple[Dict, Dict | None]:
    """
    Extract structured fields from a document using OpenRouter-compatible LLMs.
    Also returns token usage if available.
    """
    if file_path.lower().endswith(".pdf"):
        image_urls = pdf_to_data_urls(file_path, max_pages=5)
    else:
        image_urls = [image_file_to_data_url(file_path)]

    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": url} for url in image_urls
    ]

    # Decide method based on known OpenAI chat API compatibility
    if any(keyword in model.lower() for keyword in ["gpt", "claude", "gemini", "llama", "mistral"]):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            message_content = completion.choices[0].message.content
            parsed_data = parse_structured_data(message_content)

            # Include token usage if provided
            usage_info = getattr(completion, "usage", None)
            usage = {}
            if usage_info:
                usage = {
                    "prompt_tokens": usage_info.prompt_tokens,
                    "completion_tokens": usage_info.completion_tokens,
                    "total_tokens": usage_info.total_tokens
                }

            return parsed_data, usage if usage else None

        except Exception as e:
            return {"error": str(e)}, None

    # Fallback: direct API request
    else:
        payload = {"model": model, "messages": [{"role": "user", "content": content}]}
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

        resp = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=payload, headers=headers)

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            return {"error": f"{resp.status_code} - {resp.text}"}, None

        try:
            resp_json = resp.json()
            message_content = resp_json["choices"][0]["message"]["content"]
            parsed_data = parse_structured_data(message_content)

            # Include token usage if present
            usage = {}
            if "usage" in resp_json:
                usage = resp_json["usage"]

            return parsed_data, usage if usage else None

        except Exception as e:
            return {"error": str(e)}, None
