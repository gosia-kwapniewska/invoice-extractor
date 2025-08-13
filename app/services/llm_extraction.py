import io
import json
import re
import requests
import base64
import os
import fitz
from json_repair import repair_json
from PIL import Image
from openai import OpenAI


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
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

def llm_extract(file_path: str, model: str = "google/gemini-2.5-flash") -> dict:
    if file_path.lower().endswith(".pdf"):
        image_urls = pdf_to_data_urls(file_path, max_pages=5)
    else:
        # Fallback: treat as a single image
        with Image.open(file_path) as img:
            img.thumbnail((2000, 2000))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_urls = [f"data:image/jpeg;base64,{b64}"]



    prompt = """
    Extract the following fields from the provided document:
    consignor, consignee, country of origin, country of destination,
    HS Code, description of goods, means of transport, vessel.
    Respond in valid JSON. If info is missing, use null.
    """

    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": url})

    if 'openai' in model:
        completion = client.chat.completions.create(
            extra_body={},
            model="google/gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the image in 3 sentences max"
                        },
                        # {
                        #     "type": "input_image",
                        #     "image_url": f"data:image/jpeg;base64,{file_b64}"
                        # }
                        {
                            "type": "input_image",
                            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    ]
                }
            ]
        )
        return parse_structured_data(completion.choices[0].message.content)

    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}]
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        }

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers
        )

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            return {"error": f"{resp.status_code} - {resp.text}"}

        try:
            return parse_structured_data(resp.json()["choices"][0]["message"]["content"])
        except Exception as e:
            return {"error": str(e)}

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