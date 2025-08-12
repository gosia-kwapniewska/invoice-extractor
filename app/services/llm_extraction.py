import requests
import base64
import os


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