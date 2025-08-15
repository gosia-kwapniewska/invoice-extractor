import csv
import json
import time
from typing import Dict, List

import pandas as pd
from pathlib import Path
from app.services.ocr_service import ocr_and_structure
from app.services.llm_extraction import llm_extract
from dotenv import load_dotenv

load_dotenv()
IMAGE_DIR = Path("test_images")

# Pricing dict (USD per milion tokens token)
TOKEN_PRICES = {
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "google/gemini-2.5-flash": {"prompt": 0.3, "completion": 2.5},
    "google/gemini-2.0-flash-lite-001": {"prompt": 0.075, "completion": 0.3},
    "google/gemini-2.5-flash-lite": {"prompt": 0.1, "completion": 0.4},
    "meta-llama/llama-3.2-11b-vision-instruct": {"prompt": 0.049, "completion": 0.049},
    "anthropic/claude-3.5-haiku": {"prompt": 0.8, "completion": 4},
    "qwen/qwen-2.5-vl-7b-instruct": {"prompt": 0.2, "completion": 0.2},

}

IMAGE_DIR = Path("test_images")

# Ground truth JSON for each image
GROUND_TRUTH = {
    "Invoice_Style1_2.pdf": {
        "Parties": [
            {
                "PartyName": "Fresh Harvest Exports",
                "Role": "Exporter",
                "Location": {
                    "City": "Nashik",
                    "Country": "India"
                }
            },
            {
                "PartyName": "Pacific Seafood Imports",
                "Role": "Consignee",
                "Location": {
                    "City": "Tokyo",
                    "Country": "Japan"
                }
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {
                "DescriptionOfGoods": "Cotton T-Shirts",
                "HSCode": "6109.10"
            }
        ],
        "Transportation": {
            "MeansOfTransport": None,
            "VesselNumber": None
        }
    },
    "Invoice_Style1_4.pdf": {
        "Parties": [
            {
                "PartyName": "TechnoMach Tools Pvt. Ltd.",
                "Role": "Exporter",
                "Location": {"City": "Pune", "Country": "India"}
            },
            {
                "PartyName": "Precision Engineering Co.",
                "Role": "Consignee",
                "Location": {"City": "Frankfurt", "Country": "Germany"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Machine Tool Parts", "HSCode": "8466.93"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style1_5.pdf": {
        "Parties": [
            {
                "PartyName": "TechnoMach Tools Pvt. Ltd.",
                "Role": "Exporter",
                "Location": {"City": "Pune", "Country": "India"}
            },
            {
                "PartyName": "Mediterranean Spices Ltd.",
                "Role": "Consignee",
                "Location": {"City": "Rome", "Country": "Italy"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Ginger, Fresh", "HSCode": "0910.11"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style1_6.pdf": {
        "Parties": [
            {
                "PartyName": "Oceanic Seafood Exports",
                "Role": "Exporter",
                "Location": {"City": "Kochi", "Country": "India"}
            },
            {
                "PartyName": "Pacific Seafood Imports",
                "Role": "Consignee",
                "Location": {"City": "Tokyo", "Country": "Japan"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Machine Tool Parts", "HSCode": "8466.93"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style1_8.pdf": {
        "Parties": [
            {
                "PartyName": "Fresh Harvest Exports",
                "Role": "Exporter",
                "Location": {"City": "Nashik", "Country": "India"}
            },
            {
                "PartyName": "Green Valley Foods Ltd.",
                "Role": "Consignee",
                "Location": {"City": "London", "Country": "UK"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Cotton T-Shirts", "HSCode": "6109.10"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style2_3.pdf": {
        "Parties": [
            {
                "PartyName": "Oceanic Seafood Exports",
                "Role": "Exporter",
                "Location": {"City": "Kochi", "Country": "India"}
            },
            {
                "PartyName": "Green Valley Foods Ltd",
                "Role": "Consignee",
                "Location": {"City": "London", "Country": "UK"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Cotton T-Shirts", "HSCode": "6109.10"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style2_10.pdf": {
        "Parties": [
            {
                "PartyName": "Fresh Harvest Exports",
                "Role": "Exporter",
                "Location": {"City": "Nashik", "Country": "India"}
            },
            {
                "PartyName": "Global Apparel Inc.",
                "Role": "Consignee",
                "Location": {"City": "New York", "Country": "USA"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Ginger, Fresh", "HSCode": "0910.11"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style3_1.pdf": {
        "Parties": [
            {
                "PartyName": "TechnoMach Tools Pvt. Ltd.",
                "Role": "Exporter",
                "Location": {"City": "Pune", "Country": "India"}
            },
            {
                "PartyName": "Precision Engineering Co.",
                "Role": "Consignee",
                "Location": {"City": "Frankfurt", "Country": "Germany"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Frozen Shrimp", "HSCode": "0303.79"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style3_7.pdf": {
        "Parties": [
            {
                "PartyName": "Oceanic Seafood Export",
                "Role": "Exporter",
                "Location": {"City": "Kochi", "Country": "India"}
            },
            {
                "PartyName": "Global Apparel Inc.",
                "Role": "Consignee",
                "Location": {"City": "New York", "Country": "USA"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Cotton T-Shirts", "HSCode": "6109.10"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
    "Invoice_Style3_9.pdf": {
        "Parties": [
            {
                "PartyName": "ABC Textiles Pvt. Ltd.",
                "Role": "Exporter",
                "Location": {"City": "Mumbai", "Country": "India"}
            },
            {
                "PartyName": "Mediterranean Spices Ltd.",
                "Role": "Consignee",
                "Location": {"City": "Rome", "Country": "Italy"}
            }
        ],
        "CountryOverview": {
            "CountryOfOrigin": "India",
            "CountryOfDestination": None,
            "TransitCountry": None
        },
        "CommodityDetails": [
            {"DescriptionOfGoods": "Cotton T-Shirts", "HSCode": "6109.10"}
        ],
        "Transportation": {"MeansOfTransport": None, "VesselNumber": None}
    },
}
def compare_dicts(pred: dict, truth: dict) -> (int, int):
    """Return (correct_count, total_count) for predicted vs truth."""
    correct, total = 0, 0

    def recurse(p, t):
        nonlocal correct, total
        for k, tv in t.items():
            pv = p.get(k) if isinstance(p, dict) else None
            if isinstance(tv, dict) and isinstance(pv, dict):
                recurse(pv, tv)
            else:
                total += 1
                if pv == tv:
                    correct += 1

    recurse(pred, truth)
    return correct, total

def flatten_dict(d):
    for _, v in d.items():
        if isinstance(v, dict):
            yield from flatten_dict(v)
        else:
            yield v

def calculate_price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in TOKEN_PRICES:
        return 0
    pricing = TOKEN_PRICES[model]
    return (prompt_tokens / 1000000 * pricing["prompt"]) + \
           (completion_tokens / 1000000 * pricing["completion"])

def test_pipeline(models: Dict[str, List[str]], report_path, mode: str = "boolean"):

    report_rows = []

    for img_file in IMAGE_DIR.iterdir():
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".pdf"]:
            continue

        print(f"Processing {img_file.name}...")
        truth = GROUND_TRUTH.get(img_file.name)
        if not truth:
            print(f"No ground truth for {img_file.name}, skipping.\n")
            continue

        # OCR models
        for model in models.get("ocr", []):
            try:
                start_time = time.time()
                ocr_pred, usage = ocr_and_structure(str(img_file), model=model)
                elapsed = time.time() - start_time

                correct, total = compare_dicts(ocr_pred, truth)
                accuracy = correct / total * 100 if total else 0

                tokens_prompt = usage.get("prompt_tokens", 0)
                tokens_completion = usage.get("completion_tokens", 0)
                price = calculate_price(model, tokens_prompt, tokens_completion)

                report_rows.append({
                    "file": img_file.name,
                    "method": "ocr",
                    "model": model,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "time_sec": elapsed,
                    "prompt_tokens": tokens_prompt,
                    "completion_tokens": tokens_completion,
                    "total_tokens": tokens_prompt + tokens_completion,
                    "price_usd": price
                })
            except Exception as e:
                print(f"OCR ({model}) failed for {img_file.name}: {e}")

        # LLM models
        for model in models.get("llm", []):
            start_time = time.time()
            llm_pred, usage = llm_extract(str(img_file), model=model)  # usage dict from API
            elapsed = time.time() - start_time

            correct, total = compare_dicts(llm_pred, truth)
            accuracy = correct / total * 100 if total else 0

            tokens_prompt = usage.get("prompt_tokens", 0)
            tokens_completion = usage.get("completion_tokens", 0)
            price = calculate_price(model, tokens_prompt, tokens_completion)

            report_rows.append({
                "file": img_file.name,
                "method": "llm",
                "model": model,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "time_sec": elapsed,
                "prompt_tokens": tokens_prompt,
                "completion_tokens": tokens_completion,
                "total_tokens": tokens_prompt + tokens_completion,
                "price_usd": price
            })

    # Save report
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nReport saved to {report_path}")
    return report_rows

if __name__ == "__main__":
    models_to_test = {"ocr": TOKEN_PRICES.keys(), "llm": TOKEN_PRICES.keys()}
    report_rows = test_pipeline(models=models_to_test, report_path="model_eval_report.csv")

    df_report = pd.DataFrame(report_rows)  # convert list of dicts to DataFrame

    # Make sure column names match your report dict
    df_report = df_report.rename(columns={"time_sec": "speed_sec"})

    print(df_report.groupby("model").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_speed=("speed_sec", "mean"),
        total_price=("price_usd", "sum")
    ))
