"""
Neural Linguistic Suite — Cloud API Backend
Inference via the HuggingFace Inference API (no local models).
"""

import json
import os

import requests
import uvicorn
from dotenv import load_dotenv
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BASE = "https://router.huggingface.co/hf-inference/models"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRANSLATION_MODELS = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
}
SUMM_MODEL = "sshleifer/distilbart-cnn-12-6"

app = FastAPI(title="Neural Linguistic Suite")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


# ── Schemas ───────────────────────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    text: str


class SummarizeRequest(BaseModel):
    text: str


# ── HuggingFace Inference helper ──────────────────────────────────────────────


def query_hf_api(model_id: str, payload: dict):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Inference API key not configured.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(
            f"{HF_BASE}/{model_id}", headers=headers, json=payload, timeout=60
        )
    except requests.Timeout:
        raise HTTPException(
            status_code=504, detail="Inference API timed out. Please try again."
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=502, detail="Could not reach the inference API."
        )

    try:
        body = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=r.text)

    # Cold-start: model still loading
    if r.status_code == 503 or (isinstance(body, dict) and "estimated_time" in body):
        eta = int(body.get("estimated_time", 20)) if isinstance(body, dict) else 20
        raise HTTPException(
            status_code=503,
            detail=f"Model is warming up. Please retry in ~{eta}s.",
        )

    if not r.ok:
        raise HTTPException(
            status_code=r.status_code, detail=str(body)
        )

    return body


# ── Helpers ───────────────────────────────────────────────────────────────────


def _detect_reverse_key(text: str) -> str:
    """Route reverse-to-English by script: Devanagari → hi-en, else es-en."""
    return "hi-en" if any("\u0900" <= c <= "\u097f" for c in text) else "es-en"


def _load_log(filename: str) -> dict:
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Translation endpoints ─────────────────────────────────────────────────────


@app.post("/translate/multi")
def translate_multi(req: TranslateRequest, target_lang: str = "hi"):
    if target_lang == "hi":
        model = TRANSLATION_MODELS["en-hi"]
    elif target_lang == "es":
        model = TRANSLATION_MODELS["en-es"]
    elif target_lang == "en":
        model = TRANSLATION_MODELS[_detect_reverse_key(req.text)]
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported target_lang: '{target_lang}'."
        )

    result = query_hf_api(model, {"inputs": req.text})
    try:
        translation = result[0].get("translation_text", "") if isinstance(result, list) else ""
    except Exception:
        translation = str(result)
    return {"translation": translation}


@app.post("/translate/hindi-to-english")
def translate_hi_en(req: TranslateRequest):
    result = query_hf_api(TRANSLATION_MODELS["hi-en"], {"inputs": req.text})
    return {
        "translation": result[0]["translation_text"] if isinstance(result, list) else ""
    }


@app.post("/translate/spanish-to-english")
def translate_es_en(req: TranslateRequest):
    result = query_hf_api(TRANSLATION_MODELS["es-en"], {"inputs": req.text})
    return {
        "translation": result[0]["translation_text"] if isinstance(result, list) else ""
    }


# ── Summarization endpoint ────────────────────────────────────────────────────


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    # Let Hugging Face handle lengths dynamically
    payload = {"inputs": req.text}
    result = query_hf_api(SUMM_MODEL, payload)
    
    try:
        summary = result[0].get("summary_text", "") if isinstance(result, list) else str(result)
    except Exception:
        summary = str(result)
        
    ratio = round(len(req.text.split()) / max(len(summary.split()), 1), 2)
    return {"summary": summary, "compression_ratio": ratio}


# ── Model info ────────────────────────────────────────────────────────────────


@app.get("/model-info")
def model_info():
    # MarianMT representative config: d=512, 6 enc/dec layers, 8 heads, ffn=2048
    d, heads, ffn, enc_l, dec_l, vocab = 512, 8, 2048, 6, 6, 65001
    attn_p = 4 * d * d
    ffn_p = 2 * d * ffn

    enc_layers = [
        {"name": "Token Embeddings", "type": "Embedding", "params": f"{vocab * d:,}"},
        *[
            {
                "name": f"Encoder Layer {i + 1}",
                "type": f"Self-Attn + FFN  ({heads} heads)",
                "params": f"{attn_p + ffn_p:,}",
            }
            for i in range(enc_l)
        ],
        {"name": "Final Layer Norm", "type": "LayerNorm", "params": f"{d:,}"},
    ]
    dec_layers = [
        {"name": "Token Embeddings", "type": "Embedding", "params": f"{vocab * d:,}"},
        *[
            {
                "name": f"Decoder Layer {i + 1}",
                "type": f"Masked + Cross-Attn + FFN  ({heads} heads)",
                "params": f"{6 * d * d + ffn_p:,}",
            }
            for i in range(dec_l)
        ],
        {
            "name": "LM Head",
            "type": "Linear (tied weights)",
            "params": f"{vocab * d:,}",
        },
    ]

    return {
        "encoder": {
            "type": f"MarianMT Encoder — {enc_l} layers · {heads} heads · d_model={d}",
            "layers": enc_layers,
        },
        "decoder": {
            "type": f"MarianMT Decoder — {dec_l} layers · {heads} heads · d_model={d}",
            "layers": dec_layers,
        },
        "vocab_sizes": {
            "opus_mt_en_hi": 65001,
            "opus_mt_en_es": 65001,
            "opus_mt_hi_en": 59512,
            "opus_mt_es_en": 65001,
            "distilbart_cnn_12_6": 50265,
        },
        "training_config": {
            "architecture": "MarianMT  (translation)  +  DistilBART  (summarization)",
            "inference_provider": "HuggingFace Inference API",
            "translation_params": "~74M per direction",
            "summarization_params": "~306M",
            "decoding_strategy": "Beam Search",
            "beam_size": "4",
            "max_translation_tokens": "512",
        },
    }


# ── Training log endpoints ────────────────────────────────────────────────────


@app.get("/logs/translation")
def logs_translation():
    return _load_log("loss_log_translation.json")


@app.get("/logs/translation-es")
def logs_translation_es():
    return _load_log("loss_log_translation_es.json")


@app.get("/logs/summarization")
def logs_summarization():
    return _load_log("loss_log_summarization.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
