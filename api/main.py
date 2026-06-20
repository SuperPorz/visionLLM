"""
Game Vision API
---------------
Middleware FastAPI tra agenti AI e Ollama/Qwen2.5-VL.
Riceve immagini, le analizza e restituisce JSON strutturati
ottimizzati per il contesto videogame (screenshot, asset, UI, sprite).
"""

import base64
import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_TIMEOUT = 120.0  # secondi — Qwen7B può essere lento su immagini complesse

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Game Vision API",
    description="Image-to-structured-JSON via Qwen2.5-VL. Ottimizzato per screenshot e asset di videogiochi.",
    version="1.0.0",
)

# ─── Prompt templates ─────────────────────────────────────────────────────────

# Prompt di sistema: imposta il "modo" dell'analisi
SYSTEM_PROMPT = """You are a specialized game analysis AI. Your job is to analyze videogame screenshots and assets and return structured JSON descriptions that help text-only AI agents understand what is visually present.

CRITICAL RULES:
- Always respond with ONLY a valid JSON object, no markdown, no explanation outside the JSON.
- Be precise and exhaustive — a blind agent depends entirely on your description.
- Use clear, unambiguous English in all field values.
- Never invent information not visible in the image.
- For text extraction, transcribe exactly what you see, preserving original casing.
"""

# Prompt di analisi completa (default quando nessun focus è specificato)
FULL_ANALYSIS_PROMPT = """Analyze this videogame image and return a JSON object with the following structure:

{
  "scene_type": "one of: gameplay | menu | cutscene | inventory | map | loading | asset | ui_element | unknown",
  "scene_summary": "1-2 sentence plain English description of the overall scene",
  "ui_elements": [
    {
      "type": "button | label | bar | icon | panel | tooltip | minimap | dialogue_box | other",
      "label": "visible text on the element or null",
      "position": "top-left | top-center | top-right | center-left | center | center-right | bottom-left | bottom-center | bottom-right",
      "value": "for bars/counters: current value or percentage if visible, else null",
      "state": "active | inactive | highlighted | disabled | null"
    }
  ],
  "text_content": {
    "dialogue": "exact dialogue text if present, else null",
    "hud_values": { "key extracted from HUD": "value" },
    "other_text": ["list of any other visible text strings"]
  },
  "characters": [
    {
      "name": "name if identifiable, else null",
      "description": "brief visual description",
      "position": "position in frame",
      "state": "idle | moving | attacking | interacting | dead | other"
    }
  ],
  "objects_and_items": [
    {
      "name": "item/object name or description",
      "quantity": "number if visible, else null",
      "position": "position in frame"
    }
  ],
  "environment": {
    "setting": "description of the game world visible (location, biome, room, etc.)",
    "time_of_day": "day | night | dusk | dawn | indoor | unknown",
    "notable_features": ["list of notable environmental elements"]
  },
  "agent_notes": "any additional context an AI agent should know to act on this screen (e.g. 'player appears to be in combat', 'a choice is required', 'inventory is full')"
}

Return ONLY the JSON. No markdown fences, no commentary."""


# Prompt per analisi custom (quando l'agente specifica un focus)
CUSTOM_ANALYSIS_PROMPT_TEMPLATE = """Analyze this videogame image with focus on: {focus}

Return a JSON object describing what you observe relevant to that focus.
The JSON must always include at minimum:
- "scene_summary": brief overall description
- "analysis": your focused analysis result (structure it as makes sense for the focus)
- "agent_notes": anything an AI agent needs to know

Be exhaustive and precise. Return ONLY valid JSON, no markdown fences."""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def image_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def extract_json_from_response(text: str) -> dict[str, Any]:
    """
    Estrae il JSON dalla risposta del modello, gestendo
    casi in cui il modello inserisce markdown o testo extra.
    """
    text = text.strip()

    # Caso ideale: risposta è già JSON puro
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Caso: il modello ha wrappato in ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: cerca il primo { ... } valido
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Se tutto fallisce, ritorna un oggetto di errore con il testo raw
    return {
        "error": "model_did_not_return_valid_json",
        "raw_response": text,
        "agent_notes": "The vision model returned non-JSON output. See raw_response.",
    }


async def call_ollama(image_b64: str, prompt: str, mime_type: str = "image/png") -> str:
    """
    Chiama Ollama con il modello vision e restituisce il testo grezzo.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "images": [image_b64],
        "options": {
            "temperature": 0.1,      # bassa temperatura = output più deterministico
            "top_p": 0.9,
            "num_ctx": 8192,         # contesto esteso — Qwen2.5-VL usa ~4500+ token per le immagini
            "num_predict": 2048,     # sufficiente per JSON complessi
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Ollama timeout — l'immagine potrebbe essere troppo complessa o il modello non è ancora carico.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Errore Ollama: {e.response.status_code} — {e.response.text}",
            )

    data = response.json()
    return data.get("response", "")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Healthcheck — verifica che Ollama sia raggiungibile e il modello caricato."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = any(OLLAMA_MODEL.split(":")[0] in m for m in models)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "ollama": "unreachable", "error": str(e)},
        )

    return {
        "status": "healthy" if model_ready else "degraded",
        "ollama": "connected",
        "model": OLLAMA_MODEL,
        "model_ready": model_ready,
        "available_models": models,
    }


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(..., description="Screenshot o asset da analizzare (PNG, JPG, WebP)"),
    focus: str | None = Form(
        default=None,
        description=(
            "Focus opzionale per l'agente. Esempi: 'extract all text', "
            "'identify interactive UI elements', 'describe the main character', "
            "'what actions are available to the player'. "
            "Se assente, viene eseguita l'analisi completa."
        ),
    ),
):
    """
    **Endpoint principale per gli agenti AI.**

    Invia un'immagine (screenshot o asset di videogioco) e ricevi un JSON
    strutturato che descrive tutto il contenuto visivo: scena, UI, testo,
    personaggi, oggetti, ambiente e note per l'agente.

    - **image**: file immagine (multipart/form-data)
    - **focus**: stringa opzionale per orientare l'analisi su un aspetto specifico
    """
    # Validazione tipo file
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    content_type = image.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo file non supportato: {content_type}. Usa PNG, JPG o WebP.",
        )

    image_data = await image.read()
    if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=413, detail="Immagine troppo grande (max 20MB).")

    image_b64 = image_to_base64(image_data)

    # Scegli il prompt in base al focus
    if focus:
        prompt = CUSTOM_ANALYSIS_PROMPT_TEMPLATE.format(focus=focus.strip())
    else:
        prompt = FULL_ANALYSIS_PROMPT

    raw = await call_ollama(image_b64, prompt, mime_type=content_type)
    result = extract_json_from_response(raw)

    return JSONResponse(
        content={
            "model": OLLAMA_MODEL,
            "focus": focus,
            "result": result,
        }
    )


@app.post("/analyze/batch")
async def analyze_batch(
    images: list[UploadFile] = File(..., description="Lista di immagini (max 10)"),
    focus: str | None = Form(default=None),
):
    """
    Analizza più immagini in sequenza (max 10).
    Restituisce una lista di risultati nello stesso ordine delle immagini inviate.
    Utile per analizzare frame consecutivi o schermate di uno stesso flusso.
    """
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Massimo 10 immagini per batch.")

    results = []
    for img in images:
        image_data = await img.read()
        image_b64 = image_to_base64(image_data)
        prompt = (
            CUSTOM_ANALYSIS_PROMPT_TEMPLATE.format(focus=focus.strip())
            if focus
            else FULL_ANALYSIS_PROMPT
        )
        raw = await call_ollama(image_b64, prompt)
        results.append({
            "filename": img.filename,
            "result": extract_json_from_response(raw),
        })

    return JSONResponse(content={"model": OLLAMA_MODEL, "focus": focus, "results": results})


@app.get("/")
async def root():
    return {
        "service": "Game Vision API",
        "model": OLLAMA_MODEL,
        "endpoints": {
            "POST /analyze": "Analisi singola immagine con focus opzionale",
            "POST /analyze/batch": "Analisi batch (max 10 immagini)",
            "GET /health": "Stato del servizio e del modello",
            "GET /docs": "Swagger UI interattiva",
        },
    }