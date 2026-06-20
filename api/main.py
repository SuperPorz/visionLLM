"""
Game Vision API
---------------
Middleware FastAPI tra agenti AI e Ollama/Qwen2.5-VL.
Riceve immagini, le analizza e restituisce JSON strutturati
ottimizzati per il contesto videogame (screenshot, asset, UI, sprite).
"""

import base64
import io
import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_TIMEOUT = 120.0  # secondi — Qwen7B può essere lento su immagini complesse
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "4"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "2048"))
MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "2048"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(4 * 1024 * 1024)))
MAX_TOTAL_PIXELS_PER_REQUEST = int(os.getenv("MAX_TOTAL_PIXELS_PER_REQUEST", str(8 * 1024 * 1024)))

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


def build_analysis_prompt(focus: str | None, image_count: int) -> str:
    base_prompt = (
        CUSTOM_ANALYSIS_PROMPT_TEMPLATE.format(focus=focus.strip())
        if focus
        else FULL_ANALYSIS_PROMPT
    )

    if image_count == 1:
        return base_prompt

    return (
        base_prompt
        + f"""

You are given {image_count} related images in the order they were uploaded.
Analyze them jointly, compare the frames when relevant, and mention changes, continuity, or differences in the scene_summary and agent_notes.
If the images do not appear related, say so explicitly.
"""
    )


async def validate_and_encode_image(image: UploadFile) -> tuple[str, str, tuple[int, int]]:
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    content_type = image.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo file non supportato: {content_type}. Usa PNG, JPG o WebP.",
        )

    image_data = await image.read()
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Immagine troppo grande (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
        )

    try:
        with Image.open(io.BytesIO(image_data)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_data)) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="File immagine non valido o corrotto.")

    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Immagine troppo grande: {width}x{height}px. "
                f"Limite massimo: {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}px."
            ),
        )

    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Immagine troppo pesante per il budget visivo: {pixels} pixel. "
                f"Limite massimo: {MAX_IMAGE_PIXELS} pixel."
            ),
        )

    return image_to_base64(image_data), content_type, (width, height)


def validate_request_image_set(images: list[UploadFile]) -> None:
    if not images:
        raise HTTPException(status_code=400, detail="Fornisci almeno una immagine.")

    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Massimo {MAX_IMAGES_PER_REQUEST} immagini per richiesta.",
        )


def validate_total_pixel_budget(dimensions: list[tuple[int, int]]) -> None:
    total_pixels = sum(width * height for width, height in dimensions)
    if total_pixels > MAX_TOTAL_PIXELS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Budget visivo totale superato: {total_pixels} pixel. "
                f"Limite massimo per richiesta: {MAX_TOTAL_PIXELS_PER_REQUEST} pixel."
            ),
        )


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

    def repair_agent_notes(s: str) -> str:
        """Ripara agent_notes se il modello lo restituisce come dict invece di stringa."""
        return re.sub(
            r'"agent_notes"\s*:\s*\{[^}]*\}',
            lambda m: '"agent_notes": ' + json.dumps(
                " ".join(re.findall(r'"([^"]+)"', m.group(0))[1:])
            ),
            s,
        )

    # Caso: il modello ha wrappato in ```json ... ``` o ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_agent_notes(candidate))
            except json.JSONDecodeError:
                pass

    # Fallback: cerca il blocco { ... } più esterno (first { → last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_agent_notes(candidate))
            except json.JSONDecodeError:
                pass

    # Se tutto fallisce, ritorna un oggetto di errore con il testo raw
    return {
        "error": "model_did_not_return_valid_json",
        "raw_response": text,
        "agent_notes": "The vision model returned non-JSON output. See raw_response.",
    }


async def call_ollama(image_b64s: list[str], prompt: str) -> str:
    """
    Chiama Ollama con il modello vision e restituisce il testo grezzo.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "images": image_b64s,
        "options": {
            "temperature": 0.1,      # bassa temperatura = output più deterministico
            "top_p": 0.9,
            # Qui si gestisce la dimensione del contesto: alzare num_ctx aumenta il budget,
            # ma cresce anche l'uso di memoria del modello. Con GPU inferenza il KV cache
            # tende a pesare sulla VRAM; spostarlo solo in RAM non è un toggle trasparente.
            "num_ctx": OLLAMA_NUM_CTX,
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
    image: UploadFile | None = File(
        default=None,
        description="Singola immagine da analizzare (PNG, JPG, WebP)",
    ),
    images: list[UploadFile] | None = File(
        default=None,
        description="Una o più immagini da analizzare nello stesso passaggio",
    ),
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

    - **image**: file immagine singolo (multipart/form-data)
    - **images**: lista di immagini (multipart/form-data, ripetere il campo)
    - **focus**: stringa opzionale per orientare l'analisi su un aspetto specifico
    """
    request_images: list[UploadFile] = []
    if image is not None:
        request_images.append(image)
    if images:
        request_images.extend(images)

    validate_request_image_set(request_images)

    encoded_images: list[str] = []
    dimensions: list[tuple[int, int]] = []
    content_types: list[str] = []

    for upload in request_images:
        image_b64, content_type, size = await validate_and_encode_image(upload)
        encoded_images.append(image_b64)
        content_types.append(content_type)
        dimensions.append(size)

    validate_total_pixel_budget(dimensions)

    prompt = build_analysis_prompt(focus, len(encoded_images))

    raw = await call_ollama(encoded_images, prompt)
    result = extract_json_from_response(raw)

    return JSONResponse(
        content={
            "model": OLLAMA_MODEL,
            "focus": focus,
            "image_count": len(encoded_images),
            "content_types": content_types,
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
        image_b64, _, _ = await validate_and_encode_image(img)
        prompt = build_analysis_prompt(focus, 1)
        raw = await call_ollama([image_b64], prompt)
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
            "POST /analyze": "Analisi una o più immagini con focus opzionale",
            "POST /analyze/batch": "Analisi batch (max 10 immagini)",
            "GET /health": "Stato del servizio e del modello",
            "GET /docs": "Swagger UI interattiva",
        },
    }