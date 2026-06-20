# 🎮 Game Vision API

Stack: **Ollama + Qwen2.5-VL 7B + FastAPI**
Converte screenshot e asset di videogiochi in JSON strutturati per agenti AI non-vision.

---

## Requisiti

- Docker + Docker Compose v2
- NVIDIA GPU con **8GB VRAM** (RTX 3070 Ti ✓)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installato

```bash
# Verifica che Docker veda la GPU
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## Avvio

```bash
# Prima volta — scarica il modello (~5GB), ci vuole qualche minuto
docker compose up -d

# Segui il download del modello
docker logs game-vision-puller -f

# Verifica che tutto sia pronto
curl http://localhost:8000/health
```

L'API è pronta quando `/health` risponde con `"model_ready": true`.

---

## Utilizzo

### Analisi completa (default)

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@screenshot.png"
```

**Risposta:**
```json
{
  "model": "qwen2.5vl:7b",
  "focus": null,
  "result": {
    "scene_type": "gameplay",
    "scene_summary": "Third-person combat scene in a dark dungeon. Player character faces two enemies.",
    "ui_elements": [
      {
        "type": "bar",
        "label": "HP",
        "position": "bottom-left",
        "value": "73%",
        "state": "active"
      },
      {
        "type": "icon",
        "label": "Sword",
        "position": "bottom-right",
        "value": null,
        "state": "highlighted"
      }
    ],
    "text_content": {
      "dialogue": null,
      "hud_values": { "HP": "146/200", "Stamina": "80/100", "Gold": "342" },
      "other_text": ["DUNGEON LEVEL 3", "WAVE 2/5"]
    },
    "characters": [
      {
        "name": null,
        "description": "Armored knight, player character, holding a sword",
        "position": "center",
        "state": "attacking"
      }
    ],
    "objects_and_items": [
      { "name": "Health Potion", "quantity": 3, "position": "bottom-right" }
    ],
    "environment": {
      "setting": "Stone dungeon corridor, torches on walls",
      "time_of_day": "indoor",
      "notable_features": ["locked door in background", "trap mechanism visible on floor"]
    },
    "agent_notes": "Player is in active combat with 2 enemies. HP is above 70%. A locked door suggests a key or boss kill may be required to proceed."
  }
}
```

---

### Analisi con focus specifico

Il parametro `focus` orienta il modello su un aspetto preciso.
Particolarmente utile quando l'agente sa già cosa gli serve.

```bash
# Estrai solo il testo visibile
curl -X POST http://localhost:8000/analyze \
  -F "image=@screenshot.png" \
  -F "focus=extract all visible text exactly as shown"

# Identifica azioni disponibili
curl -X POST http://localhost:8000/analyze \
  -F "image=@menu.png" \
  -F "focus=list all interactive buttons and menu options available to the player"

# Analisi inventario
curl -X POST http://localhost:8000/analyze \
  -F "image=@inventory.png" \
  -F "focus=list all items in the inventory with quantities and positions"

# Analisi asset (sprite isolato)
curl -X POST http://localhost:8000/analyze \
  -F "image=@enemy_sprite.png" \
  -F "focus=describe this game asset: type, style, colors, and what it likely represents"
```

---

### Batch (più screenshot consecutivi)

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -F "images=@frame1.png" \
  -F "images=@frame2.png" \
  -F "images=@frame3.png" \
  -F "focus=what changed between these frames"
```

---

## Integrazione con agenti Python

```python
import httpx
import json

API_BASE = "http://localhost:8000"

def analyze_screenshot(image_path: str, focus: str | None = None) -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/png")}
        data = {"focus": focus} if focus else {}
        response = httpx.post(f"{API_BASE}/analyze", files=files, data=data, timeout=120)
        response.raise_for_status()
        return response.json()["result"]

# Uso diretto
scene = analyze_screenshot("screenshot.png")
print(scene["scene_summary"])
print(scene["agent_notes"])

# Con focus
actions = analyze_screenshot(
    "menu.png",
    focus="list all interactive options available to the player"
)
print(actions)
```

---

## Note operative

| Aspetto | Dettaglio |
|---|---|
| VRAM usage | ~5-6GB con Qwen2.5-VL 7B Q4 |
| Latenza tipica | 5-15s per immagine (dipende da complessità) |
| Modello persistente | `OLLAMA_KEEP_ALIVE=24h` — non viene scaricato dalla VRAM |
| Concorrenza | `NUM_PARALLEL=1` — una richiesta alla volta (safe per 8GB) |
| Modelli dati | Volume Docker `game-vision-ollama-data` — il modello non si riscarica ai restart |

---

## Comandi utili

```bash
# Logs in tempo reale
docker compose logs -f api
docker compose logs -f ollama

# Restart solo del middleware (senza perdere il modello in VRAM)
docker compose restart api

# Stop completo
docker compose down

# Stop + rimozione volumi (cancella il modello scaricato)
docker compose down -v
```

---

## Swagger UI

Interfaccia interattiva disponibile su: **http://localhost:8000/docs**
