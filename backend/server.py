import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from vosk import Model, KaldiRecognizer

from billing import router as billing_router
from database import init_db

# ---- Config ----
SAMPLE_RATE = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-en-us-0.15")

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(
                f"Vosk model not found at {MODEL_PATH}. "
                "Download and unzip a model into backend/models/ (e.g. vosk-model-small-en-us-0.15)."
            )
        _MODEL = Model(MODEL_PATH)
    return _MODEL

app = FastAPI(title="Interview Prep API")

init_db()
app.include_router(billing_router)

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/app", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    # New recognizer per connection
    model = get_model()
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                # Raw PCM16 mono @ 16k from the browser
                data = message["bytes"]
                if len(data) == 0:
                    continue

                # Feed recognizer
                # AcceptWaveform returns True when a final segment is ready
                final_ready = rec.AcceptWaveform(data)

                if final_ready:
                    result = rec.Result()  # final segment JSON
                    await websocket.send_text(json.dumps({
                        "type": "final",
                        "data": json.loads(result)  # contains 'text', 'result' (words)
                    }))
                else:
                    partial = rec.PartialResult()  # partial JSON
                    await websocket.send_text(json.dumps({
                        "type": "partial",
                        "data": json.loads(partial)  # contains 'partial'
                    }))

            elif "text" in message:
                text = message["text"]
                if text == "__end__":
                    # Flush final result on explicit end
                    final_json = json.loads(rec.FinalResult())
                    await websocket.send_text(json.dumps({"type": "final", "data": final_json}))
                    break
                elif text == "__reset__":
                    rec = KaldiRecognizer(get_model(), SAMPLE_RATE)
                    rec.SetWords(True)
                    await websocket.send_text(json.dumps({"type": "system", "data": "reset"}))
                else:
                    # Ignore other text frames
                    pass

    except WebSocketDisconnect:
        # Client closed abruptly; we’re done
        return
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
        return
