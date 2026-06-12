import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import billing

# ---- Config ----
SAMPLE_RATE = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-en-us-0.15")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Load the speech model lazily so the billing API works even before the
# Vosk model has been downloaded.
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        from vosk import Model
        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(
                f"Vosk model not found at {MODEL_PATH}. "
                "Download and unzip a model into backend/models/ (e.g. vosk-model-small-en-us-0.15)."
            )
        MODEL = Model(MODEL_PATH)
    return MODEL


billing.init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def billing_error(e: billing.BillingError, status: int = 402):
    return JSONResponse(status_code=status, content={"error": e.code, "message": e.message})


# ---- Pages ----

@app.get("/")
async def page_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/pricing")
async def page_pricing():
    return FileResponse(os.path.join(FRONTEND_DIR, "pricing.html"))


# ---- Billing API ----

class AuthBody(BaseModel):
    email: str


class PlanBody(BaseModel):
    plan: str


class BundleBody(BaseModel):
    bundle: str


class TopupBody(BaseModel):
    quantity: int


@app.get("/api/pricing")
async def api_pricing():
    return billing.pricing_catalog()


@app.post("/api/auth/start")
async def api_auth_start(body: AuthBody, request: Request):
    """Email-only signup/login. New accounts get 1 free exam
    (per email and per IP)."""
    try:
        return billing.signup_or_login(body.email, client_ip(request))
    except billing.BillingError as e:
        return billing_error(e, status=400)


@app.get("/api/me")
async def api_me(request: Request):
    try:
        return billing.get_summary(request.headers.get("x-api-key", ""))
    except billing.BillingError as e:
        return billing_error(e, status=401)


@app.post("/api/checkout/subscribe")
async def api_subscribe(body: PlanBody, request: Request):
    try:
        return billing.subscribe(request.headers.get("x-api-key", ""), body.plan)
    except billing.BillingError as e:
        return billing_error(e)


@app.post("/api/checkout/bundle")
async def api_bundle(body: BundleBody, request: Request):
    try:
        return billing.buy_bundle(request.headers.get("x-api-key", ""), body.bundle)
    except billing.BillingError as e:
        return billing_error(e)


@app.post("/api/checkout/topup")
async def api_topup(body: TopupBody, request: Request):
    try:
        return billing.buy_topup(request.headers.get("x-api-key", ""), body.quantity)
    except billing.BillingError as e:
        return billing_error(e)


@app.post("/api/exam/start")
async def api_exam_start(request: Request):
    """Consumes one credit and returns a one-time session token for /ws."""
    try:
        return billing.start_exam(request.headers.get("x-api-key", ""))
    except billing.BillingError as e:
        return billing_error(e)


# ---- Exam websocket (requires a session token from /api/exam/start) ----

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("session", "")
    if not billing.validate_session(token):
        await websocket.close(code=4001, reason="Invalid or used exam session. Start an exam first.")
        return

    await websocket.accept()

    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(get_model(), SAMPLE_RATE)
    rec.SetWords(True)

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                data = message["bytes"]
                if len(data) == 0:
                    continue

                final_ready = rec.AcceptWaveform(data)

                if final_ready:
                    result = rec.Result()
                    await websocket.send_text(json.dumps({
                        "type": "final",
                        "data": json.loads(result)
                    }))
                else:
                    partial = rec.PartialResult()
                    await websocket.send_text(json.dumps({
                        "type": "partial",
                        "data": json.loads(partial)
                    }))

            elif "text" in message:
                text = message["text"]
                if text == "__end__":
                    final_json = json.loads(rec.FinalResult())
                    await websocket.send_text(json.dumps({"type": "final", "data": final_json}))
                    break
                elif text == "__reset__":
                    rec = KaldiRecognizer(get_model(), SAMPLE_RATE)
                    rec.SetWords(True)
                    await websocket.send_text(json.dumps({"type": "system", "data": "reset"}))
                else:
                    pass

    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
        return
