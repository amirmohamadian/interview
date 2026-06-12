# Interview Prep

AI-powered mock interview practice with real-time speech-to-text feedback.

## Features

- **Real-time STT** — Stream your answers via WebSocket (Vosk backend)
- **Resume ↔ JD matching** — Notebook-based RAG pipeline (see `Untitled3.ipynb`)
- **Credit-based pricing** — Conversion-optimized plans with free trial

## Pricing Model

| Tier | Credits | Purpose |
|------|---------|---------|
| **Free** | 1 per email or API key | Hook — no card required |
| **Credit Bundles** | 3 / 5 / 10 / 20 | Try without subscription |
| **Subscriptions** | 5 / 10 / 20 per month | Primary conversion path |
| **Subscriber Refills** | 3 / 5 / 10 / 20 | ~40% cheaper top-ups for active subscribers |

- **Unlimited plan** = up to 20 exams/month (fair-use cap)
- **Max value** = 20 credits on bundles and refills
- Subscribers get discounted refills when monthly credits run out

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
# Download Vosk model into backend/models/vosk-model-small-en-us-0.15
uvicorn server:app --reload

# Open pricing page
open http://localhost:8000/app/pricing.html

# API docs
open http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pricing` | Full pricing catalog |
| POST | `/api/register` | Sign up + claim free exam |
| GET | `/api/account` | Check balance & subscription |
| POST | `/api/purchase` | Buy bundle, subscribe, or refill |
| POST | `/api/exam/consume` | Deduct 1 credit to start an exam |

## Project Structure

```
backend/
  server.py      # FastAPI app (STT + billing)
  pricing.py     # Pricing tiers & catalog
  credits.py     # Credit logic & upsell
  database.py    # SQLite persistence
  billing.py     # REST routes
frontend/
  index.html     # STT demo
  pricing.html   # Pricing & checkout UI
```
