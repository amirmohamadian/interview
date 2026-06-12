# Interview Prep

AI-powered mock interview practice with real-time speech-to-text feedback.

## Features

- **Real-time STT** — Stream your answers via WebSocket (Vosk backend)
- **Resume ↔ JD matching** — Notebook-based RAG pipeline (see `Untitled3.ipynb`)
- **Credit-based pricing** — Conversion-optimized plans with free trial

## Pricing Model (conversion-optimized)

| Tier | Credits | Purpose |
|------|---------|---------|
| **Free** | 1 per email or API key | Hook — no card required |
| **Credit Bundles** | 10 / 15 / 20 | Try without subscription (3-tier ladder) |
| **Subscriptions** | 5 / 10 / 20 per month | Primary conversion path |
| **Subscriber Refills** | 5 / 10 / 15 / 20 | ~40% cheaper top-ups when credits run out |

- **Unlimited plan** = up to 20 exams/month (fair-use cap)
- **Max bundle value** = 20 credits
- Subscribers get discounted refills when monthly credits run out
- **5-credit Quick Refill** for impulse top-ups mid-cycle

### Why 10 / 15 / 20 bundles?

Three tiers follow the "good / better / best" pattern that maximizes conversion. The middle **15-credit Practice Pack** is highlighted as Popular, while **20-credit Max Pack** anchors maximum value. Subscriptions remain clearly cheaper per exam to drive recurring revenue.

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
  pricing.py      # Product catalog & conversion tiers
  credits.py      # Credit business logic
  database.py     # SQLite persistence
  billing.py      # REST API routes
  server.py         # FastAPI app (STT + billing)
frontend/
  pricing.html    # Conversion-optimized pricing page
  index.html      # STT demo
Untitled3.ipynb   # Resume/JD matching pipeline
```

## Tests

```bash
cd backend
python -m unittest test_pricing.py -v
```
