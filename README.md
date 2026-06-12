# interview

Resume ↔ Job Description matching ("exam") tool, plus a real‑time speech‑to‑text demo.

## Pricing & monetization

One **exam** = one resume/JD match analysis. Our cost is ≈ **$0.10/exam**, so every
plan below keeps an 80%+ margin while being simple and conversion‑first.

| Plan | Price | Exams | Per‑exam | Who it's for |
|------|-------|-------|----------|--------------|
| **Free trial** | $0 | 1 (per email **or** API key) | — | First‑touch hook, no card |
| **Starter pack** | $5 | 5 | $1.00 | Testing without subscribing |
| **Value pack** ⭐ | $15 | 20 | $0.75 | Best price without subscribing |
| **Pro** ⭐ (sub) | $9/mo (or $90/yr) | 20 / month | ≈$0.45 | Active job seekers |
| **Unlimited** (sub) | $19/mo (or $190/yr) | Unlimited, fair‑use 20/day | — | Coaches & power users |
| **Top‑up** (subscribers) | $0.50/exam | up to 20 | $0.50 | Ran out mid‑month — cheaper than bundles |

Design rules baked in:

- **1 free exam** for every new email *and* every new API key → maximum top‑of‑funnel.
- **Subscription first**: Pro is the "Most popular" anchor; effective ~$0.45/exam.
- **Cheaper when you run out**: subscriber top‑ups ($0.50) undercut every bundle.
- **Unlimited** is genuinely unlimited for normal use, protected by a 20/day fair‑use cap.
- **Nothing exceeds a value of 20** (max bundle, monthly allowance, and daily cap).

Tune any number in one place: the `PRICING` catalog in `backend/pricing.py`.

### Backend (pricing API)

```bash
cd backend
pip install fastapi "uvicorn[standard]" pydantic
uvicorn pricing:app --reload          # http://localhost:8000
# persist accounts to disk: PRICING_DB=accounts.json uvicorn pricing:app
```

Endpoints: `GET /pricing`, `POST /signup`, `GET /account`, `POST /consume`
(returns `402` with an upsell payload when out of entitlements),
`POST /purchase/bundle`, `POST /subscribe`, `POST /topup`.

Run the tests:

```bash
cd backend && python3 test_pricing.py
```

### Frontend (pricing page)

`frontend/pricing.html` is a conversion‑optimized landing/pricing page that reads the
live catalog from `GET /pricing` (with a built‑in fallback so it renders standalone).
Just open it in a browser, or serve the `frontend/` folder statically.

## Speech‑to‑text demo

See `backend/server.py` (Vosk websocket) and `frontend/index.html`. Requires a Vosk
model in `backend/models/` — see the comment in `server.py`.
