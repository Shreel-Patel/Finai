# FINAI — AI Market Agent

Backend API for an AI-powered stock market analyst: full analysis (technicals + sentiment), chat agent with optional LLM tool-calling, and endpoints for news, sentiment, technicals, and comparison.

**Repo:** [github.com/Shreel-Patel/Finai](https://github.com/Shreel-Patel/Finai)

---

## Features

- **Full analysis** — Unified predictor (XGBoost + calibration) for direction (buy/sell/hold) and P(up); technicals, Reddit + news sentiment.
- **REST API** — `/analyze`, `/query`, `/chart/{ticker}`, `/tickers`, `/chat` (POST).
- **Chat agent** — Natural-language chat; intent-based flow or optional Ollama tool-calling (Phase 2) when available.
- **Compare** — Side-by-side analysis for two tickers (e.g. "compare AAPL and NVDA").

---

## Quick start

```bash
# Clone
git clone https://github.com/Shreel-Patel/Finai.git
cd Finai

# Install
pip install -r requirements.txt

# Run pipeline for a ticker (fetches price, Reddit, news, builds dataset)
# From project root, ensure data/raw, data/reddit, data/processed exist; then:
python -c "from src.pipeline.run_pipeline import ensure_data; ensure_data('AAPL')"

# Start API
uvicorn src.api.main:app --reload
```

API: [http://127.0.0.1:8000](http://127.0.0.1:8000) — docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Project layout

| Path | Description |
|------|-------------|
| `src/api/main.py` | FastAPI app: `/analyze`, `/query`, `/chart`, `/tickers`, `/chat` |
| `src/agent/` | Intent parser, chat agent (Ollama tools), decision rules, LLM explainer |
| `src/models/predictor.py` | Unified price predictor (XGBoost, calibrated prob_up) |
| `src/pipeline/run_pipeline.py` | Pipeline: price → Reddit/news → sentiment → final dataset |
| `data/final/` | Final CSVs per ticker (after pipeline run) |
| `DEPLOY.md` | Free deployment (Render + Vercel/Netlify) |

---

## Environment (optional)

| Variable | Description |
|----------|-------------|
| `FINAI_CHAT_MODEL` | Ollama model for chat agent (default: `gemma:2b`) |
| `CORS_ORIGINS` | Comma-separated origins for CORS (default: `*`) |

---

## Frontend

Use the [aura-finance](https://github.com/Shreel-Patel/aura-finance) (or your fork) React app. Set `VITE_API_URL` to this API’s URL (e.g. `http://127.0.0.1:8000` locally).

---

## License

MIT (or your choice).
