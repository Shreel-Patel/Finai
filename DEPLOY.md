# Deploy FINAI for Free

Deploy the **backend (FINAI)** and **frontend (aura-finance)** on free tiers. Frontend and backend are deployed separately; the frontend calls the backend URL via `VITE_API_URL`.

---

## 1. Frontend (aura-finance) — Vercel or Netlify

### Option A: Vercel (recommended)

1. Push your frontend code to GitHub (only the `aura-finance` app folder, or the repo that contains it).
2. Go to [vercel.com](https://vercel.com) → Sign in with GitHub → **Add New Project**.
3. Import the repo and set the **root directory** to the frontend folder (e.g. `aura-finance-98-main` or where `package.json` and `vite.config.ts` live).
4. **Build settings** (usually auto-detected):
   - Build command: `npm run build`
   - Output directory: `dist`
5. **Environment variables** (important):
   - `VITE_API_URL` = your backend URL (e.g. `https://your-finai-backend.onrender.com`)
   - Add it in Vercel: Project → Settings → Environment Variables.
6. Deploy. Your app will be at `https://your-project.vercel.app`.

### Option B: Netlify

1. Push frontend to GitHub.
2. Go to [netlify.com](https://netlify.com) → Add new site → Import from Git → choose repo and root (frontend folder).
3. Build command: `npm run build`, Publish directory: `dist`.
4. Add env var: `VITE_API_URL` = backend URL (e.g. `https://your-finai-backend.onrender.com`).
5. Deploy. App at `https://your-site.netlify.app`.

**Note:** `VITE_API_URL` must be set at build time (Vite bakes it in). If you change the backend URL later, rebuild and redeploy the frontend.

---

## 2. Backend (FINAI) — Render free tier

1. Push the **FINAI** project (this repo) to GitHub.
2. Go to [render.com](https://render.com) → Sign in with GitHub → **New** → **Web Service**.
3. Connect the FINAI repo.
4. **Settings:**
   - **Name:** e.g. `finai-api`
   - **Region:** choose closest to you.
   - **Runtime:** Python 3.
   - **Build command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start command:**
     ```bash
     uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance type:** Free.

5. **Environment variables** (optional but useful):
   - `FINAI_CHAT_MODEL` = `gemma:2b` (or leave default; Ollama won’t run on Render, so chat uses intent fallback).
   - `CORS_ORIGINS` = your frontend URL (e.g. `https://your-project.vercel.app`) if you want to restrict CORS in production.

6. **Deploy.** Your API will be at `https://finai-api.onrender.com` (or the name you chose).

**Render free tier caveats:**

- **Cold starts:** First request after idle can take 30–60 seconds; then it’s fast until it sleeps again.
- **No persistent disk:** Anything written under `data/` (e.g. `data/final/*.csv`) is lost on restart. So:
  - Pre-built data won’t persist.
  - For a “live” backend you’d need to run the pipeline on each request (or on a schedule) or use an external DB/storage.
- **No Ollama:** The server can’t run Ollama, so the Phase 2 LLM agent won’t run; chat will use the **intent-based fallback** (same behavior as when Ollama is missing).

---

## 3. Optional: Restrict CORS in production

If you want the backend to allow only your frontend origin:

1. In **FINAI** add support for an env var, e.g. in `src/api/main.py`:

```python
import os
# After creating app:
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
if _cors_origins != "*":
    _cors_origins = [o.strip() for o in _cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. On Render, set `CORS_ORIGINS` = `https://your-project.vercel.app` (or multiple origins comma-separated).

If you don’t set `CORS_ORIGINS`, the backend keeps `allow_origins=["*"]` (works for any frontend; fine for demos).

---

## 4. Quick checklist

| Step | Action |
|------|--------|
| 1 | Push FINAI to GitHub → deploy backend on Render (build: `pip install -r requirements.txt`, start: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`) |
| 2 | Copy backend URL (e.g. `https://finai-api.onrender.com`) |
| 3 | Push frontend to GitHub → deploy on Vercel/Netlify with `VITE_API_URL` = that URL |
| 4 | Open frontend URL; use **Demo mode** if backend is sleeping or has no data |

---

## 5. Other free backend options

- **Railway:** Free tier with a monthly allowance; similar idea: connect repo, set build/start commands, add `PORT` and optional env vars.
- **Fly.io:** Free allowance; deploy with a `Dockerfile` or their Python buildpack.
- **Your own PC:** Run the backend locally and expose it with [ngrok](https://ngrok.com) or [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps); set `VITE_API_URL` to the public URL. Then Ollama and full data pipeline can run locally.

For a fully free, zero-setup cloud setup: **Render (backend) + Vercel (frontend)** with Demo mode when the backend is cold or has no data.
