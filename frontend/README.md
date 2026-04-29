# Vasculitis XAI Frontend

React + TypeScript + Vite frontend for the FastAPI Vasculitis XAI backend.

## Development

Start the backend first:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies API calls such as `/health`, `/predict`, `/predict/all`, `/explain/*`, `/model/*`, `/chat`, and `/agent/*` to `http://localhost:8000`.

To override the API proxy target:

```bash
VITE_API_PROXY_TARGET=https://api.magisterka.localhost npm run dev
```

## Build

```bash
npm run build
```
