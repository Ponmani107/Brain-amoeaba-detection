# Brain Amoebic Infection Detection System

Full-stack demo app using FastAPI (backend) and React + Tailwind + Vite (frontend).

## Requirements
- Python 3.10+
- Node.js 18+
- Windows PowerShell or any shell

## Backend (FastAPI)

```powershell
cd backend
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Notes:
- Torch/torchvision are optional in `requirements.txt` for Windows; model falls back to a mock predictor if unavailable.
- API: `http://localhost:8000`
  - GET `/health`
  - POST `/predict` (multipart form with `file`)
  - POST `/report` (JSON, returns PDF)

## Frontend (React + Vite + Tailwind)

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Usage
1. Drag-and-drop a brain MRI/CT image or click Browse.
2. Click "Analyze Image" to send to the backend.
3. View prediction, confidence, and Grad-CAM overlay.
4. Click "Download Report" to get a PDF summary.

## Configuration
- Frontend uses `VITE_API_BASE` env var to target backend. Example:

```powershell
$env:VITE_API_BASE = "http://localhost:8000"
npm run dev
```

## Disclaimer
This is a demonstration system. The model may use mock predictions without medical validity. Do not use for clinical decisions.
