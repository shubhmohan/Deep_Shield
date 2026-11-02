# DeepShield — Frontend

React frontend for DeepShield, an ensemble deepfake detector (ViT-B/16 +
EfficientNet-B4) with Grad-CAM explainability. Talks to the FastAPI backend
in the main `deepshield` repo.

## Local development

```bash
npm install
cp .env.example .env    # then edit VITE_API_URL if your backend isn't on localhost:8000
npm run dev
```

Make sure the FastAPI backend is running (`uvicorn api.main:app --reload`)
before testing uploads — the app will show a network error otherwise.

## Deploying

**Frontend (Vercel or Netlify):**
1. Push this folder to a GitHub repo (or the frontend subfolder of your main repo)
2. Import it into Vercel/Netlify — both auto-detect Vite projects
3. Set the environment variable `VITE_API_URL` to your deployed backend's URL
   (see below) before the first deploy, or redeploy after adding it

**Backend (Render or Railway):**
1. Deploy the `deepshield` FastAPI project (the `api/` folder) as a web service
2. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. Copy the resulting URL (e.g. `https://deepshield-api.onrender.com`) into
   your frontend's `VITE_API_URL`

**CORS note:** `api/main.py` currently allows all origins (`allow_origins=["*"]`)
for ease of testing. Once you know your deployed frontend's URL, tighten this
to that exact origin before sharing the link widely.

**Cold starts:** Render/Railway free tiers spin down after inactivity — the
first request after idle time can take 30-60 seconds while the container
wakes up and loads both models. This is normal; consider adding a brief
"waking up the models…" note in the UI if you notice recruiters hitting this.

## What's here

| File | Purpose |
|---|---|
| `src/App.jsx` | Main app — upload flow, state machine (idle → scanning → done/error) |
| `src/api.js` | Fetch calls to `/detect/image` and `/detect/video` |
| `src/components/UploadZone.jsx` | Drag-and-drop / click-to-browse upload |
| `src/components/ImageResult.jsx` | Report panel for image results |
| `src/components/VideoResult.jsx` | Report panel for video results (frame stats, blink signal) |
| `src/components/ArchitectureFlow.jsx` | "How it works" diagram |
| `src/index.css` | Design tokens + all styling (no CSS framework dependency) |
