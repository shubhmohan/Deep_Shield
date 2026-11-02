"""
DeepShield - FastAPI Application
Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Docs auto-generated at: http://localhost:8000/docs
"""

import sys
import os
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from detectors.image_detector import ImageDetector
from detectors.video_detector import VideoDetector
from api.utils import numpy_image_to_base64
from api.schemas import ImageDetectionResponse, VideoDetectionResponse, ErrorResponse

# Models are loaded ONCE at server startup, not per-request — this is why
# ImageDetector/VideoDetector's __init__ prints "Loading models..." only once
# in your server logs, not on every upload.
detectors = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting DeepShield API — loading models...")
    detectors["image"] = ImageDetector()
    detectors["video"] = VideoDetector(max_frames=20)
    print("DeepShield API ready.")
    yield
    detectors.clear()


app = FastAPI(title="DeepShield API", version="0.1.0", lifespan=lifespan)

# Allow your Streamlit/React frontend (running on a different port) to call this API.
# Tighten allow_origins to your actual frontend URL before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(detectors.keys())}


@app.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    file_bytes = await file.read()

    try:
        result = detectors["image"].detect(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if "heatmap_overlay" in result and result["heatmap_overlay"] is not None:
        result["heatmap_overlay"] = numpy_image_to_base64(result["heatmap_overlay"])

    return result


@app.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(file: UploadFile = File(...)):
    # Video needs a real file path on disk (OpenCV's VideoCapture can't read
    # from raw bytes directly), so we save the upload to a temp file first.
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = detectors["video"].detect(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp_path)  # always clean up, even if detection failed

    result["most_suspicious_frame"]["heatmap_overlay"] = numpy_image_to_base64(
        result["most_suspicious_frame"]["heatmap_overlay"]
    )

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)