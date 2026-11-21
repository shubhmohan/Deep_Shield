"""
DeepShield - FastAPI Application
Run from the project root (the folder that contains api/, detectors/, models/):
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Docs auto-generated at: http://127.0.0.1:8000/docs
(Skip --reload: it restarts the server and reloads the big model on every file save.)
"""

import sys
import os
import tempfile
import shutil
import threading
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from detectors.image_detector import ImageDetector
from detectors.video_detector import VideoDetector
from api.utils import numpy_image_to_base64
from api.schemas import ImageDetectionResponse, VideoDetectionResponse, ErrorResponse

# Models are loaded ONCE at server startup, not per-request.
detectors = {}

# Grad-CAM keeps hook state on the shared model, so run one analysis at a time.
detect_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting DeepShield API - loading models (can take a minute on CPU)...")
    image_detector = ImageDetector()
    detectors["image"] = image_detector
    # Share the same loaded model with the video detector instead of loading it twice.
    detectors["video"] = VideoDetector(max_frames=20, image_detector=image_detector)
    print(f"DeepShield API ready. mode={image_detector.mode}")
    yield
    detectors.clear()


app = FastAPI(title="DeepShield API", version="0.1.0", lifespan=lifespan)

# Tighten allow_origins to your real frontend URL before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    image = detectors.get("image")
    return {
        "status": "ok",
        "models_loaded": list(detectors.keys()),
        "mode": getattr(image, "mode", None),
    }


# These are plain `def` (not `async def`) on purpose: inference is blocking CPU/GPU work,
# and FastAPI runs `def` endpoints in a worker thread so /health stays responsive meanwhile.
@app.post(
    "/detect/image",
    response_model=ImageDetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def detect_image(file: UploadFile = File(...)):
    file_bytes = file.file.read()

    try:
        with detect_lock:
            result = detectors["image"].detect(file_bytes, file.filename or "upload")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {e}")

    if result.get("heatmap_overlay") is not None:
        result["heatmap_overlay"] = numpy_image_to_base64(result["heatmap_overlay"])

    return result


@app.post(
    "/detect/video",
    response_model=VideoDetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def detect_video(file: UploadFile = File(...)):
    # OpenCV can't read video from raw bytes, so save the upload to a temp file first.
    suffix = os.path.splitext(file.filename or "")[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        with detect_lock:
            result = detectors["video"].detect(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {e}")
    finally:
        os.remove(tmp_path)

    frame = result["most_suspicious_frame"]
    if frame.get("heatmap_overlay") is not None:
        frame["heatmap_overlay"] = numpy_image_to_base64(frame["heatmap_overlay"])

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)