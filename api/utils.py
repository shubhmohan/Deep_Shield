"""
DeepShield - API Utilities
Helpers for converting internal numpy/PIL objects into JSON-safe API responses.
"""

import base64
import io
import numpy as np
from PIL import Image


def numpy_image_to_base64(array: np.ndarray) -> str:
    """
    Converts a numpy array (values 0-1, shape [H,W,3]) — like our Grad-CAM
    heatmap overlays — into a base64-encoded PNG string the frontend can
    drop straight into an <img src="data:image/png;base64,..."> tag.
    """
    img_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"