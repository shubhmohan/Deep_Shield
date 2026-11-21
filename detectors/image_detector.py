"""
DeepShield - Image Detector (Orchestrator)
preprocessing -> (face crop) -> model(s) -> verdict -> Grad-CAM

Two modes (pick with ImageDetector(mode=...) or the DEEPSHIELD_MODEL env var):
  "df40_clip" : the pretrained DF40 CLIP ViT-L/14 detector only (models/df40_model.py).
  "ensemble"  : your original ViT-B/16 + EfficientNet-B4 weighted ensemble.
  "auto"      : (default) df40_clip if checkpoints/df40/clip.pth exists, else ensemble.

Face cropping: video frames were already face-cropped before detection, but
single images were not, so the same model saw tight face crops in video and
full photos (with backgrounds, bodies, text) in images. Both models were
trained on face-centered images, so images now get the same crop.
If no face is found we fall back to the full image and report it.
"""

import sys
import os
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessors.image_prep import ImagePreprocessor
from preprocessors.video_prep import VideoPreprocessor
from models.ensemble import ensemble_predict, get_risk_level

DF40_FAKE_THRESHOLD = float(os.environ.get("DEEPSHIELD_THRESHOLD", "0.5"))


class ImageDetector:
    def __init__(self, use_face_crop: bool = True, mode: str = None):
        mode = mode or os.environ.get("DEEPSHIELD_MODEL", "auto")
        if mode == "auto":
            from models.df40_model import DF40_CLIP_CHECKPOINT
            mode = "df40_clip" if os.path.exists(DF40_CLIP_CHECKPOINT) else "ensemble"
        if mode not in ("df40_clip", "ensemble"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'df40_clip', 'ensemble' or 'auto'.")
        self.mode = mode

        print(f"[ImageDetector] Loading models (mode={self.mode})... (once, at server startup)")
        self.use_face_crop = use_face_crop
        self.preprocessor = ImagePreprocessor()
        self.face_cropper = VideoPreprocessor(max_frames=1) if use_face_crop else None

        if self.mode == "df40_clip":
            from models.df40_model import DF40CLIPDetector
            from detectors.vit_gradcam import ViTGradCAM
            self.clip = DF40CLIPDetector()
            self.gradcam = ViTGradCAM(self.clip)
        else:
            from models.vit_model import ViTDeepfakeDetector
            from models.efficientnet_model import EfficientNetDeepfakeDetector
            from detectors.explainer import GradCAM
            self.vit = ViTDeepfakeDetector()
            self.efficientnet = EfficientNetDeepfakeDetector()
            self.gradcam = GradCAM(self.efficientnet.model, self.efficientnet.target_layer)

        print(f"[ImageDetector] Ready. face_crop={'on' if use_face_crop else 'off'}")

    def _crop_face_or_fallback(self, image):
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face = self.face_cropper.crop_face(frame_bgr)
        if face is None:
            return image, False
        return face, True

    def detect(self, file_bytes: bytes, filename: str) -> dict:
        start = time.time()

        self.preprocessor.validate(file_bytes, filename)
        image = self.preprocessor.load_image(file_bytes)

        face_detected = False
        if self.use_face_crop:
            image, face_detected = self._crop_face_or_fallback(image)

        result = self.detect_pil_image(image, run_gradcam=True)
        result["face_detected"] = face_detected
        result["processing_time_sec"] = round(time.time() - start, 3)
        return result

    def detect_pil_image(self, image, run_gradcam: bool = True) -> dict:
        """Takes an already-loaded (and already-cropped) PIL image. Video frames use this."""
        prepped = self.preprocessor.preprocess(image)

        if self.mode == "df40_clip":
            return self._detect_df40(prepped, run_gradcam)
        return self._detect_ensemble(prepped, run_gradcam)

    def _detect_df40(self, prepped: dict, run_gradcam: bool) -> dict:
        # DF40 CLIP wants 224x224, mean/std 0.5 -> exactly the existing "vit_input" tensor.
        clip_tensor = prepped["vit_input"]
        fake_prob = self.clip.predict(clip_tensor)["fake_prob"]

        verdict = "FAKE" if fake_prob >= DF40_FAKE_THRESHOLD else "REAL"
        result = {
            "verdict": verdict,
            "confidence": round(abs(fake_prob - 0.5) * 2 * 100, 2),
            "fake_probability": round(fake_prob, 4),
            "risk_level": get_risk_level(fake_prob),
            "model_used": "df40_clip",
            "model_breakdown": {"df40_clip_score": round(fake_prob, 4)},
        }

        if run_gradcam:
            heatmap = self.gradcam.generate(clip_tensor, class_idx=self.clip.fake_idx)
            result["heatmap_overlay"] = self.gradcam.overlay_heatmap(heatmap, prepped["display_array"])
        return result

    def _detect_ensemble(self, prepped: dict, run_gradcam: bool) -> dict:
        vit_result = self.vit.predict(prepped["vit_input"])
        eff_result = self.efficientnet.predict(prepped["efficientnet_input"])

        verdict = ensemble_predict(
            vit_fake_prob=vit_result["fake_prob"],
            efficientnet_fake_prob=eff_result["fake_prob"],
        )

        result = {
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "fake_probability": verdict.fake_probability,
            "risk_level": verdict.risk_level,
            "model_used": "ensemble",
            "model_breakdown": {
                "vit_score": verdict.vit_score,
                "efficientnet_score": verdict.efficientnet_score,
            },
        }

        if run_gradcam:
            # Explain the FAKE class. ImageFolder sorts classes alphabetically (fake=0, real=1),
            # so the old default class_idx=1 was explaining "real" for locally trained checkpoints.
            fake_idx = getattr(self.efficientnet, "fake_idx", 1)
            heatmap = self.gradcam.generate(prepped["efficientnet_input"], class_idx=fake_idx)
            result["heatmap_overlay"] = self.gradcam.overlay_heatmap(heatmap, prepped["display_array"])

        return result