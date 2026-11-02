"""
DeepShield - Image Detector (Orchestrator)
preprocessing -> (face crop) -> ViT + EfficientNet -> ensemble -> Grad-CAM

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
from models.vit_model import ViTDeepfakeDetector
from models.efficientnet_model import EfficientNetDeepfakeDetector
from models.ensemble import ensemble_predict
from detectors.explainer import GradCAM


class ImageDetector:
    def __init__(self, use_face_crop: bool = True):
        print("[ImageDetector] Loading models... (this happens once at server startup)")
        self.use_face_crop = use_face_crop
        self.preprocessor = ImagePreprocessor()
        self.face_cropper = VideoPreprocessor(max_frames=1) if use_face_crop else None
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
            "model_breakdown": {
                "vit_score": verdict.vit_score,
                "efficientnet_score": verdict.efficientnet_score,
            },
        }

        if run_gradcam:
            heatmap = self.gradcam.generate(prepped["efficientnet_input"])
            result["heatmap_overlay"] = self.gradcam.overlay_heatmap(heatmap, prepped["display_array"])

        return result