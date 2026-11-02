"""
DeepShield - Video Detector (Orchestrator)
Entry point for video analysis: sample frames -> crop faces -> run each
face through the existing image detection pipeline -> aggregate per-frame
scores into one video-level verdict -> attach a single representative
Grad-CAM (on the most suspicious frame, for speed) -> attach the blink
signal as a supplementary note.
"""

import sys
import os
import time
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.image_detector import ImageDetector
from detectors.blink_analyzer import BlinkAnalyzer
from preprocessors.video_prep import VideoPreprocessor
from models.ensemble import get_risk_level


class VideoDetector:
    """
    Reuses a single ImageDetector instance (and therefore its already-loaded
    models) across all sampled frames — do NOT instantiate a new
    ImageDetector per frame, that would reload both models every time.
    """

    def __init__(self, max_frames: int = 20):
        print("[VideoDetector] Initializing (reuses ImageDetector's already-loaded models)...")
        self.image_detector = ImageDetector()
        self.blink_analyzer = BlinkAnalyzer()
        self.video_preprocessor = VideoPreprocessor(max_frames=max_frames)
        print("[VideoDetector] Ready.")

    def detect(self, video_path: str) -> dict:
        start = time.time()

        # 1. Extract raw sampled frames (needed both for face-cropping AND blink analysis)
        raw_frames = self.video_preprocessor.extract_frames(video_path)

        # 2. Crop faces per frame — frames with no detectable face are skipped
        face_crops = []
        for i, frame in enumerate(raw_frames):
            face = self.video_preprocessor.crop_face(frame)
            if face is not None:
                face_crops.append((i, face))

        if len(face_crops) == 0:
            raise ValueError(
                "No face could be detected in any sampled frame. DeepShield's current "
                "pipeline is face-centric — this video may not contain a clear, "
                "front-facing subject, or the face is too small/obscured."
            )

        # 3. Run each cropped face through the image pipeline (no Grad-CAM yet — too slow to do every frame)
        frame_results = []
        for frame_idx, face_image in face_crops:
            result = self.image_detector.detect_pil_image(face_image, run_gradcam=False)
            frame_results.append({"frame_idx": frame_idx, **result})

        # 4. Aggregate: mean fake_probability across frames is our primary video-level score.
        #    We also track max and variance — high variance suggests inconsistent frames,
        #    which itself can be a tell (real videos tend to score consistently low).
        fake_probs = [r["fake_probability"] for r in frame_results]
        mean_fake_prob = statistics.mean(fake_probs)
        max_fake_prob = max(fake_probs)
        std_fake_prob = statistics.stdev(fake_probs) if len(fake_probs) > 1 else 0.0

        video_verdict = "FAKE" if mean_fake_prob >= 0.5 else "REAL"
        video_confidence = round(abs(mean_fake_prob - 0.5) * 2 * 100, 2)
        video_risk = get_risk_level(mean_fake_prob)

        # 5. Grad-CAM on ONLY the single most-suspicious frame — this is the "why" visual for the report
        most_suspicious = max(frame_results, key=lambda r: r["fake_probability"])
        most_suspicious_idx = most_suspicious["frame_idx"]
        most_suspicious_image = dict(face_crops)[most_suspicious_idx]
        gradcam_result = self.image_detector.detect_pil_image(most_suspicious_image, run_gradcam=True)

        # 6. Blink analysis (supplementary signal — see blink_analyzer.py docstring on why this
        #    is NOT folded into the ML score)
        blink_result = self.blink_analyzer.analyze(raw_frames)

        elapsed = round(time.time() - start, 3)

        return {
            "verdict": video_verdict,
            "confidence": video_confidence,
            "fake_probability": round(mean_fake_prob, 4),
            "risk_level": video_risk,
            "frame_stats": {
                "frames_sampled": len(raw_frames),
                "frames_with_face_detected": len(face_crops),
                "mean_fake_probability": round(mean_fake_prob, 4),
                "max_fake_probability": round(max_fake_prob, 4),
                "std_fake_probability": round(std_fake_prob, 4),
            },
            "most_suspicious_frame": {
                "frame_index": most_suspicious_idx,
                "fake_probability": most_suspicious["fake_probability"],
                "heatmap_overlay": gradcam_result.get("heatmap_overlay"),
            },
            "blink_analysis": {
                "signal": blink_result["signal"],
                "note": blink_result["note"],
            },
            "per_frame_results": [
                {"frame_idx": r["frame_idx"], "fake_probability": r["fake_probability"], "verdict": r["verdict"]}
                for r in frame_results
            ],
            "processing_time_sec": elapsed,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detectors/video_detector.py <path_to_video.mp4>")
        sys.exit(1)

    detector = VideoDetector(max_frames=20)
    result = detector.detect(sys.argv[1])

    print("\n--- VIDEO DETECTION RESULT ---")
    for k, v in result.items():
        if k not in ("most_suspicious_frame", "per_frame_results"):
            print(f"{k}: {v}")
    print(f"\nMost suspicious frame: #{result['most_suspicious_frame']['frame_index']} "
          f"(fake_prob={result['most_suspicious_frame']['fake_probability']:.3f})")
    print(f"Blink signal: {result['blink_analysis']['signal']} — {result['blink_analysis']['note']}")
