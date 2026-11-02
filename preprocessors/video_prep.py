"""
DeepShield - Video Preprocessing Module
Extracts a sampled set of frames from an input video and crops the face
region from each using MediaPipe Face Detection, ready to feed into the
same image-detection pipeline used for single images.
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from typing import Optional

# How many frames to sample per video. More = more accurate but slower.
# 16-24 frames is a reasonable balance for a hackathon demo — enough to
# catch temporal inconsistency without taking forever per video.
DEFAULT_MAX_FRAMES = 20

# Crop the face with this much margin around MediaPipe's bounding box,
# as a fraction of the box's width/height — a tight crop on just the box
# often clips the chin/forehead, which hurts both models' accuracy.
FACE_CROP_MARGIN = 0.35

MIN_DETECTION_CONFIDENCE = 0.5


class VideoPreprocessor:
    def __init__(self, max_frames: int = DEFAULT_MAX_FRAMES):
        self.max_frames = max_frames
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = better for faces farther from camera / full video frames
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )

    def extract_frames(self, video_path: str) -> list:
        """
        Returns a list of raw frames (numpy arrays, BGR) evenly sampled
        across the video's duration.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video appears to have no frames (corrupt file or unsupported codec).")

        # Evenly space our samples across the whole video rather than just
        # grabbing the first N frames — deepfake artifacts can be inconsistent
        # across a clip, so we want coverage from start to end.
        num_samples = min(self.max_frames, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError("Could not extract any readable frames from this video.")

        return frames

    def crop_face(self, frame_bgr: np.ndarray) -> Optional[Image.Image]:
        """
        Detects the largest face in the frame and returns it as a cropped
        PIL Image (RGB), or None if no face is detected in this frame.
        Frames with no detected face are skipped by the caller rather than
        forcing a whole-frame prediction, which would be unreliable — both
        our models expect a face-centered image.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_detector.process(frame_rgb)
        if not results.detections:
            return None

        # Pick the largest detected face (most likely the subject, not a background face)
        best_detection = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        box = best_detection.location_data.relative_bounding_box

        x1 = box.xmin * w
        y1 = box.ymin * h
        box_w = box.width * w
        box_h = box.height * h

        # Add margin around the tight face box
        margin_x = box_w * FACE_CROP_MARGIN
        margin_y = box_h * FACE_CROP_MARGIN

        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(w, int(x1 + box_w + 2 * margin_x))
        y2 = min(h, int(y1 + box_h + 2 * margin_y))

        if x2 <= x1 or y2 <= y1:
            return None

        cropped = frame_rgb[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    def get_face_crops(self, video_path: str) -> list:
        """
        Full pipeline: extract sampled frames, crop faces, return a list of
        (frame_index, PIL.Image) tuples for frames where a face was found.
        Frame index is preserved so the caller can report WHICH part of the
        video looked suspicious (e.g. "frame 14 of 20, ~timestamp X").
        """
        frames = self.extract_frames(video_path)
        crops = []
        for i, frame in enumerate(frames):
            face = self.crop_face(frame)
            if face is not None:
                crops.append((i, face))
        return crops


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_prep.py <path_to_video.mp4>")
        sys.exit(1)

    prep = VideoPreprocessor()
    crops = prep.get_face_crops(sys.argv[1])
    print(f"Extracted {len(crops)} face crops from sampled frames.")
    if crops:
        print(f"First crop size: {crops[0][1].size}")
