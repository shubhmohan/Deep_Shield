"""
DeepShield - Blink Irregularity Analyzer
Uses MediaPipe Face Mesh eye landmarks to compute the Eye Aspect Ratio (EAR)
across sampled video frames and flags suspiciously abnormal blink patterns —
e.g. no detected blinks at all across the whole sample, which is a classic
early-generation-GAN tell (many older deepfake models struggled to render
natural blinking).

IMPORTANT — SCOPE HONESTY FOR YOUR PITCH:
This is a SUPPLEMENTARY heuristic signal, not a trained classifier, and it
is NOT fed into the ensemble's fake_probability score. Reasons:
  1. We have no labeled training data for "natural vs unnatural blink
     timing" — building a real classifier for this needs a dataset we
     don't have time to assemble this week.
  2. With only ~20 sampled frames per video (not every frame), blink
     detection is inherently approximate — a real blink lasts 100-400ms
     and could fall between our samples entirely.
Present this to judges as "an additional forensic signal we surface
alongside the ML verdict" — that's accurate and still impressive — rather
than claiming it drives the classification, which would be overstating it
under technical questioning.
"""
import mediapipe as mp
import numpy as np
import cv2

# Standard 6-point eye landmark indices from MediaPipe Face Mesh (468 landmarks)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_BLINK_THRESHOLD = 0.21  # below this = eye considered closed (standard value from EAR literature)


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    vertical_1 = _euclidean(pts[1], pts[5])
    vertical_2 = _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    if horizontal == 0:
        return None
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class BlinkAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def analyze(self, frames_bgr: list) -> dict:
        """
        frames_bgr: list of raw video frames (numpy arrays, BGR) — pass the
        SAME sampled frames used for face cropping in video_prep.py, before
        cropping (Face Mesh needs the full frame, not a tight face crop).

        Returns a dict with the EAR time series and a plain-language flag.
        """
        ear_series = []

        for frame in frames_bgr:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                ear_series.append(None)
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)

            if left_ear is None or right_ear is None:
                ear_series.append(None)
            else:
                ear_series.append((left_ear + right_ear) / 2.0)

        valid_ears = [e for e in ear_series if e is not None]
        blink_frames = sum(1 for e in valid_ears if e < EAR_BLINK_THRESHOLD)

        if len(valid_ears) < 3:
            return {
                "signal": "inconclusive",
                "note": "Too few frames had a clearly detected face to assess blinking.",
                "ear_series": ear_series,
            }

        if blink_frames == 0:
            signal = "suspicious"
            note = ("No blinks detected across all sampled frames. This can indicate an "
                    "AI-generated face (older generation models often fail to render natural "
                    "blinking), but can also just mean the subject didn't blink during your "
                    "sampled frames — treat as a secondary signal, not standalone proof.")
        else:
            signal = "normal"
            note = f"Detected {blink_frames} likely-closed-eye frame(s) out of {len(valid_ears)} analyzed — consistent with natural blinking."

        return {
            "signal": signal,
            "note": note,
            "blink_frame_count": blink_frames,
            "frames_analyzed": len(valid_ears),
            "ear_series": ear_series,
        }


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from preprocessors.video_prep import VideoPreprocessor

    if len(sys.argv) < 2:
        print("Usage: python blink_analyzer.py <path_to_video.mp4>")
        sys.exit(1)

    prep = VideoPreprocessor()
    frames = prep.extract_frames(sys.argv[1])

    analyzer = BlinkAnalyzer()
    result = analyzer.analyze(frames)
    print(f"Signal: {result['signal']}")
    print(f"Note: {result['note']}")
