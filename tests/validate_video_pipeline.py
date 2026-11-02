import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Replace lines 9-16 with just this:
from detectors.video_detector import VideoDetector

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def scan_videos(dir_path: Path) -> List[Path]:
    """Recursively collect all video files from a directory."""
    if not dir_path.exists():
        return []
    return [p for p in dir_path.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]


def run_validation(
    real_dir: str,
    fake_dir: str,
    threshold: float = 0.5,
):
    real_path = Path(real_dir)
    fake_path = Path(fake_dir)

    real_videos = scan_videos(real_path)
    fake_videos = scan_videos(fake_path)

    if not real_videos and not fake_videos:
        print(f"[!] No videos found in:\n  - {real_path}\n  - {fake_path}")
        return

    # Pair each video with its label: 0 for REAL, 1 for FAKE
    dataset = [(p, 0) for p in real_videos] + [(p, 1) for p in fake_videos]

    print(f"[*] Initializing VideoDetector pipeline...")
    detector = VideoDetector() if callable(VideoDetector) else None

    print(f"[*] Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")
    print(f"[*] Total to process: {len(dataset)}\n")
    print(f"{'Video File':<35} {'Ground Truth':<12} {'Pred Prob':<12} {'Verdict':<8}")
    print("-" * 72)

    total_samples = 0
    skipped = 0
    tp = fp = tn = fn = 0

    for video_file, gt_is_fake in dataset:
        total_samples += 1
        raw_label = "FAKE" if gt_is_fake else "REAL"

        try:
            if hasattr(detector, "predict"):
                pred = detector.predict(str(video_file))
            elif hasattr(detector, "detect"):
                pred = detector.detect(str(video_file))
            else:
                pred = detector(str(video_file))

            # Extract confidence score for FAKE class
            if isinstance(pred, dict):
                fake_prob = pred.get("fake_probability", pred.get("confidence", 0.0))
                if "label" in pred and isinstance(pred["label"], str):
                    is_pred_fake = 1 if pred["label"].upper() == "FAKE" else 0
                else:
                    is_pred_fake = 1 if fake_prob >= threshold else 0
            else:
                fake_prob = float(pred)
                is_pred_fake = 1 if fake_prob >= threshold else 0

        except Exception as e:
            print(f"[!] Error processing {video_file.name}: {e}")
            skipped += 1
            total_samples -= 1
            continue

        # Confusion Matrix updates (Positive = FAKE, Negative = REAL)
        if gt_is_fake and is_pred_fake:
            tp += 1
            status = "CORRECT"
        elif not gt_is_fake and not is_pred_fake:
            tn += 1
            status = "CORRECT"
        elif not gt_is_fake and is_pred_fake:
            fp += 1
            status = "WRONG (FP)"
        else:
            fn += 1
            status = "WRONG (FN)"

        print(f"{video_file.name:<35} {raw_label:<12} {fake_prob:<12.4f} {status:<8}")

    if total_samples == 0:
        print("\n[!] Zero videos successfully evaluated.")
        return

    # Metrics
    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n" + "=" * 50)
    print("           VALIDATION SUMMARY REPORT             ")
    print("=" * 50)
    print(f"Total Evaluated : {total_samples}")
    print(f"Skipped / Failed: {skipped}")
    print("-" * 50)
    print(f"True Positives  (Fake correctly caught) : {tp}")
    print(f"True Negatives  (Real correctly cleared): {tn}")
    print(f"False Positives (Real misidentified)    : {fp}")
    print(f"False Negatives (Fake missed)           : {fn}")
    print("-" * 50)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1_score * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Folder-based Batch Video Pipeline Validator"
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default="sample_data/real_videos",
        help="Path to directory containing real videos",
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="sample_data/fake_videos",
        help="Path to directory containing fake videos",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold to classify as FAKE",
    )

    args = parser.parse_args()
    run_validation(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        threshold=args.threshold,
    )