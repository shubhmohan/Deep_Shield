"""
DeepShield - Pipeline Validation Script
Runs the full detector against a folder of known-REAL and known-FAKE images
and reports accuracy, confusion matrix, and per-image results.

HOW TO GET TEST IMAGES:
  Real faces:  any 10-15 photos of real people (your own photos are fine, or
               use a public face dataset like FFHQ samples)
  Fake faces:  https://thispersondoesnotexist.com/  -- refresh the page to
               generate a new AI face each time, save 10-15 of them

USAGE:
  python tests/validate_pipeline.py --real_dir sample_data/real --fake_dir sample_data/fake
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.image_detector import ImageDetector


def load_images_from_dir(dir_path):
    valid_ext = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(valid_ext)]
    return sorted(files)


def run_validation(real_dir: str, fake_dir: str):
    detector = ImageDetector()

    results = []
    correct = 0
    total = 0

    print(f"\n{'='*70}\nTesting REAL images from: {real_dir}\n{'='*70}")
    for fname in load_images_from_dir(real_dir):
        path = os.path.join(real_dir, fname)
        with open(path, "rb") as f:
            file_bytes = f.read()
        try:
            result = detector.detect(file_bytes, fname)
            is_correct = result["verdict"] == "REAL"
            correct += is_correct
            total += 1
            results.append((fname, "REAL", result["verdict"], result["fake_probability"], is_correct))
            status = "✓" if is_correct else "✗ MISCLASSIFIED"
            bd = result["model_breakdown"]
            print(f"  {fname:30s} | verdict: {result['verdict']:5s} | ensemble: {result['fake_probability']:.3f} | "
                  f"vit: {bd['vit_score']:.3f} | eff: {bd['efficientnet_score']:.3f} | {status}")
        except Exception as e:
            print(f"  {fname:30s} | ERROR: {e}")

    print(f"\n{'='*70}\nTesting FAKE images from: {fake_dir}\n{'='*70}")
    for fname in load_images_from_dir(fake_dir):
        path = os.path.join(fake_dir, fname)
        with open(path, "rb") as f:
            file_bytes = f.read()
        try:
            result = detector.detect(file_bytes, fname)
            is_correct = result["verdict"] == "FAKE"
            correct += is_correct
            total += 1
            results.append((fname, "FAKE", result["verdict"], result["fake_probability"], is_correct))
            status = "✓" if is_correct else "✗ MISCLASSIFIED"
            bd = result["model_breakdown"]
            print(f"  {fname:30s} | verdict: {result['verdict']:5s} | ensemble: {result['fake_probability']:.3f} | "
                  f"vit: {bd['vit_score']:.3f} | eff: {bd['efficientnet_score']:.3f} | {status}")
        except Exception as e:
            print(f"  {fname:30s} | ERROR: {e}")

    print(f"\n{'='*70}")
    if total > 0:
        acc = correct / total * 100
        print(f"RESULT: {correct}/{total} correct ({acc:.1f}% accuracy)")
        if acc < 70:
            print("\n⚠️  Accuracy is low. Likely causes, check in this order:")
            print("   1. A checkpoint silently fell back to untrained weights —")
            print("      re-check the console output above for 'WARNING: could not load'")
            print("   2. The label convention flip in efficientnet_model.py is backwards —")
            print("      try inverting fake_prob there and re-running")
            print("   3. The images aren't clear frontal faces (both models expect faces)")
    else:
        print("No images found — check your --real_dir and --fake_dir paths.")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", default="sample_data/real")
    parser.add_argument("--fake_dir", default="sample_data/fake")
    args = parser.parse_args()

    os.makedirs(args.real_dir, exist_ok=True)
    os.makedirs(args.fake_dir, exist_ok=True)

    run_validation(args.real_dir, args.fake_dir)
