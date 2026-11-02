"""
DeepShield - Branch diagnostics + threshold calibration

USAGE (from the repo root):
  python tests/calibrate_threshold.py --real_dir sample_data/real --fake_dir sample_data/fake
  python tests/calibrate_threshold.py --no_face_crop      # compare with cropping off
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.image_detector import ImageDetector

EXT = (".jpg", ".jpeg", ".png", ".webp")


def auc(pos, neg):
    """Probability a random fake scores higher than a random real (ties = 0.5)."""
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def collect(detector, folder):
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(EXT):
            continue
        with open(os.path.join(folder, f), "rb") as fh:
            r = detector.detect(fh.read(), f)
        rows.append((f, r["fake_probability"], r["model_breakdown"]["vit_score"],
                     r["model_breakdown"]["efficientnet_score"], r.get("face_detected")))
    return rows


def best_threshold(real_scores, fake_scores):
    best_t, best_acc = 0.5, 0.0
    total = len(real_scores) + len(fake_scores)
    for i in range(1, 100):
        t = i / 100
        correct = sum(s < t for s in real_scores) + sum(s >= t for s in fake_scores)
        if correct / total > best_acc:
            best_acc, best_t = correct / total, t
    return best_t, best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="sample_data/real")
    ap.add_argument("--fake_dir", default="sample_data/fake")
    ap.add_argument("--no_face_crop", action="store_true")
    args = ap.parse_args()

    det = ImageDetector(use_face_crop=not args.no_face_crop)
    real = collect(det, args.real_dir)
    fake = collect(det, args.fake_dir)

    if not args.no_face_crop:
        found = sum(1 for r in real + fake if r[4])
        print(f"\nFace found in {found}/{len(real) + len(fake)} images "
              f"(the rest were analyzed uncropped)")

    print(f"\n{'branch':<14}{'AUC':>8}{'acc@0.5':>10}{'best thr':>10}{'acc@best':>10}")
    for name, idx in [("ensemble", 1), ("vit", 2), ("efficientnet", 3)]:
        rs = [r[idx] for r in real]
        fs = [r[idx] for r in fake]
        acc05 = (sum(s < 0.5 for s in rs) + sum(s >= 0.5 for s in fs)) / (len(rs) + len(fs))
        t, a = best_threshold(rs, fs)
        print(f"{name:<14}{auc(fs, rs):>8.3f}{acc05*100:>9.1f}%{t:>10.2f}{a*100:>9.1f}%")

    print("\nHow to read this:")
    print("  AUC ~0.5        -> that branch has no signal on your images (needs retraining)")
    print("  AUC >0.8 but acc@0.5 low -> signal exists, cutoff is wrong; use 'best thr'")
    print("  Note: 'acc@best' is tuned on these same images, so treat it as optimistic.")