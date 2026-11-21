"""
DeepShield - quick evaluation of the DF40 CLIP detector on your own images.

USAGE (from the project root):
  python tests/test_df40.py --real_dir sample_data/real --fake_dir sample_data/fake
  python tests/test_df40.py --real_dir ... --fake_dir ... --no_face_crop

Prints per-image fake probability, accuracy at the 0.5 cutoff, and AUC.
AUC is the number to trust: if AUC is high but accuracy is low, the model separates
real from fake fine and only the cutoff (DEEPSHIELD_THRESHOLD) needs adjusting.
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.image_detector import ImageDetector

EXT = (".jpg", ".jpeg", ".png", ".webp")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score_folder(det, folder):
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(EXT):
            continue
        with open(os.path.join(folder, f), "rb") as fh:
            r = det.detect(fh.read(), f)
        rows.append((f, r["fake_probability"], r.get("face_detected")))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="sample_data/real")
    ap.add_argument("--fake_dir", default="sample_data/fake")
    ap.add_argument("--no_face_crop", action="store_true")
    args = ap.parse_args()

    det = ImageDetector(use_face_crop=not args.no_face_crop, mode="df40_clip")
    real = score_folder(det, args.real_dir)
    fake = score_folder(det, args.fake_dir)

    for label, rows in (("REAL", real), ("FAKE", fake)):
        print(f"\n--- known {label} ---")
        for name, p, face in rows:
            print(f"  {name:<32} fake_prob={p:.3f}  face={'yes' if face else 'NO'}")

    rs, fs = [r[1] for r in real], [r[1] for r in fake]
    total = len(rs) + len(fs)
    acc = (sum(s < 0.5 for s in rs) + sum(s >= 0.5 for s in fs)) / max(total, 1)
    print(f"\nImages: {len(rs)} real, {len(fs)} fake")
    print(f"Accuracy @0.5: {acc*100:.1f}%    AUC: {auc(fs, rs):.3f}")
    if rs and fs:
        print(f"Mean fake_prob  real={sum(rs)/len(rs):.3f}  fake={sum(fs)/len(fs):.3f}")