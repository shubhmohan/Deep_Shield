"""
DeepShield - Vision Transformer Wrapper
Loads a ViT-B/16 backbone fine-tuned for real-vs-fake deepfake classification.

UPDATED after real-world testing (see tests/validate_pipeline.py results):
the remote dima806/deepfake_vs_real_image_detection checkpoint only caught
2 of 18 known-fake test images — real but weak, likely concept drift on
modern StyleGAN-style faces. Priority order now:
  1. Local checkpoint you trained yourself (models/train_vit.py) — BEST,
     trained on the exact dataset family your test fakes come from.
  2. Remote dima806 checkpoint — kept as a fallback if you haven't trained
     yet, but treat its output with skepticism (see test results above).
  3. Untrained ImageNet backbone — last resort, effectively random.
"""

import os
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig

LOCAL_CHECKPOINT_PATH = "./checkpoints/vit_deepfake.pt"
REMOTE_HF_CHECKPOINT = "dima806/deepfake_vs_real_image_detection"

# Model card recommends lowering this from 0.5 if you see false negatives —
# only relevant when using the remote checkpoint, not your own trained one.
FAKE_THRESHOLD = 0.5


class ViTDeepfakeDetector(nn.Module):
    def __init__(self, local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 use_remote_fallback: bool = True, device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.source = None

        if os.path.exists(local_checkpoint_path):
            # Priority 1: your own trained checkpoint
            checkpoint = torch.load(local_checkpoint_path, map_location="cpu")
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
            config.num_labels = 2
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", config=config, ignore_mismatched_sizes=True
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])

            class_to_idx = checkpoint["class_to_idx"]  # e.g. {"fake": 0, "real": 1}
            self.fake_idx = class_to_idx["fake"]
            self.source = "local_trained"
            print(f"[ViTDeepfakeDetector] Loaded YOUR trained checkpoint from {local_checkpoint_path} "
                  f"(val_acc={checkpoint.get('val_accuracy', '?')}, fake_idx={self.fake_idx})")

        elif use_remote_fallback:
            # Priority 2: remote checkpoint — works, but weak per our own testing
            self.model = ViTForImageClassification.from_pretrained(REMOTE_HF_CHECKPOINT)
            id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
            self.fake_idx = next(i for i, label in id2label.items() if "fake" in label)
            self.source = "remote_fallback"
            print(f"[ViTDeepfakeDetector] No local checkpoint found — using remote "
                  f"'{REMOTE_HF_CHECKPOINT}' (fake_idx={self.fake_idx}). "
                  f"WARNING: this checkpoint tested weak on modern deepfakes — "
                  f"train your own with models/train_vit.py when you can.")
        else:
            # Priority 3: untrained fallback
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
            config.num_labels = 2
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", config=config, ignore_mismatched_sizes=True
            )
            self.fake_idx = 1
            self.source = "untrained"
            print("[ViTDeepfakeDetector] WARNING: no checkpoint available — predictions are near-random.")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> dict:
        """
        pixel_values: preprocessed tensor of shape [1, 3, 224, 224]
        Returns: {"fake_prob": float, "logits": tensor}
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1)
        fake_prob = probs[0, self.fake_idx].item()
        return {"fake_prob": fake_prob, "logits": outputs.logits}

    def get_last_attention_layer(self):
        """Returns the last transformer block for Grad-CAM / attention rollout hooking."""
        return self.model.vit.encoder.layer[-1]


if __name__ == "__main__":
    detector = ViTDeepfakeDetector()
    dummy_input = torch.randn(1, 3, 224, 224)
    result = detector.predict(dummy_input)
    print(f"Source: {detector.source} | Fake probability (random noise, expect ~meaningless): {result['fake_prob']:.4f}")
