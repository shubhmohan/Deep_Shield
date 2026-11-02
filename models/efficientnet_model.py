"""
DeepShield - EfficientNet-B4 Wrapper
Loads an EfficientNet-B4 backbone (via timm) with a binary real/fake head.
This branch specializes in catching pixel-level GAN artifacts and blending
seams that CNNs pick up on more reliably than transformers.

UPDATED after real-world testing (see tests/validate_pipeline.py results):
the remote Yashikaysn29/deepshield checkpoint scored EXACTLY 0.000 fake
probability on all 36 test images (real and fake alike) — zero variation
means it wasn't discriminating at all, i.e. broken or degenerate. It is
NO LONGER the default. Priority order now:
  1. Local checkpoint you trained yourself (models/train_efficientnet.py) —
     BEST, trained on the exact dataset family your test fakes come from.
  2. Untrained ImageNet backbone — fallback if you haven't trained yet,
     effectively random, exists so the pipeline doesn't crash.
  (The remote Yashikaysn29 checkpoint can still be loaded manually via
  load_broken_remote_checkpoint_anyway=True if you want to re-test it
  yourself, but it is not recommended.)
"""

import os
import torch
import torch.nn as nn
import timm

LOCAL_CHECKPOINT_PATH = "./checkpoints/efficientnet_b4_deepfake.pt"

# Kept only for reference / manual re-testing — proven non-discriminative
# in our own validation run, do not use as your primary path.
BROKEN_REMOTE_HF_CHECKPOINT_REPO = "Yashikaysn29/deepshield"
BROKEN_REMOTE_HF_CHECKPOINT_FILE = "best_model.pth"


class _DeepshieldHead(nn.Module):
    """Only used if you explicitly opt into re-testing the broken remote checkpoint."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class EfficientNetDeepfakeDetector(nn.Module):
    def __init__(self, local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 load_broken_remote_checkpoint_anyway: bool = False, device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.using_finetuned = False
        self.source = None

        if os.path.exists(local_checkpoint_path):
            # Priority 1: your own trained checkpoint (standard 2-class timm model)
            checkpoint = torch.load(local_checkpoint_path, map_location="cpu")
            self.model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            class_to_idx = checkpoint["class_to_idx"]  # e.g. {"fake": 0, "real": 1}
            self.fake_idx = class_to_idx["fake"]
            self.using_finetuned = True
            self.source = "local_trained"
            self.target_layer = self.model.conv_head
            print(f"[EfficientNetDeepfakeDetector] Loaded YOUR trained checkpoint from "
                  f"{local_checkpoint_path} (val_acc={checkpoint.get('val_accuracy', '?')}, "
                  f"fake_idx={self.fake_idx})")

        elif load_broken_remote_checkpoint_anyway:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=BROKEN_REMOTE_HF_CHECKPOINT_REPO, filename=BROKEN_REMOTE_HF_CHECKPOINT_FILE
            )
            self.model = _DeepshieldHead()
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            self.using_finetuned = True
            self.source = "remote_broken"
            self.target_layer = self.model.backbone.conv_head
            print("[EfficientNetDeepfakeDetector] Loaded the REMOTE checkpoint you explicitly "
                  "asked to re-test — this scored 0.000 on every image in our validation run.")

        else:
            self.model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)
            self.fake_idx = 1
            self.source = "untrained"
            self.target_layer = self.model.conv_head
            print("[EfficientNetDeepfakeDetector] WARNING: no local checkpoint found — "
                  "predictions are near-random. Run models/train_efficientnet.py first.")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> dict:
        """
        pixel_values: preprocessed tensor of shape [1, 3, 224, 224] (matches
        train_efficientnet.py's IMAGE_SIZE) if using a local trained checkpoint
        or the untrained fallback; [1, 3, 224, 224] also for the broken remote
        checkpoint's sigmoid path (kept for compatibility, not recommended).
        Returns: {"fake_prob": float, "logits": tensor}
        """
        pixel_values = pixel_values.to(self.device)

        if self.source == "remote_broken":
            prob_real = self.model(pixel_values)
            fake_prob = 1.0 - prob_real.item()
            return {"fake_prob": fake_prob, "logits": prob_real}
        else:
            logits = self.model(pixel_values)
            probs = torch.softmax(logits, dim=-1)
            fake_prob = probs[0, self.fake_idx].item()
            return {"fake_prob": fake_prob, "logits": logits}

    def predict_with_grad(self, pixel_values: torch.Tensor):
        """
        Same as predict() but WITHOUT no_grad — required for Grad-CAM, which
        needs gradients to flow back to the target conv layer.
        """
        pixel_values = pixel_values.to(self.device)
        pixel_values.requires_grad_(True)
        logits = self.model(pixel_values)
        return logits, pixel_values


if __name__ == "__main__":
    detector = EfficientNetDeepfakeDetector()
    dummy_input = torch.randn(1, 3, 224, 224)
    result = detector.predict(dummy_input)
    print(f"Source: {detector.source} | Fake probability (random noise, expect ~meaningless): {result['fake_prob']:.4f}")
