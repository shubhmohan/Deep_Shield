"""
DeepShield - DF40 CLIP Wrapper
Loads the CLIP ViT-L/14 deepfake detector released with DF40
(https://github.com/YZY-stack/DF40, CC BY-NC 4.0) WITHOUT needing the
DeepfakeBench_DF40 codebase.

The original detector (DeepfakeBench_DF40/training/detectors/clip_detector.py) is:
    backbone = CLIPModel("openai/clip-vit-large-patch14").vision_model
    head     = nn.Linear(1024, 2)
    forward  = head(backbone(image).pooler_output)
    prob     = softmax(logits)[:, 1]        # index 1 = FAKE
We rebuild exactly that structure from a hard-coded CLIP-L/14 config, so the
checkpoint's keys (backbone.*, head.*) line up and nothing is downloaded from
Hugging Face at runtime.

Preprocessing the model expects (from training/config/detector/clip.yaml):
    224x224, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], face-cropped.
That is identical to the ViT branch in preprocessors/image_prep.py, so the
existing "vit_input" tensor can be fed straight in.

Put your downloaded file at:  checkpoints/df40/clip.pth
(or set the DF40_CLIP_CHECKPOINT environment variable to another path).
"""

import os
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF40_CLIP_CHECKPOINT = os.environ.get(
    "DF40_CLIP_CHECKPOINT",
    os.path.join(PROJECT_ROOT, "checkpoints", "df40", "clip.pth"),
)

CLIP_INPUT_SIZE = 224
CLIP_MEAN = [0.5, 0.5, 0.5]
CLIP_STD = [0.5, 0.5, 0.5]


def _clip_large_patch14_vision_config() -> CLIPVisionConfig:
    """openai/clip-vit-large-patch14 vision tower, hard-coded so no download is needed."""
    return CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        projection_dim=768,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
    )


class DF40CLIPDetector(nn.Module):
    def __init__(self, checkpoint_path: str = DF40_CLIP_CHECKPOINT, device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.source = "df40_clip"
        self.fake_idx = 1  # DeepfakeBench convention: label 1 = fake

        # DF40 used CLIPModel.vision_model, so checkpoint keys are backbone.* / head.*
        # transformers 4.x: CLIPVisionModel wraps it as .vision_model
        # transformers 5.x: CLIPVisionModel IS that module (same layer names, no wrapper)
        vision = CLIPVisionModel(_clip_large_patch14_vision_config())
        self.backbone = getattr(vision, "vision_model", vision)
        self.head = nn.Linear(1024, 2)

        self._load_checkpoint(checkpoint_path)
        self.to(self.device)
        self.eval()

    def _load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DF40 CLIP checkpoint not found at '{path}'. Download clip.pth from the DF40 "
                f"README (df40_weights/train_on_fs/clip.pth) and put it there, or set "
                f"DF40_CLIP_CHECKPOINT."
            )
        ckpt = torch.load(path, map_location="cpu")

        # Some checkpoints wrap the weights; DeepfakeBench's own files are a plain state_dict.
        if isinstance(ckpt, dict):
            for wrapper_key in ("state_dict", "model_state_dict", "model"):
                if wrapper_key in ckpt and isinstance(ckpt[wrapper_key], dict):
                    ckpt = ckpt[wrapper_key]
                    break

        # Trained with DataParallel/DDP -> keys start with "module."
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}

        missing, unexpected = self.load_state_dict(state, strict=False)
        # position_ids is a non-persistent buffer in newer transformers; harmless either way.
        missing = [k for k in missing if "position_ids" not in k]
        unexpected = [k for k in unexpected if "position_ids" not in k]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint does not match the DF40 CLIP-L/14 detector layout.\n"
                f"  missing keys   ({len(missing)}): {missing[:5]}\n"
                f"  unexpected keys({len(unexpected)}): {unexpected[:5]}\n"
                "Is this the CLIP checkpoint (not the Xception one)? Xception files need a "
                "different wrapper."
            )
        print(f"[DF40CLIPDetector] Loaded DF40 CLIP-L/14 weights from {path}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(pixel_values=pixel_values).pooler_output
        return self.head(pooled)

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> dict:
        """pixel_values: [1, 3, 224, 224], normalized with mean/std 0.5. Returns fake probability."""
        logits = self(pixel_values.to(self.device))
        fake_prob = torch.softmax(logits, dim=-1)[0, self.fake_idx].item()
        return {"fake_prob": fake_prob, "logits": logits}


if __name__ == "__main__":
    det = DF40CLIPDetector()
    out = det.predict(torch.randn(1, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))
    print(f"fake_prob on random noise (meaningless, just a load check): {out['fake_prob']:.4f}")