import os, torch, torch.nn as nn
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF40_CLIP_CHECKPOINT = os.environ.get(
    "DF40_CLIP_CHECKPOINT", os.path.join(PROJECT_ROOT, "checkpoints", "df40", "clip.pth"))

def _cfg():
    return CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
        num_attention_heads=16, image_size=224, patch_size=14, projection_dim=768,
        hidden_act="quick_gelu", layer_norm_eps=1e-5)

class DF40CLIPDetector(nn.Module):
    def __init__(self, checkpoint_path=DF40_CLIP_CHECKPOINT, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fake_idx = 1
        self.backbone = CLIPVisionTransformer(_cfg())
        self.head = nn.Linear(1024, 2)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Put clip.pth at {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        for k in ("state_dict", "model_state_dict", "model"):
            if isinstance(ckpt, dict) and isinstance(ckpt.get(k), dict):
                ckpt = ckpt[k]; break
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        unexpected = [k for k in unexpected if "position_ids" not in k]
        if missing or unexpected:
            raise RuntimeError(f"Not the DF40 CLIP layout. missing={missing[:5]} unexpected={unexpected[:5]}")
        self.to(self.device).eval()

    def forward(self, x):
        return self.head(self.backbone(pixel_values=x).pooler_output)

    @torch.no_grad()
    def predict(self, pixel_values):
        logits = self(pixel_values.to(self.device))
        return {"fake_prob": torch.softmax(logits, -1)[0, self.fake_idx].item(), "logits": logits}