"""
DeepShield - Grad-CAM for the DF40 CLIP (ViT) branch.

The CNN Grad-CAM in explainer.py hooks a conv layer. A ViT has no conv maps,
so we use the standard ViT adaptation: take the patch-token activations at an
encoder layer, weight each channel by its average gradient, sum, ReLU, and
reshape the 256 patch tokens to a 16x16 grid (224 / patch 14 = 16).

Why layers[-2] and not the last layer: the classifier only reads the CLS token
of the LAST layer's output, so the gradient w.r.t. that layer's patch tokens is
exactly zero. The second-to-last layer's output is what the final attention
block mixes into CLS, so its patch gradients carry real signal.

Same interface as explainer.GradCAM (generate / overlay_heatmap), so
ImageDetector can swap it in without other changes.
"""

import numpy as np
import torch
import torch.nn.functional as F

from detectors.explainer import GradCAM


class ViTGradCAM:
    def __init__(self, detector, layer_index: int = -2):
        self.detector = detector
        self.layer = detector.backbone.encoder.layers[layer_index]
        self.activations = None
        self.gradients = None
        self.layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        tokens = output[0] if isinstance(output, (tuple, list)) else output
        self.activations = tokens.detach()
        if tokens.requires_grad:
            tokens.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """Returns a [H, W] heatmap in 0-1 explaining evidence for `class_idx` (1 = fake)."""
        self.detector.zero_grad()
        x = input_tensor.to(self.detector.device)
        with torch.enable_grad():
            logits = self.detector(x)
            logits[0, class_idx].backward()

        acts = self.activations[0, 1:, :]    # drop CLS -> [256, C]
        grads = self.gradients[0, 1:, :]     # [256, C]
        weights = grads.mean(dim=0)          # [C]
        cam = F.relu((acts * weights).sum(dim=-1))   # [256]

        side = int(round(cam.numel() ** 0.5))
        cam = cam.reshape(1, 1, side, side)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        return cam / (cam.max() + 1e-8)

    # Reuse the exact overlay code from the CNN Grad-CAM (it never touches self state).
    overlay_heatmap = GradCAM.overlay_heatmap