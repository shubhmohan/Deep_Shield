"""
DeepShield - Grad-CAM Explainability Module
Generates a heatmap over the input image showing WHICH regions drove the
EfficientNet-B4 branch's fake/real decision. This is what turns DeepShield
from a black-box classifier into an explainable one — judges will respond
strongly to seeing *why*, not just *what*.

We hook Grad-CAM onto EfficientNet-B4's final conv layer rather than ViT's
attention, because conv activation maps line up spatially with the image
much more directly, and are simpler & faster to compute under hackathon
time pressure. ViT explainability (attention rollout) is a good "if time
permits" stretch goal — see notes at the bottom.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        input_tensor: [1, 3, H, W], requires_grad should be enabled upstream
        class_idx: which class to explain when the model outputs multi-class
            logits (1 = "fake" in our two-class convention). IGNORED when the
            model has a single sigmoid output (our fine-tuned Yashikaysn29/deepshield
            checkpoint) — in that case we always explain "evidence pointing
            toward fake", regardless of this argument.
        Returns: normalized heatmap as a numpy array, shape [H, W], values 0-1
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        if output.shape[-1] == 1:
            # Single sigmoid output = P(real) for our fine-tuned checkpoint.
            # To explain "why fake", we backprop the negative of P(real) —
            # i.e. what pushes the score DOWN toward fake.
            score = -output[0, 0]
        else:
            score = output[0, class_idx]

        score.backward()

        # Global-average-pool the gradients to get per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps, then ReLU (we only care about positive influence)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1 for visualization
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def overlay_heatmap(self, heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        original_image: numpy array [H, W, 3], values 0-1 (from ImagePreprocessor.display_array)
        Returns: RGB numpy array [H, W, 3], values 0-1, ready to save/display
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

        original_uint8 = (original_image * 255).astype(np.uint8)
        if original_uint8.shape[:2] != heatmap_color.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (original_uint8.shape[1], original_uint8.shape[0]))

        overlay = (1 - alpha) * original_image + alpha * heatmap_color
        overlay = np.clip(overlay, 0, 1)
        return overlay


def get_top_suspicious_regions(heatmap: np.ndarray, top_n: int = 3) -> list:
    """
    Utility for the report generator: finds the top-N hottest bounding boxes
    in the heatmap so the PDF report can call out "manipulation detected near
    left eye / mouth boundary" style text instead of just showing a blob.
    """
    h, w = heatmap.shape
    grid_size = 4  # divide image into a 4x4 grid, rank cells by average heat
    cell_h, cell_w = h // grid_size, w // grid_size

    cell_scores = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = heatmap[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_scores.append((cell.mean(), i, j))

    cell_scores.sort(reverse=True)
    return cell_scores[:top_n]


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from models.efficientnet_model import EfficientNetDeepfakeDetector

    detector = EfficientNetDeepfakeDetector()
    cam = GradCAM(detector.model, detector.target_layer)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = cam.generate(dummy_input)
    print("Heatmap shape:", heatmap.shape, "min/max:", heatmap.min(), heatmap.max())

    dummy_display = np.random.rand(224, 224, 3)
    overlay = cam.overlay_heatmap(heatmap, dummy_display)
    print("Overlay shape:", overlay.shape)
