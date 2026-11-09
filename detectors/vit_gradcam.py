import numpy as np, torch, torch.nn.functional as F
from detectors.explainer import GradCAM

class ViTGradCAM:
    def __init__(self, detector, layer_index=-2):
        self.detector = detector
        self.acts = self.grads = None
        detector.backbone.encoder.layers[layer_index].register_forward_hook(self._hook)

    def _hook(self, m, i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        self.acts = t.detach()
        if t.requires_grad:
            t.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def generate(self, x, class_idx=1):
        self.detector.zero_grad()
        with torch.enable_grad():
            logits = self.detector(x.to(self.detector.device))
            logits[0, class_idx].backward()
        a, g = self.acts[0, 1:], self.grads[0, 1:]          # drop CLS -> [256, C]
        cam = F.relu((a * g.mean(0)).sum(-1))
        s = int(round(cam.numel() ** 0.5))
        cam = F.interpolate(cam.reshape(1, 1, s, s), size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy(); cam -= cam.min()
        return cam / (cam.max() + 1e-8)

    overlay_heatmap = GradCAM.overlay_heatmap