"""
DeepShield - Image Preprocessing Module
Loads, validates, resizes, and normalizes images for the ViT / EfficientNet branches.

IMPORTANT: every constant below must match what the training scripts used.
  - train_vit.py          -> 224x224, Normalize([0.5]*3, [0.5]*3)
  - train_efficientnet.py -> 380x380, Normalize(ImageNet mean/std)
A mismatch here does not crash anything; it silently wrecks accuracy on
unseen images while Kaggle validation still looks great.
"""

from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import io

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]

VIT_INPUT_SIZE = 224
EFFICIENTNET_INPUT_SIZE = 380

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE_MB = 20


class ImagePreprocessor:
    def __init__(self):
        self.vit_transform = transforms.Compose([
            transforms.Resize((VIT_INPUT_SIZE, VIT_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
        ])
        self.efficientnet_transform = transforms.Compose([
            transforms.Resize((EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load_image(self, file_bytes: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not decode image: {e}")

    def validate(self, file_bytes: bytes, filename: str) -> None:
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
        ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""
        if ext == "JPG":
            ext = "JPEG"
        if ext not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Allowed: {ALLOWED_FORMATS}")

    def preprocess(self, image: Image.Image) -> dict:
        vit_tensor = self.vit_transform(image).unsqueeze(0)
        eff_tensor = self.efficientnet_transform(image).unsqueeze(0)
        display_img = image.resize((EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE))
        display_array = np.array(display_img) / 255.0
        return {
            "vit_input": vit_tensor,
            "efficientnet_input": eff_tensor,
            "display_array": display_array,
        }