"""
DeepShield - ViT-B/16 Fine-Tuning Script
Fine-tunes ViT-B/16 on the same Kaggle "140k Real and Fake Faces" dataset
used for the EfficientNet branch (see train_efficientnet.py for download
instructions).

WHY WE'RE NOW TRAINING THIS OURSELVES TOO:
Real-world validation (tests/validate_pipeline.py) showed the pretrained
dima806/deepfake_vs_real_image_detection checkpoint only reliably caught
2 of 18 known-fake test images — its training data doesn't generalize well
to these faces (likely StyleGAN2 output, same family as our Kaggle dataset).
Training on the exact dataset that matches our test distribution should
perform much better, and gives you real, defensible metrics for your report.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig

DATASET_ROOT = "./data/real_vs_fake/real-vs-fake"  # same dataset as train_efficientnet.py
BATCH_SIZE = 32
NUM_EPOCHS = 6
LEARNING_RATE = 5e-5   # lower than EfficientNet's — ViT fine-tunes are more sensitive to LR
IMAGE_SIZE = 224
CHECKPOINT_OUT = "./checkpoints/vit_deepfake.pt"
FREEZE_EARLY_LAYERS = True


def build_model():
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    config.num_labels = 2
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", config=config, ignore_mismatched_sizes=True
    )

    if FREEZE_EARLY_LAYERS:
        # ViT-B/16 has 12 transformer layers (0-11). Freeze the first 8,
        # fine-tune the last 4 + classifier head — same rationale as the
        # EfficientNet trainer: faster, less prone to overfitting.
        for name, param in model.named_parameters():
            if any(f"encoder.layer.{i}." in name for i in range(8, 12)) or "classifier" in name or "layernorm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # ViT's expected normalization
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "valid"), transform=eval_transform)

    print(f"Class-to-index mapping: {train_ds.class_to_idx}")
    print("^ This gets saved into the checkpoint so vit_model.py knows which index is 'fake' — no guessing.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. ViT fine-tuning on CPU will be very slow.")

    model = build_model().to(device)
    train_loader, val_loader = get_dataloaders()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(CHECKPOINT_OUT), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images).logits
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} done in {elapsed/60:.1f}min | "
              f"Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "epoch": epoch + 1,
                "class_to_idx": train_loader.dataset.class_to_idx,
            }, CHECKPOINT_OUT)
            print(f"  -> New best model saved (val_acc={val_acc*100:.2f}%) to {CHECKPOINT_OUT}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved at: {CHECKPOINT_OUT}")


if __name__ == "__main__":
    train()
