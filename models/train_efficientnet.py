"""
DeepShield - EfficientNet-B4 Fine-Tuning Script
Fine-tunes the EfficientNet-B4 branch on the Kaggle "140k Real and Fake Faces"
dataset (https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).

WHY WE TRAIN THIS BRANCH OURSELVES INSTEAD OF IMPORTING A CHECKPOINT:
1. You have a full week now, not 24 hours — there's time to do this properly.
2. It gives you YOUR OWN confusion matrix, ROC curve, and accuracy numbers to
   put in your report/pitch — much more credible to judges than "we used a
   pretrained model" for both branches.
3. Avoids depending on a fragile, low-visibility third-party checkpoint that
   requires vendoring someone else's custom class definitions.

HOW TO GET THE DATASET (do this first):
  1. Go to https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
  2. Download it (requires free Kaggle account) — it's ~4GB, structured as:
       real_vs_fake/real-vs-fake/train/real/*.jpg
       real_vs_fake/real-vs-fake/train/fake/*.jpg
       real_vs_fake/real-vs-fake/valid/real|fake/*.jpg
       real_vs_fake/real-vs-fake/test/real|fake/*.jpg
  3. Extract it and set DATASET_ROOT below to point at the extracted folder.

Expect this script to take a few hours on a consumer GPU for 5-8 epochs.
Run it once, then reuse the saved checkpoint — you don't need to retrain
every time you test the pipeline.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ---- CONFIG: adjust these for your machine ----
DATASET_ROOT = "./data/real_vs_fake/real-vs-fake"  # <- point this at your extracted Kaggle download
BATCH_SIZE = 32          # lower to 16 if you hit CUDA out-of-memory
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
IMAGE_SIZE = 380         # EfficientNet-B4 native resolution
CHECKPOINT_OUT = "./checkpoints/efficientnet_b4_deepfake.pt"
FREEZE_EARLY_LAYERS = True  # freeze early conv blocks, only fine-tune later blocks + head (faster, less overfitting risk)


def build_model():
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)

    if FREEZE_EARLY_LAYERS:
        # Freeze everything except the last few blocks and the classifier head.
        # This is faster to train and less prone to overfitting on a single dataset.
        for name, param in model.named_parameters():
            if "blocks.5" in name or "blocks.6" in name or "conv_head" in name \
               or "bn2" in name or "classifier" in name:
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, "valid"), transform=eval_transform)

    # ImageFolder assigns label indices alphabetically: "fake"=0, "real"=1 typically.
    # PRINT AND VERIFY THIS — do not assume. Our ensemble.py expects index 1 = fake.
    print(f"Class-to-index mapping: {train_ds.class_to_idx}")
    print("^ If 'fake' is not mapped to 1, flip the labels in ensemble_predict() or remap here.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. This will be extremely slow — consider Colab/Kaggle notebooks instead.")

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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        # Validation pass
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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
    print("Next: load this checkpoint in efficientnet_model.py (see load_finetuned_checkpoint below)")


if __name__ == "__main__":
    train()
