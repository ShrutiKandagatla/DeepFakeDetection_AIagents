import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm import tqdm
import os
import random
import numpy as np
from collections import Counter
import csv
from datetime import datetime

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
DATA_DIR = "Data Set 1"   # should contain train/real, train/fake, validation/real, validation/fake
BATCH_SIZE = 32
IMG_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_deepfake_v2s.pth"
LAST_CHECKPOINT_PATH = "last_checkpoint.pth"
EARLY_STOP_PATIENCE = 5
SEED = 42

# Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==========================================================
# 2. DATA PREPROCESSING (simplified augmentation)
# ==========================================================
def main():
    # Transforms
    train_transform = transforms.Compose([
        # Stronger but safe augmentations
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=val_transform)

    # Class balancing via WeightedRandomSampler (helps on imbalanced datasets)
    label_list = train_data.targets  # list of int labels
    class_counts = Counter(label_list)
    class_sample_counts = [class_counts[i] for i in range(len(train_data.classes))]
    print(f"📊 Class counts (train): {class_counts}")

    weights_per_class = [0.0] * len(train_data.classes)
    for i, c in enumerate(class_sample_counts):
        weights_per_class[i] = 1.0 / max(c, 1)

    sample_weights = [weights_per_class[label] for label in label_list]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ Classes: {train_data.classes}")
    print(f"🖼️ Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Model — EfficientNet V2-S
    print("🔧 Loading EfficientNetV2-S pretrained on ImageNet...")
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    # Unfreeze strategy: train head + last blocks
    for name, param in model.features.named_parameters():
        if "6" in name or "7" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(DEVICE)

    # Loss, optimizer, scheduler, AMP scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    use_amp = DEVICE.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_val_acc = 0.0
    epochs_no_improve = 0

    # Metrics history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch [{epoch+1}/{EPOCHS}]")

        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100 * correct / total
        print(f"✅ Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        epoch_train_loss = train_loss/len(train_loader)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        class_correct = [0 for _ in range(len(train_data.classes))]
        class_total = [0 for _ in range(len(train_data.classes))]
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

                for i in range(labels.size(0)):
                    label_i = labels[i].item()
                    class_total[label_i] += 1
                    if preds[i].item() == label_i:
                        class_correct[label_i] += 1

        val_acc = 100 * correct / total
        epoch_val_loss = val_loss/len(val_loader)
        print(f"📊 Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        for idx, cls in enumerate(train_data.classes):
            if class_total[idx] > 0:
                pc_acc = 100.0 * class_correct[idx] / class_total[idx]
                print(f"   └─ Class '{cls}' Acc: {pc_acc:.2f}%  ({class_correct[idx]}/{class_total[idx]})")

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)

        # Quick epoch verdict
        gap = train_acc - val_acc
        if len(history['epoch']) >= 3:
            last3_val = history['val_loss'][-3:]
            last3_train = history['train_loss'][-3:]
            val_trend_up = last3_val[-1] > last3_val[0] + 0.02  # small margin
            train_trend_down = last3_train[-1] < last3_train[0] - 0.02
            if gap > 10 and val_trend_up and train_trend_down:
                print("⚠️ Potential overfitting detected (train>>val and diverging losses)")
            elif train_acc < 70 and val_acc < 70 and not train_trend_down:
                print("ℹ️ Potential underfitting (both accuracies low; training not improving)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Best model saved! New Val Acc: {best_val_acc:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save last checkpoint (resume support)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'seed': SEED
        }, LAST_CHECKPOINT_PATH)

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"⏹️ Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.")
            break

    # Summary
    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📁 Model saved as: {MODEL_PATH}")

    # Save metrics to CSV and JSON
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"training_metrics_{ts}.csv"
    json_path = f"training_metrics_{ts}.json"
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(history.keys())
            for i in range(len(history['epoch'])):
                writer.writerow([history[k][i] for k in history.keys()])
        import json
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 Metrics saved: {csv_path}, {json_path}")
    except Exception as e:
        print(f"❌ Failed to save metrics: {e}")

    # Final over/underfitting diagnosis
    verdict = "undetermined"
    reason = []
    if len(history['epoch']) >= 3:
        gap = history['train_acc'][-1] - history['val_acc'][-1]
        val_loss_trend = history['val_loss'][-1] - history['val_loss'][max(0, len(history['val_loss'])-4)]
        train_loss_trend = history['train_loss'][-1] - history['train_loss'][max(0, len(history['train_loss'])-4)]
        if gap > 10 and val_loss_trend > 0 and train_loss_trend < 0:
            verdict = "overfitting"
            reason.append(f"Train-Validation acc gap {gap:.1f}% with val loss rising and train loss falling")
        elif history['train_acc'][-1] < 70 and history['val_acc'][-1] < 70 and train_loss_trend > -0.01:
            verdict = "underfitting"
            reason.append("Both accuracies <70% and training loss not decreasing meaningfully")
        else:
            verdict = "well-fit / balanced"
            reason.append("Small gap and no divergent loss trends")
    print(f"🧪 Fit diagnosis: {verdict} — {'; '.join(reason) if reason else 'insufficient epochs for robust diagnosis'}")


if __name__ == "__main__":
    main()
    