import os
import time
import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

SAVE_DIR = "weights_ResNet18_CIFAR_10"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "resnet18_best.pth")
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")
PLOT_PATH = os.path.join(SAVE_DIR, "train_vs_test_plot.png")

# =========================================================
# DATA TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# =========================================================
# CIFAR-10 DATASET
# =========================================================

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =========================================================
# MODEL
# =========================================================

model = models.resnet18(weights=None)

# Modify final layer for CIFAR-10
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(DEVICE)

# =========================================================
# LOSS & OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# =========================================================
# TRAINING VARIABLES
# =========================================================

best_test_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())

early_stop_counter = 0

train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

# =========================================================
# CREATE CSV LOG FILE
# =========================================================

with open(LOG_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Epoch",
        "Train Loss",
        "Train Accuracy",
        "Test Loss",
        "Test Accuracy"
    ])

# =========================================================
# TRAINING LOOP
# =========================================================

start_time = time.time()

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # =====================================================
    # TRAINING
    # =====================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training")

    for images, labels in train_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_bar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # =====================================================
    # TESTING
    # =====================================================

    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        test_bar = tqdm(test_loader, desc="Testing")

        for images, labels in test_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Test Loss  : {test_loss:.4f}")
    print(f"Test Acc   : {test_acc:.2f}%")

    # =====================================================
    # SAVE TRAIN LOG
    # =====================================================

    with open(LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1,
            train_loss,
            train_acc,
            test_loss,
            test_acc
        ])

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if test_loss < best_test_loss:

        best_test_loss = test_loss
        best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, MODEL_PATH)

        print("Best model saved.")

        early_stop_counter = 0

    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{PATIENCE}")

    # =====================================================
    # LEARNING RATE SCHEDULER
    # =====================================================

    scheduler.step(test_loss)

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if early_stop_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# =========================================================
# TOTAL TRAINING TIME
# =========================================================

end_time = time.time()

total_time = end_time - start_time

print(f"\nTotal Training Time: {total_time/60:.2f} minutes")

# =========================================================
# LOAD BEST MODEL
# =========================================================

model.load_state_dict(best_model_wts)

# =========================================================
# PLOT TRAIN VS TEST
# =========================================================

epochs_range = range(1, len(train_losses) + 1)

# ---------------- Loss Plot ----------------

plt.figure(figsize=(10, 5))

plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss (ResNet-18 on CIFAR-10)")
plt.legend()

plt.grid(True)

plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))

plt.close()

# ---------------- Accuracy Plot ----------------

plt.figure(figsize=(10, 5))

plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy (ResNet-18 on CIFAR-10)")

plt.legend()

plt.grid(True)

plt.savefig(os.path.join(SAVE_DIR, "accuracy_plot.png"))

plt.close()

print("\nTraining completed successfully.")

print(f"\nSaved Files Inside: {SAVE_DIR}")
print("1. Best Model (.pth)")
print("2. Training Log (.csv)")
print("3. Loss Plot (.png)")
print("4. Accuracy Plot (.png)")