import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os

# ==========================================================
# CONFIG
# ==========================================================
BATCH_SIZE = 16          # Reduce further if CUDA OOM
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
IMG_SIZE = 64
NUM_CLASSES = 47

torch.backends.cudnn.benchmark = True
os.makedirs("weights_EMNIST", exist_ok=True)

# ==========================================================
# TRANSFORMS
# ==========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ==========================================================
# DATASET
# ==========================================================
full_trainset = torchvision.datasets.EMNIST(
    root='./Dataset/EMNIST',
    split='balanced',
    train=True,
    download=True,
    transform=train_transform
)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

# Validation transform
valset.dataset.transform = val_transform

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

valloader = DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ==========================================================
# MODEL : LSNet-B
# ==========================================================
def load_model():
    from model.lsnet import LSNet

    model = LSNet(
        img_size=IMG_SIZE,
        patch_size=8,
        embed_dim=[96, 192, 384, 768],
        depth=[0, 6, 18, 24],
        num_heads=[3, 6, 12, 24],
        num_classes=NUM_CLASSES
    )

    return model.to(DEVICE)

# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model(model):
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    train_losses = []
    val_losses = []

    best_val = float('inf')
    patience = 12
    counter = 0

    for epoch in range(EPOCHS):

        # ================= TRAIN =================
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(
            trainloader,
            desc=f"Train Epoch {epoch+1}/{EPOCHS}"
        )

        for images, labels in train_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_bar.set_postfix(loss=loss.item())

        train_loss = running_train_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)

        # ================= VALIDATION =================
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in valloader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / len(valloader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # ================= SAVE BEST =================
        if val_loss < best_val:
            best_val = val_loss
            counter = 0

            torch.save(
                model.state_dict(),
                "weights_EMNIST/best_model_B.pth"
            )
        else:
            counter += 1

        # ================= EARLY STOPPING =================
        if counter >= patience:
            print("Early stopping triggered.")
            break

    # ==========================================================
    # SAVE FINAL MODEL
    # ==========================================================
    torch.save(
        model.state_dict(),
        "weights_EMNIST/lsnet_B.pth"
    )

    # ==========================================================
    # PLOT LOSS CURVE
    # ==========================================================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSNet-B EMNIST Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("weights_EMNIST/loss_curve_B.png")
    plt.close()

    # ==========================================================
    # SAVE LOG FILE
    # ==========================================================
    with open("weights_EMNIST/train_log_LsNet_B.txt", "w") as f:
        for i in range(len(train_losses)):
            f.write(
                f"Epoch {i+1}: "
                f"Train Loss={train_losses[i]:.4f}, "
                f"Val Loss={val_losses[i]:.4f}\n"
            )

# ==========================================================
# MAIN
# ==========================================================
print("\nTraining LSNet-B on EMNIST")

model = load_model()
train_model(model)

del model
torch.cuda.empty_cache()
gc.collect()