import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os

# -----------------------------
# CONFIG
# -----------------------------
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
IMG_SIZE = 64

torch.backends.cudnn.benchmark = True
os.makedirs("weights_EMNIST", exist_ok=True)

# -----------------------------
# TRANSFORMS (NO AUGMENTATION)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# -----------------------------
# DATASET
# -----------------------------
from torch.utils.data import random_split

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
valset.dataset.transform = val_transform

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

NUM_CLASSES = 47

# -----------------------------
# MODEL (NO DROPOUT)
# -----------------------------
def load_model():
    from model.lsnet import LSNet

    model = LSNet(
        img_size=IMG_SIZE,
        patch_size=8,
        embed_dim=[64,128,256,384],
        depth=[0,4,12,14],   # increased capacity
        num_heads=[3,3,3,4],
        num_classes=NUM_CLASSES
    )

    return model.to(DEVICE)

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(model):
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,        # increased LR
        weight_decay=0  # removed regularization
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    train_losses = []
    val_losses = []

    best_val = float('inf')
    patience = 8
    counter = 0

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for images, labels in tqdm(trainloader, desc=f"Train Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss /= len(valloader)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        # ---- EARLY STOPPING ----
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), "weights_EMNIST/best_model_T.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    # ---------------- SAVE ----------------
    torch.save(model.state_dict(), "weights_EMNIST/lsnet_T.pth")

    # ---------------- PLOT ----------------
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LSNet-T EMNIST Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig("weights_EMNIST/loss_curve_T.png")
    plt.close()

    # ---------------- LOG ----------------
    with open("weights_EMNIST/train_log_LsNet_T.txt", "w") as f:
        for i in range(len(train_losses)):
            f.write(f"Epoch {i+1}: Train={train_losses[i]:.4f}, Val={val_losses[i]:.4f}\n")


# -----------------------------
# MAIN
# -----------------------------
print("\nTraining LSNet-T on EMNIST")

model = load_model()
train_model(model)

del model
torch.cuda.empty_cache()
gc.collect()