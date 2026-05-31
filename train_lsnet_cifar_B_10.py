import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
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

torch.backends.cudnn.benchmark = True

# create folder
os.makedirs("weights_CIFAR_10", exist_ok=True)

# -----------------------------
# TRANSFORM
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# -----------------------------
# DATASET (Train + Val Split)
# -----------------------------
full_trainset = torchvision.datasets.CIFAR10(
    root='./Dataset/CIFAR-10', train=True, download=True, transform=train_transform
)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

# -----------------------------
# MODEL
# -----------------------------
def load_model():
    from model.lsnet import LSNet

    model = LSNet(
        img_size=224,
        patch_size=8,
        embed_dim=[128,256,384,512],
        depth=[4,6,8,10],
        num_heads=[3,3,3,4]
    )

    return model.to(DEVICE)

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for images, labels in tqdm(trainloader, desc=f"Train Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    torch.save(model.state_dict(), "weights_CIFAR_10/lsnet_B.pth")

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LSNet-B Loss Curve')
    plt.legend()
    plt.grid()

    plt.savefig("weights_CIFAR_10/loss_curve_B.png")
    plt.close()

    # -----------------------------
    # SAVE LOG
    # -----------------------------
    with open("weights_CIFAR_10/train_log_B.txt", "w") as f:
        for i in range(EPOCHS):
            f.write(f"Epoch {i+1}: Train={train_losses[i]:.4f}, Val={val_losses[i]:.4f}\n")


# -----------------------------
# MAIN
# -----------------------------
print("\nTraining LSNet-B")

model = load_model()
train_model(model)

# clear memory
del model
torch.cuda.empty_cache()
gc.collect()
