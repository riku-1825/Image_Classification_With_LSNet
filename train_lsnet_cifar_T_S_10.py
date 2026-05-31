import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import gc

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100

torch.backends.cudnn.benchmark = True

# -----------------------------
# DEFINE TRANSFORM FIRST
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


# -----------------------------
# DATASET
# -----------------------------
from torch.utils.data import random_split

full_trainset = torchvision.datasets.CIFAR10(
    root='./Dataset/CIFAR-10', train=True, download=True, transform=train_transform
)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

trainset = torchvision.datasets.CIFAR10(
    root='./Dataset/CIFAR-10', train=True, download=True, transform=train_transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

# -----------------------------
# MODEL
# -----------------------------
def load_model(variant='T'):
    from model.lsnet import LSNet

    if variant == 'T':
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[64,128,256,384],
                      depth=[0,2,8,10],
                      num_heads=[3,3,3,4])

    elif variant == 'S':
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[96,192,320,448],
                      depth=[1,2,8,10],
                      num_heads=[3,3,3,4])

    elif variant == 'B':
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[128,256,384,512],
                      depth=[4,6,8,10],
                      num_heads=[3,3,3,4])

    return model.to(DEVICE)

# -----------------------------
# TRAIN FUNCTION
# -----------------------------

def train_model(model, variant):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for images, labels in tqdm(trainloader, desc=f"{variant} Train Epoch {epoch+1}"):
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

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # SAVE MODEL
    torch.save(model.state_dict(), f"weights_CIFAR_10/lsnet_{variant}.pth")

    # ---- PLOT ----
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'LSNet-{variant} Loss Curve')
    plt.legend()
    plt.grid()

    # SAVE PLOT
    plt.savefig(f"weights_CIFAR_10/loss_curve_{variant}.png")
    plt.close()

    # SAVE LOG
    with open(f"weights_CIFAR_10/train_log_{variant}.txt", "w") as f:
        for i in range(EPOCHS):
            f.write(f"Epoch {i+1}: Train={train_losses[i]:.4f}, Val={val_losses[i]:.4f}\n")


# -----------------------------
# MAIN
# -----------------------------
variants = ['T', 'S', 'B']

for v in variants:
    print(f"\nTraining LSNet-{v}")
    model = load_model(v)
    train_model(model, v)
    del model
    torch.cuda.empty_cache()
    gc.collect()

