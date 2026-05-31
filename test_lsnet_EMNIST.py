import os
import time
import csv
import gc
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
IMG_SIZE = 64
NUM_CLASSES = 47

RESULT_DIR = "evaluation_results_EMNIST"

os.makedirs(RESULT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ==========================================================
# TEST TRANSFORM
# ==========================================================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# ==========================================================
# TEST DATASET
# ==========================================================
testset = torchvision.datasets.EMNIST(
    root='./Dataset/EMNIST',
    split='balanced',
    train=False,
    download=True,
    transform=test_transform
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ==========================================================
# MODEL CONFIGS
# IMPORTANT:
# These MUST EXACTLY MATCH training configs
# ==========================================================
MODEL_CONFIGS = {

    # ======================================================
    # LSNet-T
    # ======================================================
    "LSNet-T": {

        "embed_dim": [64, 128, 256, 384],

        "depth": [0, 4, 12, 14],

        "num_heads": [3, 3, 3, 4],

        "weight_path": "/mnt/DATA/SPLAB/EE25M305/MLSI_Project/lsnet/weights_EMNIST/best_model_T.pth"
    },

    # ======================================================
    # LSNet-S
    # ======================================================
    "LSNet-S": {

        "embed_dim": [64, 128, 320, 512],

        "depth": [0, 4, 12, 18],

        "num_heads": [3, 3, 5, 8],

        "weight_path": "/mnt/DATA/SPLAB/EE25M305/MLSI_Project/lsnet/weights_EMNIST/best_model_S.pth"
    },

    # ======================================================
    # LSNet-B
    # ======================================================
    "LSNet-B": {

        "embed_dim": [96, 192, 384, 768],

        "depth": [0, 6, 18, 24],

        "num_heads": [3, 6, 12, 24],

        "weight_path": "/mnt/DATA/SPLAB/EE25M305/MLSI_Project/lsnet/weights_EMNIST/best_model_B.pth"
    }
}

# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model(config):

    from model.lsnet import LSNet

    model = LSNet(
        img_size=IMG_SIZE,
        patch_size=8,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        num_classes=NUM_CLASSES
    )

    # ------------------------------------------------------
    # LOAD WEIGHTS
    # ------------------------------------------------------
    state_dict = torch.load(
        config["weight_path"],
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)

    model.eval()

    return model

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================
def evaluate_model(model, model_name):

    # ------------------------------------------------------
    # PARAMS
    # ------------------------------------------------------
    params = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------
    # FLOPs
    # ------------------------------------------------------
    dummy_input = torch.randn(
        1,
        3,
        IMG_SIZE,
        IMG_SIZE
    ).to(DEVICE)

    flops, _ = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

    # ------------------------------------------------------
    # GPU WARMUP
    # ------------------------------------------------------
    with torch.no_grad():

        for _ in range(10):
            _ = model(dummy_input)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    # ------------------------------------------------------
    # TESTING
    # ------------------------------------------------------
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():

        for images, labels in tqdm(
            testloader,
            desc=f"Testing {model_name}"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------
    accuracy = 100.0 * correct / total

    total_time = end_time - start_time

    throughput = total / total_time

    avg_time = total_time / total

    return {

        "Model": model_name,

        "Top-1 Acc (%)": accuracy,

        "Params": float(params),

        "FLOPs": float(flops),

        "Throughput (img/s)": float(throughput),

        "Total Time (s)": float(total_time),

        "Avg Time/Image (s)": float(avg_time)
    }

# ==========================================================
# MAIN EVALUATION
# ==========================================================
all_results = []

for model_name, config in MODEL_CONFIGS.items():

    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    model = load_model(config)

    results = evaluate_model(
        model,
        model_name
    )

    all_results.append(results)

    print("\nRESULTS")

    for k, v in results.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------
    # CLEAR MEMORY
    # ------------------------------------------------------
    del model

    gc.collect()

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# ==========================================================
# SAVE TXT REPORT
# ==========================================================
txt_path = os.path.join(
    RESULT_DIR,
    "LSNet_EMNIST_Evaluation.txt"
)

with open(txt_path, "w") as f:

    f.write(
        "LSNet Evaluation Results on EMNIST\n"
    )

    f.write("=" * 50 + "\n\n")

    for r in all_results:

        f.write(
            f"Model: {r['Model']}\n"
        )

        f.write(
            f"Top-1 Accuracy: "
            f"{r['Top-1 Acc (%)']:.4f} %\n"
        )

        f.write(
            f"Params: "
            f"{r['Params']}\n"
        )

        f.write(
            f"FLOPs: "
            f"{r['FLOPs']}\n"
        )

        f.write(
            f"Throughput: "
            f"{r['Throughput (img/s)']:.2f} images/sec\n"
        )

        f.write(
            f"Total Inference Time: "
            f"{r['Total Time (s)']:.4f} sec\n"
        )

        f.write(
            f"Avg Time/Image: "
            f"{r['Avg Time/Image (s)']:.8f} sec\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

# ==========================================================
# SAVE CSV REPORT
# ==========================================================
csv_path = os.path.join(
    RESULT_DIR,
    "LSNet_EMNIST_Evaluation.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as csvfile:

    fieldnames = [

        "Model",

        "Top-1 Acc (%)",

        "Params",

        "FLOPs",

        "Throughput (img/s)",

        "Total Time (s)",

        "Avg Time/Image (s)"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for r in all_results:
        writer.writerow(r)

# ==========================================================
# FINAL PRINT
# ==========================================================
print("\n" + "=" * 60)
print("Evaluation Completed.")
print("=" * 60)

print(f"\nTXT saved at:\n{txt_path}")

print(f"\nCSV saved at:\n{csv_path}")