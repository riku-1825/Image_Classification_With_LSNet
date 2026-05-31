import os
import time
import csv
import gc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 128
IMG_SIZE = 32
NUM_CLASSES = 10

RESULT_DIR = "evaluation_results_ResNet18_CIFAR10"

os.makedirs(RESULT_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ==========================================================
# TEST TRANSFORM
# ==========================================================
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

# ==========================================================
# TEST DATASET
# ==========================================================
testset = torchvision.datasets.CIFAR10(
    root='./data',
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
# MODEL CONFIG
# ==========================================================
MODEL_CONFIGS = {

    "ResNet18": {

        "weight_path":
        "weights_ResNet18_CIFAR_10/resnet18_best.pth"
    }
}

# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model(config):

    model = models.resnet18(weights=None)

    # ------------------------------------------------------
    # MODIFY FINAL LAYER
    # ------------------------------------------------------
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
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
    params = sum(
        p.numel() for p in model.parameters()
    )

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
    "ResNet18_CIFAR10_Evaluation.txt"
)

with open(txt_path, "w") as f:

    f.write(
        "ResNet18 Evaluation Results on CIFAR-10\n"
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
    "ResNet18_CIFAR10_Evaluation.csv"
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