import torch
import torchvision
import torchvision.transforms as transforms
import time
import csv
from tqdm import tqdm
from thop import profile

# -----------------------------
# CONFIG
# -----------------------------
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# DATASET
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

testset = torchvision.datasets.CIFAR10(
    root='./Dataset/CIFAR-10',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

# -----------------------------
# MODEL LOADER
# -----------------------------
def load_model(variant):
    from model.lsnet import LSNet

    if variant == 'T':
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[64,128,256,384],
                      depth=[0,2,8,10],
                      num_heads=[3,3,3,4])
        weight_path = "weights_CIFAR_10/lsnet_T.pth"

    elif variant == 'S':
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[96,192,320,448],
                      depth=[1,2,8,10],
                      num_heads=[3,3,3,4])
        weight_path = "weights_CIFAR_10/lsnet_S.pth"

    elif variant == 'B':
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[128,256,384,512],
                      depth=[4,6,8,10],
                      num_heads=[3,3,3,4])
        weight_path = "weights_CIFAR_10/lsnet_B.pth"

    elif variant == 'B_1': 
        model = LSNet(img_size=224, patch_size=8,
                      embed_dim=[128,256,384,512],
                      depth=[4,6,8,10],
                      num_heads=[3,3,3,4])
        weight_path = "weights_CIFAR_10/lsnet_B_1.pth"

    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# -----------------------------
# METRICS
# -----------------------------
def get_model_stats(model):
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return params, flops

def evaluate(model):
    correct = 0
    total = 0

    start = time.time()

    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    end = time.time()

    acc = 100 * correct / total
    total_time = end - start
    avg_time = total_time / total
    throughput = total / total_time

    return acc, total_time, avg_time, throughput

# -----------------------------
# MAIN
# -----------------------------
variants = ['T', 'S', 'B', 'B_1']
results = []

for v in variants:
    print(f"\nEvaluating LSNet-{v}")

    model = load_model(v)

    params, flops = get_model_stats(model)
    acc, total_time, avg_time, throughput = evaluate(model)

    results.append({
        "Model": f"LSNet-{v}",
        "Top-1 Acc (%)": acc,
        "Params": params,
        "FLOPs": flops,
        "Throughput (img/s)": throughput,
        "Total Time (s)": total_time,
        "Avg Time/Image (s)": avg_time
    })

# -----------------------------
# PRINT RESULTS
# -----------------------------
for r in results:
    print(r)

with open("cifar_10_trained_lsnet_results.txt", "w") as f:
    f.write("LSNet Evaluation Results on CIFAR-10\n")
    f.write("="*50 + "\n\n")
    
    for r in results:
        f.write(f"Model: {r['Model']}\n")
        f.write(f"Top-1 Accuracy: {r['Top-1 Acc']:.4f} %\n")
        f.write(f"Params: {r['Params']:,}\n")
        f.write(f"FLOPs: {r['FLOPs']:,}\n")
        f.write(f"Throughput: {r['Throughput']:.2f} images/sec\n")
        f.write(f"Total Inference Time: {r['Total Time']:.4f} sec\n")
        f.write(f"Avg Time/Image: {r['Avg Time/Image']:.8f} sec\n")
        f.write("-"*50 + "\n")

import csv

with open("cifar_10_trained_lsnet_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # Header
    writer.writerow([
        "Model", "Top-1 Acc (%)", "Params", "FLOPs",
        "Throughput (img/s)", "Total Time (s)", "Avg Time/Image (s)"
    ])
    
    # Data
    for r in results:
        writer.writerow([
            r['Model'],
            r['Top-1 Acc'],
            r['Params'],
            r['FLOPs'],
            r['Throughput'],
            r['Total Time'],
            r['Avg Time/Image']
        ])