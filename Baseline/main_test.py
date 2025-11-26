import os
import toml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import Net
from data import CMHADDataset
from torch.utils.data import DataLoader


def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())
    return total / 1024 / 1024  

def compute_flops(model, device):

    model.eval()


    imu_dummy = torch.randn(1, 60, 6).to(device)
    video_dummy = torch.randn(1, 3, 60, 112, 112).to(device)


    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            model(imu_dummy, video_dummy)


    flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            flops += evt.flops

    return flops


@torch.no_grad()
def collect_once(model, options):
    print("========== FIRST PASS: MODEL INFERENCE ONLY ==========")

    root_dir = options["validation"]["dataset"]
    device = torch.device(
        f"cuda:{options['general']['gpuid']}"
        if options["general"]["usecudnn"] else "cpu"
    )
    model.to(device)
    model.eval()
    print("\n====== Model Complexity ======")
    model_size = count_parameters(model)
    print(f"Model size: {model_size:.2f} MB")

    flops = compute_flops(model, device)
    print(f"FLOPs per forward: {flops/1e9:.3f} GFLOPs")
    print("================================\n")

    test_dataset = CMHADDataset(root_dir, "test", augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=options["input"]["numworkers"],
        drop_last=False
    )

    print(f"[TEST] num windows = {len(test_dataset)}")

    all_labels = []
    all_preds  = []
    all_scores = []

    for i_batch, sample in enumerate(test_loader):
        if i_batch % 50 == 0:
            print(f"[TEST] batch {i_batch}/{len(test_loader)}")

        x = sample["temporalvolume_x"].to(device)
        y = sample["temporalvolume_y"].to(device)
        label = int(sample["label"].item())

        logits = model(x, y)
        probs = torch.softmax(logits, dim=1)

        max_prob, pred = torch.max(probs, dim=1)
        score = float(max_prob.item())
        pred = int(pred.item())

        all_labels.append(label)
        all_preds.append(pred)
        all_scores.append(score)


    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_scores = np.array(all_scores)

    print(" 推理完成（仅 1 次 GPU 计算）\n")
    return all_labels, all_preds, all_scores


def eval_threshold(all_labels, all_preds, all_scores, th):
    mask = all_scores >= th
    kept = mask.sum()
    total = len(mask)
    coverage = kept / total

    print(f"\n===== Threshold = {th:.2f} =====")
    print(f"Kept windows: {kept}/{total} ({coverage*100:.2f}%)")

    if kept == 0:
        print(" No windows kept, cannot compute metrics.")
        return

    y_true = all_labels[mask]
    y_pred = all_preds[mask]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    print("Loading options...")
    with open("options1.toml", "r") as f:
        options = toml.loads(f.read())

    model = Net()
    device = torch.device(
        f"cuda:{options['general']['gpuid']}"
        if options["general"]["usecudnn"] else "cpu"
    )
    print("Loading model:", options["general"]["modelsavepath"])
    model.load_state_dict(torch.load(options["general"]["modelsavepath"], map_location=device))

    labels, preds, scores = collect_once(model, options)

    thresholds = [0.1, 0.2, 0.5, 0.9]
    for th in thresholds:
        eval_threshold(labels, preds, scores, th)

    print("\n=== ALL DONE ===")
