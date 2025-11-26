from __future__ import print_function
import os
from collections import Counter

import torch
import toml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from training import Trainer
from validation import Validator
from data import CMHADDataset
from model import Net


from main_test import collect_once

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def build_class_counts(options):

    print("Building train dataset just for class count ...")
    train_root = options["training"]["dataset"]
    tmp_train_dataset = CMHADDataset(train_root, "train", augment=False)

    class_counts = Counter()
    for entry in tmp_train_dataset.file_list:

        label = entry[0]
        class_counts[label] += 1

    print(f">>> Train class counts = {class_counts}")
    return class_counts


def eval_threshold_local(all_labels, all_preds, all_scores, th):

    mask = all_scores >= th
    kept = mask.sum()
    total = len(mask)
    coverage = kept / total if total > 0 else 0.0

    if kept == 0:

        return coverage, 0.0, 0.0, 0.0, 0.0

    y_true = all_labels[mask]
    y_pred = all_preds[mask]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return coverage, acc, prec, rec, f1


def train_and_eval_single_fusion(options, fusion_type, class_counts, thresholds):

    print("\n" + "=" * 80)
    print(f"============  Running Fusion Type: {fusion_type}  ============")
    print("=" * 80)


    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    usecudnn = options["general"]["usecudnn"]
    gpuid = options["general"]["gpuid"]

    if options["general"]["usecudnnbenchmark"] and usecudnn:
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    model = Net(
        sensor_input_dim=6,
        video_feature_dim=256,
        fused_hidden=128,
        num_classes=2,
        video_backbone="tsn_mbv2",
        fusion_type=fusion_type,
    )

    if usecudnn and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpuid}")
    else:
        device = torch.device("cpu")
        print(">>> WARNING: CUDA not available, using CPU.")

    model = model.to(device)


    trainer = Trainer(options, class_counts)
    trainer.attach_model(model)

    validator = Validator(options)

    val_interval = options["training"].get("val_interval", 1)
    num_epochs = options["training"]["epoch"]

    print(f"Validation interval = {val_interval}")
    print(f"Num epochs = {num_epochs}")

    best_f1 = -1.0
    savemodel = options["general"].get("savemodel", False)
    base_model_path = options["general"].get("modelsavepath", None)
    if base_model_path is not None:
        root, ext = os.path.splitext(base_model_path)
        modelsavepath = f"{root}_{fusion_type}{ext}"
    else:
        modelsavepath = None


    for epoch in range(num_epochs):
        print("\n========== Fusion: {} | Epoch {:d} / {:d} =========="
              .format(fusion_type, epoch + 1, num_epochs))


        model.train()
        trainer.epoch(model, epoch)
        torch.cuda.empty_cache()


        if (epoch + 1) % val_interval == 0:
            print(f">>> Running validation at epoch {epoch} ...")
            model.eval()
            acc, prec, rec, f1 = validator.epoch(model)
            torch.cuda.empty_cache()


            if f1 > best_f1:
                best_f1 = f1
                print(f">>> [Fusion={fusion_type}] New best Val F1 = {best_f1:.4f}")
                if savemodel and modelsavepath is not None:
                    torch.save(model.state_dict(), modelsavepath)
                    print(f">>> [Fusion={fusion_type}] Best model saved to: {modelsavepath}")


    if savemodel and modelsavepath is not None and os.path.isfile(modelsavepath):
        print(f"\n[Fusion={fusion_type}] Loading best model from: {modelsavepath}")
        state = torch.load(modelsavepath, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        print(f"\n[Fusion={fusion_type}] No saved model found or savemodel=False, "
              f"using current model for TEST.")


    print(f"\n>>> [Fusion={fusion_type}] Running final TEST (window-level thresholds) ...")
    model.eval()


    labels, preds, scores = collect_once(model, options)


    labels = np.asarray(labels)
    preds = np.asarray(preds)
    scores = np.asarray(scores)

    print(f"\n### Window-level Metrics for Fusion = {fusion_type} ###")
    print("Fusion\tThresh\tCoverage\tAcc\tPrec(macro)\tRec(macro)\tF1(macro)")

    for th in thresholds:
        coverage, acc, prec, rec, f1 = eval_threshold_local(labels, preds, scores, th)
        print(f"{fusion_type}\t{th:.2f}\t{coverage*100:6.2f}%\t"
              f"{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}")

    torch.cuda.empty_cache()
    print(f"\n=== [Fusion={fusion_type}] DONE ===\n")


if __name__ == '__main__':
    print("Loading options...")
    with open('options1.toml', 'r', encoding='utf-8') as optionsFile:
        options = toml.loads(optionsFile.read())


    class_counts = build_class_counts(options)


    fusion_types = ["gmu","concat", "film", "xattn", "bilinear"]#"gmu", 


    thresholds = [0.1, 0.2, 0.5, 0.9]


    for fusion in fusion_types:
        train_and_eval_single_fusion(options, fusion, class_counts, thresholds)

    print("\n=== ALL FUSION EXPERIMENTS DONE ===")
