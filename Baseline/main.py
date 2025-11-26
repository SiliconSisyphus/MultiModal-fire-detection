from __future__ import print_function
import os
from collections import Counter

import torch
import toml

from training import Trainer
from validation import Validator
from data import CMHADDataset
from model import Net

from main_test import collect_once, eval_threshold

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


if __name__ == '__main__':
    print("Loading options...")
    with open('options1.toml', 'r', encoding='utf-8') as optionsFile:
        options = toml.loads(optionsFile.read())

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
        video_backbone="tsn_mbv2"
    )

    if usecudnn and torch.cuda.is_available():
        device = torch.device("cuda", gpuid)
    else:
        device = torch.device("cpu")
        print(">>> WARNING: CUDA not available, using CPU.")

    model = model.to(device)


    print("Building train dataset just for class count ...")
    train_root = options["training"]["dataset"]
    tmp_train_dataset = CMHADDataset(train_root, "train", augment=False)

    class_counts = Counter()
    for entry in tmp_train_dataset.file_list:
        label = entry[0]
        class_counts[label] += 1
    print(f">>> Train class counts = {class_counts}")

    trainer = Trainer(options, class_counts)
    trainer.attach_model(model)

    validator = Validator(options)

    val_interval = options["training"].get("val_interval", 1)
    print(f"Validation interval = {val_interval}")

    num_epochs = options["training"]["epoch"]

    best_f1 = -1.0
    modelsavepath = options["general"].get("modelsavepath", None)
    savemodel = options["general"].get("savemodel", False)

    for epoch in range(num_epochs):
        print("\n========== Epoch {:d} / {:d} ==========".format(epoch + 1, num_epochs))


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
                print(f">>> New best F1 = {best_f1:.4f}")


    if savemodel and modelsavepath is not None:
        torch.save(model.state_dict(), modelsavepath)
        print(f">>> Best model saved to: {modelsavepath}")


    if savemodel and modelsavepath is not None and os.path.isfile(modelsavepath):
        print(f"\nLoading best model from: {modelsavepath}")
        state = torch.load(modelsavepath, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        print("\nNo saved model found or savemodel=False, using current model for TEST.")
    

    print("\n>>> Running final TEST ...")
    model.eval()



    print("\n>>> Running window-level threshold scan (0.1 / 0.2 / 0.5 / 0.9) ...")
    labels, preds, scores = collect_once(model, options)
    for th in [0.1, 0.2, 0.5, 0.9]:
        eval_threshold(labels, preds, scores, th)

    print("\n=== ALL DONE ===")
    torch.cuda.empty_cache()
