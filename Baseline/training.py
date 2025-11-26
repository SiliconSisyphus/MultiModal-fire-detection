import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from data import CMHADDataset
import numpy as np
import time


class Trainer():

    def __init__(self, options, class_counts):

        self.batchsize = options["input"]["batchsize"]
        self.shuffle = options["input"]["shuffle"]
        self.numworkers = options["input"]["numworkers"]
        self.usecudnn = options["general"]["usecudnn"]
        self.gpuid = options["general"]["gpuid"]

        self.lr = options["training"]["learningrate"]
        self.momentum = options["training"]["momentum"]
        self.weight_decay = options["training"]["weightdecay"]

        train_dataset = CMHADDataset(options["training"]["dataset"], "train", augment=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=self.shuffle,
            num_workers=self.numworkers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.numworkers > 0 else False
        )

        cnt0 = class_counts.get(0, 1)
        cnt1 = class_counts.get(1, 1)
        inv = torch.tensor([1/(cnt0+1e-6), 1/(cnt1+1e-6)], dtype=torch.float)
        self.class_weights = (inv / inv.sum()).cuda(self.gpuid)
        print(f">>> Class Weights = {self.class_weights}")


        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)


        self.optimizer = None


        self.scaler = GradScaler()



    def attach_model(self, model):

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        print(">>> Optimizer attached (AdamW)")



    def epoch(self, model, epoch_idx):

        if self.optimizer is None:
            raise RuntimeError("You must call trainer.attach_model(model) before training!")

        model.train()
        running_loss = 0.0

        for batch_idx, sample in enumerate(self.train_loader):

            imu = sample["temporalvolume_x"]
            video = sample["temporalvolume_y"]
            labels = sample["label"]

            if self.usecudnn:
                imu = imu.cuda(self.gpuid, non_blocking=True)
                video = video.cuda(self.gpuid, non_blocking=True)
                labels = labels.cuda(self.gpuid, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast():
                outputs = model(imu, video)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch_idx} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(self.train_loader)
        print(f"=== Epoch {epoch_idx} Done | Avg Loss = {avg_loss:.4f} ===")

        return avg_loss
