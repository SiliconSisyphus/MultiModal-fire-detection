from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


class Validator():
    def __init__(self, options):


        self.validationdataset = CMHADDataset(
            options["validation"]["dataset"],
            "val",
            augment=False
        )

        self.validationdataloader = DataLoader(
            self.validationdataset,
            batch_size=options["input"]["batchsize"],
            shuffle=False,
            num_workers=options["input"]["numworkers"],
            drop_last=False
        )

        self.usecudnn = options["general"]["usecudnn"]
        self.gpuid = options["general"]["gpuid"]
        self.accuracyfilelocation = options["validation"]["accuracyfilelocation"]


        self.use_amp = True

    def epoch(self, model):
        print("\n===== Starting Validation =====")
        model.eval()

        all_labels = []
        all_predictions = []


        with torch.no_grad():
            for batch_idx, sample in enumerate(self.validationdataloader):


                imu = sample["temporalvolume_x"]
                video = sample["temporalvolume_y"]
                labels = sample["label"]

                if self.usecudnn:
                    imu = imu.cuda(self.gpuid, non_blocking=True)
                    video = video.cuda(self.gpuid, non_blocking=True)
                    labels = labels.cuda(self.gpuid, non_blocking=True)


                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(imu, video)
                else:
                    outputs = model(imu, video)


                _, predicted = torch.max(outputs, 1)


                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())


        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)


        accuracy = accuracy_score(all_labels, all_predictions)


        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)


        msg = (
            f"Val | Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )
        print(msg)

        with open(self.accuracyfilelocation, "a") as f:
            f.write(msg + "\n")

        return accuracy, precision, recall, f1
