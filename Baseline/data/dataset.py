from torch.utils.data import Dataset
from .preprocess import *
import os
import xlrd
import pandas
import torch
import random


class CMHADDataset(Dataset):
    def __init__(self, directory, split, augment=True):

        assert split in ["train", "val", "test"], f"Unknown split: {split}"
        self.split = split
        self.label_list, self.file_list = self.build_file_list(directory, split)
        self.augment = augment
        


    def build_file_list(self, root_dir, split):
        labels = ['Action1', 'Action2']
        complete_list = []

        subject = 1
        while subject <= 1:
            train_dir = os.path.join(root_dir, "Train Data", f"Subject{subject}")
            test_dir  = os.path.join(root_dir, "Test Data",  f"Subject{subject}")


            train_xls_path = os.path.join(
                train_dir,
                f"ActionOfInterestTraSubject{subject}.xlsx"
            )
            if not os.path.exists(train_xls_path):
                print(f"[WARN] Train Excel not found: {train_xls_path}")
                subject += 1
                continue

            train_wb = xlrd.open_workbook(train_xls_path)
            train_sheet = train_wb.sheet_by_index(0)


            train_trials = []
            for m in range(1, train_sheet.nrows):
                trial_id  = int(train_sheet.cell_value(m, 0))
                action_id = int(train_sheet.cell_value(m, 1)) - 1
                t_start   = float(train_sheet.cell_value(m, 2))
                t_end     = float(train_sheet.cell_value(m, 3))
                train_trials.append((trial_id, action_id, t_start, t_end))

            n_total = len(train_trials)
            if n_total == 0:
                print(f"[CMHADDataset] Subject{subject} has 0 train trials!")
                subject += 1
                continue



            train_ids, val_ids = set(), set()
            if split in ["train", "val"]:

                unique_trial_ids = sorted(set([t[0] for t in train_trials]))
                n_trials = len(unique_trial_ids)

                if n_trials == 0:
                    print(f"[CMHADDataset] Subject{subject} has 0 trials in Train Data!")
                    subject += 1
                    continue


                rng = random.Random(42 + subject)
                rng.shuffle(unique_trial_ids)


                n_train_trials = max(1, int(n_trials * 0.8))
                n_val_trials   = max(1, n_trials - n_train_trials)

                train_ids = set(unique_trial_ids[:n_train_trials])
                val_ids   = set(unique_trial_ids[n_train_trials:n_train_trials + n_val_trials])

                print(f"[CMHADDataset] Subject{subject} trial split (by trial_id):")
                print(f"  total trials: {n_trials}")
                print(f"  train_ids: {sorted(list(train_ids))}")
                print(f"  val_ids  : {sorted(list(val_ids))}")



            test_trials = []
            if split == "test":
                test_xls_path = os.path.join(
                    test_dir,
                    f"ActionOfInterestTraSubject{subject}.xlsx"
                )
                if os.path.exists(test_xls_path):
                    test_wb = xlrd.open_workbook(test_xls_path)
                    test_sheet = test_wb.sheet_by_index(0)
                    for m in range(1, test_sheet.nrows):
                        trial_id  = int(test_sheet.cell_value(m, 0))
                        action_id = int(test_sheet.cell_value(m, 1)) - 1
                        t_start   = float(test_sheet.cell_value(m, 2))
                        t_end     = float(test_sheet.cell_value(m, 3))
                        test_trials.append((trial_id, action_id, t_start, t_end))
                    print(f"[CMHADDataset] Subject{subject} test trials: {len(test_trials)}")
                else:
                    print(f"[WARN] Test Excel not found: {test_xls_path}")


            if split in ["train", "val"]:

                for (trial_id, action_id, t_start, t_end) in train_trials:
                    midtime = (t_start + t_end) / 2.0

                    if split == "train" and trial_id not in train_ids:
                        continue
                    if split == "val" and trial_id not in val_ids:
                        continue

                    Idirpath = os.path.join(
                        train_dir, "InertialData",
                        f"inertial_sub{subject}_tr{trial_id}.csv"
                    )
                    Vdirpath = os.path.join(
                        train_dir, "VideoData",
                        f"video_sub{subject}_tr{trial_id}.avi"
                    )

                    df = pandas.read_csv(Idirpath)
                    MissFrames = 5435 - len(df.index)

                    if split == "train":

                        startframe = int(64 * midtime) - 200
                        endframe   = int(64 * midtime) + 199
                        startframe, endframe = self.check_overflow(startframe, endframe)

                        Vstart = int(12 * midtime) - 40
                        Vend   = int(12 * midtime) + 39
                        Vstart, Vend = self.Vcheck_overflow(Vstart, Vend)

                        for n in range(12):
                            imu_s = startframe + 5 * n
                            imu_e = imu_s + 299
                            vid_s = Vstart + n
                            vid_e = vid_s + 59
                            entry = (
                                action_id,
                                trial_id,
                                Idirpath,
                                imu_s,
                                imu_e,
                                MissFrames,
                                Vdirpath,
                                vid_s,
                                vid_e,
                                subject
                            )
                            complete_list.append(entry)

                    else:
                        startframe = int(64 * midtime) - 150
                        endframe   = int(64 * midtime) + 149
                        startframe, endframe = self.check_overflow(startframe, endframe)

                        Vstart = int(12 * midtime) - 29
                        Vend   = int(12 * midtime) + 30
                        Vstart, Vend = self.Vcheck_overflow(Vstart, Vend)

                        entry = (
                            action_id,
                            trial_id,
                            Idirpath,
                            startframe,
                            startframe+299,
                            MissFrames,
                            Vdirpath,
                            Vstart,
                            Vstart+59,
                            subject
                        )
                        complete_list.append(entry)

            elif split == "test":

                for (trial_id, action_id, t_start, t_end) in test_trials:

                    print(f"[TEST] Sub{subject}, Trial {trial_id}, Action {action_id+1}")

                    Idirpath = os.path.join(
                        test_dir, "InertialData",
                        f"inertial_sub{subject}_tr{trial_id}.csv"
                    )
                    Vdirpath = os.path.join(
                        test_dir, "VideoData",
                        f"video_sub{subject}_tr{trial_id}.avi"
                    )


                    midtime = (t_start + t_end) / 2.0

                    df = pandas.read_csv(Idirpath)
                    MissFrames = 5435 - len(df.index)


                    startframe = int(64 * midtime) - 150
                    endframe   = startframe + 299
                    startframe, endframe = self.check_overflow(startframe, endframe)


                    Vstart = int(12 * midtime) - 29
                    Vend   = Vstart + 59
                    Vstart, Vend = self.Vcheck_overflow(Vstart, Vend)

                    entry = (
                        action_id,
                        trial_id,
                        Idirpath,
                        startframe,
                        endframe,
                        MissFrames,
                        Vdirpath,
                        Vstart,
                        Vend,
                        subject
                    )
                    complete_list.append(entry)


            subject += 1

        print(f"[CMHADDataset] split={split}, Size of data: {len(complete_list)}")
        return labels, complete_list


    def check_overflow(self, startframe, endframe):
        if startframe < 84: 
            endframe = endframe + 84 - startframe
            startframe = 84
        elif endframe > 5404:
            startframe = startframe - (endframe - 5404)
            endframe = 5404
        return startframe, endframe


    def Vcheck_overflow(self, startframe, endframe):
        if startframe < 0:
            endframe = endframe - startframe
            startframe = 0
        elif endframe > 920:
            startframe = startframe - (endframe - 920)
            endframe = 920
        return startframe, endframe

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        (
            label,
            trial_id,
            filename,
            startframe,
            endframe,
            MissFrames,
            Vdirpath,
            Vstartframe,
            Vendframe,
            subject
        ) = self.file_list[idx]



        Inerframes = load_inertial(filename, startframe - MissFrames)
        vidframes  = load_video(Vdirpath, Vstartframe)
        temporalvolume = bbc(vidframes, self.augment)

        sample = {
            'temporalvolume_x': Inerframes,
            'temporalvolume_y': temporalvolume,
            'label':      torch.tensor(label, dtype=torch.long),
            'trial_id':   torch.tensor(trial_id, dtype=torch.long),
            'MiddleTime': torch.tensor((startframe-MissFrames + 150) / 64.0, dtype=torch.float32),
            'subject':    torch.tensor(subject, dtype=torch.long),
        }
        return sample
