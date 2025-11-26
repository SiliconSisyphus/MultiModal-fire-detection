import imageio

import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import pandas
import pdb
import numpy as np

def load_video(filename, startframe):


    try:
        vid = imageio.get_reader(filename,  'ffmpeg')
    except:
        print("Error at: "+filename)    
    frames = []
    for i in range(0, 60):
        image = vid.get_data(startframe+i)
        image = functional.to_tensor(image)
        frames.append(image)
    return frames



def bbc(vidframes, augmentation=True):
    frames = []
    for i in range(60):
        result = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])(vidframes[i])
        frames.append(result)


    temporalvolume = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

    return temporalvolume.unsqueeze(0)

def load_inertial(filename, startframe):
    if startframe<0:
        startframe=0
    df = pandas.read_csv(filename)
    if df.shape[1] >= 6:

        df = df.iloc[:, :6]
    else:
        raise ValueError(f"Inertial file {filename} has only {df.shape[1]} columns, expected >= 6")

    df.columns = ['A', 'B', 'C', 'D', 'E', 'F']
    df = df.astype(float).to_numpy()


    frames = df[startframe:startframe+300:5,:]

    frames = torch.from_numpy(frames).float()
    return frames

