import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from torchvision import transforms
from pathlib import Path
import pandas as pd

class FeatureDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        if mode == 'train':
            self.df = pd.read_json(os.path.join(cfg.file_prefix, cfg.train_file), lines = True)
        elif mode == 'val':
            self.df = pd.read_json(os.path.join(cfg.file_prefix, cfg.val_file), lines = True)
        elif mode == 'test':
            self.df = pd.read_json(os.path.join(cfg.file_prefix, cfg.test_file), lines = True)
        self.num_classes = cfg.num_classes
        self.mode = mode

    def __getitem__(self, index):
        entry = self.df.iloc[index % len(self.df)]
        img_path = entry.img
        feature = torch.tensor(entry.features).float()
        if feature.dim() == 2:
            feature = feature.squeeze(0)
        label = entry.label
        return {'x': feature, 'y': label, 'cls': label, 'file': img_path}

    def __len__(self):
        return len(self.df)
