import os
import glob
import pandas as pd
import numpy as np
from scipy import interpolate
from sys import exit as e

import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class OpenData(Dataset):
  def __init__(self, args):
    super().__init__()

    self.label_dict = {1: 0, 2: 0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:1, 10:0, 11:0, 13:0, 14:0, 15:1, 16:0, 17:1, 18:1, 19:1, 20:1}
    self.train_id = [3, 4, 6, 7, 8, 9, 13, 15, 17, 2, 5, 11, 16, 18]
    self.test_id = [1, 10, 19, 20]

    self.au_occ = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
    self.au_int = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]

    self.au_occ_cols = [f"AU{str(AU).zfill(2)}_c" for AU in self.au_occ]
    self.au_int_cols = [f"AU{str(AU).zfill(2)}_r" for AU in self.au_int]
    self.lm_2d_cols = [f"x_{str(i)}" for i in range(68)] + [f"y_{str(i)}" for i in range(68)]
    self.lm_3d_cols = [f"x_{str(i)}" for i in range(68)] + [f"y_{str(i)}" for i in range(68)] + [f"z_{str(i)}" for i in range(68)]

    self.args = args
    self.all_files = []
    self.getAllFiles()
    self.transform = transforms.Compose([
      transforms.ToTensor()
    ])


  def resample_by_col(self, fd, s, idx=None):
    len, cols = fd.shape
    fd_new = np.zeros((s, cols))
    x = np.linspace(0, len, len)
    x_new = np.linspace(0, len, s)
    for i in range(cols):
      fd_new[:, i] = interpolate.interp1d(x, fd[:, i])(x_new)
    return fd_new

  def getAllFiles(self):
    for sub in os.listdir(os.path.join(self.args.root)):
      sub_dir = os.path.join(self.args.root, sub)
      if not os.path.isdir(sub_dir):
        continue
      for fl in os.listdir(sub_dir):
        fl_path = os.path.join(sub_dir, fl)
        if os.path.splitext(fl_path)[-1] != ".csv":
          continue
        self.all_files.append(fl_path)

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    fname = self.all_files[idx].split('\\')[-1].split('.')[0]
    sub = self.all_files[idx].split('\\')[-2]
    fl = f"{sub}_{fname}"
    
    df = pd.read_csv(self.all_files[idx], skipinitialspace = True)
    if self.args.feat == "all":
      x = self.resample_by_col(df.iloc[:, 2:].to_numpy(), self.args.scale)
    else:
      df = df[self.lm_2d_cols]
      x = self.resample_by_col(df.to_numpy(), self.args.scale)

    x = self.transform(x).squeeze()
    x = F.normalize(x, p=1, dim=1)
    target = self.label_dict[int(sub)]
    
    return x.to(torch.float32), target, fl


    
