import os 
from sys import exit as e

import torch 
from torch import nn 
from torch.utils.data import DataLoader

from args import get_args
from dataset import OpenData
from model import Attention

if __name__ == "__main__":
  args = get_args()
  dataset = OpenData(args)
  dataloader = DataLoader(dataset, batch_size = args.batch, shuffle=True)

  model = Attention(712, 4)

  for b, (x, target, fl) in enumerate(dataloader, 0):
    print(x.size(), target, fl, x.dtype)
    out = model(x)
    print(out.size())
    e()
