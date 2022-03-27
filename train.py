import os 
import numpy as np
import random
from sys import exit as e

import torch 
from torch import nn 
from torch.utils.data import DataLoader

from args import get_args
from dataset import OpenData
from model import MultiHeadAttention, Encoder

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  dataset = OpenData(args)
  dataloader = DataLoader(dataset, batch_size = args.batch, shuffle=True)
  # model = MultiHeadAttention(712, 4)
  model = Encoder(712, 4)

  for b, (x, target, fl) in enumerate(dataloader, 0):
    out = model(x)
    print(out.size())
    e()
