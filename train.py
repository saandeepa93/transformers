import os 
import numpy as np
import random
from sys import exit as e

import torch 
from torch import nn 
from torch.utils.data import DataLoader

from args import get_args
from dataset import OpenData
from model import MultiHeadAttention, Encoder, Transformer
import utils as ut

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

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

  print("GPU: ", torch.cuda.is_available())
  args = get_args()
  dataset = OpenData(args)
  dataloader = DataLoader(dataset, batch_size = args.batch, shuffle=True)
  # model = MultiHeadAttention(712, 4)
  if args.feat == "all":
    d_model = 712
  else:
    d_model = 136
  model = Transformer(d_model, args.n_head, args.nx)

  print("Total Parameters: ", sum(p.numel() for p in model.parameters()))
  print("Total Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  

  for b, (x, target, fl) in enumerate(dataloader, 0):
    out = model(x)
    print(out, target)
    e()