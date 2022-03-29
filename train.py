import os 
import numpy as np
import random
from tqdm import tqdm
from sys import exit as e

import torch 
from torch import nn, optim
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
  train_dataset = OpenData(args, "train")
  trainloader = DataLoader(train_dataset, batch_size = args.batch, shuffle=True)
  
  test_dataset = OpenData(args, "test")
  testloader = DataLoader(test_dataset, batch_size = args.batch, shuffle=True)

  print(f"Training isntances: {len(trainloader)}")
  print(f"Test instances: {len(testloader)}")

  # model = MultiHeadAttention(712, 4)
  if args.feat == "all":
    d_model = 712
  elif args.feat == "lm2d":
    d_model = 136
  elif args.feat == "lm3d":
    d_model = 204
  model = Transformer(d_model, args.n_head, args.nx, args.n_class)

  print("Total Parameters: ", sum(p.numel() for p in model.parameters()))
  print("Total Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.NLLLoss()

  pbar = tqdm(range(args.iter))
  for epoch in pbar:
    for b, (x, target, fl) in enumerate(tqdm(trainloader), 0):
      out = model(x)

      loss = criterion(out, target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      pbar.set_description(f"epoch: {epoch}; loss: {round(loss.item(), 3)}")