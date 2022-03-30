import os 
import numpy as np
import random
from tqdm import tqdm
from sys import exit as e

import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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


def get_valScores(testloader, model):
  accuracy_detail = {}
  model.eval()
  y_pred = []
  y_true= []
  for b, (x, target, sub) in enumerate(tqdm(testloader), 0):
    accuracy_detail[sub] = {'pred': 0, 'true': 0}
    x = x.to(device)
    with torch.no_grad():
      out = model(x)
    y_true.append(target[0].item())
    y_pred.append(torch.argmax(out, dim=1)[0].item())

    accuracy_detail[sub]['pred'] = torch.argmax(out, dim=1)[0].item()
    accuracy_detail[sub]['true'] = target[0].item()

  return accuracy_score(y_pred, y_true), accuracy_detail


if __name__ == "__main__":
  seed_everything(42)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)

  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
  os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

  print("GPU: ", torch.cuda.is_available())
  args = get_args()
  train_dataset = OpenData(args, "train")
  trainloader = DataLoader(train_dataset, batch_size = args.batch, shuffle=True)
  
  test_dataset = OpenData(args, "test")
  testloader = DataLoader(test_dataset, batch_size = 1, shuffle=True)

  writer = SummaryWriter()

  print(f"Training instances: {len(trainloader)}")
  print(f"Test instances: {len(testloader)}")

  # model = MultiHeadAttention(712, 4)
  if args.feat == "all":
    d_model = 712
  elif args.feat == "lm2d":
    d_model = 136
  elif args.feat == "lm3d":
    d_model = 204
  
  model = Transformer(d_model, args.n_head, args.nx, args.n_class)
  model = nn.DataParallel(model)
  model = model.to(device)

  print("Total Parameters: ", sum(p.numel() for p in model.parameters()))
  print("Total Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.NLLLoss()

  y_true_train = []
  y_pred_train = []
  pbar = tqdm(range(args.iter))
  for epoch in pbar:
    for b, (x, target, fl) in enumerate(tqdm(trainloader), 0):
      model.train()
      x = x.to(device)
      target = target.to(device)
      out = model(x)

      loss = criterion(out, target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        y_true_train.append(target[0].item())
        y_pred_train.append(torch.argmax(out, dim=1)[0].item())

    if epoch%5 == 0:
      torch.save(
              model.state_dict(), f"./checkpoint/model_{epoch}.pt"
        )

    with torch.no_grad():
      train_acc = accuracy_score(y_pred_train, y_true_train)
      val_acc, _ = get_valScores(testloader, model)

    writer.add_scalar("Loss/Train", loss.item(), epoch)
    writer.add_scalar("Acc/Val", val_acc, epoch)
    writer.add_scalar("Acc/Train", train_acc, epoch)

    pbar.set_description(f"epoch: {epoch}; Train loss: {round(loss.item(), 3)}; Val Acc: {val_acc}; Train_acc: {train_acc}")
  
  _, val_predictions = get_valScores(testloader, model)
  print(val_predictions)
  writer.flush()
  writer.close()