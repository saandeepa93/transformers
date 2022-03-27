import torch 
from torch import nn 
import torch.nn.functional as F

from sys import exit as e


if __name__ == "__main__":

  a = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
  print(a.size(), torch.cuda.is_available())

  b = F.normalize(a, p =1, dim=1)
  print(b)