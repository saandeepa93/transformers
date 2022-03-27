import torch 
from torch import nn 
import torch.nn.functional as F
import math

from sys import exit as e


if __name__ == "__main__":
  max_seq_len = 2
  d_model = 4
  pe = torch.zeros(max_seq_len, d_model)
  for pos in range(max_seq_len):
    for i in range(0, d_model, 2):
      pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
      pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

  print(pe)
  e()      
  pe = pe.unsqueeze(0)

  a = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
  print(a.size(), torch.cuda.is_available())

  b = F.normalize(a, p =1, dim=1)
  print(b)
