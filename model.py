import torch 
from torch import nn 
import torch.nn.functional as F

from math import sqrt
from sys import exit as e

class MultiHeadAttention(nn.Module):
  def __init__(self, dim, n_heads):
    super().__init__()

    self.Wq = nn.Linear(dim, n_heads * dim, bias=False)
    self.Wk = nn.Linear(dim, n_heads * dim, bias=False)
    self.Wv = nn.Linear(dim, n_heads * dim, bias=False)

    self.finalDense = nn.Linear(n_heads*dim, dim, bias=False)
    self.n_heads = n_heads

  def forward(self, x):
    b, t, k = x.size()
    h = self.n_heads

    K = self.Wk(x).view(b, t, h, k).contiguous().view(b*h, t, k)
    Q = self.Wq(x).view(b, t, h, k).contiguous().view(b*h, t, k)
    V = self.Wv(x).view(b, t, h, k).contiguous().view(b*h, t, k)

    W = torch.einsum('btk, bkn -> btn', K, Q.transpose(1, 2))/sqrt(k)
    W = F.softmax(W, dim=2)

    Z = torch.einsum('btn, bnk -> btk', W, V).view(b, h, t, k).contiguous().view(b, t, h*k)
    Z = self.finalDense(Z)
    return Z

class Transformer(nn.Module):
  def __init__(self, dim, n_heads):
    super().__init__()
    self.attention = MultiHeadAttention(dim, n_heads)
    self.norm1 = nn.LayerNorm(dim)

    self.ff = nn.Sequential(
      nn.Linear(dim, n_heads * dim),
      nn.ReLU(),
      nn.Linear(n_heads * dim, dim) 
    )
    self.norm2 = nn.LayerNorm(dim)

    
  def forward(self, x):
    attn = self.attention(x)
    out = self.norm1(attn + x)

    ff_out = self.ff(out)
    out = self.norm2(ff_out + x)
    return out
