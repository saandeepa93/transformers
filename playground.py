import torch 
from torch import nn 
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sys import exit as e


if __name__ == "__main__":

  transt = transforms.ToTensor()
  transp = transforms.ToPILImage()
  img = transt(Image.open('./images/1/Baseline_Start_aligned/frame_det_00_000001.bmp'))
  c, h, w = img.size()
  img = img.view(1, c, h, w)
  print(img.size())

  unfold = nn.Unfold(kernel_size=(16, 16), stride = 16)
  patches = unfold(img)
  print(patches.size())
  e()


  patches = img.data.unfold(0, 3, 3).unfold(1, 16, 16).unfold(2, 16, 16)
  print(patches.size())

  def visualize(patches):
    """Imshow for Tensor."""    
    fig = plt.figure(figsize=(4, 4))
    for i in range(4):
        for j in range(4):
            inp = transp(patches[0][i][j])
            inp = np.array(inp)

            ax = fig.add_subplot(4, 4, ((i*4)+j)+1, xticks=[], yticks=[])
            plt.imshow(inp)
    plt.show()

  visualize(patches)