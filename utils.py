import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

from sys import exit as e


def plot_lm(arr, idx):
  fig = plt.figure()
  ax = Axes3D(fig)
  lm_2d = arr[idx]
  ax.scatter(lm_2d[:68], lm_2d[68:136], -lm_2d[136:])
  ax.view_init(elev=90., azim=90.)
  plt.savefig(f"./plots/orig/{idx}.png")

def show_plot(x, x_new, y, y_new):
  # y = (y * 5 )
  # y_new = (y_new * 5 )
  plt.plot(x, y, 'bo')
  plt.xlabel("Frames", fontsize=18)
  plt.ylabel("AU Intensity", fontsize=18)
  plt.savefig(f"./plots/orig.png")
  plt.plot(x_new, y_new, 'r.')
  plt.xlabel("Frames", fontsize=18)
  plt.ylabel("AU Intensity", fontsize=18)
  plt.savefig("./plots/res.png")
  plt.close()
  # plt.show()

def resample_by_col(fd, s, idx=None):
  len, cols = fd.shape
  fd_new = np.zeros((s, cols))
  x = np.linspace(0, len, len)
  x_new = np.linspace(0, len, s)
  for i in range(cols):
    fd_new[:, i] = interpolate.interp1d(x, fd[:, i])(x_new)
  # if dir == "SN007":
  # show_plot(x, x_new, fd[:,idx], fd_new[:, idx])
  return fd_new

def save_plots(arr):
  for i in tqdm(range(arr.shape[0])):
    plot_lm(arr, i)