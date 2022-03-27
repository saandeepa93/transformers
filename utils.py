import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def res_signal(arr, idx):
  # plt.plot(arr[:, idx], 'r-')
  arr_resample = resample_by_col(arr, 1000, idx)
  print(arr_resample.shape)
  # plt.plot(arr_resample[:, idx], 'bo')
  # plt.show()


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