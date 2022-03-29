import argparse

def get_args():
  parser = argparse.ArgumentParser(description="Normalizing Flow")
  parser.add_argument('--root', type=str, default='../../fg/glow_revisited/data/test/', help='Root path of dataset')
  parser.add_argument('--batch', type=int, default=32, help='Batch size')
  parser.add_argument('--n_chan', type=int, default=3, help='# of channels')
  parser.add_argument('--n_class', type=int, default=2, help='# of class')
  parser.add_argument('--iter', type=int, default=20, help='# of epochs')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  
  parser.add_argument('--n_head', type=int, default=4, help='Learning rate')
  parser.add_argument('--nx', type=int, default=2, help='Learning rate')
  parser.add_argument('--scale', type=int, default=1000, help='Learning rate')
  parser.add_argument('--feat', type=str, default="lm", help='Learning rate')
  args = parser.parse_args()
  return args