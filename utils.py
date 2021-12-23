import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {
        n: {
            "method": torch.autograd.Variable(
                torch.IntTensor(np.array([method])), requires_grad=False
            ),
            "scale": torch.autograd.Variable(
                torch.IntTensor(np.array([scale])), requires_grad=False
            ),
            "bitwidth": torch.autograd.Variable(
                torch.IntTensor(np.array([bitwidth])), requires_grad=False
            ),
        }
        for n in names
    }

# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g

# https://discuss.pytorch.org/t/how-to-override-the-gradients-for-parameters/3417/6
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

def calc_zero_ratio(weights, scale):
    step = 2 ** (scale - 8)
    #x = Round.apply(weights / step) * step

    y = Round.apply(weights.abs() / step).data.cpu().numpy()
    b1 = np.floor(y/64)
    b2 = np.floor((y-b1*64)/16)
    b3 = np.floor((y-b1*64-b2*16)/4)
    b4 = y-b1*64-b2*16-b3*4

    zero_cnt = np.array([np.count_nonzero(b1==0), np.count_nonzero(b2==0), np.count_nonzero(b3==0), np.count_nonzero(b4==0)])
    total_param_cnt = np.array([np.size(b1), np.size(b2), np.size(b3), np.size(b4)])
    return zero_cnt, total_param_cnt

def calc_l1_and_zero_ratio(weights, scale):
    x = Round.apply(weights.abs() / 2 ** (scale - 8))

    b1 = Floor.apply(x/64)
    b2 = Floor.apply((x-b1.detach()*64)/16)
    b3 = Floor.apply((x-b1.detach()*64-b2.detach()*16)/4)
    b4 = x-b1.detach()*64-b2.detach()*16-b3.detach()*4

    l1_norm = b1.abs().sum() + b2.abs().sum() + b3.abs().sum() + b4.abs().sum()

    b1_ = b1.data.cpu().numpy()
    b2_ = b2.data.cpu().numpy()
    b3_ = b3.data.cpu().numpy()
    b4_ = b4.data.cpu().numpy()

    zero_cnt = np.array([np.count_nonzero(b1_==0), np.count_nonzero(b2_==0), np.count_nonzero(b3_==0), np.count_nonzero(b4_==0)])
    total_param_cnt = np.array([np.size(b1_), np.size(b2_), np.size(b3_), np.size(b4_)])

    return l1_norm, zero_cnt, total_param_cnt

