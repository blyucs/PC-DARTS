import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from utils import *
from torch.autograd import Variable
from model import NetworkCIFAR as Network

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"   # batchsize

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/tmp', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
# parser.add_argument('--init_channels', type=int, default=20, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
# parser.add_argument('--layers', type=int, default=16, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP/', help='experiment name')
parser.add_argument('--load-dir', type=str, default='EXP/pretrain-20211229-105459', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='BS_V1', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument(
  "--th",
  type=float,
  default=0.02,
  help="threshold for prunning"
)
parser.add_argument(
  "--alpha",
  type=float,
  # default=0.2,
  default=0,
)
args = parser.parse_args()

args.save = '{}finetune-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

if args.set=='cifar100':
    CIFAR_CLASSES = 100
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = nn.DataParallel(model)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  filename = os.path.join(args.load_dir,
                          "model_best.pth.tar")
  checkpoint = torch.load(filename)
  model.load_state_dict(checkpoint["state_dict"])

  model = model.cuda()


  fix_cfg = checkpoint["fix_cfg"]
  model.module.load_fix_configs(fix_cfg["data"])
  model.module.load_fix_configs(fix_cfg["grad"], grad=True)

  # Prune
  masks = {}
  prune(model, args, masks)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  #valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, masks)
    logging.info('train_acc %f', train_acc)

    if epoch % 10 == 0:
      prune(model, args, masks)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer, masks):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    l1_reg = 0
    zero_cnt = np.zeros((4, ))
    total_param_cnt = np.zeros((4, ))
    cur_fix_cfg = model.module.get_fix_configs(data_only=True)
    for name, param in model.module.named_parameters():
      # Using quantization module for batch norm will yield weird behavior,
      # which we suspect originates from the fixed-point training codes
      # So we don't quantize the bn module
      if "bn" in name or "downsample.1" in name:
        continue

      length = len(name.split('.'))
      temp = cur_fix_cfg
      for j in range(length):
        temp = temp[name.split('.')[j]]
      scale = temp['scale']

      l1_reg_, zero_cnt_, total_param_cnt_ = calc_l1_and_zero_ratio(param, scale)
      l1_reg += l1_reg_
      zero_cnt += zero_cnt_
      total_param_cnt += total_param_cnt_

    loss = loss + args.alpha * l1_reg / np.sum(total_param_cnt)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

    # set gradients of pruned weights to 0
    for name, p in model.named_parameters():
      p.grad.data = p.grad.data*masks[name]

    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def prune(model, args, masks):
  for name, p in model.named_parameters():
    tensor = p.data.cpu().numpy()
    threshold = args.th #np.std(tensor) * args.sensitivity
    #print(f'Pruning with threshold : {threshold} for layer {name}')
    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
    p.data = torch.from_numpy(new_mask).cuda()
    mask = np.where(abs(tensor) < threshold, 0., 1.)
    masks[name] = torch.from_numpy(mask).float().cuda()
  print("--- Finished Pruning ---")

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():    
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      logits,_ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      l1_reg = 0
      zero_cnt = np.zeros((4, ))
      total_param_cnt = np.zeros((4, ))
      cur_fix_cfg = model.module.get_fix_configs(data_only=True)
      for name, param in model.module.named_parameters():
        # Using quantization module for batch norm will yield weird behavior,
        # which we suspect originates from the fixed-point training codes
        # So we don't quantize the bn module
        if "bn" in name or "downsample.1" in name:
          continue

        length = len(name.split('.'))
        temp = cur_fix_cfg
        for i in range(length):
          temp = temp[name.split('.')[i]]
        scale = temp['scale']

        zero_cnt_, total_param_cnt_ = calc_zero_ratio(param, scale)
        l1_reg += torch.norm(param, p=1)
        zero_cnt += zero_cnt_
        total_param_cnt += total_param_cnt_

      z1 = 100-zero_cnt[0] / total_param_cnt[0] * 100.0
      z2 = 100-zero_cnt[1] / total_param_cnt[1] * 100.0
      z3 = 100-zero_cnt[2] / total_param_cnt[2] * 100.0
      z4 = 100-zero_cnt[3] / total_param_cnt[3] * 100.0

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f, Nonzero_ratio: a:%.2f, b:%.2f, c:%.2f, d:%.2f',
                     step, objs.avg, top1.avg, top5.avg, z1, z2, z3, z4)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

