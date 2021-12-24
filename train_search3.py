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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import nics_fix_pt as nfp
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from bitslice_sparsity import BitSparsity
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"   # batchsize

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--batch_size', type=int, default=192, help='batch size')
# parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--bit_learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--report_freq', type=float, default=40, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=12, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()

args.save = '{}search-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
  #torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  # logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = nn.DataParallel(model)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  # num_train = (int)(len(train_data)*0.2)
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.val_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  bitsparsity = BitSparsity(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.module.genotype()
    logging.info('genotype = %s', genotype)

    logging.info(F.softmax(model.module.alphas_normal, dim=-1))
    logging.info(F.softmax(model.module.alphas_reduce, dim=-1))
    logging.info(F.softmax(model.module.betas_normal[2:5], dim=-1))
    #model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch, bitsparsity)
    logging.info('train_acc %f', train_acc)

    # validation
    if args.epochs-epoch<=1:
      valid_acc, valid_obj, z1, z2, z3, z4 = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f NonZero_ratio: a:%.2f, b:%.2f, c:%.2f, d:%.2f', valid_acc, 100-z1, 100-z2, 100-z3, 100-z4)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch, bitsparsity):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  # objs_bit = utils.AvgrageMeter()
  # top1_bit = utils.AvgrageMeter()
  # top5_bit = utils.AvgrageMeter()

  model.module.set_fix_method(nfp.FIX_AUTO)

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    # get a random minibatch from the search queue with replacement
    #input_search, target_search = next(iter(valid_queue))
    try:
      input_search, target_search = next(valid_queue_iter)
    except:
      valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    if epoch>=10:
    # if epoch>=3:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
      # torch.cuda.empty_cache()
      bitsparsity.step(valid_queue, logging, 1, args.report_freq, step)

    input = input.cuda()
    target = target.cuda(non_blocking=True)
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('Train accuracy %03d %e top1:%f top5:%f', step, objs.avg, top1.avg, top5.avg)

  # if epoch >= 5:
  #   bitsparsity.step(valid_queue, logging, 10)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.module.set_fix_method(nfp.FIX_FIXED)
  model.eval()

  with torch.no_grad():    
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  l1_reg = 0
  zero_cnt = np.zeros((4, ))
  total_param_cnt = np.zeros((4, ))
  cur_fix_cfg = model.module.get_fix_configs(data_only=True)
  for name, param in model.module.named_parameters():
    # Using quantization module for batch norm will yield weird behavior,
    # which we suspect originates from the fixed-point training codes
    # So we don't quantize the bn module
    if "bn" in name or "downsample.1" in name or "alphas_normal" in name or "alphas_reduce" in name:
      continue

    length = len(name.split('.'))
    temp = cur_fix_cfg
    for i in range(length):
      # if name.split('.')[i] == 'bias':
      #     i = 1
      temp = temp[name.split('.')[i]]
    scale = temp['scale']

    zero_cnt_, total_param_cnt_ = calc_zero_ratio(param, scale)
    # l1_reg += torch.norm(param, p=1)
    zero_cnt += zero_cnt_
    total_param_cnt += total_param_cnt_

  z1 = zero_cnt[0] / total_param_cnt[0] * 100.0
  z2 = zero_cnt[1] / total_param_cnt[1] * 100.0
  z3 = zero_cnt[2] / total_param_cnt[2] * 100.0
  z4 = zero_cnt[3] / total_param_cnt[3] * 100.0

  return top1.avg, objs.avg, z1, z2, z3, z4


if __name__ == '__main__':
  main() 

