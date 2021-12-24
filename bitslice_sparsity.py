import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils
from utils import *
class BitSparsity(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.grad_clip = args.grad_clip

    self.optimizer_bit = torch.optim.SGD(
      self.model.parameters(),
      lr=args.bit_learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  def step(self, valid_queue, logging, epochs, report_frequency, step):
    self.optimizer_bit.zero_grad()
    for i in range(epochs):
      l1_reg = 0
      zero_cnt = np.zeros((4, ))
      total_param_cnt = np.zeros((4, ))
      cur_fix_cfg = self.model.module.get_fix_configs(data_only=True)
      for name, param in self.model.module.named_parameters():
        # Using quantization module for batch norm will yield weird behavior,
        # which we suspect originates from the fixed-point training codes
        # So we don't quantize the bn module
        if "bn" in name or "downsample.1" in name or "alphas_normal" in name or "alphas_reduce" in name:
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

      l1_reg.backward()
      nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
      self.optimizer_bit.step()

    if step % report_frequency== 0:
      # torch.cuda.empty_cache()
      z1 = zero_cnt[0] / total_param_cnt[0] * 100.0
      z2 = zero_cnt[1] / total_param_cnt[1] * 100.0
      z3 = zero_cnt[2] / total_param_cnt[2] * 100.0
      z4 = zero_cnt[3] / total_param_cnt[3] * 100.0

      try:
        input_search, target_search = next(valid_queue_iter)
      except:
        valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
      input_search = input_search.cuda()
      target_search = target_search.cuda(non_blocking=True)
      logits = self.model(input_search)

      prec1, prec5 = utils.accuracy(logits, target_search, topk=(1, 5))

      # if i % interval == 0:
      logging.info('NonZero_ratio: a:%.2f, b:%.2f, c:%.2f, d:%.2f', 100-z1, 100-z2, 100-z3, 100-z4)
      logging.info('After sparsity accuracy step:%03d prec1:%f prec5:%f', step, prec1, prec5)
