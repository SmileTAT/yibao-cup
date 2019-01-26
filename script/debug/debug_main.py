from __future__ import print_function

import sys
sys.path.insert(0, ".")
sys.path.append("../../")
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import pickle

import torch
from torchsummary import summary
from torch.nn.parallel import DataParallel
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.nn import BCELoss, BCEWithLogitsLoss

from segmentation_package.dataset import TrainSet, ValidationSet, TestSet
from segmentation_package.model.unet import UNet
from segmentation_package.model.loss import weighted_bce_dice_loss
from segmentation_package.model.loss import dice_loss
from segmentation_package.model.loss import match_loss
from segmentation_package.utils.utils import str2bool
from segmentation_package.utils.utils import tight_float_str as tfs
from segmentation_package.utils.utils import time_str
from segmentation_package.utils.utils import ReDirectSTD
from segmentation_package.utils.utils import set_devices
from segmentation_package.utils.utils import set_seed
from segmentation_package.utils.utils import load_ckpt
from segmentation_package.utils.utils import adjust_lr_exp
from segmentation_package.utils.utils import adjust_lr_staircase
from segmentation_package.utils.utils import AverageMeter
from segmentation_package.utils.utils import may_set_mode
from segmentation_package.utils.utils import to_scalar
from segmentation_package.utils.utils import save_ckpt


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--sys_device_ids", type=eval, default=(0,))
    parser.add_argument("-r", "--run", type=int, default=1)
    parser.add_argument("--set_seed", type=str2bool, default=False)
    parser.add_argument("--dataset", type=str, default="yibao-cup-data")
    parser.add_argument("--pattern", type=str, default="trainval",
                        choices=["trainval", "train", "test"])
    parser.add_argument("--resize_h_w", type=eval, default=(400, 400))
    parser.add_argument("--base_channel", type=int, default=16)
    # add_argument for preprocess and augmentation
    # ...

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--epochs_per_val', type=int, default=1e10)

    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=151)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(101, 201,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=300)

    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    # If you want to make your results exactly reproducible, you have
    # to also set num of threads to 1 during training.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    # Image processing anf augmentation ...
    #
    self.dataset = args.dataset
    self.pattern = args.pattern
    self.base_channel = args.base_channel
    # self.train_batch_size = 32
    # self.train_final_batch = False
    # self.train_shuffle = True
    #
    # self.valid_batch_size = 32
    # self.valid_final_batch = True
    # self.valid_shuffle = False
    #
    # dataset_kwargs = dict(
    #   name=self.dataset,
    #   batch_dims="NCHW",
    # )
    #
    # train_set_kwargs = dict(
    #
    # )

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in epochs) to test on val set.
    self.epochs_per_val = args.epochs_per_val

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.steps_per_log = args.steps_per_log

    # Only test and without training.
    # self.only_test = args.only_test

    self.resume = args.resume


    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        'pt_{}_'.format(self.pattern) +
        'bc_{}_'.format(self.base_channel) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


def creat_dataset():
  im_dir = "/data/data_smile/img_mask/img/"
  mask_dir = "/data/data_smile/img_mask/mask/"
  im2p_dir = "/data/data_smile/img_mask/im2p.pkl"
  with open(im2p_dir, 'rb') as pickle_file:
    im2p = pickle.load(pickle_file)

  total_sample_size = 40000
  serial_nums = np.arange(total_sample_size)
  train_serial_nums, valid_serial_nums = train_test_split(serial_nums, test_size=0.1, random_state=42)
  # test_serial_nums

  train_set = TrainSet(im_dir=im_dir,
                       mask_dir=mask_dir,
                       serial_nums=train_serial_nums,
                       batch_size=16,
                       im2p=im2p,
                       num_prefetch_threads=8)

  valid_set = ValidationSet(im_dir=im_dir,
                            mask_dir=mask_dir,
                            serial_nums=valid_serial_nums,
                            batch_size=4,
                            num_prefetch_threads=8)
  return train_set, valid_set


class classifier(object):
  """A function to be called in the val/test set, to segmentation.
  Args:
    TVT: A callable to transfer images to specific device.
  """
  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    mask = self.model(ims)
    mask = mask.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return mask


def main():
  """
  baseline: 1. there is no test set, only train set, and valid set.
            len(train set) / len(valid set) = 4 / 1
            5-fold cross validation --> maybe get a more stable metric result.
            2. no preprocess, no augmentation
            3. ... just baseline :)
  :return:
  """
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)
  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########
  train_set, valid_set = creat_dataset()

  ###########
  # Model   #
  ###########
  model = UNet(2, base_channel=cfg.base_channel)
  model_w = DataParallel(model, )

  #############################
  # Criteria and Optimizers   #
  #############################
  # loss_fn = weighted_bce_dice_loss(weight=1e-1)
  # loss_fn = BCELoss(weight=None, size_average=True)
  # loss_fn = BCEWithLogitsLoss()
  # loss_fn = dice_loss
  loss_fn = match_loss(weight=1e-1)
  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  def validate():
    if valid_set.classifier is None:
      valid_set.set_classifier(classifier(model_w, TVT))
    print('\n=========> Test on validation set <=========\n')
    max_iou, miou = valid_set.eval(verbose=False)
    print()
    return max_iou, miou

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    if cfg.lr_decay_type == 'exp':
      adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.total_epochs,
        cfg.exp_decay_at_epoch)
    else:
      adjust_lr_staircase(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    # For recording loss, iou
    iou_meter = AverageMeter()
    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, masks, im_names, mask_names, epoch_done = train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      masks_t = TVT(torch.from_numpy(masks).float())

      pred_mask = model_w(ims_var)

      loss = loss_fn(masks_t, pred_mask)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ############
      # Step Log #
      ############

      # IOU
      # loss, ...
      # iou_meter.update()
      loss_meter.update(to_scalar(loss))

      if step % cfg.steps_per_log == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st, )

        loss_log = (', loss {:.4f}'.format(loss_meter.val, ))

        log = time_log + loss_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st)

    loss_log = (', loss {:.4f}'.format(loss_meter.avg,))

    log = time_log + loss_log
    print(log)

    ##########################
    # Test on Validation Set #
    ##########################

    max_iou, miou = 0, 0
    if ((ep + 1) % cfg.epochs_per_val == 0) and (valid_set is not None):
      max_iou, miou = validate()

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(max_iou=max_iou,
             miou=miou),
        ep)
      writer.add_scalars(
        'loss',
        dict(loss=loss_meter.avg, ),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  # test(load_model_weight=False)


if __name__ == "__main__":
  main()
