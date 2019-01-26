from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import numpy as np

from .Dataset import Dataset
from ..utils.utils import measure_time


class ValidationSet(Dataset):
  """
  validation set.
  Args:
    im_dir: char, the direction of images
    mask_dir: char, the direction of masks
    serial_num: char, the serial number of image and mask
    classifier:
    **kwargs: ..., include batch_size, shuffle, ..., and preprocess parameters
  """

  def __init__(
    self,
    im_dir=None,
    mask_dir=None,
    serial_nums=None,
    classifier=None,
    **kwargs):

    super(ValidationSet, self).__init__(dataset_size=len(serial_nums),
                                        shuffle=False,
                                        **kwargs)

    self.im_dir = im_dir
    self.mask_dir = mask_dir
    self.serial_nums = serial_nums
    self.classifier = classifier

  def set_classifier(self, classifier):
    self.classifier = classifier

  def get_sample(self, ptr):
    """Here one sample means one image and one mask.
    :param: ptr: ...
    Returns:
      im: np.array, one image
      mask: np.array, one mask
    """
    im, mask, im_name, mask_name = self.open_as_gray(self.serial_nums[ptr])
    # if self.preprocess:
    #   ...
    return im, mask, im_name, mask_name

  def next_batch(self):
    """Next batch of images and masks.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      masks: a numpy array of masks, same shape with ims
      mask_names: a numpy array of mask names, len(mask_name) >= 1
      self.epoch_done: whether the epoch is over
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done and self.shuffle:
      np.random.shuffle(self.serial_nums)

    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, mask_list, im_name_list, mask_name_list = zip(*samples)
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(im_list)
    masks = np.stack(mask_list)
    im_names = np.hstack(im_name_list)
    mask_names = np.hstack(mask_name_list)
    return ims, masks, im_names, mask_names, self.epoch_done

  def open_as_gray(self, serial_num):
    # open an image and convert to gray
    serial_num = str(serial_num)
    im_name = "img_"+serial_num+".jpg"
    mask_name = "label_"+serial_num+".png"
    # im = np.asarray(Image.open(osp.join(self.im_dir, im_name)).convert("L"))
    im = np.asarray(Image.open(osp.join(self.im_dir, im_name)))
    # im = np.expand_dims(im, axis=-1)
    _mask = np.asarray(Image.open(osp.join(self.mask_dir, mask_name)).convert("L"))
    _mask = np.expand_dims(_mask, axis=-1)
    mask_0 = np.where(np.logical_or(_mask == 60, _mask == 180), 1, 0)
    mask_1 = np.where(np.logical_or(_mask == 120, _mask == 180), 1, 0)
    mask = np.concatenate([mask_0, mask_1], axis=-1)
    return im.transpose(2, 0, 1), mask.transpose(2, 0, 1), im_name, mask_name

  def predict(self, verbose=True):
    """segment the whole image set.
    Args:
      verbose: whether to print the progress
    Returns:
      preds: numpy array with shape [N, C]
      masks: numpy array with shape [N]
      im_names: numpy array with shape [N]
      mask_names: numpy array with shape [N]
      ...: numpy array with shape [N]
    """
    ims, masks, im_names, mask_names, pred = [], [], [], [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()
    while not done:
      ims_, masks_, im_names_, mask_names_, done = self.next_batch()
      pred_ = self.classifier(ims_)
      pred.append(pred_)
      masks.append(masks_)
      im_names.append(im_names_)
      mask_names.append(mask_names_)

      if verbose:
        # Print the progress of extracting feature
        total_batches = (self.prefetcher.dataset_size
                         // self.prefetcher.batch_size + 1)
        step += 1
        if step % 20 == 0:
          if not printed:
            printed = True
          else:
            # Clean the current line
            sys.stdout.write("\033[F\033[K")
          print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
                .format(step, total_batches,
                        time.time() - last_time, time.time() - st))
          last_time = time.time()

    pred = np.vstack(pred)
    masks = np.vstack(masks)
    im_names = np.hstack(im_names)
    mask_names = np.hstack(mask_names)
    return pred, masks, im_names, mask_names


  def predict_batch(self, verbose=True):
    """segment the whole image set.
    Args:
      verbose: whether to print the progress
    Returns:
      preds: numpy array with shape [N, C]
      masks: numpy array with shape [N]
      im_names: numpy array with shape [N]
      mask_names: numpy array with shape [N]
      ...: numpy array with shape [N]
    """
    ims, masks, im_names, mask_names, pred = [], [], [], [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()

    ims_, masks_, im_names_, mask_names_, done = self.next_batch()
    pred_ = self.classifier(ims_)
    pred.append(pred_)
    masks.append(masks_)
    im_names.append(im_names_)
    mask_names.append(mask_names_)

    if verbose:
      # Print the progress of extracting feature
      total_batches = (self.prefetcher.dataset_size
                       // self.prefetcher.batch_size + 1)
      step += 1
      if step % 20 == 0:
        if not printed:
          printed = True
        else:
          # Clean the current line
          sys.stdout.write("\033[F\033[K")
        print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
              .format(step, total_batches,
                      time.time() - last_time, time.time() - st))
        # last_time = time.time()

    pred = np.vstack(pred)
    masks = np.vstack(masks)
    im_names = np.hstack(im_names)
    mask_names = np.hstack(mask_names)
    return pred, masks, im_names, mask_names, done


  def eval(self, verbose=True):
    """Evaluate using metric IOD or mIOU.
    Args:
      verbose: whether to print the intermediate information
    """

    # A helper function just for avoiding code duplication.
    def sigmoid(x):
      return 1. / (1 + np.exp(-x))

    def compute_mIOU_score(y_true, y_pred, smooth=1):
      """
      :param y_true:
      :param y_pred:
      :return:
      """
      y_true_f, y_pred_f = y_true.flatten().astype(int), y_pred.flatten().astype(int)
      I = (y_true_f & y_pred_f).sum()
      U = (y_true_f | y_pred_f).sum()
      return (I + smooth) / (U + smooth)

    def compute_maxIOU_score(y_true, y_pred, smooth=1):
      size = len(y_true)
      max_iou = []
      for i in range(size):
        max_iou.append(max(compute_mIOU_score(y_true[i], y_pred[i]), compute_mIOU_score(y_true[i], y_pred[i, ::-1])))
      return np.mean(max_iou)

    def get_mIOU_score(
      masks=None,
      pred=None,
      threshold=0.5
    ):
      pred = np.where(sigmoid(pred) > threshold, 1, 0)
      miou = 0 # compute_mIOU_score(masks, pred)
      maxiou = compute_maxIOU_score(masks, pred, smooth=1)
      return maxiou, miou

    def print_scores(maxiou, miou):
      print('[maxiou: {:5.2%}], [miou_0: {:5.2%}]'
            .format(maxiou, miou))

    done = False
    result_maxiou = []
    result_miou = []
    while not done:
      with measure_time('Predicting images...', verbose=verbose):
        pred, masks, im_names, mask_names, done = self.predict_batch(verbose)
        print("done: ", done)
      with measure_time('Computing scores...', verbose=verbose):
        maxiou, miou = get_mIOU_score(masks, pred)
        result_maxiou.append(maxiou)
        result_miou.append(miou)

      print('{:<30}'.format('valid iou score:'), end='')
      print_scores(np.mean(result_maxiou), np.mean(result_miou))
    return np.mean(result_maxiou), np.mean(result_miou)
