from .Dataset import Dataset

import os.path as osp
from PIL import Image
import numpy as np
import cv2


class TrainSet(Dataset):
  """
  Training set.
  # Take care of mirrored should be False
  # And, Augmentation should also be more careful.
  Args:
    im_dir: char, the direction of images
    mask_dir: char, the direction of masks
    serial_num: char, the serial number of image and mask
    im2p: dict, key=img_*.jpg, value=[num0, num1]
    **kwargs: ..., include batch_size, shuffle, ..., and preprocess parameters
  """

  def __init__(
    self,
    im_dir=None,
    mask_dir=None,
    serial_nums=None,
    im2p=None,
    **kwargs):

    self.im_dir = im_dir
    self.mask_dir = mask_dir
    self.serial_nums = serial_nums
    self.im2p = im2p

    super(TrainSet, self).__init__(
      dataset_size=len(self.serial_nums),
      **kwargs)

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
    # if self.augmentation:
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
    ims = np.stack(im_list, axis=0)
    masks = np.stack(mask_list, axis=0)
    im_names = np.hstack(im_name_list)
    mask_names = np.hstack(mask_name_list)
    return ims, masks, im_names, mask_names, self.epoch_done

  def open_as_gray(self, serial_num, resize_h_w=(400, 400)):
    # open an image and convert to gray
    serial_num = str(serial_num)
    im_name = "img_"+serial_num+".jpg"
    mask_name = "label_"+serial_num+".png"
    # im = np.asarray(Image.open(osp.join(self.im_dir, im_name)).convert("L"))
    im = np.asarray(Image.open(osp.join(self.im_dir, im_name)))
    im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # im = np.expand_dims(im, axis=-1)
    _mask = np.asarray(Image.open(osp.join(self.mask_dir, mask_name)).convert("L"))
    # print(_mask.shape)
    _mask = np.expand_dims(_mask, axis=-1)
    # print(_mask.shape)
    mask_0 = np.where(np.logical_or(_mask == 60, _mask == 180), 1, 0)
    # print(mask_0.shape)
    mask_1 = np.where(np.logical_or(_mask == 120, _mask == 180), 1, 0)
    # print(mask_1.shape)
    mask = np.concatenate([mask_0, mask_1], axis=-1)
    # if self.im2p[im_name][0] <= self.im2p[im_name][1]:
    #   mask = np.concatenate([mask_0, mask_1], axis=-1)
    # else:
    #   mask = np.concatenate([mask_1, mask_0], axis=-1)
    # print(mask.shape)
    return im.transpose(2, 0, 1), mask.transpose(2, 0, 1), im_name, mask_name

