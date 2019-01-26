from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import numpy as np

from .Dataset import Dataset


class TestSet(Dataset):
  """
  testing set.
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

    super(TestSet, self).__init__(dataset_size=len(serial_nums),
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
    im, _, im_name, _ = self.open_as_gray(self.serial_nums[ptr])
    # if self.preprocess:
    #   ...
    return im, _, im_name, _

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
    im_list, _, im_name_list, _ = zip(*samples)
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(im_list)
    im_names = np.concatenate(im_name_list)
    return ims, None, im_names, None, self.epoch_done

  def open_as_gray(self, serial_num):
    # open an image and convert to gray
    serial_num = str(serial_num)
    im_name = "img_"+serial_num+".jpg"
    im = np.asarray(Image.open(osp.join(self.im_dir, im_name)).convert("L"))
    im = np.expand_dims(im, axis=-1)
    return im.transpose(2, 0, 1), None, im_name, None

  def predict(self, verbose=True):
    """segment the whole image set.
    Args:
      verbose: whether to print the progress
    Returns:
      feat: numpy array with shape [N, C]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    ims, _, im_names, _, pred = [], [], [], [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()
    while not done:
      ims_, _, im_names_, _, done = self.next_batch()
      pred_ = self.classifier(ims_)
      pred.append(pred_)
      im_names.append(im_names_)

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
    im_names = np.hstack(im_names)
    return pred, im_names
