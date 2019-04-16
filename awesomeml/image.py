# -*- coding: utf-8 -*-
"""
Image processing.
"""
import collections.abc
import numpy as np
from PIL import Image
import imageio
from . import utils as _utils

def center_crop(img, size):
    """Crop the given image at the center

    Args:

      img (ndarray): Input image.

      size (int, Seq[int]): Desired output size of the crop. If size
        is an int instead of sequence like (h, w), a square crop
        (size, size) is made.

    Examples:

    >>> import numpy as np
    >>> img = np.ones((100, 200), dtype=np.uint8)
    >>> center_crop(img, 50).shape
    (50, 50)
    >>> img = np.ones((100, 200, 3), dtype=np.uint8)
    >>> center_crop(img, 50).shape
    (50, 50, 3)
    >>> center_crop(img, (50, 61)).shape
    (50, 61, 3)
    """
    H, W = img.shape[0], img.shape[1]
    size = _utils.validate_int_seq(size, 2, name='size')
    rmH, rmW = H-size[0], W-size[1]
    iH1 = (H - size[0]) // 2
    iH2 = iH1 + size[0]
    iW1 = (W - size[1]) // 2
    iW2 = iW1 + size[1]
    return img[iH1:iH2, iW1:iW2, ...]

def resize(img, size, interpolation='bilinear'):
    """Resize an image

    Args:

      img (ndarray): Input image.

      size (int, Seq[int]): Desired output size. If size is a sequence
        like (h, w), output size will be matched to this. If size is
        an int, smaller edge of the image will be matched to this
        number. i.e, if height > width, then image will be rescaled to
        (size * height / width, size)

      interpolation (str): Interpolation method. Default to 'bilinear'.

    Examples:

    >>> img = np.ones((100, 200, 3), dtype=np.uint8)
    >>> resize(img, 50).shape
    (50, 100, 3)
    >>> resize(img, (50, 101)).shape
    (50, 101, 3)
    >>> resize(0.2, 50)
    Traceback (most recent call last):
      ...
    TypeError: ...
    >>> resize(img, (50, 50, 50))
    Traceback (most recent call last):
      ...
    TypeError: ...
    >>> resize(img, 50, interpolation='nearest').shape
    (50, 100, 3)
    >>> resize(img, 50, interpolation='bicubic').shape
    (50, 100, 3)
    >>> import uuid
    >>> resize(img, 50, interpolation=uuid.uuid4()).shape
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> resize(img, 100).shape
    (100, 200, 3)
    >>> img = np.ones((200, 100, 3), dtype=np.uint8)
    >>> resize(img, 50).shape
    (100, 50, 3)
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if not isinstance(img, Image.Image):
        raise TypeError('img should be PIL Image or ndarray. Got {}'.format(
            type(img)))
    if not (isinstance(size, int) or
            (isinstance(size, collections.abc.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    interpolation = _utils.validate_option(
        interpolation, ['nearest', 'bilinear', 'bicubic'],
        name='interpolation', str_normalizer=str.lower)
    if interpolation == 'nearest':
        interpolation = Image.NEAREST
    elif interpolation == 'bilinear':
        interpolation = Image.BILINEAR
    elif interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    else: #pragma no cover
        assert False
        
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            pass
        if w < h:
            ow = size
            oh = int(size * h / w)
            img = img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            img = img.resize((ow, oh), interpolation)
    else:
        img = img.resize(size[::-1], interpolation)
    return np.array(img)
    
def img_as(img, target_dtype):
    """Convert dtype of an image (or images) represetned as arrays

    For floating point dtypes, pixel values are in range [0, 1], where
    0 represents 'black' and 1 represents 'white'. For integral
    dtypes, pixel values are in range [min, max], where min and max
    are the minimum and maximum possible values of this dtype.

    Args:

      img (array-like): Input image (or images) represented as an
        array.
    
      target_dtype (dtype): Dtype of the converted image.

    Examples:

    >>> img = np.array([0, 127, 255], dtype=np.uint8)
    >>> img2 = np.array([-128, -1, 127], dtype=np.int8)
    >>> img3 = np.array([0, 127/255, 255], dtype=np.float32)
    >>> np.all(img_as(img2, np.uint8) == img)
    True
    >>> np.all(img_as(img, np.int8) == img2)
    True
    >>> np.all(img_as(img3, np.float64) == img3)
    True
    """
    target_dtype = np.dtype(target_dtype)
    if img.dtype == target_dtype:
        res = img
    elif (issubclass(img.dtype.type, np.floating) and
          issubclass(target_dtype.type, np.floating)):
        res = img.astype(target_dtype)
    else:
        res = img.astype(np.float64)
        if not issubclass(img.dtype.type, np.floating):
            # from integer type
            info1 = np.iinfo(img.dtype)
            res = (res - info1.min) / (info1.max - info1.min)
        if not issubclass(target_dtype.type, np.floating):
            # to integer type
            info2 = np.iinfo(target_dtype)
            res = res * (info2.max -info2.min) + info2.min
        res = res.astype(target_dtype)
    assert res.dtype == target_dtype
    return res
        

def img_as_float32(img):
    """Convert an image (or images) to float32 format.

    Args:

      img (array-like): Input image (or images) as a tensor.

    Returns:

    Tensor: Result tensor with dtype torch.float32, pixel values are
      in range [0, 1].

    Examples:

    >>> img = np.array([0, 127, 255], dtype=np.uint8)
    >>> img2 = np.array([0, 127/255, 1], dtype=np.float32)
    >>> np.all(img_as_float32(img) == img2)
    True
    """
    return img_as(img, np.float32)


def img_as_float64(img):
    """Convert an image (or images) to float32 format.

    Args:

      img (array-like): Input image (or images) as a tensor.

    Returns:

    Tensor: Result tensor with dtype torch.float32, pixel values are
      in range [0, 1].

    Examples:

    >>> img = np.array([0, 127, 255], dtype=np.uint8)
    >>> np.all(img_as_float64(img) == np.array([0, 127/255, 1]))
    True
    """
    
    return img_as(img, np.float64)


def img_as_uint8(img):
    """Convert an image (or images) to uint8 format.

    Args:

      img (array-like): Input image (or images) as a tensor.

    Returns:

    Tensor: Result tensor with dtype torch.float32, pixel values are
      in range [0, 255].

    Examples:

    >>> img = np.array([0, 127, 255], dtype=np.uint8)
    >>> np.all(img_as_uint8(img) == img)
    True
    >>> img2 = np.array([0, 127/255, 1], dtype=np.float32)
    >>> np.all(img_as_uint8(img2) == img)
    True
    """
    return img_as(img, np.uint8)
