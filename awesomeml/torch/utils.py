# -*- coding: utf-8 -*-
"""
PyTorch related utilities.
"""
import collections
import torch
import decoutils
from pathlib import Path
from ..datasets import utils as _ds_utils
from .. import utils as _utils
from .. import config as _config

def validate_data_home(value, create=False, name='data_home'):
    """Check and return a valid value for argument data_home
    
    Args:

      value: Value of argument data_home

      create (bool): Whether create a directory if value is not None
        and does not point to the path of an existing directory.

      name (str): Name of the argument. Default to 'data_home'.

    Returns:

    path-like: If value is None, return DATA_HOME, else return Path(value).

    Examples:

    >>> import tempfile
    >>> import uuid
    >>> p = Path(tempfile.gettempdir())/str(uuid.uuid4())
    >>> validate_data_home(p, create=False)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> validate_data_home(p, create=True) == p
    True
    """
    return _utils.validate_dir(value, default_path=_config.TORCH_DATA_HOME,
                               create=create, name=name)


def validate_model_home(value, create=False, name='model_home'):
    """Check and return a valid value for argument model_home
    
    Args:

      value: Value of argument model_home

      create (bool): Whether create a directory if value is not None
        and does not point to the path of an existing directory.

      name (str): Name of the argument. Default to 'model_home'.

    Returns:

    path-like: If value is None, return MODEL_HOME, else return Path(value).

    Examples:

    >>> import tempfile
    >>> import uuid
    >>> p = Path(tempfile.gettempdir())/str(uuid.uuid4())
    >>> validate_model_home(p, create=False)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> validate_model_home(p, create=True) == p
    True
    """
    return _utils.validate_dir(value, default_path=_config.TORCH_MODEL_HOME,
                               create=create, name=name)



class iinfo:
    """Augumenting torch.iinfo by providing a 'min' property

    Examples:

    >>> iinfo(torch.uint8).min
    0
    >>> iinfo(torch.int8).min
    -128
    >>> iinfo(torch.int16).min
    -32768
    >>> iinfo(torch.int32).min
    -2147483648
    >>> iinfo(torch.int64).min
    -9223372036854775808
    >>> iinfo(torch.int64).max == torch.iinfo(torch.int64).max
    True
    >>> iinfo(torch.int64).bits == torch.iinfo(torch.int64).bits
    True
    """
    def __init__(self, dtype):
        self._iinfo = torch.iinfo(dtype)
        self.dtype = dtype
        

    @property
    def min(self):
        if self.dtype == torch.uint8:
            min_val = 0
        elif self.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            min_val = -self.max-1
        else: #pragma no cover
            raise TypeError('Wrong dtype:', self.dtype)
        return min_val
    
    @property
    def max(self):
        return self._iinfo.max

    @property
    def bits(self):
        return self._iinfo.bits



def img_as(img, dtype):
    """Convert dtype of an image (or images) represented as tensor

    For floating point dtypes, pixel values are in range [0, 1], where
    0 represents 'black' and 1 represents 'white'. For integral
    dtypes, pixel values are in range [min, max], where min and max
    are the minimum and maximum possible values of this dtype.

    Args:

      img (Tensor): Input image (or images) represented as a tensor.

      dtype (dtype): Dtype of the converted image.

    Examples:

    >>> img = torch.tensor([0, 127, 255], dtype=torch.uint8)
    >>> img2 = torch.tensor([-128, -1, 127], dtype=torch.int8)
    >>> img3 = torch.tensor([0, 127/255, 255], dtype= torch.float32)
    >>> (img_as(img2, torch.uint8) == img).all().item()
    1
    >>> (img_as(img, torch.int8) == img2).all().item()
    1
    >>> (img_as(img3, torch.float64) == img3.to(torch.float64)).all().item()
    1
    """
    #target_dtype = torch.dtype(target_dtype)
    if img.dtype == dtype:
        res = img
    elif img.dtype.is_floating_point and dtype.is_floating_point:
        res = img.to(dtype=dtype)
    else:
        res = img.to(dtype=torch.float64)
        if not img.dtype.is_floating_point:
            # from integer type
            info1 = iinfo(img.dtype)
            res = (res - info1.min) / (info1.max - info1.min)
        if not dtype.is_floating_point:
            # to integer type
            info2 = iinfo(dtype)
            res = res * (info2.max - info2.min) + info2.min
        res = res.to(dtype=dtype)
    assert res.dtype == dtype
    return res
    
    
def img_as_float32(img):
    """Convert an image (or images) to float32 format.

    Args:

      img (Tensor): Input image (or images) as a tensor.

    Returns:

    Tensor: Result tensor with dtype torch.float32, pixel values are
      in range [0, 1].

    Examples:

    >>> img = torch.tensor([0, 127, 255], dtype=torch.uint8)
    >>> img_as_float32(img)
    tensor([0.0000, 0.4980, 1.0000])
    """
    return img_as(img, torch.float32)


def img_as_float64(img):
    """Convert an image (or images) to float32 format.

    Args:

      img (Tensor): Input image (or images) as a tensor.

    Returns:

    Tensor: Result tensor with dtype torch.float32, pixel values are
      in range [0, 1].

    Examples:

    >>> img = torch.tensor([0, 127, 255], dtype=torch.uint8)
    >>> img_as_float64(img)
    tensor([0.0000, 0.4980, 1.0000], dtype=torch.float64)
    """
    return img_as(img, torch.float64)


def img_as_uint8(img):
    """Convert an image (or images) to uint8 format.

    Args:

      img (Tensor): Input image (or images) as a tensor.

    Returns:

    Tensor: Result tensor with dtype torch.float32, pixel values are
      in range [0, 255].

    Examples:

    >>> img = torch.tensor([0, 127, 255], dtype=torch.uint8)
    >>> (img_as_uint8(img) == img).all().item()
    1
    >>> img2 = torch.tensor([0, 127/255, 1], dtype=torch.float32)
    >>> (img_as_uint8(img2) == img).all().item()
    1
    """
    return img_as(img, torch.uint8)


def _validate_conv_args(*args):
    args = [_utils.validate_int_seq(x) for x in args]
    ndims = set(len(x) for x in args)
    ndims_max = max(ndims)
    if not (len(ndims) == 1 or len(ndims) == 2 and min(ndims) == 1):
        raise ValueError('Length of args mismatch, should be the same or 1.')
    args = [x * ndims_max if len(x) == 1 else x for x in args]
    return args


def conv_out_size(in_size, kernel_size, stride=1, padding=0, dilation=1,
                  transposed=False):
    """Calculate output size of the convolution operator
    
    Args:

      in_size (int, Seq[int]): Size of inputs to the convolution

      kernel_size (int, Seq[int]): Size of the convolutional kernel

      stride (int, Seq[int]): Stride for the convolution operator

      padding (int, Seq[int]): Amount of implicit paddings points

      dilation (int, Seq[int]): Spacing between kernel points

      transposed (bool): If true, calculate the output size of
        transposed convolution instead of ordinary convolution.

    Returns:

    int, Seq[int]: Size of output of the convolution

    Examples:

    >>> conv_out_size(4, 2, 2)
    2
    >>> conv_out_size([5, 4], 2, 2)
    [2, 2]
    >>> conv_out_size([5, 5], [5, 5, 5])
    Traceback (most recent call last):
    ...
    ValueError: ...
    >>> conv_out_size(4, 4, 2, 1, transposed=True)
    8
    """
    if not isinstance(in_size, collections.abc.Collection):
        single_in = True
    else:
        single_in = False
    in_size, kernel_size, stride, padding, dilation = _validate_conv_args(
        in_size, kernel_size, stride, padding, dilation)
    ndims = len(in_size)
    kernel_size = list(kernel_size)
    for i in range(ndims):
        kernel_size[i] += (kernel_size[i]-1) * (dilation[i]-1)
    out_size = [0] * ndims
    for i in range(ndims):
        if transposed:
            out_size[i] = (in_size[i]-1)*stride[i]+kernel_size[i]-2*padding[i]
        else:
            out_size[i] = (in_size[i]+2*padding[i]-kernel_size[i])//stride[i]+1
    if single_in and ndims == 1:
        assert len(out_size) == 1
        out_size = out_size[0]
    return out_size
    
    
def same_padding(in_size, kernel_size, stride=1, dilation=1):
    """Calculate the padding values for the 'same' padding mode

    Args:

      in_size (int, Seq[int]): Size of inputs to the convolution

      kernel_size (int, Seq[int]): Size of the convolutional kernel

      stride (int, Seq[int]): Stride for the convolution operator

      dilation (int, Seq[int]): Spacing between kernel points

    Returns:

    int, Seq[int]: Amount of implicit padding points for both sides.

    Examples:

    >>> same_padding(8, 4, 2)
    1
    >>> same_padding([8, 16], 4, 2)
    [1, 1]
    >>> same_padding(8, 7, 7)
    3
    >>> same_padding(8, 3, 3)
    Traceback (most recent call last):
    ...
    ValueError: ...
    """
    if not isinstance(in_size, collections.abc.Collection):
        single_in = True
    else:
        single_in = False
    in_size, kernel_size, stride, dilation = _validate_conv_args(
        in_size, kernel_size, stride, dilation)
    ndims = len(in_size)
    kernel_size = list(kernel_size)
    for i in range(ndims):
        kernel_size[i] += (kernel_size[i]-1) * (dilation[i]-1)
    padding = [0] * ndims
    for i in range(ndims):
        if in_size[i] % stride[i] == 0:
            p_tot = max(kernel_size[i]-stride[i], 0)
        else:
            p_tot = max(kernel_size[i]-(in_size[i]%stride[i]), 0)
        if p_tot % 2 == 0:
            padding[i] = p_tot // 2
        else:
            raise ValueError('Cannot equals padding both sides.')
    if single_in and ndims == 1:
        assert len(padding) == 1
        padding = padding[0]
    return padding
