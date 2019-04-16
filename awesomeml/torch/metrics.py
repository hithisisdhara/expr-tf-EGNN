# -*- coding: utf-8 -*-
"""
Metrics for models implemented with PyTorch.
"""
import collections
import functools
import PIL
import numpy as np
import skimage
from scipy.stats import entropy
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose,ToTensor,Resize,Lambda,Normalize
from torchvision.transforms import ToPILImage, CenterCrop
from . import datasets as _torch_ds
from . import utils as _torch_utils
from .. import utils as _utils
from .. import stats as _stats
from .. import datasets as _ds
from ..datasets import utils as _ds_utils


def _validate_images(data, use_std_data, std_data_path, std_data_home,
                     std_data_subsets, dtype=None): #pragma no cover
    """
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    assert dtype.is_floating_point

    if use_std_data:
        extra_kwargs = dict()
        if 'imagenet' not in data.lower() and 'ilsvrc' not in data.lower():
            extra_kwargs['download'] = True
        data = _torch_ds.load_dataset(
            data, data_path=std_data_path, data_home=std_data_home,
            subsets=std_data_subsets, **extra_kwargs)
    elif isinstance(data, np.ndarray) or torch.is_tensor(data):
        if isinstance(data, np.ndarray):
            if data.ndim == 4:
                data = np.transpose(data, [0, 2, 3, 1])
            data = torch.from_numpy(data)
        if data.dim() == 3:
            data.unsqueeze_(1)
        assert data.dim() == 4
        if data.size(1) == 1:
            data = data.repeat([1, 3, 1, 1])
        assert data.size(1) == 3
        data = torch.utils.data.TensorDataset(data)
    elif not isinstance(data, torch.utils.data.Dataset):
        data = _torch_ds.DatasetAdapter(_ds.image.ImageFolder(data))
    assert isinstance(data, torch.utils.data.Dataset)
    if not isinstance(data[0], collections.abc.Sequence):
        data = _torch_ds.TransformDataset(data, lambda x: (x,))

    def trans_func(x):
        if torch.is_tensor(x):
            x = _torch_utils.img_as(x, dtype=dtype)
            if x.dim() == 2:
                x = x.unsqueeze(0)
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = np.expand_dims(x, -1)
            # use copy() to avoid issue: ValueError: some of the
            # strides of a given numpy array are negative. This is
            # currently not supported
            x = ToTensor()(x.copy())
        else:
            x = ToTensor()(x)
        return x

    trans = Compose([
        Lambda(trans_func),
        ToPILImage(), Resize(299), CenterCrop(299), ToTensor(),
        Lambda(lambda x: x.repeat([3,1,1]) if x.dim()==3 and x.size(0) == 1
               else x),
        Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])
    data = _torch_ds.TransformLabeledDataset(data, trans)
    return data


def inception_score(data, n_splits=10, batch_size=128, gpu_ids=None,
                    use_std_data=False,
                    std_data_path=None,
                    std_data_home=None,
                    std_data_subsets=None,
                    dtype=None): #pragma no cover
    """Calculate Inception score from a set of images

    Args:

      data (Dataset, path-like, str): Input images, will be
        interpreted differently depending on its type and the value of
        use_std_data. If use_std_data is True, *data* will be
        interpreted as the name of the standard dataset, otherwise, it
        will be interpreted differently depending on its type:

        * ndarray: Image content represented as a shape [N,H,W,C]
          array.

        * Tensor: Image content represented as a shape [N,C,H,W]
          tensor.

        * Dataset: Content of images as a Dataset object similar as
          torchvision.datasets. The pixel values should be already
          normalized as described in:
          https://pytorch.org/docs/stable/torchvision/models.html

      n_split (int): Number of splits to apply to the dataset

      batch_size (int): Batch size to evaluate the inception model

      gpu_ids (Collection[int], NoneType): Specify ids of GPUs to
        use. If is None or empty, use CPU.

      use_std_data (bool): Whether use standard dataset.

      std_data_path (path-like): Path to the standard dataset. Will be
        passed to dataset loader API as argument data_path.

      std_data_home (path-like): Path to the folder where the data
        will be searched and downloaded to. Will be passed to dataset
        loader API as argument data_home.

      std_data_subsets (Seq[str], str): Subsets of standard dataset to
        use, will be passed to dataset loader API as argument subsets.

    """
    if dtype is None:
        dtype = torch.get_default_dtype()
        
    if gpu_ids:
        device = torch.device('cuda:'+str(gpu_ids[0]))
    else:
        device = 'cpu'

    data = _validate_images(data, use_std_data, std_data_path,
                            std_data_home, std_data_subsets, dtype=dtype)
    
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)

    # Load inception model
    model = torchvision.models.inception_v3(pretrained=True,
                                            transform_input=False)
    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, gpu_ids)
    model.to(device, dtype=dtype)
    model.eval()

    Y = []
    for X in dataloader:
        X = X[0]
        X = X.to(device, dtype=dtype)
        #X = F.interpolate(X, 299, mode='bilinear', align_corners=False)
        with torch.no_grad():
            Y.append(F.softmax(model(X), dim=-1).cpu().numpy())

    Y = np.array_split(np.concatenate(Y, 0), n_splits)
    scores = [np.exp(np.mean(entropy(y.T, np.mean(y, 0, keepdims=True).T)))
              for y in Y]
    
    return np.mean(scores), np.std(scores)



def inception_stats(data,
                    batch_size=128,
                    use_stats_file=False,
                    use_std_data=False,
                    std_data_path=None,
                    std_data_home=None,
                    std_data_subsets=None,
                    std_stats_home=None,
                    gpu_ids=None,
                    dtype=None): #pragma no cover
    """Calculate or load Inception statistics

    Args:

      data: Images, path to an image folder, statistics, or path to a
        precalculated statistics file. The value of data1 will be
        treated differently depends on different values of
        use_stats_file and use_std_data.

        * False, False: data are images represented as torchvision
          dataset, ndarray, Tensor or path to an image folder.

        * False, True: data is name of a standard dataset, the
          dataset will be loaded to memory as ndarray for statistics
          calculation.

        * True, False: data is path a file contains the statistics
          which will be loaded from the file.

        * True, True: data is name of a standard dataset and
          precalculated statistics of this dataset will be loaded from
          a file. The file will be searched at stats_path, and will be
          donwloaded if not exist.

      batch_size (int): Size of minibatch to calculate Inception outputs.

      use_stats_file (bool): Whether load statistics from a file.

      use_std_data (bool): Whether data is name of a standard dataset.

      std_data_path (path-like): Path to the standard dataset
        archive. If None, the archive will be searched at
        *std_data_home*. If the archive does not exist, it will be
        downloaded.

      std_data_home (path-like): Path to the folder where standard
        dataset archive will be searched or downloaded to. If None,
        config.DATA_HOME will be used. Defaults to None.

      std_data_subsets (Seq[str], str, NoneType): Subsets of standard
        dataset to use, e.g., 'train', 'test' or ['train', 'test']. If
        None, the whole dataset will be used.

      std_stats_home (path-like): Path to the directory where the
        statistics files of standard datasets reside. If None, use
        std_data_home/'inception_stats'.

      gpu_ids (Collection[int], NoneType): Specify ids of GPUs to
        use. If is None or empty, use CPU.

    Returns:

    tuple: (mu, sigma) as mean vectors and covariance matrix.

    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if gpu_ids:
        device = torch.device('cuda:'+str(gpu_ids[0]))
    else:
        device = 'cpu'

    if use_std_data:
        data = _utils.validate_str(data, 'data')
        #std_data_home = _torch_utils.validate_data_home(std_data_home)
        std_data_subsets = _ds_utils.validate_tvt(std_data_subsets,
                                                  return_list=True)
        
    if use_stats_file:
        if use_std_data:
            std_stats_home = _utils.validate_dir(
                std_stats_home,
                _torch_utils.validate_data_home(std_data_home)/'inception_stats')
            # get path to std stats file
            name = data.lower()
            if 'imagenet' in name or 'ilsvrc' in name:
                name = 'ilsvrc2012'
            if std_data_subsets is not None:
                name = '-'.join([name] + std_data_subsets)
            fname = name + '.npz'
            std_stats_path = std_stats_home/fname
            if not std_stats_path.is_file():
                url = ('https://sourceforge.net/projects/std-inception'
                       '-stats/files/pytorch/'+fname)
                _utils.download(url, std_stats_path)
            data = std_stats_path
        with np.load(std_stats_path) as f:
            return f['mu'], f['sigma']

    data = _validate_images(data, use_std_data, std_data_path,
                            std_data_home, std_data_subsets, dtype)


    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    
    # load the pretrained Inception_v3 model
    incep_v3 = torchvision.models.inception_v3(pretrained=True,
                                               transform_input=False)

    # construct model to calculate statistics
    model = torch.nn.Sequential(
        incep_v3.Conv2d_1a_3x3,
        incep_v3.Conv2d_2a_3x3,
        incep_v3.Conv2d_2b_3x3,
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        incep_v3.Conv2d_3b_1x1,
        incep_v3.Conv2d_4a_3x3,
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        incep_v3.Mixed_5b,
        incep_v3.Mixed_5c,
        incep_v3.Mixed_5d,
        incep_v3.Mixed_6a,
        incep_v3.Mixed_6b,
        incep_v3.Mixed_6c,
        incep_v3.Mixed_6d,
        incep_v3.Mixed_6e,
        incep_v3.Mixed_7a,
        incep_v3.Mixed_7b,
        incep_v3.Mixed_7c,
        torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    for param in model.parameters():
        param.requires_grad = False

    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, gpu_ids)
    model.to(device, dtype=dtype)
    model.eval()

    Y = []
    for X in dataloader:
        X = X[0]
        X = X.to(device, dtype=dtype)
        #X = F.interpolate(X, 299, mode='bilinear', align_corners=True)
        with torch.no_grad():
            Y.append(model(X).squeeze().cpu().numpy())
    Y = np.concatenate(Y)
    mu = np.mean(Y, axis=0)
    sigma = np.cov(Y, rowvar=False)
    return mu, sigma
    

def inception_distance(data1, data2, metric='frechet',
                       batch_size=32,
                       data1_use_stats_file=False,
                       data1_use_std=False,
                       data1_std_path=None,
                       data1_std_subsets=None,
                       data2_use_stats_file=False,
                       data2_use_std=False,
                       data2_std_path=None,
                       data2_std_subsets=None,
                       std_stats_home=None,
                       std_data_home=None,
                       gpu_ids=None,
                       dtype=None): #pragma no cover
    """Frechet Inception distance between generated and real images

    Args:

      data1: Images, path to an image folder, statistics, or path to a
        precalculated statistics file. The value of data1 will be
        treated differently depends on different values of
        data1_use_stats_file and data1_use_std:
    
        * False, False: data1 are images represented as a torchvision
          dataset, or statistics represented as tuple or list of
          ndarray: (mu, sigma).

        * False, True: data1 is name of a standard dataset, the
          dataset will be loaded to memory as ndarray for statistics
          calculation.

        * True, False: data1 is path a file contains the statistics
          which will be loaded from the file.

        * True, True: data1 is name of a standard dataset and
          precalculated statistics of this dataset will be loaded from
          a file. The file will be searched at stats_path, and will be
          donwloaded if not exist.

      data2 : Same as data1, but for another data set.

      metric (str): Distance metric to use. Currently, only frechet
        distance is supported.

      batch_size (int): Size of minibatch to calculate Inception outputs.

      data1_use_stats_file (bool): Whether load statistics from a
        file.

      data1_use_std (bool): Whether data1 is name of a standard dataset.

      data1_std_path (path-like): Path to the standard dataset for
        data1. Will be passed to dataset loader API as argument
        *data_path*.

      data1_std_subsets (Seq[str], str, NoneType): Subsets of standard
        dataset to use, e.g., 'train', 'test' or ['train', 'test']. If
        None, the whole dataset will be used.

      data2_use_stats_file (bool): Same as data1_use_stats_file, but
        for the second dataset.

      data2_use_std (bool): Same as data1_use_std, but for the second
        dataset.

      data2_std_path (path-like): Same as data1_std_path, but for the
        second dataset.

      data2_std_subsets (Seq[str], str): Same as data1_std_subsets, but
        for the second dataset.

      std_stats_home (path-like): Path to the directory where the
        statistics files of standard datasets reside. If None, use
        std_data_home/'inception_stats'.

      std_data_home (path-like): Path to the folder where the data
        will be searched and downloaded to. Will be passed to dataset
        loader API as argument data_home.

      gpu_ids (Collection[int], NoneType): Specify ids of GPUs to
        use. If is None or empty, use CPU.

    """
    mu1, sigma1 = inception_stats(data1,
                                  batch_size=batch_size,
                                  use_stats_file=data1_use_stats_file,
                                  use_std_data=data1_use_std,
                                  std_data_path=data1_std_path,
                                  std_data_home=std_data_home,
                                  std_data_subsets=data1_std_subsets,
                                  std_stats_home=std_stats_home,
                                  gpu_ids=gpu_ids,
                                  dtype=dtype)

    mu2, sigma2 = inception_stats(data2,
                                  batch_size=batch_size,
                                  use_stats_file=data2_use_stats_file,
                                  use_std_data=data2_use_std,
                                  std_data_path=data2_std_path,
                                  std_data_home=std_data_home,
                                  std_data_subsets=data2_std_subsets,
                                  std_stats_home=std_stats_home,
                                  gpu_ids=gpu_ids,
                                  dtype=dtype)
    
    return _stats.gaussian_frechet_distance(mu1, sigma1, mu2, sigma2)    


if __name__ == '__main__': # pragma no cover
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands')

    # inception score
    def cmd_inception_score(**kwargs):
        output_decimal_length = kwargs.pop('output_decimal_length')
        score = inception_score(**kwargs)
        fmt = ''.join(('%.', str(output_decimal_length), 'f'))
        print(fmt % score)
        
    IS = subparsers.add_parser('inception-score', help='Inception score.')
    IS.add_argument('data', type=str)
    IS.add_argument('--n-splits', type=int, default=10)
    IS.add_argument('--batch-size', type=int, default=32)
    IS.add_argument('--use-std-data', action='store_true')
    IS.add_argument('--std-data-path', type=Path, default=None)
    IS.add_argument('--std-data-home', type=Path, default=None)
    IS.add_argument('--std-data-subsets', type=str, default=None)
    IS.add_argument('--output-decimal-length', type=str, default=6,
                    help='Number of digits of the decimal.')
    IS.set_defaults(cmd=cmd_inception_score)

    # inception statistics
    def cmd_inception_stats(**kwargs):
        output_file = kwargs.pop('output-file')
        output_name_mean = kwargs.pop('output_name_mean')
        output_name_covariance = kwargs.pop('output_name_covariance')
        mu, sigma = inception_stats(**kwargs)
        kwargs2 = dict(zip([output_name_mean, output_name_covariance],
                           [mu, sigma]))
        np.savez(output_file, **kwargs2)
        
    STAT = subparsers.add_parser('inception-statistics',
                                 aliases=['inception-stats'],
                                 help='Inception statistics.')
    STAT.add_argument('data', type=str)
    STAT.add_argument('output-file', type=Path)
    STAT.add_argument('--batch-size', type=int, default=32)

    STAT.add_argument('--use-std-data', action='store_true')
    STAT.add_argument('--std-data-path', type=Path, default=None)
    STAT.add_argument('--std-data-home', type=Path, default=None)
    STAT.add_argument('--std-data-subsets', type=str, default=None)
    STAT.add_argument('--output-name-mean', type=str, default='mu',
                      help='Name of mean vector in saved .npz file.')
    STAT.add_argument('--output-name-covariance', type=str, default='sigma',
                      help='Name of covariance matrix in saved .npz file.')
    STAT.set_defaults(cmd=cmd_inception_stats)

    # inception distance
    def cmd_inception_distance(**kwargs):
        output_decimal_length = kwargs.pop('output_decimal_length')
        d = inception_distance(**kwargs)
        fmt = ''.join(('%.', str(output_decimal_length), 'f'))
        print(fmt % d)

    DIST = subparsers.add_parser('inception-distance', help='Inception distance')
    DIST.add_argument('data1', type=str)
    DIST.add_argument('data2', type=str)
    DIST.add_argument('--metric', type=str, default='frechet')
    DIST.add_argument('--batch-size', type=int, default=32)
    DIST.add_argument('--data1-use-stats-file', action='store_true')
    DIST.add_argument('--data1-use-std', action='store_true')
    DIST.add_argument('--data1-std-path', type=Path, default=None)
    DIST.add_argument('--data1-std-subsets', type=str, default=None)
    DIST.add_argument('--data2-use-stats-file', action='store_true')
    DIST.add_argument('--data2-use-std', action='store_true')
    DIST.add_argument('--data2-std-path', type=Path, default=None)
    DIST.add_argument('--data2-std-subsets', type=str, default=None)
    DIST.add_argument('--output-decimal-length', type=str, default=6,
                    help='Number of digits of the decimal.')
    DIST.set_defaults(cmd=cmd_inception_distance)
    
    kwargs = vars(parser.parse_args())
    cmd = kwargs.pop('cmd')
    cmd(**kwargs)
