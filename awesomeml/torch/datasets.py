# -*- coding: utf-8 -*-
"""PyTorch datasets.

The common interface is load_dataset, which will return a
torch.data.Dataset object.

"""
import functools
from pathlib import Path
import torch
import torch.utils.data
from PIL import Image
import decoutils
from .. import datasets as _ds
from ..datasets import utils as _ds_utils
from .. import utils as _utils
from .. import config as _config
from . import utils as _torch_utils

# dataset loader registration functionality
_dataset_loaders = _utils.SafeDict(
    key_normalizer=_ds_utils.normalize_dataset_name,
    value_checker=callable)


for name in ['register_dataset_loader', 'unregister_dataset_loader',
             'dataset_has_loader', 'assert_dataset_has_loader',
             'get_dataset_loader']:
    globals()[name] = functools.partial(
        _ds_utils.__dict__[name], storage=_dataset_loaders)


class TransformDataset(torch.utils.data.Dataset):
    """Generate new dataset by adding a transform to an existing dataset

    Args:

      dataset (Dataset): The input dataset.

      transform (callable): The transform that will be applied to
        each example of the input dataset.

      target_transform (callable): The transform that will be applied
        to the target of each example. Default to None.

    Returns:

    Dataset: A new dataset that each example will be transformed.

    Examples:

    >>> ds = torch.utils.data.TensorDataset(torch.ones(3,4))
    >>> trans = lambda x: (x[0] * 2,)
    >>> ds2 = TransformDataset(ds, trans)
    >>> len(ds2)
    3
    >>> (ds2[0][0] == 2).all().item()
    1

    """
    def __init__(self, dataset, transform):
        self.original_dataset = dataset
        self.transform = transform


    def __getitem__(self, index):
        return self.transform(self.original_dataset[index])

    def __len__(self):
        return len(self.original_dataset)


class TransformLabeledDataset(torch.utils.data.Dataset):
    """Generate new labeled dataset by adding a transform to an existing
labeled dataset

    Args:

      dataset (Dataset): The input dataset.

      transform (callable): The transform that will be applied to
        each example of the input dataset.

      target_transform (callable): The transform that will be applied
        to the target of each example. Default to None.

    Returns:

    Dataset: A new dataset that each example will be transformed.

    Examples:

    >>> ds = torch.utils.data.TensorDataset(torch.ones(3,4), torch.ones(3))
    >>> trans = lambda x: x * 2
    >>> trans2 = lambda x: x * 3
    >>> ds2 = TransformLabeledDataset(ds, trans, trans2)
    >>> len(ds2)
    3
    >>> (ds2[0][0] == 2).all().item()
    1
    >>> ds2[0][1].item()
    3.0

    """
    def __init__(self, dataset, transform, target_transform=None):
        self.original_dataset = dataset
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        example = list(self.original_dataset[index])
        example[0] = self.transform(example[0])
        if self.target_transform is not None:
            example[1] = self.target_transform(example[1])
        return tuple(example)

    def __len__(self):
        return len(self.original_dataset)


class DatasetAdapter(torch.utils.data.Dataset): #pragma no cover
    """Adapter that convert a numpy dataset to a torch dataset

    Args:

      dataset: Input numpy dataset
    """
    def __init__(self, dataset):
        self.numpy_dataset = dataset
        

    def __getitem__(self, index):
        return self.numpy_dataset[index]


    def __len__(self):
        return len(self.numpy_dataset)


def load_dataset(name, *args, **kwargs): #pragma no cover
    """Load a dataset specified by name

    Args:

      name (str): Dataset name.

      args, kwargs: Arguments and keyword arguments passed to the
        corresponding loader loader.

    Returns:

      Dataset: Loaded dataset represented as torch.utils.data.Dataset.

    """
    if dataset_has_loader(name):
        D = get_dataset_loader(name)(*args, **kwargs)
    elif _ds_utils.dataset_has_loader(name):
        D = _ds_utils.get_dataset_loader(name)(*args, **kwargs)
        trans = kwargs.pop('transform', None)
        trans2 = kwargs.pop('target_transform', None)
        if trans is not None or trans2 is not None:
            D = TransformLabeledDataset(D, trans, trans2)
    else:
        raise RuntimeError('Cannot find either pytorch or numpy dataset '
                           'loader.')
    return D

    
@register_dataset_loader
def load_mnist(data_path=None, data_home=None, subsets=None,
               *args, **kwargs): #pragma no cover
    """Load the MNIST dataset as a torchvision dataset

    Args:

      data_path (path_like): Path to the dataset, should be a valid
        directory, zip, or tar file.

      data_home (path_like): Path to the root where the package looks
        for dataset

      subsets (str, Seq[str]): Subsets (e.g. 'training', ['training',
        'test']) of the data to load. If None, the whole dataset will
        be loaded.

      args, kwargs: Arguments parsed to the torchvision.datasets.MNIST
        constructor.

    """
    _config.assert_has_package('torchvision')
    from torchvision.datasets import MNIST
    if data_path is None:
        data_home = _torch_utils.validate_data_home(data_home)
        data_path = data_home/'mnist'
    if subsets is None:
        subsets = ['training', 'test']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)
    ds = []
    for subset in subsets:
        if subset == 'training':
            ds.append(MNIST(data_path, True, *args, **kwargs))
        elif subset == 'test':
            ds.append(MNIST(data_path, False, *args, **kwargs))
        else:
            assert False
    if len(ds) == 1:
        ds = ds[0]
    else:
        assert len(ds) > 1
        ds = torch.utils.data.ConcatDataset(ds)
    return ds


@register_dataset_loader
def load_cifar10(data_path=None, data_home=None, subsets=None,
                 *args, **kwargs): #pragma no cover
    """Load the CIFAR10 dataset as a torchvision dataset

    Args:

      data_path (path_like): Path to the dataset, should be a valid
        directory, zip, or tar file.

      data_home (path_like): Path to the root where the package looks
        for dataset

      subsets (str, Seq[str]): Subsets (e.g. 'training', ['training',
        'test']) of the data to load. If None, the whole dataset will
        be loaded.

      args, kwargs: Arguments parsed to the torchvision.datasets.MNIST
        constructor.

    """
    _config.assert_has_package('torchvision')
    from torchvision.datasets import CIFAR10
    if data_path is None:
        data_home = _torch_utils.validate_data_home(data_home)
        data_path = data_home/'cifar10'
    if subsets is None:
        subsets = ['training', 'test']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)
    ds = []
    for subset in subsets:
        if subset == 'training':
            ds.append(CIFAR10(data_path, True, *args, **kwargs))
        elif subset == 'test':
            ds.append(CIFAR10(data_path, False, *args, **kwargs))
        else:
            assert False
    if len(ds) == 1:
        ds = ds[0]
    else:
        assert len(ds) > 1
        ds = torch.utils.data.ConcatDataset(ds)
    return ds


@register_dataset_loader
def load_cifar100(data_path=None, data_home=None, subsets=None,
                  *args, **kwargs): #pragma no cover
    """Load the CIFAR100 dataset as a torchvision dataset

    Args:

      data_path (path_like): Path to the dataset, should be a valid
        directory, zip, or tar file.

      data_home (path_like): Path to the root where the package looks
        for dataset

      subsets (str, Seq[str]): Subsets (e.g. 'training', ['training',
        'test']) of the data to load. If None, the whole dataset will
        be loaded.

      args, kwargs: Arguments parsed to the torchvision.datasets.MNIST
        constructor.

    """
    _config.assert_has_package('torchvision')
    from torchvision.datasets import CIFAR100
    if data_path is None:
        data_home = _torch_utils.validate_data_home(data_home)
        data_path = data_home/'cifar100'
    if subsets is None:
        subsets = ['training', 'test']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)
    ds = []
    for subset in subsets:
        if subset == 'training':
            ds.append(CIFAR100(data_path, True, *args, **kwargs))
        elif subset == 'test':
            ds.append(CIFAR100(data_path, False, *args, **kwargs))
        else:
            assert False
    if len(ds) == 1:
        ds = ds[0]
    else:
        assert len(ds) > 1
        ds = torch.utils.data.ConcatDataset(ds)
    return ds


@register_dataset_loader(
    name=['ilsvrc2012', 'ilsvrc-2012', 'ilsvrc', 'imagenet2012',
          'imagenet-2012', 'imagenet'])
def load_ilsvrc2012(data_path=None, data_home=None, subsets=None,
                    label_encoder='index',
                    image_size=None,
                    transform=None,
                    target_transform=None): #pragma no cover
    """Load the Imagenet2012 (aka. ILSVRC2012) dataset

    Args:

      data_path (path-like): Path to the dataset folder. If None, the
        following folders will be searched under the path of
        data_home: 'ILSVRC2012', 'ILSVRC-2012', 'ILSVRC',
        'imagenet2012', 'imagenet-2012', 'imagenet', where folder
        names are case-insensitive.

      data_home (path-like): Path to a folder where the dataset will
        be searched. If None, the package default will be used.

      subsets (str, Seq[str]): Subset names, e.g. 'training', 'test',
        or ['training', 'test']. If None, the whole dataset will be
        loaded.

      label_encoder (str, dict, callable): Method to encode raw labels
        (i.e. subfolder names) into integers. Possible values could
        be:
        
        * 'none' or None: No encoding will be performed, the raw str
          labels will be returned.

        * 'index': Encode labels as indices to the whole label
          set. Note that the ordering of labels in the label set might
          be implementation dependent.

        * A dictionary D that will map a label L to integer D[L].

        * A callable func that will map a label L to integer func(L).

        * A sklearn.LabelEncoder enc that will map a label L to
          integer enc.transform(L).

      image_size (int, Seq[int]): Desired output image size. If size
        is a sequence like (h, w), output size will be matched to
        this. If size is an int, smaller edge of the image will be
        matched to this number. i.e, if height > width, then image
        will be rescaled to (size * height / width, size). If None,
        the image will not be resized and cropped. Default to None.

      transform (callable): The transform that will be applied to
        each example of the input dataset.

      target_transform (callable): The transform that will be applied
        to the target of each example. Default to None.
    """
    D = _ds.image.load_ilsvrc2012(data_path=data_path, data_home=data_home,
                                  subsets=subsets,
                                  label_encoder=label_encoder,
                                  image_size=image_size)
    D = DatasetAdapter(D)
    if transform is not None or target_transform is not None:
        D = TransformLabeledDataset(D, transform, target_transform)
    return D
