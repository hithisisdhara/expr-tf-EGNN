# -*- coding: utf-8 -*-
"""
Image datasets.
"""
import os
import sys
import pickle
import imageio
import functools
import numpy as np
from pathlib import Path
import arlib
from . import utils as _ds_utils
from . import core as _ds_core
from .. import utils as _utils
from .. import config as _config
from .. import image as _image


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']


class ImageFolder(_ds_core.DataFileFolder): #pragma no cover
    """Image dataset that represented as a folder contains image files

    Args:

      data_path (path-like): Path to the folder.

      transform (callable): A transformation that will be applied to
        the loaded data

      recursive (bool): If True, recursives look for files. Otherwise,
        search files in the top level folder only.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.

    See also:

      :class:`LabeledImageFolder`

    """
    def __init__(self, data_path, transform=None, recursive=False,
                 loader=None,
                 extensions=None):
        if loader is None:
            loader = imageio.imread
        if extensions is None:
            extensions = IMG_EXTENSIONS
        super().__init__(data_path,
                         loader=loader,
                         extensions=extensions,
                         transform=transform,
                         recursive=recursive)


class LabeledImageFolder(_ds_core.LabeledDataFileFolder):
    """Labeled image datasets represented as subfolders of image files.

      data_path (path-like): Path to the folder.

      label_encoder (str, dict, callable): Method to encode raw labels
        into integers. Possible values could be:
        
        * 'none' or None: No encoding will be performed, the raw str
          labels will be returned.

        * 'index': Encode labels as indices to the whole label
          set. Note that the ordering of labels in the label set might
          be implementation dependent.

        * A dictionary D that will map a label L to integer D[L].

        * A callable func that will map a label L to integer func(L).

        * A sklearn.LabelEncoder enc that will map a label L to
          integer enc.transform(L).

      transform (callable): A transformation that will be applied to
        the loaded data

      target_transform (callable): A transformation that will be
        applied to the target values (labels) whose values initially
        are integers.

      skip_containing_folder (bool): Whether skip the containing
        folder, i.e. the only folder without siblings in the
        archive. Default to True.

      category_recursive (bool): Whether recursively search for files
        for each category in subfolders. Default to False.

      category_skip_containing_folder (bool): Whether skip the
        containing folder when searching for data files for each
        category. If None, use the same value as
        skip_containing_folder. Default to None.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.

    """
    def __init__(self, data_path,
                 label_encoder='index',
                 transform=None, target_transform=None,
                 skip_containing_folder=True,
                 category_recursive=False,
                 category_skip_containing_folder=None,
                 loader=None, extensions=None):
        if loader is None:
            loader = imageio.imread
        if extensions is None:
            extensions = IMG_EXTENSIONS
        super().__init__(data_path, loader=loader, extensions=extensions,
                         label_encoder=label_encoder, 
                         transform=transform,
                         target_transform=target_transform,
                         skip_containing_folder=skip_containing_folder,
                         category_recursive=category_recursive,
                         category_skip_containing_folder=category_skip_containing_folder)



class ImageArchive(_ds_core.DataFileArchive):
    """Image datasets stored as an archive file

    Args:

      data_path (file-like): Path to or file object of the archive
        file.

      transform (callable): A transformation that will be applied to
        the loaded data

      recursive (bool): Whether recursively search files in
        folders. Default to False.

      skip_containing_folder (bool): Whether skip the containing
        folder, i.e. the only folder without siblings in the
        archive. Default to True.

      engine (arlib.Archive): The *engine* argument passed to the
        arlib.open function. Default to None.

      cache_archive_obj (bool): Whether open the dataset archive file
        at the construction point and cache the arlib.Archive object
        for later use. Otherwise, the archive file obj will be closed
        after initialization and will be opened each time the
        __getitem__ method is called. Default to True.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.
    """
    def __init__(self, data_path, transform=None, recursive=False,
                 skip_containing_folder=True, engine=None,
                 cache_archive_obj=True,
                 loader=None, extensions=None):
        if loader is None:
            loader = imageio.imread
        if extensions is None:
            extensions = IMG_EXTENSIONS
        super().__init__(data_path, loader=loader, extensions=extensions,
                         transform=transform, recursive=recursive,
                         skip_containing_folder=skip_containing_folder,
                         engine=engine, cache_archive_obj=cache_archive_obj)



class LabeledImageArchive(_ds_core.LabeledDataFileArchive):
    """Labeled image datasets represented as subfolders of image files.

    Args:

      data_path (path-like): Root directory path.

      label_encoder (str, dict, callable): Method to encode raw labels
        into integers. Possible values could be:
        
        * 'none' or None: No encoding will be performed, the raw str
          labels will be returned.

        * 'index': Encode labels as indices to the whole label
          set. Note that the ordering of labels in the label set might
          be implementation dependent.

        * A dictionary D that will map a label L to integer D[L].

        * A callable func that will map a label L to integer func(L).

        * A sklearn.LabelEncoder enc that will map a label L to
          integer enc.transform(L).

      transform (callable): A transformation that will be applied to
        the loaded images. If None, no transform will be
        applied. Default to None.

      target_transform (callable): A transformation that will be
        applied to the target values (encoded labels). If None, no
        transform will be applied. Default to None.

      member_archives_as_categories (bool): Whether use member archive
        (i.e. archive file in the dataset archive file) as
        categories. If False, only member folders will be considered
        as categories. Default to True.

      skip_containing_folder (bool): Whether skip the containing
        folder (which is the only folder without siblings) when
        searching for member folder or member archive as categories,
        . Default to True.

      category_recursive (bool): Whether recursively search for files
        for each category in a member folder or member
        archive. Default to False.

      category_skip_containing_folder (bool): Whether skip the
        containing folder when search data file from member archive or
        member folder. If None, use same value as
        skip_containing_folder. Default to None.

      cache_archive_obj (bool): Whether open the dataset archive file
        at the construction point and cache the arlib.Archive object
        for later use. Otherwise, the archive file obj will be closed
        after initialization and will be opened each time the
        __getitem__ method is called. Default to True.

      loader (callable): A function to load a sample given its path.
        files. If None, use imageio.imread. Default to None.

      extensions (str, Seq[str]): A list (or single) of allowed
        extensions. If None, use IMG_EXTENSIONS. Default to None.

      engine (arlib.Archive): The *engine* argument passed to the
        arlib.open function. Default to None.

    See also:

     :class:`ImageFolder`

    """
    def __init__(self, data_path,
                 label_encoder='index',
                 transform=None, target_transform=None,
                 member_archives_as_categories=True,
                 skip_containing_folder=True,
                 category_recursive=False,
                 category_skip_containing_folder=None,
                 cache_archive_obj=True,
                 loader=None,
                 extensions=None,
                 engine=None):
        if loader is None:
            loader = imageio.imread
        if extensions is None:
            extensions = IMG_EXTENSIONS
        super().__init__(data_path=data_path,
                         loader=loader,
                         extensions=extensions,
                         label_encoder=label_encoder,
                         transform=transform,
                         target_transform=target_transform,
                         member_archives_as_categories=member_archives_as_categories,
                         skip_containing_folder=skip_containing_folder,
                         category_recursive=category_recursive,
                         category_skip_containing_folder=category_skip_containing_folder,
                         cache_archive_obj=cache_archive_obj,
                         engine=engine)
        
@_ds_utils.register_dataset_loader
def load_mnist(data_path=None, data_home=None, subsets=None):
    """Load the MNIST dataset

    Args:

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): Path to a folder where the dataset archive
        will be searched when data_path is None.
    
      subsets (str, list[str]): Subset names, e.g. 'training', 'test',
        or ['training', 'test']. If None, the whole dataset will be
        loaded.

    """
    if data_path is None:
        data_path = _utils.validate_data_home(data_home)
        data_path /= 'mnist.npz'
        url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
        _ds_utils.get_file(data_path, url)
    
    if subsets is None:
        subsets = ['training', 'test']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)
    X, Y = [], []
    with np.load(data_path) as f:
        for subset in subsets:
            if subset == 'training':
                X.append(f['x_train'])
                Y.append(f['y_train'])
            elif subset == 'test':
                X.append(f['x_test'])
                Y.append(f['y_test'])
            else:
                raise ValueError('Subset:', subset, ' not supported.')
    return np.concatenate(X), np.concatenate(Y)


def _load_cifar_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    
    Args:
    
        fpath: path the file to parse.

        label_key: key for label data in the retrieve
            dictionary.

    Returns:

        A tuple `(data, labels)`.
    """
    if isinstance(fpath, (os.PathLike, str, bytes)):
        with open(fpath, 'rb') as f:
            return _load_cifar_batch(f, label_key)

    d = pickle.load(fpath, encoding='bytes')
    # decode utf8
    d_decoded = {}
    for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
    d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32).transpose([0, 2, 3, 1])
    return data, labels    

@_ds_utils.register_dataset_loader
def load_cifar10(data_path=None, data_home=None, subsets=None):
    """Load the CIFAR10 dataset

    Args:

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): Path to a folder where the dataset archive
        will be searched when data_path is None.
    
      subsets (str, list[str]): Subset names, e.g. 'training', 'test',
        or ['training', 'test']. If None, the whole dataset will be
        loaded.

    """
    if data_path is None:
        data_path = _utils.validate_data_home(data_home)
        data_path /= 'cifar-10-python.tar.gz'
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        _ds_utils.get_file(data_path, url)
    
    if subsets is None:
        subsets = ['training', 'test']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)
    X, Y = [], []
    with arlib.open(data_path) as ar:
        for subset in subsets:
            if subset == 'training':
                for i in range(1, 6):
                    mname = [x for x in ar.member_names
                             if x.endswith('data_batch_'+str(i))]
                    assert len(mname) == 1
                    mname = mname[0]
                    tmp = _load_cifar_batch(ar.open_member(mname,'rb'))
                    X.append(tmp[0])
                    Y.append(tmp[1])
            elif subset == 'test':
                mname = [x for x in ar.member_names if x.endswith('test_batch')]
                assert len(mname) == 1
                mname = mname[0]
                tmp = _load_cifar_batch(ar.open_member(mname, 'rb'))
                X.append(tmp[0])
                Y.append(tmp[1])
            else:
                raise ValueError('Subset:', subset, ' not supported.')
    return np.concatenate(X), np.concatenate(Y)

@_ds_utils.register_dataset_loader
def load_cifar100(data_path=None, data_home=None, subsets=None,
                  label_mode='fine'):
    """Load the CIFAR10 dataset

    Args:

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): Path to a folder where the dataset archive
        will be searched when data_path is None.
    
      subsets (str, list[str]): Subset names, e.g. 'training', 'test',
        or ['training', 'test']. If None, the whole dataset will be
        loaded.

      label_mode (str): Either 'fine', or 'coarse', load fine-level
        labels or coarse-level labels.

    """
    if data_path is None:
        data_path = _utils.validate_data_home(data_home)
        data_path /= 'cifar-100-python.tar.gz'
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        _ds_utils.get_file(data_path, url)
    
    if subsets is None:
        subsets = ['training', 'test']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)

    label_mode = _utils.validate_option(label_mode, ['fine', 'coarse'],
                                        name='label_mode')
    
    X, Y = [], []
    with arlib.open(data_path) as ar:
        for subset in subsets:
            if subset == 'training':
                name = [x for x in ar.member_names if x.endswith('train')]
            elif subset == 'test':
                name = [x for x in ar.member_names if x.endswith('test')]
            else:
                raise ValueError('Subset:', subset, ' not supported.')
            assert len(name) == 1
            name = name[0]
            tmp = _load_cifar_batch(ar.open_member(name, 'rb'),
                                    label_key=label_mode + '_labels')
            X.append(tmp[0])
            Y.append(tmp[1])
    return np.concatenate(X), np.concatenate(Y)


@_ds_utils.register_dataset_loader(
    name=['ilsvrc2012', 'ilsvrc-2012', 'ilsvrc', 'imagenet2012',
          'imagenet-2012', 'imagenet'])
def load_ilsvrc2012(data_path=None, data_home=None, subsets=None,
                    label_encoder='index',
                    image_size=None): #pragma no cover
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

    """
    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        allow_names = ['ilsvrc2012', 'ilsvrc-2012', 'ilsvrc',
                       'imagenet2012', 'imagenet-2012', 'imagenet']
        for p in data_home.iterdir():
            if p.name.lower() in allow_names:
                data_path = p
                break
        if data_path is None:
            raise RuntimeError('Cannot find the standard path of the '
                               'ILSVRC2012 dataset.')
    else:
        data_path = Path(data_path)
        if not data_path.is_dir():
            raise ValueError(str(data_path)+' is not an existing dir')

    if subsets is None:
        subsets = ['training', 'validation']
    subsets = _ds_utils.validate_tvt(subsets, return_list=True)
    assert all(x in ['training', 'validation'] for x in subsets)

    train_path, val_path = None, None
    for p in data_path.iterdir():
        if p.is_dir() and p.name.lower() in ['training', 'train']:
            train_path = p
            break
    for p in data_path.iterdir():
        if p.is_dir() and p.name.lower() in ['validation', 'valid', 'val']:
            val_path = p
            break
    if (isinstance(label_encoder, (str, bytes)) and
        label_encoder.lower()=='index'):
        p = train_path if train_path is not None else val_path
        assert p is not None
        labels = [x.name for x in p.iterdir() if x.is_dir()]
        label_encoder = {x: i for i, x in enumerate(labels)}
    
    D = []
    if image_size is None:
        trans = None
    else:
        trans = lambda x: _image.center_crop(_image.resize(x, image_size), image_size)

    img_loader = functools.partial(imageio.imread, pilmode='RGB')
    for subset in subsets:
        if subset.lower() not in ['training', 'validation']:
            raise ValueError('ILSVRC2012 does not have subset:'+subset)
        subset_path = train_path if subset == 'training' else val_path
        if subset_path is None:
            raise ValueError('Cannot find the folder of subset: '+subset)
        D.append(LabeledImageFolder(subset_path,label_encoder=label_encoder,
                                    transform=trans, loader=img_loader))
    assert len(D) > 0
    if len(D) == 1:
        D = D[0]
    else:
        D = _ds_core.ConcatDataset(D)
    return D
