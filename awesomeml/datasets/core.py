# -*- coding: utf-8 -*-
"""
Core functionalities for dataset manipulation: loading, registration, etc.
"""
import os
import bisect
import io
import sys
import pickle
import itertools
import collections
import random
import arlib
import compfile
import scipy.sparse

import numpy as np
import networkx as nx

from pathlib import Path

from . import utils as _ds_utils
from .. import utils as _utils

def load_dataset(name, *args,
                 cache=False,
                 data_cache_home=None,
                 cache_name = None,
                 **kwargs): # pragma: no cover
    """Load a dataset which is specified by `name`

    This is the unique interface for loading datasets by name. It will dispatch the procedure to different loader 
    
    Args:

        name (str): dataset name

        args: positional arguments that will be passed to the loader

        kwargs: keyword arguments that will be passed to the loader

        cache (bool): whether cache the dataset or not

        data_cache_home (path_like, NoneType): root directory for
          cache files. If it is None, use the default cache directory.

        cache_name (str): name of the cache file. If it is None, use
          dataset name as cache file name.
    
    Returns:

        Results returned by the loader

    Raises:
      KeyError: if no loader was registered for the dataset

    """
    if cache:
        data_cache_home = _utils.validate_dir(
            data_cache_home, default_path=_config.DATA_HOME/'cache')
        if cache_name is None:
            cache_name = name
        fpath = data_cache_home/name
        if fpath.exists() and fpath.is_file():
            # load from cache
            with open(fpath, 'rb') as f:
                return pickle.load(f)
    loader = _ds_utils.get_dataset_loader(name)
    data = loader(*args, **kwargs)
    if cache:
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)
    return data


def load_dataset_tvt(name, splitting,
                     *args, **kwargs): # pragma: no cover
    """Load a dataset training-validation-testing splitting
    
    This is the unique interface for loading dataset TVT splitting by
    name. It will dispatch the procedure to different loader
    
    Args:

      name (str): Dataset name

      splitting (str): Training-validation-testing splitting name

      args: Positional arguments that will be passed to the loader

      kwargs: Keyword arguments that will be passed to the loader

    Returns:

      Results returned by tvt loader

    """
    if _ds_utils.dataset_tvt_has_loader(name, splitting):
        loader = _ds_utils.get_dataset_tvt_loader(name, splitting)
        data = loader(*args, **kwargs)
    else:
        data = load_dataset_tvt_default(name, splitting)
    return data


def load_dataset_tvt_txt(fpath): # pragma: no cover
    """Load a train-validation-test splitting file
    
    Args:

      fpath (path_like, stream): the path or stream of the tvt file
    
    Returns:
    
      tuple[list]: A tuple containing id_training, id_validation and id_test.

    Raises:
        
      ValueError: if input `fpath` is not a valid path or stream
    """
    if isinstance(fpath, (str, bytes, os.PathLike)):
        with compfile.open(fpath, 'rt') as f:
            return load_dataset_tvt_txt(f)
    
    if isinstance(fpath, io.IOBase):
        if isinstance(fpath, io.TextIOBase):
            id_train, id_val, id_test = [], [], []
            for line in fpath:
                node_id, tvt_label = line.split()
                if tvt_label.lower() in ['training', 'train', 'tr']:
                    id_train.append(node_id)
                elif tvt_label.lower() in ['validation', 'val']:
                    id_val.append(node_id)
                elif tvt_label.lower() in ['test', 'testing', 'te']:
                    id_test.append(node_id)
                else:
                    raise ValueError('Unexpected label:'+tvt_label)
            return id_train, id_val, id_test
        else:
            return load_dataset_tvt_txt(io.TextIOWrapper(fpath))
    
    raise ValueError('Input fpath must be a file fpath or opened file '
                     ' stream')


def load_dataset_tvt_pickle(fpath): # pragma: no cover
    """Load dataset tvt splitting from pickled file

    Args:

      fpath (path_like, stream): the path or stream of the tvt file
    
    Returns:
    
      tuple[list]: A tuple containing id_training, id_validation and id_test.
    """
    if isinstance(fpath, (str, bytes, os.PathLike)):
        if compfile.is_compressed_file(fpath):
            with compfile.open(fpath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(fpath, 'rb') as f:
                return pickle.load(f)

    if isinstance(fpath, io.RawIOBase):
        return pickle.load(f)

    raise ValueError('fpath must be a path-like object or a binary readable'
                     ' file object.')


def load_dataset_tvt_default(name, splitting, data_home=None,
                             download=True): # pragma: no cover
    """Load dataset tvt splitting from the default file location
    
    Args:

      name (str): dataset name

      splitting (str): splitting name

      root (path_like): root directory of the splitting file
    
    Return:

      Results returned by load_dataset_tvt_txt or load_dataset_tvt_pickle.

    See also

      :func:`load_dataset_tvt_txt`, :func:`load_dataset_tvt_pickle`.

    """
    data_home = _utils.validate_data_home(data_home)
    fpath = str(data_home/(name+'-tvt-'+splitting))
    exts1 = ['', '.bz2', '.gz', '.lzma', '.xz']
    exts2 = exts1 + ['.pickle'+x for x in exts1] 
    for ext in exts2:
        fpath2 = Path(fpath+ext)
        if fpath2.is_file():
            return load_dataset_tvt_pickle(fpath2)

    exts2 = ['.txt'+x for x in exts1]
    for ext in exts2:
        fpath2 = Path(fpath+ext)
        if fpath2.is_file():
            return load_dataset_tvt_txt(fpath2)

    raise RuntimeError('Failed to find default tvt file for dataset: '+
                       name+' and splitting: '+splitting+'.')


def split(Y, k, stratified=None, return_residual_set=False):
    """Split a dataset into subsets
    
    Args:

      Y (int, Seq): An integer specify the total number of examples,
        or a sequence specify the labels of each example. Labels will
        be used for stratified splitting.

      k (int, float, Seq[int,float]): Specify the numbers or
        proportions of examples in each subset.
    
      stratified (bool, Seq[bool], None): specify whether stratified
        sampling should be used to sample each subset. Possible values:
    
        * None: Automatically determined depending on Y. If Y is a
          sequence, all subsets are stratified random samples, else
          all subsets are simple random samples.
    
        * bool: If True, all subsets are stratified random samples,
          else all subsets are simple random samples.
    
        * Seq[bool]: Specify stratified or not for each subset separately.
    
      return_residual_set (bool): Whether return indices of the
        residual set or not

    Return: 
    
      tuple[Seq[int]]: tuple of sequences of int represent indices of
        the subsets

    Examples:

    >>> I, = split(10, 2)
    >>> len(I) == 2
    True
    >>> I1, I2 = split(10, [0.2, 3])
    >>> len(I1) == 2
    True
    >>> len(I2) == 3
    True
    >>> Y = [1]*10 + [2]*20 + [3]*30
    >>> I1, = split(Y, 2, True)
    >>> sorted(Y[i] for i in I1) == [1]*2 + [2]*2 + [3]*2
    True
    >>> I1, = split(Y, 0.2, True)
    >>> sorted(Y[i] for i in I1) == [1]*2 + [2]*4 + [3]*6
    True
    >>> I1, I2 = split(Y, [2, 0.3])
    >>> sorted(Y[i] for i in I1) == [1]*2 + [2]*2 + [3]*2
    True
    >>> sorted(Y[i] for i in I2) == [1]*3 + [2]*6 + [3]*9
    True
    >>> len(split(10, 2, return_residual_set=False))
    1
    >>> len(split(10, 2, return_residual_set=True))
    2
    >>> len(split(10, [2,0.2], return_residual_set=False))
    2
    >>> len(split(10, [2,0.2], return_residual_set=True))
    3
    >>> I1, I2, I3 = split(Y, [2, 0.3], return_residual_set=True)
    >>> sorted(I1+I2+I3) == list(range(60))
    True
    >>> split(None, 10)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> split(10, 2, True)
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    if (isinstance(Y, collections.Sequence) and
        not isinstance(Y, (str, bytes, bytearray))):
        N = len(Y)
    elif isinstance(Y, int):
        N, Y = Y, None
    else:
        raise ValueError('The first argument should be an int specifying '
                         'the total number of examples, or a sequence '
                         'specifying the labels of each example.')

    # normalize k so that proportions are converted to integers
    k = _utils.validate_seq(k, None, 'k', element_types=(int, float))
    nsets = len(k)
    
    if stratified is None:
        stratified = False if Y is None else True

    stratified = _utils.validate_seq(
        stratified, len(k), 'stratified', element_types=bool)

    if Y is None and any(stratified):
        raise ValueError('Labels are required for stratified splitting.')

    I = list(range(N))
    # np.nan and None are treated as unlabled elements, we will not
    # sample from these elements
    if Y is not None:
        I = [i for i in I if not (Y[i] is None or isinstance(Y[i], float)
                                  and np.isnan(Y[i]))]
        Ys = list(set(Y[i] for i in I))
        Ns = collections.Counter(Y[i] for i in I)
        nC = len(Ys)
        N = len(I)

    res_set = set(I)
    ret = [] # subsets to return
    for m in range(nsets):
        ret_m = [] # the m^th subset
        km =  k[m]
        if stratified[m]:
            indices = []
            for n in range(nC):
                res_set_n = [i for i in res_set if Y[i] == Ys[n]]
                km_n = km if isinstance(km, int) else round(km * Ns[Ys[n]])
                indices += random.sample(res_set_n, km_n)
        else:
            # simple random sampling
            if isinstance(km, float):
                km = round(km * N)
            indices = random.sample(res_set, km)
                
        res_set -= set(indices)
        ret_m += (indices)
        ret.append(ret_m)
    if return_residual_set:
        ret.append(list(res_set))
    return ret


class Dataset(object): #pragma no cover
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    
class ArrayDataset(Dataset):
    """Dataset wrapping arrays.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments: 

      arrays (array, tuple[array]): a single array or tuple of arrays
        that have the same size of the first dimension.

    Examples:

    >>> D = ArrayDataset(np.diag(np.arange(3)))
    >>> len(D)
    3
    >>> D[0]
    array([0, 0, 0])
    >>> D = ArrayDataset((np.diag(np.arange(3)), np.arange(3)))
    >>> len(D)
    3
    >>> D[0]
    (array([0, 0, 0]), 0)
    """
    def __init__(self, arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
        self.arrays = arrays

    def __getitem__(self, index):
        if isinstance(self.arrays, np.ndarray):
            item = self.arrays[index]
        else:
            item = tuple(array[index] for array in self.arrays)
        return item

    def __len__(self):
        if isinstance(self.arrays, np.ndarray):
            n = self.arrays.shape[0]
        else:
            n = self.arrays[0].shape[0]
        return n


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets.

    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:

        datasets (sequence): List of datasets to be concatenated

    Examples:

    >>> D = ArrayDataset(np.diag(np.arange(3)))
    >>> D = ConcatDataset((D, D))
    >>> len(D)
    6
    >>> D[3]
    array([0, 0, 0])
    >>> D[0]
    array([0, 0, 0])
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]        


class TransformDataset(Dataset):
    """Generate new dataset by adding a transform to an existing dataset

    Args:

      dataset (Dataset): The input dataset.

      transform (callable): The transform that will be applied to
        each example of the input dataset.

    Returns:

    Dataset: A new dataset that each example will be transformed.
    
    Examples:

    >>> trans = lambda x: x * 2
    >>> D = TransformDataset(ArrayDataset(np.ones((3,4))), trans)
    >>> len(D)
    3
    >>> np.all(D[0] == 2)
    True
    """
    def __init__(self, dataset, transform=None):
        self.original_dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        item = self.original_dataset[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.original_dataset)
    

class TransformLabeledDataset(Dataset):
    """Generate new dataset by transforming an existing labeled dataset

    Args:

      dataset (Dataset): The input dataset.

      transform (callable): The transform that will be applied to
        each example of the input dataset.

      target_transform (callable): The transform that will be applied
        to the target of each example. Default to None.

    Returns:

    Dataset: A new dataset that each example will be transformed.

    Examples:

    >>> D = ArrayDataset((np.ones((3,4)), np.ones((3,))))
    >>> trans = lambda x: x * 2
    >>> trans2 = lambda x: x * 3
    >>> D = TransformLabeledDataset(D, trans, trans2)
    >>> len(D)
    3
    >>> np.all(D[0][0] == 2)
    True
    >>> D[0][1]
    3.0
    """
    def __init__(self, dataset, transform, target_transform=None):
        self.original_dataset = dataset
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        x = self.original_dataset[index]
        assert len(x)==2, 'only works with 2-tuple data'
        x = list(x)
        x[0] = self.transform(x[0])
        x[1] = self.target_transform(x[1])
        return tuple(x)

    def __len__(self):
        return len(self.original_dataset)
        

class DataFileFolder(Dataset):
    """Dataset represented as folder contains data files

    Args:

      data_path (path-like): Path to the folder.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.

      transform (callable): A transformation that will be applied to
        the loaded data

      recursive (bool): If True, recursives look for files. Otherwise,
        search files in the top level folder only.

      skip_containing_folder (bool): Whether skip the containing
        folder, i.e. the only folder without siblings in the
        archive. Default to True.

    See also:

      :class:`LabeledDataFileFolder`

    """
    def __init__(self, data_path, loader, extensions, transform=None,
                 recursive=False, skip_containing_folder=True):
        data_path = Path(data_path)
        if isinstance(extensions, (str,bytes)):
            extensions = [extensions]
        extensions = [x.lower() for x in extensions]

        if recursive:
            samples = []
            for p, _, files in os.walk(data_path):
                p = Path(p)
                samples += [p/x  for x in files if any(x.lower().endswith(y) for y in extensions)]
        else:
            if skip_containing_folder:
                tmp = list(data_path.iterdir())
                while len(tmp) == 1 and tmp[0].is_dir():
                    data_path = tmp[0]
                    tmp = list(data_path.iterdir())
            samples = [x for x in data_path.iterdir()
                       if x.suffix.lower() in extensions]
        if len(samples) == 0:
            raise RuntimeError(
                "Found 0 files in folder: " + str(data_path) + "\n"
                "Supported extensions are: " + ",".join(extensions))
        self.data_path = data_path
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


    def __len__(self):
        return len(self.samples)


class LabeledDataFileFolder(Dataset):
    """Labeled image dataset represented as subfolders of image files

    Args:

      data_path (path-like): Path to the folder.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.

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


    See also:

      :class:`DataFileFolder`

    """
    
    def __init__(self, data_path, loader, extensions,
                 label_encoder='index',
                 transform=None, target_transform=None,
                 skip_containing_folder=True,
                 category_recursive=False,
                 category_skip_containing_folder=None):
        data_path = Path(data_path)
        if isinstance(extensions, (str, bytes)):
            extensions = [extensions]
        if category_skip_containing_folder is None:
            category_skip_containing_folder = skip_containing_folder
        
        subfolders = [x for x in data_path.iterdir() if x.is_dir()]
        if skip_containing_folder:
            pre_subfolders = subfolders
            while len(subfolders) == 1:
                pre_subfolders = subfolders
                subfolders = [x for x in subfolders[0].iterdir()
                              if x.is_dir()]
            subfolders = subfolders if len(subfolders)>0 else pre_subfolders
        
        labels = [x.name for x in subfolders]
        samples = [DataFileFolder(
            subfolder, loader, extensions,
            transform=transform,
            recursive=category_recursive,
            skip_containing_folder=category_skip_containing_folder)
                   for subfolder in subfolders]
        nsamples = [len(x) for x in samples]
        cum_n = list(itertools.accumulate(nsamples))

        self.data_path = data_path
        self.loader = loader
        self.extensions = extensions
        self.labels = labels
        self.samples = samples
        self.cum_n = cum_n
        self.transform = transform
        self.target_transform = target_transform
        
        if (label_encoder is None or isinstance(label_encoder, str)
            and label_encoder.lower() == 'none'):
            self.label_encoder = None
        elif (isinstance(label_encoder, str) and
              label_encoder.lower() == 'index'):
            self.label_encoder = {y: i for i, y in enumerate(labels)}
        else:
            self.label_encoder = label_encoder


    def _encode_label(self, label):
        if self.label_encoder is None:
            pass
        elif isinstance(self.label_encoder, collections.abc.Mapping):
            label = self.label_encoder[label]
        elif callable(self.label_encoder):
            label = self.label_encoder(label)
        else:
            assert hasattr(self.label_encoder, 'transform')
            label = self.label_encoder.transform([label])[0]
        return label


    def __getitem__(self, index):
        index1 = bisect.bisect_right(self.cum_n, index)
        index2 = index - (self.cum_n[index1-1] if index1>0 else 0)
        sample = self.samples[index1][index2]
        label = self._encode_label(self.labels[index1])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label
    

    def __len__(self):
        return self.cum_n[-1]


class DataFileArchive(Dataset):
    """Dataset stored in an archive file

    Args:

      data_path (file-like): Path to or file object of the archive
        file.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.

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

    See also:

      :class:`DataFileFolder`

    """
    def __init__(self, data_path, loader, extensions, transform=None,
                 recursive=False, skip_containing_folder=True,
                 engine=None, cache_archive_obj=True):
        if isinstance(extensions, (str, bytes)):
            extensions = [extensions]
        extensions = [x.lower() for x in extensions]

        self.data_path = data_path
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.recursive = recursive
        self.skip_containing_folder = skip_containing_folder
        self.engine = engine
        self.cache_archive_obj = cache_archive_obj
        self._prepare()
        if not cache_archive_obj:
            self.close_archive()

    def _prepare(self):
        self.archive = arlib.open(self.data_path, engine=self.engine)
        ar = self.archive
        extensions = self.extensions
        fnames = ar.member_names
        if not self.recursive:
            if self.skip_containing_folder:
                dirs = [x for x in fnames if ar.member_is_dir(x)]
                prefix = os.path.commonprefix(dirs)
                while (prefix.endswith('/') and prefix in fnames and
                       all(x.startswith(prefix) for x in fnames)):
                    assert ar.member_is_dir(prefix)
                    fnames.remove(prefix)
                    dirs.remove(prefix)
                    prefix = os.path.commonprefix(dirs)
            dirs = [x for x in fnames if ar.member_is_dir(x)]
            fnames = [x for x in fnames
                      if not any(x.startswith(y) for y in dirs)]
        fnames = [x for x in fnames
                  if (ar.member_is_file(x) and
                      any(x.lower().endswith(y) for y in extensions))]
        if len(fnames) == 0:
            raise RuntimeError(
                'Found 0 files in archive:'+str(self.data_path)+'\n'
                'Supported extensions are:'+','.join(extensions))
        self.fnames = fnames
        
    def close_archive(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        
        
    def __getitem__(self, index):
        if self.archive is None:
            self._prepare()
        name = self.fnames[index]
        sample = self.loader(self.archive.open_member(name, 'rb'))
        if self.transform is not None:
            sample = self.transform(sample)

        if not self.cache_archive_obj:
            self.close_archive()
            
        return sample

    def __len__(self):
        return len(self.fnames)


class LabeledDataFileArchive(Dataset):
    """Labeled data files stored in an archive file

    Args:

      data_path (path-like): Path to the archive file.

      loader (callable): A function to load a sample given its path.
        files.

      extensions (str, Seq[str]): A list (or single) of allowed extensions.

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

      member_archives_as_categories (bool): Whether use member archive
        (i.e. archive file in the dataset archive file) as
        categories. Default to True. If False, only member folders
        will be considered as categories.

      label_ignore_member_archive_suffix (bool): Whether ignore the
        suffix of member archive name in labels. For example, samples
        in xyz.abc will be labeled 'xyz' instead of 'xyz.abc'. Default
        to True.

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

    """
    def __init__(self, data_path, loader, extensions,
                 label_encoder='index',
                 transform=None, target_transform=None,                 
                 member_archives_as_categories=True,
                 label_ignoare_member_archive_suffix=True,
                 skip_containing_folder=True,
                 category_recursive=False,
                 category_skip_containing_folder=None,
                 engine=None,     
                 cache_archive_obj=True):
        
        if isinstance(extensions, (str, bytes)):
            extensions = [extensions]
        extensions = [x.lower() for x in extensions]
        if category_skip_containing_folder is None:
            category_skip_containing_folder = skip_containing_folder
        self.data_path = data_path
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
        self.member_archives_as_categories = member_archives_as_categories
        self.label_ignore_member_archive_suffix = label_ignoare_member_archive_suffix
        self.skip_containing_folder = skip_containing_folder
        self.category_recursive = category_recursive
        self.category_skip_containing_folder =category_skip_containing_folder
        self.engine = engine
        self.cache_archive_obj = cache_archive_obj
        self._prepare()
        if not cache_archive_obj:
            self.close_archive()
            
        if (label_encoder is None or isinstance(label_encoder, str)
            and label_encoder.lower() == 'none'):
            self.label_encoder = None
        elif (isinstance(label_encoder, str) and
              label_encoder.lower() == 'index'):
            self.label_encoder = {y: i for i, y in enumerate(self.labels)}
        else:
            self.label_encoder = label_encoder

            
    def _prepare(self):
        self.archive = arlib.open(self.data_path, engine=self.engine)
        ar = self.archive
        fnames = ar.member_names
        extensions = self.extensions
        if self.skip_containing_folder:
            pre_fnames = fnames
            dirs = [x for x in fnames if ar.member_is_dir(x)]
            prefix = os.path.commonprefix(dirs)
            level = 1
            while (prefix.endswith('/') and prefix in fnames and
                   all(x.startswith(prefix) for x in fnames)):
                assert ar.member_is_dir(prefix)
                pre_fnames = fnames
                fnames.remove(prefix)
                dirs.remove(prefix)
                level += 1
                prefix = os.path.commonprefix(dirs)
            dirs = [x for x in dirs if len(x.split('/')) == level+1]
            fnames = pre_fnames

        ars = [] # list of list [name, engine]
        if self.member_archives_as_categories:
            for x in fnames:
                if ar.member_is_file(x) and len(x.split('/')) == level:
                    engine = arlib.auto_engine(ar.open_member(x, 'rb'), 'r')
                    if engine is None:
                        # try determine engine by name
                        engine = arlib.auto_engine(x, 'w')
                    if engine is not None:
                        ars.append([x, engine])

        samples = dict()
        for d in dirs:
            if self.category_recursive:
                names = [x for x in fnames if x.startswith(d)]
            else:
                if self.category_skip_containing_folder:
                    pred = lambda x: (
                        x.startswith(d) and
                        (x.endswith('/') and
                         len(x.split('/'))==len(d.split('/'))+1 or
                         not x.endswith('/') and
                         len(x.split('/'))==len(d.split('/'))))
                    names_all = [x for x in fnames if pred(x)]
                    while len(names_all) == 1 and names_all[0].endswith('/'):
                        d = names_all[0]
                        names_all = [x for x in fnames if pred(x)]

                names = [x for x in fnames if x.startswith(d) and
                         len(x.split('/')) == len(d.split('/'))]
            names = [x for x in names
                     if any(x.endswith(y) for y in extensions)]
            d = d.split('/')[-2]
            assert d not in samples
            samples[d] = names

        self.mem_ars = dict()
        for fname, engine in ars:
            name = fname.split('/')[-1]
            if self.label_ignore_member_archive_suffix:
                name = name.split('.')[0]
            assert name not in samples
            samples[name] = DataFileArchive(
                ar.open_member(fname, 'rb'), loader=self.loader,
                extensions=extensions, transform=self.transform,
                recursive=self.category_recursive,
                skip_containing_folder=self.category_skip_containing_folder,
                engine=engine)
                
        labels = list(samples.keys())
        samples = [samples[x] for x in labels]
        nsamples = [len(x) for x in samples]
        cum_n = list(itertools.accumulate(nsamples))
        self.labels = labels
        self.samples = samples
        self.cum_n = cum_n
        if self.cum_n[-1] == 0: #pragma no cover
            raise RuntimeError('No data file was found in: '+
                               str(self.data_path))

    def _encode_label(self, label):
        if self.label_encoder is None:
            pass
        elif isinstance(self.label_encoder, collections.abc.Mapping):
            label = self.label_encoder[label]
        elif callable(self.label_encoder):
            label = self.label_encoder(label)
        else:
            assert hasattr(self.label_encoder, 'transform')
            label = self.label_encoder.transform([label])[0]
        return label
        
        
    def close_archive(self):
        self.archive.close()
        self.archive = None

    def __len__(self):
        return self.cum_n[-1]
                        
    
    def __getitem__(self, index):
        index = bisect.bisect_right(self.cum_n, index)
        index2 = index - (self.cum_n[index-1] if index>0 else 0)
        if self.archive is None:
            self._prepare()
        sub_ar = self.samples[index]
        if isinstance(sub_ar, (list, tuple)):
            sample=self.loader(self.archive.open_member(sub_ar[index2],'rb'))
        else:
            assert isinstance(sub_ar, DataFileArchive)
            sample = sub_ar[index2]
            
        if not self.cache_archive_obj:
            self.close_archive()

        label = self._encode_label(self.labels[index])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label
