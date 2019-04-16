# -*- coding: utf-8 -*-
"""TensorFlow datasets

The common interface is :func:`load_dataset`, which returns a
tf.data.Dataset object.

"""
import functools
from pathlib import Path
import numpy as np
import decoutils
import collections.abc
import tensorflow as tf
from . import utils as _tf_utils
from ..datasets import utils as _ds_utils
from .. import utils as _utils
from .. import config as _config
from .. import datasets as _ds


# dataset loader registration functionality
_dataset_loaders = _utils.SafeDict(
    key_normalizer=_ds_utils.normalize_dataset_name,
    value_checker=callable)


for name in ['register_dataset_loader', 'unregister_dataset_loader',
             'dataset_has_loader', 'assert_dataset_has_loader',
             'get_dataset_loader']:
    globals()[name] = functools.partial(
        _ds_utils.__dict__[name], storage=_dataset_loaders)


def load_dataset_default(name, *args, subsets=None, data_home=None,
                         **kwargs): #pragma no cover
    """Default loader for dataset without registered loader

    Args:

      name (str): Name of the dataset.

      kwargs: Other arguments.

    Returns:

    Dataset: A tensorflow dataset.
    """
    _config.assert_has_package('tensorflow_datasets')
    import tensorflow_datasets as tfds
    subsets = _ds_utils.validate_tvt(subsets, training='train')
    data_home = _tf_utils.validate_data_home(data_home)
    D = tfds.load(name, split=subsets, data_dir=data_home,
                  as_supervised=True,
                  as_dataset_kwargs=dict(shuffle_files=False))
    if isinstance(D, tf.data.Dataset):
        pass # this is what we want
    elif isinstance(D, collections.abc.Sequence):
        assert all(isinstance(x, tf.data.Dataset) for x in D)
        assert len(D) > 0
        tmp = D[0]
        for i in range(1, len(D)):
            tmp = tmp.concatenate(D[i])
        D = tmp
    else:
        assert isinstance(D, collections.abc.Mapping)
        tmp = None
        for name in D:
            if tmp is None:
                tmp = D[name]
            else:
                tmp = tmp.concatenate(D[name])
        D = tmp
    return D


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
    else:
        D = load_dataset_default(name, *args, **kwargs)
    return D


def _deduce_otypes(sample):
    """

    Examples:
    >>> x = np.ones((3, 4), dtype=np.float32)
    >>> y = 3.0
    >>> _deduce_otypes((x, y))
    (tf.float32, tf.float64)
    >>> _deduce_otypes(dict(x=x, y=y))
    {'x': tf.float32, 'y': tf.float64}
    >>> _deduce_otypes(x)
    tf.float32
    """
    if isinstance(sample, collections.abc.Sequence):
        otypes = [x.dtype if hasattr(x, 'dtype') else np.dtype(type(x))
                      for x in sample]
        otypes = tuple(tf.as_dtype(x) for x in otypes)
    elif isinstance(sample, collections.abc.Mapping): #pragma no cover
        otypes = dict()
        for name in sample:
            x = sample[name]
            otypes[name] = tf.as_dtype(x.dtype if hasattr(x,'dtype') else
                                       np.dtype(type(x)))
    else:
        otypes = tf.as_dtype(sample.dtype if hasattr(sample, 'dtype')
                             else np.dtype(type(sample)))
    return otypes

def _deduce_oshapes(sample):
    """

    Examples:

    >>> x = np.ones((3, 4), dtype=np.float32)
    >>> y = 3
    >>> _deduce_oshapes((x, y))
    (TensorShape([Dimension(3), Dimension(4)]), TensorShape([]))
    >>> _deduce_oshapes(dict(x=x, y=y))
    {'x': TensorShape([Dimension(3), Dimension(4)]), 'y': TensorShape([])}
    >>> _deduce_oshapes(x)
    TensorShape([Dimension(3), Dimension(4)])
    """
    if isinstance(sample, collections.abc.Sequence):
        oshapes = [x.shape if hasattr(x, 'shape') else () for x in sample]
        oshapes = tuple(tf.TensorShape(x) for x in oshapes)
    elif isinstance(sample, collections.abc.Mapping): #pragma no cover
        oshapes = dict()
        for name in sample:
            x = sample[name]
            oshapes[name] = tf.TensorShape(x.shape if hasattr(x, 'shape')
                                          else ())
    else:
        oshapes = tf.TensorShape(sample.shape if hasattr(sample, 'shape')
                                 else ())
    return oshapes
        
        
def from_numpy_dataset(D, output_types='deduce', output_shapes='deduce'):
    """Convert a datasets.Dataset to tf.data.Dataset

    Args:

      D (Dataset): The input numpy Dataset.

      output_types: A nested structure of tf.DType objects
        corresponding to each component of an element yielded by
        resultant dataset. If None, 'none' or 'deduce', the dtypes
        will be deduced from the type of the first sample in the
        dataset. Default to None.

      output_shapes: A nested structure of tf.TensorShape objects
        corresponding to each component of an element yielded by the
        resultant dataset. If is 'deduce', the shapes will be deduced
        from the shape of the first sample in the dataset. If None or
        'none', the output tensor will have no fixed shape. Default to
        'deduce'.

    Examples:

    >>> D = _ds.ArrayDataset(np.ones((2, 3)))
    >>> g = tf.Graph()
    >>> with g.as_default():
    ...     D2 = from_numpy_dataset(D)
    ...     value = D2.make_one_shot_iterator().get_next()
    >>> with tf.Session(graph=g) as sess:
    ...     for i in range(2):
    ...         x = sess.run(value)
    ...         assert x.shape == (3, )
    ...         assert np.all(x == 1)
    ...     sess.run(value)
    Traceback (most recent call last):
      ...
    tensorflow.python.framework.errors_impl.OutOfRangeError: ...

    """
    # generator wrapper of the numpy dataset
    def _gen(): #pragma no cover
        for x in D:
            yield x
    
    sample = None
    if (output_types is None or isinstance(output_types, (str, bytes)) and
        (output_types.lower() in ['none', 'deduce'])):
        sample = D[0] if sample is None else sample
        output_types = _deduce_otypes(sample)
    if isinstance(output_shapes, (str, bytes)):
        if output_shapes.lower() == 'none':#pragma no cover
            output_shapes = None
        elif output_shapes.lower() == 'deduce':
            sample = D[0] if sample is None else sample
            output_shapes = _deduce_oshapes(sample)
    return tf.data.Dataset.from_generator(_gen, output_types=output_types,
                                          output_shapes=output_shapes)

@register_dataset_loader(
    name=['ilsvrc2012', 'ilsvrc-2012', 'ilsvrc', 'imagenet2012',
          'imagenet-2012', 'imagenet'])
def load_ilsvrc2012(*args, **kwargs): #pragma no cover
    """Load the Imagenet2012 (aka. ILSVRC2012) dataset

    Basically, this is a wrapper of the numpy version of
    load_ilsvrc. It calls the numpy version, then convert the result
    to a tf.data.Dataset.

    """
    return from_numpy_dataset(_ds.image.load_ilsvrc2012(*args, **kwargs))
