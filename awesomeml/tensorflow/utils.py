# -*- coding: utf-8 -*-
"""
Utility functions for TensorFlow related operations.
"""
import collections
import collections.abc
from pathlib import Path
import tensorflow as tf
import numpy as np


from .. import utils as _utils
from .. import config as _config
from .. import ops as _ops

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
    return _utils.validate_dir(value, default_path=_config.TF_DATA_HOME,
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
    return _utils.validate_dir(value, default_path=_config.TF_MODEL_HOME,
                               create=create, name=name)



def dense_to_sparse(value):
    """Convert a dense tensor to sparse tensor
    
    Args:
      value (Tensor): input dense Tensor
    
    Return:
      SparseTensor: converted sparse tensor

    Examples:

    >>> import tensorflow as tf
    >>> x = np.array([[1,0],[0,1]], dtype=np.float32)
    >>> y = dense_to_sparse(x)
    >>> y = tf.sparse.to_dense(y)
    >>> with tf.Session():
    ...     np.testing.assert_array_equal(y.eval(), x)
    
    """
    x = tf.convert_to_tensor(value)
    x.shape.assert_has_rank(2)
    indices = tf.where(tf.not_equal(x, 0))
    res = tf.SparseTensor(indices=indices,
                          values=tf.gather_nd(x, indices),
                          dense_shape=x.get_shape())
    return tf.sparse_reorder(res)


def as_sparse_tensor(value):
    """
    """
    #assert_is_sparse(value)
    if isinstance(value, tf.SparseTensor):
        return value
    value = _utils.as_tf_sparse(value)
    return tf.convert_to_tensor_or_sparse_tensor(value)


def is_sparse(value):
    return (_utils.is_sparse(value) or
            isinstance(value, (tf.SparseTensor, tf.SparseTensorValue)))

def is_sparse_seq(value):
    """
    """
    return (isinstance(value, collections.abc.Sequence) and
            all(is_sparse(x) for x in value))


def as_tensor_or_sparse_tensor(value):
    """Convert value to a SparseTensor or Tensor
    
    Args:

      value (any-tensor-like): A SparseTensor, SparseTensorValue,
        scipy sparse matrix, pydata sparse array, or an object whose
        type has a registered Tensor conversion function.

    Returns:

    A SparseTensor or Tensor based on value.

    Examples:

    >>> import awesomeml as aml
    >>> import tensorflow as tf
    >>> import scipy.sparse
    >>> sess = tf.Session()
    >>> a = scipy.sparse.random(3, 4, 0.8)
    >>> b = as_tensor_or_sparse_tensor(a)
    >>> isinstance(b, tf.SparseTensor)
    True
    >>> a2 = aml.utils.as_numpy_array(sess.run(b))
    >>> np.testing.assert_array_almost_equal(a.todense(), a2)
    """
    if _utils.is_sparse(value):
        value = _utils.as_tf_sparse(value)

    return tf.convert_to_tensor_or_sparse_tensor(value)


def as_tensor_or_sparse_tensor_or_seq(value, axis=0,
                                             seq_type=list):
    """Convert value to Tensor, SparseTensor or Collection of
       SparseTensors
    
    Similar to tf.convert_to_tensor_or_sparse_tensor, but will convert
    a sparse array with ndim==3 to a collection of SparseTensors.

    Args:

      value (any_tensor_like): input array

      axis (int): axis to unstack if value has ndim==3

    Examples:

    >>> import tensorflow as tf
    >>> import awesomeml as aml
    >>> import sparse
    >>> sess = tf.Session()
    >>> a = np.random.random((3,4))
    >>> b = as_tensor_or_sparse_tensor_or_seq(sparse.as_coo(a))
    >>> isinstance(b, tf.SparseTensor)
    True
    >>> b = aml.utils.as_numpy_array(sess.run(b))
    >>> np.testing.assert_array_almost_equal(a, b)

    >>> a = np.random.random((2,3,4))
    >>> b = list(a)
    >>> b2 = as_tensor_or_sparse_tensor_or_seq(sparse.as_coo(a))
    >>> isinstance(b2, list)
    True
    >>> all(isinstance(x, tf.SparseTensor) for x in b2)
    True
    >>> b2 = [aml.utils.as_numpy_array(sess.run(x)) for x in b2]
    >>> np.testing.assert_array_almost_equal(b[0], b2[0])
    >>> np.testing.assert_array_almost_equal(b[1], b2[1])

    >>> a = sparse.random((2,3,4,5))
    >>> as_tensor_or_sparse_tensor_or_seq(a)
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    if (_config.has_package('sparse') and _utils.is_pydata_sparse(value)):
        if value.ndim == 2:
            return as_sparse_tensor(value)
        elif value.ndim == 3:
            values = _ops.unstack(value, axis=axis)
            return seq_type(as_sparse_tensor(x) for x in values)
        else:
            raise ValueError('ndim of value must not be greater than 3.')
    elif is_sparse(value):
        return as_sparse_tensor(value)
    elif is_sparse_seq(value):
        return seq_type(as_sparse_tensor(x) for x in value)
    else:
        return tf.convert_to_tensor(value)



def as_tensor_or_sparse_tensor_seq(value, axis=0, seq_type=list):
    """Convert value to Tensor or sequence of SparseTensors

    Similar to as_tensor_or_sparse_tensor_or_seq, but never
    return SparseTensor, but collection of SparseTensor. That's to
    say, if value can be converted to a SparseTensor, it will be
    returned as a list/tuple with only one element.

    Examples:

    >>> import tensorflow as tf
    >>> import awesomeml as aml
    >>> import scipy.sparse
    >>> sess = tf.Session()
    >>> a = scipy.sparse.random(3, 4, 0.8)
    >>> b = as_tensor_or_sparse_tensor_seq(a)
    >>> isinstance(b, list)
    True
    >>> b = b[0]
    >>> isinstance(b, tf.SparseTensor)
    True
    >>> b = aml.utils.as_numpy_array(sess.run(b))
    >>> np.all(b == a.todense())
    True

    """
    res = as_tensor_or_sparse_tensor_or_seq(value, axis, seq_type)
    if isinstance(res, tf.SparseTensor):
        res = seq_type([res])
    return res
