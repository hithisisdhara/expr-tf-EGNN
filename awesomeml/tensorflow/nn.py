# -*- coding: utf-8 -*-
"""
TensorFlow neural network ops, extending tf.nn.
"""
import tensorflow as tf
import numpy as np
from .. import utils as _utils

def bias_add(value, bias, axis=-1):
    """Add `bias` to the specified axis (axes) of `value`
    
    Args:

      value (tensor_like): input tensor

      bias (tensor_like): bias tensor to add

      axis (int, Seq[int]): axis (axes) to add the bias
    
    Return:

      Tensor: Resultant Tensor with bias added to input

    Examples:

    >>> import tensorflow as tf
    >>> import awesomeml as aml
    >>> sess = tf.Session()
    >>> a = np.random.random((2,3,4,5))
    >>> b = np.random.random((4,))
    >>> c = a + np.reshape(b, (1,1,4,1))
    >>> c2 = bias_add(a, b, axis=2)
    >>> np.testing.assert_array_almost_equal(c, sess.run(c2))

    >>> b = np.random.random((3,4))
    >>> c = a + np.reshape(b, (1,3,4,1))
    >>> c2 = bias_add(a, b, axis=(1,2))
    >>> np.testing.assert_array_almost_equal(c, sess.run(c2))

    >>> b = np.random.random((4,3))
    >>> c = a + np.reshape(b.T, (1,3,4,1))
    >>> c2 = bias_add(a, b, axis=(2,1))
    >>> np.testing.assert_array_almost_equal(c, sess.run(c2))

    >>> b = np.random.random((4,3))
    >>> bias_add(a, b, axis=(1,2,3))
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> bias_add(a, b, axis=(1,1))
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    value = tf.convert_to_tensor(value)
    bias = tf.convert_to_tensor(bias)
    ndims = value.get_shape().ndims
    axis = _utils.validate_axis(axis, ndims, accept_none=False,
                                    scalar_to_seq=True)
    bias_shape = bias.get_shape().as_list()
    if len(bias_shape) != len(axis):
        raise ValueError('ndims of bias does not match number of axes: '+
                         '{}!={}'.format(len(bias_shape), len(axis)))
    if len(set(axis)) != len(axis):
        raise ValueError('repeated axes are specified.')
    if not all(axis[i] < axis[i+1] for i in range(len(axis)-1)):
        perm = list(np.argsort(axis))
        axis = [axis[i] for i in perm]
        bias = tf.transpose(bias, perm)
        bias_shape = bias.get_shape().as_list()

    bias_new_shape = [1] * ndims
    for i, axis_1 in enumerate(axis):
        bias_new_shape[axis_1] = bias_shape[i]
    bias = tf.reshape(bias, bias_new_shape)
    return tf.add(value, bias)
