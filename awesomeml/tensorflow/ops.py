# -*- coding: utf-8 -*-
"""
TensorFlow tensor and sparse tensor ops
"""
import scipy.sparse
import numpy as np
import tensorflow as tf

from .. import ops as _ops
from .. import config as _config
from .. import utils as _utils
from . import utils as _tf_utils

def sparse_op(op, n, *args, reorder=None, **kwargs):
    """Apply a operator to SparseTensor
    
    Args:

      op: a callable object to apply

      n: int, specify the number of operands

      args: positional arguments passed to `op`, including the
        operands. So the first n positional arguments should be
        SparseTensor

      reorder (str, NoneType): Whether reorder the operands or
        result. Possible values:

        * 'operand': Reorder operand before applying the op.

        * 'result': Reorder result after applying the op.

        * None: Do not reorder. Default value.
    
      kwargs: other keyword arguments passed to `op`
    
    Returns:

    Resultant SparseTensor

    Raises:

      ValuerError: if length of args is less than n

      TypeError: if not all the first n positional arguments are SparseTensor

    Examples:

    >>> import tensorflow as tf
    >>> sess = tf.Session()
    >>> indices = np.array([[0, 0], [2, 1], [1, 2]], dtype=np.int64)
    >>> values = np.array([1, 3, 2], dtype=np.float32)
    >>> dense_shape = np.array([3, 3], dtype=np.int64)
    >>> a = tf.SparseTensor(indices, values, dense_shape)
    >>> b = sparse_op(tf.negative, 1, a, reorder=None)
    >>> isinstance(b, tf.SparseTensor)
    True
    >>> b = sess.run(b)
    >>> np.all(b.indices == indices)
    True
    >>> np.all(b.values == -values)
    True
    >>> np.all(b.dense_shape == dense_shape)
    True
    >>> sparse_op(tf.add, 3, a) # not enough operands
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> a = tf.ones((3,4))
    >>> b2 = sparse_op(tf.negative, 1, a) # a1 is not SparseTensor
    Traceback (most recent call last):
      ...
    ValueError: ...

    """
    assert n > 0
    if len(args) < n:
        raise ValueError(n-len(args),
                         ' more positional arguments are required.')
    
    for i in range(n):
        if not isinstance(args[i], tf.SparseTensor):
            raise ValueError('The ',i+1,'th operand is not a SparseTensor')

    reorder = _utils.validate_option(reorder, ['operand', 'result', None],
                                     name='reorder',
                                     str_normalizer=str.lower)
    if reorder == 'operand':
        args = [tf.sparse_reorder(x) for x in args]

    indices = args[0].indices
    dense_shape = args[0].dense_shape
    operands = [args[i].values for i in range(n)]
    res = tf.SparseTensor(indices=indices,
                          values=op(*operands, *args[n:], **kwargs),
                          dense_shape=dense_shape)
    if reorder == 'result':
        res = tf.sparse_reorder(res)
    return res


def sparse_op1(op, operand, *args, reorder=None, **kwargs):
    """Apply uniary operator to a SparseTensor operand

    Args:

      op: a callable object to apply

      operand: SparseTensor, the operand to apply `op`

      args: other positional arguments passed to `op`

      reorder (bool): Whether reorder the operands before applying op.

      kwargs: other keyword arguments passed to `op`

    Returns:

    Resultant SparseTensor
    
    Raises:

      TypeError: if `operand` is not a SparseTensor

    Examples:

    >>> import tensorflow as tf
    >>> sess = tf.Session()
    >>> indices = np.array([[0, 0], [2, 1], [1, 2]], dtype=np.int64)
    >>> values = np.array([1, 3, 2], dtype=np.float32)
    >>> dense_shape = np.array([3, 4], dtype=np.int64)
    >>> a = tf.SparseTensor(indices, values, dense_shape)
    >>> b = sparse_op1(tf.negative, a, reorder=None)
    >>> isinstance(b, tf.SparseTensor)
    True
    >>> b = sess.run(b)
    >>> np.all(b.indices == indices)
    True
    >>> np.all(b.values == -values)
    True
    >>> np.all(b.dense_shape == dense_shape)
    True
    >>> b = sparse_op1(tf.negative, a, reorder='result')
    >>> indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    >>> values = np.array([-1, -2, -3], dtype=np.float32)
    >>> isinstance(b, tf.SparseTensor)
    True
    >>> b = sess.run(b)
    >>> np.all(b.indices == indices)
    True
    >>> np.all(b.values == values)
    True
    >>> np.all(b.dense_shape == dense_shape)
    True
    """
    return sparse_op(op, 1, operand, *args, reorder=reorder, **kwargs)

def sparse_op2(op, operand1, operand2, *args, reorder=None, **kwargs):
    """Apply binary operator to two SparseTensor operands

    Arguments:

      op: a callable object to apply

      operand1: SparseTensor, the first operand to apply `op`

      operand2: SparseTensor, the second operand to apply `op`

      args: other positional arguments passed to `op`

      reorder (bool): Whether reorder the operands before applying op.

      kwargs: other keyword arguments passed to `op`

    Returns:

    Resultant SparseTensor
    
    Raises:

    TypeError: if `operand1` or `operand2` is not SparseTensor

    Examples:

    >>> import tensorflow as tf
    >>> sess = tf.Session()
    >>> indices = np.array([[0, 0], [2, 1], [1, 2]], dtype=np.int64)
    >>> values = np.array([1, 3, 2], dtype=np.float32)
    >>> dense_shape = np.array([3, 4], dtype=np.int64)
    >>> a = tf.SparseTensor(indices, values, dense_shape)
    >>> b = tf.SparseTensor(indices, values, dense_shape)
    >>> c = sparse_op2(tf.multiply, a, b, reorder=None)
    >>> isinstance(c, tf.SparseTensor)
    True
    >>> c = sess.run(c)
    >>> np.all(c.indices == indices)
    True
    >>> np.all(c.values == values * values)
    True
    >>> np.all(c.dense_shape == dense_shape)
    True

    >>> indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=np.int64)
    >>> values = np.array([1, 2, 3], dtype=np.float32)
    >>> c = sparse_op2(tf.multiply, a, b, reorder='operand')
    >>> isinstance(c, tf.SparseTensor)
    True
    >>> c = sess.run(c)
    >>> np.all(c.indices == indices)
    True
    >>> np.all(c.values == values * values)
    True
    >>> np.all(c.dense_shape == dense_shape)
    True
    """
    return sparse_op(op, 2, operand1, operand2, *args, reorder=reorder,
                     **kwargs)

def reduce_sum(a, *args, **kwargs):
    """Wraper of tf.reduce_sum and tf.sparse_reduce_sum

    Examples:

    >>> import tensorflow as tf
    >>> import scipy.sparse
    >>> sess = tf.Session()
    >>> a = np.random.random((3,4))
    >>> b = np.sum(a, 1)
    >>> b2 = sess.run(reduce_sum(a, 1))
    >>> np.testing.assert_array_almost_equal(b, b2)

    >>> b2 = sess.run(reduce_sum(scipy.sparse.coo_matrix(a), 1, False))
    >>> np.testing.assert_array_almost_equal(b, b2)

    >>> b2 = sess.run(reduce_sum(scipy.sparse.coo_matrix(a), axis=1, 
    ...                          keepdims=False))
    >>> np.testing.assert_array_almost_equal(b, b2)
    
    >>> b = np.sum(a, keepdims=True)
    >>> b2 = sess.run(reduce_sum(scipy.sparse.coo_matrix(a), keepdims=True))
    >>> np.testing.assert_array_almost_equal(b, b2)

    >>> b = np.sum(a, keepdims=False)
    >>> b2 = sess.run(reduce_sum(scipy.sparse.coo_matrix(a), keepdims=False))
    >>> np.testing.assert_array_almost_equal(b, b2)

    """
    a = _tf_utils.as_tensor_or_sparse_tensor(a)
    if isinstance(a, tf.Tensor):
        return tf.reduce_sum(a, *args, **kwargs)
    else:
        tf_v = tf.__version__.split('.')
        if int(tf_v[0]) > 1 or int(tf_v[0]==1) and int(_tf_v[1]) > 12: # pragma: no cover
            return tf.sparse_reduce_sum(a, *args, **kwargs);
            
        # ************************************************************
        # workaround to issue:
        # https://github.com/tensorflow/tensorflow/issues/23114
        # ************************************************************
        s = a.get_shape()
        ndim = s.ndims
        if ndim is None: # pragma: no cover
            return tf.sparse_reduce_sum(a, *args, **kwargs)
        s = s.as_list()
        if len(args) > 0:
            axis = args[0]
        else:
            axis = kwargs.get('axis', None)

        if len(args) > 1:
            keepdims = args[1]
        else:
            keepdims = kwargs.get('keepdims', False)
        if axis is None:
            if keepdims:
                s = [1] * len(s)
            else:
                s = []
        else:
            axis = _utils.validate_axis(axis, ndim)
            if not isinstance(axis, (tuple, list)):
                axis = [axis]
            axis = list(axis)
            axis = sorted(axis, reverse=True)
            for i in axis:
                if keepdims:
                    s[i] = 1
                else:
                    del s[i]
        a = tf.sparse_reduce_sum(a, *args, **kwargs)
        #a.set_shape(s)
        a = tf.reshape(a, s)
        return a


def swapaxes(a, axis1, axis2):
    """Interchange two axes of a Tensor

    Args:

      a: input Tensor

      axis1: int, first axis

      axis2: int, second axis

    Returns:

    A tensor with axes swapped

    Examples:

    >>> import tensorflow as tf
    >>> import awesomeml.utils as aml_utils
    >>> sess = tf.Session()
    >>> a = np.random.random((2,3,4,5))
    >>> b = np.swapaxes(a, axis1=1, axis2=2)
    >>> b2 = sess.run(swapaxes(a, axis1=1, axis2=2))
    >>> np.testing.assert_array_almost_equal(b, b2)

    >>> import sparse
    >>> a = np.random.random((3,4))
    >>> b = np.swapaxes(a, axis1=0, axis2=1)
    >>> b2 = swapaxes(sparse.as_coo(a), axis1=0, axis2=1)
    >>> isinstance(b2, tf.SparseTensor)
    True
    >>> b2 = sess.run(b2)
    >>> b2 = aml_utils.as_numpy_array(b2)
    >>> np.testing.assert_array_almost_equal(b, b2)
    """
    a = _tf_utils.as_tensor_or_sparse_tensor(a)
    ndim = a.get_shape().ndims
    if ndim is None: # pragma: no cover
        raise ValueError('can not swapaxes for tensors with unknown shape')
    axis1 = _utils.validate_axis(axis1, ndim)
    axis2 = _utils.validate_axis(axis2, ndim)

    if axis1 == axis2:
        return a

    perm = list(range(ndim))
    perm[axis1] = axis2
    perm[axis2] = axis1
    if isinstance(a, tf.Tensor):
        return tf.transpose(a, perm)
    else:
        return tf.sparse_transpose(a, perm)


def matmul(a, b, *args, **kwargs):
    """Wrapper of tf.matmul and tf.sparse.matmul

    Examples:

    >>> import tensorflow as tf
    >>> import awesomeml as aml
    >>> sess = tf.Session()
    >>> a = np.random.random((3,4))
    >>> b = np.random.random((4,3))
    >>> c = np.matmul(a, b)
    >>> c2 = aml.utils.as_numpy_array(sess.run(matmul(a, b)))
    >>> np.testing.assert_array_almost_equal(c, c2)

    >>> import scipy.sparse
    >>> c2 = matmul(scipy.sparse.coo_matrix(a), b)
    >>> c2 = aml.utils.as_numpy_array(sess.run(c2))
    >>> np.testing.assert_array_almost_equal(c, c2)

    >>> c2 = matmul(a, scipy.sparse.coo_matrix(b))
    >>> c2 = aml.utils.as_numpy_array(sess.run(c2))
    >>> np.testing.assert_array_almost_equal(c, c2)

    >>> c2 = matmul(scipy.sparse.coo_matrix(a), scipy.sparse.coo_matrix(b))
    >>> isinstance(c2, tf.SparseTensor)
    True
    >>> c2 = aml.utils.as_numpy_array(sess.run(c2))
    >>> np.testing.assert_array_almost_equal(c, c2)
    """
    a = _tf_utils.as_tensor_or_sparse_tensor(a)
    b = _tf_utils.as_tensor_or_sparse_tensor(b)
    if isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
        return tf.matmul(a, b, *args, **kwargs)
    elif isinstance(a, tf.Tensor) and isinstance(b, tf.SparseTensor):
        return tf.matmul(a, tf.sparse_tensor_to_dense(b), *args, **kwargs)
    elif isinstance(a, tf.SparseTensor) and isinstance(b, tf.Tensor):
        return tf.sparse_tensor_dense_matmul(a, b, *args, **kwargs)
    else:
        b = tf.sparse.to_dense(b)
        res = tf.sparse_tensor_dense_matmul(a, b, *args, **kwargs)
        return _tf_utils.dense_to_sparse(res)

def moveaxis(a, axis_src, axis_dst):
    """Move an axis of a tensor to new position, similar to np.moveaxis
    
    Other axes remain in the original order

    Args:

      a (Tensor): the tensor whose axes should be reordered
    
      axis_src (int, Seq[int]): Original positions of the axes to
    move. These must be unique.
    
      axis_dst (int, Seq[int]): Destination position for each of the
      origianl axes. These must also be unique.

    Examples:

    >>> a = np.zeros((3, 4, 5))
    >>> moveaxis(a, 0, -1).get_shape().as_list()
    [4, 5, 3]
    >>> moveaxis(a, -1, 0).get_shape().as_list()
    [5, 3, 4]
    >>> moveaxis(a, [0, 1], [-1, -2]).get_shape().as_list()
    [5, 4, 3]
    >>> moveaxis(a, [0, 1, 2], [-1, -2, -3]).get_shape().as_list()
    [5, 4, 3]
    >>> moveaxis(a, [0, 1], [-1, -2, -3])
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> sa = scipy.sparse.random(3, 4)
    >>> moveaxis(sa, [0, 1], [-1, -2]).get_shape().as_list()
    [4, 3]
    """
    a = _tf_utils.as_tensor_or_sparse_tensor(a)
    ndims = a.get_shape().ndims
    src = _utils.validate_axis(
        axis_src, ndims, 'axis_src', accept_none=False,
        scalar_to_seq=True)
    dst = _utils.validate_axis(
        axis_dst, ndims, 'axis_dst', accept_none=False,
        scalar_to_seq=True)
    if len(src) != len(dst):
        raise ValueError('`axis_src` and `axis_dst` arguments must have the '
                         'same number of elements')
    order = [i for i in range(ndims) if i not in src]
    for dst_1, src_1 in sorted(zip(dst, src)):
        order.insert(dst_1, src_1)
    if isinstance(a, tf.Tensor):
        res = tf.transpose(a, order)
    else:
        res = tf.sparse_transpose(a, order)
    return res
