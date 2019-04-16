# -*- coding: utf-8 -*-
"""
Graph (network) data processing using TensorFlow.
"""
import numpy as np
import tensorflow as tf

from . import ops as _tf_ops
from . import utils as _tf_utils
from .. import utils as _utils

def _normalize_adj_1(A, method='sym', *, axis1=-2, axis2=-1, eps=0.0,
                     assume_symmetric_input=True):
    if not A.dtype.is_floating:
        A = tf.cast(A, tf.float32)
    
    if method in ['row', 'col', 'column']:
        axis_to_sum = axis2 if method == 'row' else axis1
        norm = _tf_ops.reduce_sum(A, axis_to_sum, keepdims=True)
        norm = 1.0 / (norm + eps)
        res = A * norm
    elif method in ['sym', 'symmetric']:
        norm1 = _tf_ops.reduce_sum(A, axis=axis2, keepdims=True)
        norm1 = 1.0 / (tf.sqrt(norm1) + eps)

        if assume_symmetric_input:
            norm2 = _tf_ops.swapaxes(norm1, axis1, axis2)
        else:
            norm2 = _tf_ops.reduce_sum(A, axis=axis1, keepdims=True)
            norm2 = 1.0 / (tf.sqrt(norm2) + eps)
        res = A * norm1 * norm2
    else:
        assert method in ['dsm', 'ds', 'doubly_stochastic']

        # step 1: row normalize
        norm = _tf_ops.reduce_sum(A, axis=axis2, keepdims=True)
        norm = 1.0 / (norm + eps)
        P = A * norm

        # step 2: P @ P^T / column_sum
        P = _tf_ops.swapaxes(P, axis2, -1)
        P = _tf_ops.swapaxes(P, axis1, -2)
        norm = _tf_ops.reduce_sum(P, axis=-2, keepdims=True)
        norm = 1.0 / (norm + eps)
        PT = _tf_ops.swapaxes(P, -1, -2)
        P = P * norm
        T = _tf_ops.matmul(P, PT)
        T = _tf_ops.swapaxes(T, axis1, -2)
        T = _tf_ops.swapaxes(T, axis2, -1)
        res = T
    
    # if isinstance(A, tf.SparseTensor) and isinstance(res, tf.Tensor):
    #     res = dense_to_sparse(res)
    return res
        

def normalize_adj(A, method='sym', *, axis1=-2, axis2=-1, eps=0.0,
                  assume_symmetric_input=True):
    """Normalize adjacency matrix defined by axis0 and axis1 in a tensor or sparse tensor
    
    Args:
    
      A (any-tensor): Input adjacency matrix or matrices. 
    
      method (str): Normalization method, could be:

        * 'sym', 'symmetric': Symmetric normalization, i.e. A' =
          D^-0.5 * A * D^-0.5
    
        * 'row': Row normalizatiion, i.e. A' = D^-1 * A

        * 'col', 'column': Column normalization, i.e. A' = A * D^-1

      axis1 (int): Specify the first axis of the adjacency matrices. Note
        that the input A could be a batch of matrices.

      axis2 (int): Specify the second axis of the adjacency matrices.

      eps (float): Regularization small value to avoid dividing by
        zero. Defaults to 0.0.

      assume_symmetric_input (bool): Whether assume the input
        adjacency matrices are symmetric or not. It affects results of
        symmetric normalization only. When it is True, it will reuse
        the row sum as col sum, which will avoid the computation of
        column sum. Will need to be set as False when the inputs is
        not symmetric, otherwise the result will be incorrect. Default
        to True.

    Returns:

      any-tensor: Normalized adjacency matrix or matrices.

    Examples:

    >>> A = np.array([[1,2],[3,4]], dtype=np.float32)
    >>> sess = tf.Session()
    >>> B = np.array([[1/3, 2/3], [3/7,4/7]])
    >>> B2 = normalize_adj(A, 'row')
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))

    >>> B2 = normalize_adj(A.astype(np.int), 'row')
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))

    >>> import scipy.sparse
    >>> B2 = normalize_adj(scipy.sparse.coo_matrix(A), 'row')
    >>> isinstance(B2, tf.SparseTensor)
    True
    >>> import awesomeml as aml
    >>> B2 = aml.utils.as_numpy_array(sess.run(B2))
    >>> np.testing.assert_array_almost_equal(B, B2)
    
    >>> sA = scipy.sparse.coo_matrix(A)
    >>> B21, B22 = normalize_adj([sA, sA], 'row')
    >>> isinstance(B21, tf.SparseTensor)
    True
    >>> isinstance(B22, tf.SparseTensor)
    True
    >>> B2 = aml.utils.as_numpy_array(sess.run(B21))
    >>> np.testing.assert_array_almost_equal(B, B2)
    >>> B2 = aml.utils.as_numpy_array(sess.run(B22))
    >>> np.testing.assert_array_almost_equal(B, B2)

    >>> B = np.array([[1/4,1/3],[3/4,2/3]])
    >>> B2 = normalize_adj(A, 'col')
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))

    >>> B = np.array([[0.28867513,0.47140452],[0.56694671,0.6172134]])
    >>> B2 = normalize_adj(A, 'sym', assume_symmetric_input=False)
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))

    >>> B = np.array([[0.33333333,0.43643578],[0.65465367,0.57142857]])
    >>> B2 = normalize_adj(A, 'sym', assume_symmetric_input=True)
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))

    >>> B = np.array([[0.50480769,0.49519231],[0.49519231,0.50480769]])
    >>> B2 = normalize_adj(A, 'ds')
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))
    
    >>> B = np.reshape(B, (1,2,1,2,1))
    >>> B2 = normalize_adj(np.reshape(A,(1,2,1,2,1)), 'ds', axis1=1, axis2=3)
    >>> np.testing.assert_array_almost_equal(B, sess.run(B2))
    
    """
    A = _tf_utils.as_tensor_or_sparse_tensor_or_seq(A)
    if isinstance(A, (tf.Tensor, tf.SparseTensor)):
        ndims = A.get_shape().ndims
    else:
        ndims = A[0].get_shape().ndims
    
    method = _utils.validate_option(
        method,
        ['row', 'column', 'col', 'sym', 'symmetric',
         'ds', 'dsm', 'doubly_symmetric'],
        'method', str_normalizer=str.lower)

    axis1,axis2 = _utils.validate_axis([axis1,axis2], ndims)

    if isinstance(A, (tuple,list)):
        return type(A)(
            _normalize_adj_1(a, method=method, axis1=axis1, axis2=axis2,
                             eps=eps,
                             assume_symmetric_input=assume_symmetric_input)
            for a in A)
    else:
        return _normalize_adj_1(
            A, method=method, axis1=axis1, axis2=axis2, eps=eps,
            assume_symmetric_input=assume_symmetric_input)

def _adj_to_laplacian_1(A, adj_norm='sym', axis1=-2, axis2=-1, eps=0.0,
                        assume_symmetric_input=True):
    shape = A.get_shape()
    shape.assert_is_fully_defined()
    shape = shape.as_list()
    K = shape[axis1]
    if K != shape[axis2]:
        raise ValueError('axis1 and axis2 must have the same length.')

    I = tf.sparse.eye(K, dtype=A.dtype)
    shape_I = [1] * len(shape)
    shape_I[axis1] = K
    shape_I[axis2] = K
    I = tf.sparse_reshape(I, shape_I)

    A = normalize_adj(A, adj_norm, axis1=axis1, axis2=axis2, eps=eps,
                      assume_symmetric_input=assume_symmetric_input)
    A = tf.negative(A)
    return tf.sparse_add(I, A)
    
    
def adj_to_laplacian(A, adj_norm='sym', *, axis1=-2, axis2=-1, eps=0.0,
                     assume_symmetric_input=True):
    """Calculate Laplacian(s) from adjacency matrix (matrices)
    
    Args:

      A (any-tensor): Input adjacency matrix or matrices. 

      adj_norm (str): Adjacency matrix normalization method. See
        :func:`normalize_adj` for details.

      axis1 (int): Same as in :func:`normalize_adj`.

      axis2 (int): Same as in :func:`normalize_adj`.

      eps (float): Same as in :func:`normalize_adj`.

      assume_symmetric_input (bool): Same as in :func:`normalize_adj`.

    Returns:

    any-tensor: Laplacian matrix or matrices.

    Examples:

    >>> A = np.array([[1,2],[3,4]], dtype=np.float)
    >>> L = np.array([[2/3, -2/3], [-3/7,3/7]])
    >>> L2 = adj_to_laplacian(A, 'row')
    >>> sess = tf.Session()
    >>> np.testing.assert_array_almost_equal(L, sess.run(L2))

    >>> # axis1 and axis 2 must have the same length
    >>> adj_to_laplacian(np.random.random((3,2)), 'row')
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> L2 = adj_to_laplacian(A.astype(np.int), 'row')
    >>> np.testing.assert_array_almost_equal(L, sess.run(L2))

    >>> import scipy.sparse
    >>> sA = scipy.sparse.coo_matrix(A)
    >>> L2 = adj_to_laplacian(sA, 'row')
    >>> isinstance(L2, tf.SparseTensor)
    True
    >>> import awesomeml as aml
    >>> L2 = aml.utils.as_numpy_array(sess.run(L2))
    >>> np.testing.assert_array_almost_equal(L, L2)

    >>> L21, L22 = adj_to_laplacian([sA, sA], 'row')
    >>> isinstance(L21, tf.SparseTensor)
    True
    >>> isinstance(L22, tf.SparseTensor)
    True
    >>> L2 = aml.utils.as_numpy_array(sess.run(L21))
    >>> np.testing.assert_array_almost_equal(L, L2)
    >>> L2 = aml.utils.as_numpy_array(sess.run(L22))
    >>> np.testing.assert_array_almost_equal(L, L2)
    """
    A = _tf_utils.as_tensor_or_sparse_tensor_or_seq(A)
    if (isinstance(A, (tf.Tensor, tf.SparseTensor)) and
        not A.dtype.is_floating):
        A = tf.to_float(A)
    
    if isinstance(A, (tuple, list)):
        return type(A)(
            _adj_to_laplacian_1(
                a, adj_norm=adj_norm, axis1=axis1, axis2=axis2, eps=eps,
                assume_symmetric_input=assume_symmetric_input)
            for a in A)
    else:
        return _adj_to_laplacian_1(
            A, adj_norm=adj_norm, axis1=axis1, axis2=axis2, eps=eps,
            assume_symmetric_input=assume_symmetric_input)
    
    

