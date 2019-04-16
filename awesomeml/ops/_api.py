# -*- coding: utf-8 -*-

import scipy.sparse
import numpy as np

from .. import config as _config
from .. import utils as _utils


def matmul(a, b, array_mode=None, array_default_mode='numpy',
           array_homo_mode=None):
    """Similar to np.matmul, but can handle sparse arrays
    
    Args:

      a (any_array): First input array.

      b (any_array): Second input array.

    Return:

      sparse_array: matrix multiplication of a and b.

    Examples:

    >>> a = np.random.random((3,4))
    >>> b = np.random.random((4,3))
    >>> c1 = matmul(a, b, array_mode='numpy')
    >>> isinstance(c1, np.ndarray)
    True
    >>> c2 = matmul(a, b, array_mode='scipy-sparse')
    >>> scipy.sparse.issparse(c2)
    True
    >>> c3 = matmul(a, b, array_mode='pydata-sparse')
    >>> import sparse
    >>> isinstance(c3, sparse.SparseArray)
    True
    >>> np.testing.assert_array_almost_equal(c1, c2.todense())
    >>> np.testing.assert_array_almost_equal(c1, c3.todense())
    """
    a, b = _utils.validate_array_args(a, b,
                                      mode=array_mode,
                                      default_mode=array_default_mode,
                                      homo_mode=array_homo_mode)
    if _config.has_package('sparse') and _utils.is_pydata_sparse(a):
        import sparse
        c = sparse.matmul(a, b)
    elif scipy.sparse.issparse(a):
        c = a @ b
    else:
        c = np.matmul(a, b)
    return c


def concatenate(arrays, axis=0, array_mode=None, array_default_mode='numpy',
                array_homo_mode=None):
    """Join a sequence of arrays along an existing axis.

    Args:

      arrays (Seq[array]): Input arrays

      axis (int): The axis along which the arrays will be joined. If
        axis is None, arrays are flattened before use. Default is 0.

      array_mode (str, NoneType): See :func:`validate_array_args`.

      array_default_mode (str, NoneType): See :func:`validate_array_args`.

      array_homo_mode (str, NoneType): See :func:`validate_array_args`.

    Returns:

    The concatenated array.

    Examples:

    >>> a = np.random.random((3, 4))
    >>> b = np.random.random((3, 4))
    >>> c1 = np.concatenate((a,b))
    >>> c2 = concatenate((a,b))
    >>> isinstance(c2, np.ndarray)
    True
    >>> np.all(c1 == c2)
    True

    >>> c2 = concatenate((a,b), array_mode='scipy-sparse')
    >>> scipy.sparse.issparse(c2)
    True
    >>> c3 = concatenate((a,b), array_mode='pydata-sparse')
    >>> import sparse
    >>> isinstance(c3, sparse.SparseArray)
    True
    >>> np.all(c1 == c2.todense())
    True
    >>> np.all(c1 == c3.todense())
    True

    >>> c1 = np.concatenate((a,b), axis=1)
    >>> c2 = concatenate((a,b), array_mode='scipy-sparse', axis=1)
    >>> scipy.sparse.issparse(c2)
    True
    >>> np.all(c1 == c2.todense())
    True
    """
    arrays = _utils.validate_array_args(*arrays,
                                      mode=array_mode,
                                      default_mode=array_default_mode,
                                      homo_mode=array_homo_mode)
    if all(scipy.sparse.issparse(a) for a in arrays):
        axis = axis+2 if axis < 0 else axis
        if axis == 0:
            c = scipy.sparse.vstack(arrays)
        else:
            assert axis == 1
            c = scipy.sparse.hstack(arrays)
    elif (_config.has_package('sparse') and
          all(_utils.is_pydata_sparse(a) for a in arrays)):
        import sparse
        c = sparse.concatenate(arrays, axis)
    else:
        c = np.concatenate(arrays, axis)
    return c



def transpose(a, axes=None, array_mode=None, array_default_mode='numpy',
              array_homo_mode=None):
    """Transpose an array or sparse array
    
    Arguments:

      a: array_like or sparse array. Input array.

      axes: list of ints, optional. By default, reverse the
        dimensions, otherwise permute the axes according to the values
        given.
    
    Returns:

    Tranposed array or sparse array

    Examples:

    >>> a = np.random.random((3,4))
    >>> b = np.transpose(a)
    >>> b2 = transpose(a, array_mode='scipy')
    >>> scipy.sparse.issparse(b2)
    True
    >>> np.all(b == b2.todense())
    True

    >>> b2 = transpose(a, array_mode='pydata')
    >>> import sparse
    >>> isinstance(b2, sparse.SparseArray)
    True
    >>> np.all(b == b2.todense())
    True

    >>> b2 = transpose(a, array_mode='scipy', axes=[0, 1])
    >>> scipy.sparse.issparse(b2)
    True
    >>> np.all(a == b2.todense())
    True

    >>> b2 = transpose(a, array_mode='scipy', axes=[1, 0])
    >>> scipy.sparse.issparse(b2)
    True
    >>> np.all(b == b2.todense())
    True

    >>> transpose(a, array_mode='scipy', axes=[1,1])
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    a = _utils.validate_array_args(a, mode=array_mode,
                                   default_mode=array_default_mode,
                                   homo_mode=array_homo_mode)
                                   
    if scipy.sparse.issparse(a):
        if axes is None:
            return a.T
        else:
            axes = _utils.validate_axis(axes, 2)
            if len(np.unique(axes)) < len(axes):
                raise ValueError('axis repeated in transpose.')
            if axes[0] == 0 and axes[1] == 1:
               return a
            else:
               return a.T
    else:
        return np.transpose(a, axes)


def swapaxes(a, axis1, axis2, array_mode=None, array_default_mode='numpy',
             array_homo_mode=None):
    """Interchange two axes of array or sparse array

    Args:

      axis1 (int): The first axis to swap.
    
      axis2 (int): The second axis to swap.

    Examples:

    >>> a = np.random.random((1,2,3,4))
    >>> axis1, axis2 = 1, 3
    >>> b = np.swapaxes(a, axis1, axis2)
    >>> b2 = swapaxes(a, axis1, axis2, 'pydata')
    >>> import sparse
    >>> isinstance(b2, sparse.SparseArray)
    True
    >>> np.all(b == b2.todense())
    True
    """
    ndim = np.ndim(a)
    axis1 = _utils.validate_axis(axis1, ndim)
    axis2 = _utils.validate_axis(axis2, ndim)

    if axis1 == axis2:
        return a

    axes = list(range(ndim))
    axes[axis1] = axis2
    axes[axis2] = axis1
    return transpose(a, axes, array_mode=array_mode,
                     array_default_mode=array_default_mode,
                     array_homo_mode=array_homo_mode)


def unstack(a, axis=0, array_mode=None, array_default_mode='numpy',
            array_homo_mode=None):
    """Unpack the given dimesnion of a rank-N pydata sparse array to
    rank-(N-1)
    
    Arguments:

      a: a pydata sparse array

      axis (int): the axis along which to unstack
    
    Returns:

    A list of pydata sparse arrays unstacked from x

    Examples:

    >>> a = np.random.random((2,2))
    >>> b, c = unstack(a, axis=1)
    >>> b2, c2 = unstack(a, array_mode='scipy', axis=1)
    >>> scipy.sparse.issparse(b2) and scipy.sparse.issparse(c2)
    True
    >>> np.all(b == b2.todense())
    True
    >>> b3, c3 = unstack(a, array_mode='pydata', axis=1)
    >>> import sparse
    >>> isinstance(b3, sparse.SparseArray)
    True
    >>> isinstance(c3, sparse.SparseArray)
    True
    """
    a = _utils.validate_array_args(a, mode=array_mode,
                                   default_mode=array_default_mode,
                                   homo_mode=array_homo_mode)
    
    rank = len(a.shape)
    axis = _utils.validate_axis(axis, rank)
    if axis != 0:
        # move axis to the front-most
        perm = [axis] + sorted(set(range(rank)) - set([axis]))
        a = transpose(a, perm)
    if scipy.sparse.issparse(a):
        a = a.tocsr()
    return list(a)


def sum(a, *args, array_mode=None, array_default_mode='numpy',
        array_homo_mode=None, **kwargs):
    """Same as np.sum, but can handle `keepdims` argument for scipy sparse
    matrix

    Examples:

    >>> a = np.random.random((3,4))
    >>> b = np.sum(a, axis=0, keepdims=True)
    >>> b2 = sum(scipy.sparse.coo_matrix(a), axis=0, keepdims=True)
    >>> np.all(b == b2)
    True

    >>> b2 = sum(scipy.sparse.coo_matrix(a), 0, None, None, True)
    >>> np.all(b == b2)
    True

    >>> b = np.sum(a, axis=0, keepdims=False)
    >>> b2 = sum(scipy.sparse.coo_matrix(a), 0, None, None, False)
    >>> np.all(b == b2)
    True
    """
    a = _utils.validate_array_args(a, mode=array_mode,
                                   default_mode=array_default_mode,
                                   homo_mode=array_homo_mode)
    if scipy.sparse.issparse(a):
        if len(args) > 3:
            keepdims = args[3]
            args = list(args)
            del args[3]
        else:
            keepdims = kwargs.pop('keepdims', False)

        if len(args) > 0:
            axis = args[0]
            args = list(args)
            del args[0]
        else:
            axis = kwargs.pop('axis', None)

        res = np.sum(a, axis, *args, **kwargs)
            
        if keepdims:
            return np.asarray(res)
        else:
            if axis is not None:
                res = np.asarray(res).squeeze(axis)
    else:
        res = np.sum(a, *args, **kwargs)
    return res



def multiply(a, b, array_mode=None, array_default_mode='numpy',
             array_homo_mode=None):
    """Element-wise multiplication which can handle sparse matrices

    Args:

      a (array-like): First operand of multiplication.

      b (array-like): Second operand of multiplication.

      array_mode, array_default_mode, array_homo_mode (str, NoneType):
        See :func:`utils.validate_array_args`.

    Examples:

    >>> a = np.random.random((3,4))
    >>> b = np.random.random((3,4))
    >>> c = a * b

    >>> c2 = multiply(a, b)
    >>> np.all(c == c2)
    True

    >>> c2 = multiply(a, b, array_mode='scipy')
    >>> import scipy.sparse
    >>> scipy.sparse.issparse(c2)
    True
    >>> np.all(c == c2)
    True

    >>> c2 = multiply(scipy.sparse.coo_matrix(a), b)
    >>> scipy.sparse.issparse(c2)
    True
    >>> np.all(c == c2)
    True

    >>> c2 = multiply(a, scipy.sparse.coo_matrix(b))
    >>> scipy.sparse.issparse(c2)
    True
    >>> np.all(c == c2)
    True
    """
    a, b = _utils.validate_array_args(a, b, mode=array_mode,
                                      default_mode=array_default_mode,
                                      homo_mode=array_homo_mode)

    if scipy.sparse.issparse(a):
        return a.multiply(b)
    if scipy.sparse.issparse(b):
        return b.multiply(a)
    return a * b
