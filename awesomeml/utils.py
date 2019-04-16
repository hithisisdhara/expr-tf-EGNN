# -*- coding: utf-8 -*-
"""
Utilities: argument validation, array type check and conversion etc.
"""
import urllib
import pathlib
import collections.abc
import scipy.sparse
import decoutils

import numpy as np
from pathlib import Path
from . import config as _config

def validate_seq(value,
                 n=None,
                 name='value',
                 *,
                 seq_type=None,
                 element_types=None,
                 element_converter=None,
                 repeat_scalar=True,
                 accept_scalar=True,
                 str_as_scalar=True):

    """Try to convert argument to required sequence type

    Arguments:

      value: Argument value.

      n (int, Nonetype): required number of elements of the
        argument. If n is None, any number of elements is
        valid. Defaults to `None`.

      name (str): The name of the argument. Defaults to "value".

      seq_type (type): Type of sequence to return. If None
        and `value` is a sequence, use the type of `value`. If None
        and `value` is a scalar, use `tuple`. Default to None.

      element_types (type, Iterable[type]): Allowed element types. If
        None, do not check element type.

      element_converter (callable, NoneTyep): A callable to convert
        elements to the required type. If None, will not convert
        sequence element. Defaults to `None`.

      accept_scalar (bool): Whether accept scalar as length 1
        sequence. Default to True.

      str_as_scalar (bool): Whether treat str as collection or
        scalar. Defaults to `True`.

      repeat_scalar (bool): If the argument value is a scalar, repeat
        it to create a sequence or not. Defaults to True. Will be
        ignored if accept_scalar is False.
    
    Returns:

      Sequence: A sequence with type of `seq_type`, which has
      been validated.

    Raises:

      ValueError: If length of `value` is not satisfied

      TypeError: If element type is not satisfied, or failed to be
        converted, or `value` is a scalar but accept_scalar is
        `False`, or required return sequence type is `str` but the
        elements are not converted to `str`.

    Examples:

      >>> validate_seq((1, 1))
      (1, 1)
      >>> validate_seq((1, 1), 2)
      (1, 1)
      >>> validate_seq((1, 1), 3)
      Traceback (most recent call last):
        ...
      ValueError: ...
      >>> validate_seq((1, 1), 2, element_types=int)
      (1, 1)
      >>> validate_seq((1, 1), 2, element_types=float)
      Traceback (most recent call last):
        ...
      TypeError: ...
      >>> validate_seq((1, 1), 2, element_converter=str)
      ('1', '1')
      >>> validate_seq((1, 1), 2, seq_type=list)
      [1, 1]
      >>> validate_seq(('a','b'), element_converter=int)
      Traceback (most recent call last):
        ...
      TypeError: ...
      >>> validate_seq((1, 1), seq_type=str,
      ... element_converter=str)
      '11'
      >>> validate_seq((1, 1), seq_type=str)
      Traceback (most recent call last):
        ...
      TypeError: ...
      >>> validate_seq(1, accept_scalar=True)
      (1,)
      >>> validate_seq(1, accept_scalar=False)
      Traceback (most recent call last):
        ...
      TypeError: ...
      >>> validate_seq(1, 2, accept_scalar=True, repeat_scalar=True)
      (1, 1)

    See also:

      :func:`validate_int_seq`,
      :func:`validate_str_seq`, :func:`validate_str`

    """

    # process scalar input
    if (isinstance(value, collections.abc.Iterable) and
        not (str_as_scalar and isinstance(value, str))):
        valid_value = tuple(value)
        if seq_type is None:
            seq_type = type(value) if isinstance(value, collections.abc.Collection) else tuple
    else:
        if not accept_scalar:
            raise TypeError('The `'+name+'` argument should be a sequence'
                            '. Received: '+str(value)+'. Note that '
                            'str_as_scalar is '+str(str_as_scalar)+'.')
        valid_value = (value,)
        if seq_type is None:
            seq_type = tuple
        if n is not None and repeat_scalar:
            valid_value *= n

    # check length
    if n and len(valid_value) != n:
        raise ValueError('The `'+name+'` argument does not has '+str(n)+
                         ' elements. Received: '+str(value))
    # check element types
    if element_types is not None:
        for element in valid_value:
            if not isinstance(element_types, collections.abc.Iterable):
                element_types = (element_types,)
            if type(element) not in element_types:
                raise TypeError('The `'+name+'` argument should be '
                                'sequence of types:'+str(element_types)+
                                '. Received: '+str(value))
            
    # convert elements
    if element_converter is not None:
        try:
            valid_value = tuple(element_converter(x) for x in valid_value)
        except (TypeError,ValueError):
            raise TypeError('The `'+name+'` argument has elements which can'
                             'not be converted. Received: '+str(value))
    # convert to required sequence type
    if seq_type is str:
        # str is a special sequence type, need to be treated carefully
        for x in valid_value:
            if not isinstance(x,str):
                raise TypeError(
                    'For required sequence type `str`, the elements of '
                    'the `'+name+'` argument should be converted to `str`'
                    '. However, element_converter:'+str(element_converter)+
                    'does not produce `str`.')
        return ''.join(valid_value)
    else:
        return seq_type(valid_value)


def validate_int_seq(value,
                            n=None,
                            name='value',
                            *,
                            seq_type=None,
                            element_types=int,
                            accept_scalar=True,
                            str_as_scalar=True,
                            repeat_scalar=True):
    """Check an argument and convert it to a collection of integers

    Arguments:

      value: Argument value.

      n (int, Nonetype): required number of elements of the
        argument. If n is None, any number of elements is
        valid. Defaults to `None`.

      name (str): The name of the argument. Defaults to "value".

      seq_type (type): Type of sequence to return. If None
        and `value` is a sequence, use the type of `value`. If None
        and `value` is a scalar, use `tuple`. Default to None.

      element_types (type, Iterable[type]): Allowed element types. If
        None, do not check element type.

      accept_scalar (bool): Whether accept scalar as length 1
        sequence. Default to True.

      str_as_scalar (bool): Whether treat str as collection or
        scalar. Defaults to `True`.

      repeat_scalar (bool): If the argument value is a scalar, repeat
        it to create a sequence or not. Defaults to True. Will be
        ignored if accept_scalar is False.

    Returns:

      Seq[int]: A sequence of integers.
    
    Examples:

    >>> validate_int_seq((1, 2))
    (1, 2)
    """
    return validate_seq(value, n, name,
                        element_converter=int,
                        seq_type=seq_type,
                        element_types=element_types,
                        accept_scalar=accept_scalar,
                        str_as_scalar=True,
                        repeat_scalar=repeat_scalar)


def validate_int_none_seq(value,
                          n=None,
                          name='value',
                          *,
                          seq_type=None,
                          element_types=[int,type(None)],
                          accept_scalar=True,
                          str_as_scalar=True,
                          repeat_scalar=True):
    """Check an argument and convert it to a collection of integers or None

    Arguments:

      value: Argument value.

      n (int, Nonetype): required number of elements of the
        argument. If n is None, any number of elements is
        valid. Defaults to `None`.

      name (str): The name of the argument. Defaults to "value".

      seq_type (type): Type of sequence to return. If None and `value`
        is a sequence, use the type of `value`. If None and `value` is
        a scalar, use `tuple`. Default to None.

      element_types (type, Iterable[type]): Allowed element types. If
        None, do not check element type.

      accept_scalar (bool): Whether accept scalar as length 1
        sequence. Default to True.

      str_as_scalar (bool): Whether treat str as collection or
        scalar. Defaults to `True`.

      repeat_scalar (bool): If the argument value is a scalar, repeat
        it to create a sequence or not. Defaults to True. Will be
        ignored if accept_scalar is False.

    Returns:

    Seq[int]: A sequence of integers.
    
    Examples:

    >>> validate_int_none_seq((1, None))
    (1, None)

    """
    converter = lambda x: None if x is None else int(x)
    return validate_seq(value, n, name,
                        element_converter=converter,
                        seq_type=seq_type,
                        element_types=element_types,
                        accept_scalar=accept_scalar,
                        str_as_scalar=True,
                        repeat_scalar=repeat_scalar)


def validate_str_seq(value, n=None, name='value',
                            normalizer=None,
                            seq_type=None,
                            accept_scalar=True,
                            repeat_scalar=True):
    """Make sure an argument is a collection of strs

    Args:

      value: Argument value.

      n (int, Nonetype): required number of elements of the
        argument. If n is None, any number of elements is
        valid. Defaults to `None`.

      name (str): The name of the argument. Defaults to "value".
    
      normalizer (callable, NoneType): Str normalize method. If None,
        no normalization will be performed. Default to None.

      seq_type (type): Type of collection to return. If None
        and `value` is a sequence, use the type of `value`. If None
        and `value` is a scalar, use `tuple`. Default to None.

      accept_scalar (bool): Whether accept scalar as length 1
        sequence. Default to True.

      repeat_scalar (bool): If the argument value is a scalar, repeat
        it to create a sequence or not. Defaults to True. Will be
        ignored if accept_scalar is False.
    
    Return:

      Seq[str]: A sequence of strs.

    Examples:
    
      >>> validate_str_seq('a', 2)
      ('a', 'a')

    See also:

      :func:`validate_str`

    """
    return validate_seq(value, n, name,
                        element_types=str,
                        element_converter=normalizer,
                        seq_type=seq_type,
                        accept_scalar=accept_scalar,
                        repeat_scalar=repeat_scalar)


def validate_str(value, name='value', accept_none=False,
                 normalizer=str.lower):
    """Check str type argument
    
    Args:

      value: argument value

      name (str): argument name

      accept_none (bool): True if None is also a valid value, False
        otherwise. Default to False.

      normalizee (callable, NoneTyep): Str normalizer. Default to str.lower.

    Raises:

      TypeError: if `value` is not a `str`

    Examples:

      >>> validate_str('abc')
      'abc'

      >>> validate_str('abc', normalizer=None)
      'abc'

      >>> validate_str('AbC', normalizer=str.lower)
      'abc'

      >>> validate_str(None, accept_none=True)

      >>> validate_str(None, accept_none=False)
      Traceback (most recent call last):
        ...
      TypeError: ...

      >>> validate_str('a', normalizer=1.023)
      Traceback (most recent call last):
        ...
      ValueError: ...

    See also: 

      :func:`validate_option_argument`, :func:`validate_str_seq`.

    """
    if value is None and accept_none:
        return value

    if not isinstance(value, str):
        raise TypeError('Argument '+name+' is not a str. Received type:'+
                        str(type(value)))

    if normalizer is None:
        pass
    elif callable(normalizer):
        value = normalizer(value)
    else:
        raise ValueError('Normalizer should be a callable')
    return value


def validate_option(value, options, name='value', str_normalizer=None):
    """Validate arguments to make sure it is in a given options
    
    Arguments:

      value: The value of the argument to validate

      options (Iterable): Allowed values for the argument.

      name (str): Name of the argument (for error message
        use). Default to 'value'.

      str_normalizer (callable, NoneType): Method to normalize str
        before comparing. If None, no normalization will be
        performed. Default to None.

    Return:
      
      A valid option
    
    Raises:

      ValueError: If value is not in options, or str_normalizer is
        neither None nor callable.

    Examples:

    >>> validate_option('Abc', ['aBc', None], str_normalizer=str.lower)
    'abc'
    >>> validate_option('abc', ['def', None])
    Traceback (most recent call last):
      ...
    ValueError: Value is not in options.
    >>> validate_option('abc', ['abc', None], str_normalizer=1.02)
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    if isinstance(value, str):
        if str_normalizer is None:
            pass
        elif callable(str_normalizer):
            value = str_normalizer(value)
        else:
            raise TypeError('str_normalizer must be None or callable')

    options = frozenset(str_normalizer(x) if isinstance(x,str)
                        and str_normalizer is not None
                        else x for x in options)
        
    if value not in options:
        raise ValueError('Value is not in options.')

    return value



def validate_axis(axis, ndims=None, name='axis',
                  accept_none=False,
                  scalar_to_seq=False,
                  sequence_type=None):
    """Check axis argument
    
    Arguments:

      axis (int, Seq[int]): axis argument value

      ndims (int): the number of total dimensions, aka rank

      name (str): The name of the argument, Defaults to 'axis'.

      accept_none (bool): Whether accept None as valide value of axis,
      defaults to False.

      scalar_to_seq (bool): Whether convert a scalar input to a
        collection. This is useful when some function accept a
        collection of axis only. Default to False.
    
    Returns:

      int, Seq[int], None: validated axis (axes), could be:

        * None: if axis is None and accept_none is True

        * int: if axis is int and scalar_to_sequence is False

        * collection of int: if axis is sequence of int, or axis is int
          and scalar_to_sequence is True
    
    Raises:

      TypeError: if axis is None but accept_none is False

    Examples:

    >>> validate_axis(1)
    1
    >>> validate_axis(-1)
    -1
    >>> validate_axis(-1, 2)
    1
    >>> validate_axis(None, accept_none=True) is None
    True
    >>> validate_axis(None, accept_none=False)
    Traceback (most recent call last):
      ...
    TypeError: ...
    >>> validate_axis([-1, -2], 3)
    [2, 1]
    >>> validate_axis([4, 5], 3)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> validate_axis(5, 3)
    Traceback (most recent call last):
      ...
    ValueError: ...

    """
    if axis is None:
        if accept_none:
            return axis
        else:
            raise TypeError('The `'+name+' argument is None, but accept_none'
                            ' is False.')

    assert ndims is None or ndims > 0
    if isinstance(axis, int) and not scalar_to_seq:
        if ndims is not None:
            if axis < -ndims or axis >= ndims:
                raise ValueError('The `'+name+' argument should be in range'+
                                 '['+str(-ndims)+','+str(ndims-1)+']. '
                                 'Received:'+str(axis))
            if axis < 0:
                axis += ndims
        return axis
    
    axis = validate_int_seq(axis, name=name)
    if ndims is not None and any(i<-ndims or i >=ndims for i in axis):
        raise ValueError('The `'+name+' argument should be in range'+
                         '['+str(-ndims)+','+str(ndims-1)+']. Received:'+str(axis))
    if ndims is not None:
        axis = [i+ndims if i<0 else i for i in axis]
    return axis



def validate_data_format(value, name='data_format'):
    """Check data_format argument
    
    Examples:

    >>> validate_data_format('channels_first')
    'channels_first'
    """
    return validate_option(
        value=value,
        options=('channels_first', 'channels_last'),
        name=name,
        str_normalizer=str.lower)


def validate_array_mode(mode):
    """Check and normalize the array_mode argument for APIs

    Args:

      mode (str): Value of the array_mode argument

    Return:

      str: Valid value of mode argument.

    Examples:

    >>> validate_array_mode('sparse') # suppose sparse package was installed
    'pydata-sparse'
    >>> validate_array_mode('scipy_sparse')
    'scipy-sparse'
    >>> validate_array_mode('array')
    'numpy'
    >>> validate_array_mode('scipy')
    'scipy-sparse'
    >>> validate_array_mode('pydata')
    'pydata-sparse'

    See also:

    :func:`validate_array_args`
    """
    if mode is not None:
        mode = mode.replace('_', '-')
        mode = mode.replace('.', '-')
    opts = ['sparse', 'numpy', 'array', 'scipy-sparse', 'pydata-sparse',
            'scipy', 'pydata', None]
    mode = validate_option(mode, opts, 'mode', str.lower)
    # normalize
    if mode == 'array':
        mode = 'numpy'
    if mode == 'scipy':
        mode = 'scipy-sparse'
    if mode == 'pydata':
        mode = 'pydata-sparse'
    if mode == 'sparse':
        if _config.has_package('sparse'):
            mode = 'pydata-sparse'
        else: # pragma: no cover
            mode = 'scipy-sparse'
    return mode



def validate_array_args(*args, mode=None, default_mode='numpy',
                        homo_mode=None):
    """Converting args to specified array types

    The function will do the following:

    1. If mode is not None, convert each item in args to array type
       specified by mode.

    2. If mode is None and default_mode is not None, convert items
       that are not any sparse array or array to array type specified
       by default_mode.

    3. If mode is None and homo_mode is not None and items of args
       does not have homogeneous array type (e.g. some of them are
       pydata sparse array, but others are scipy sparse matrices),
       convert all items to array type specified by homo_mode.

    Arrays are classified into four types:
    
    * numpy arrays
    * scipy sparse matrices
    * pydata sparse arrays
    * not array (i.e. other types)
    
    Args:

      *args: Arguments need to be validated.

      mode (str, NoneType): Array mode, arguments will be converted to
        a specific type (i.e. scipy.sparse.coo_matrix, sparse.COO or
        np.ndarray) according to the value of mode. Possible values:

        * 'scipy-sparse', 'scipy': Convert argument to
          scipy.sparse.coo_matrix.

        * 'pydata-sparse', 'pydata': Convert argument to pydata sparse
          arrays.

        * 'numpy', 'array': Convert argument to numpy arrays.

        * 'sparse': Convert to pydata sparse if the package has been
          installed, otherwise scipy.sparse.coo_matrix.

        * None: Leave the argument unchanged.

      default_mode (str, NoneType): Default array mode, arguments that
        are not arrays will be converted to array type specified by
        this value.

      homo_mode (str, NoneType): Array homogenize mode, arguments will
        be converted to a specific type if they are not
        homegeneous. Possible values are the same as mode.

    
    Returns:

    list, object: validated args.

    Examples:

    >>> args = [np.array([1]), scipy.sparse.coo_matrix(np.array([1])), [1]]
    >>> args2 = validate_array_args(*args, mode='scipy')
    >>> all(scipy.sparse.issparse(x) for x in args2)
    True

    >>> args2 = validate_array_args(*args, mode='pydata')
    >>> all(is_pydata_sparse(x) for x in args2)
    True

    >>> args2 = validate_array_args(*args, mode='numpy')
    >>> all(isinstance(x, np.ndarray) for x in args2)
    True

    >>> args2 = validate_array_args(*args, default_mode='numpy')
    >>> isinstance(args2[0], np.ndarray)
    True
    >>> scipy.sparse.issparse(args2[1])
    True
    >>> isinstance(args2[2], np.ndarray)
    True

    >>> args2 = validate_array_args(*args, default_mode='scipy')
    >>> scipy.sparse.issparse(args2[2])
    True

    >>> args2 = validate_array_args(*args, default_mode='pydata')
    >>> import sparse
    >>> isinstance(args2[2], sparse.SparseArray)
    True

    >>> args2 = validate_array_args(*args, homo_mode='scipy')
    >>> all(scipy.sparse.issparse(x) for x in args2)
    True
    """
    mode = validate_array_mode(mode)
    default_mode = validate_array_mode(default_mode)
    homo_mode = validate_array_mode(homo_mode)

    if mode == 'scipy-sparse':
        args = [x if scipy.sparse.issparse(x) else as_scipy_coo(x)
                for x in args]
    elif mode == 'pydata-sparse':
        _config.assert_has_package('sparse')
        args = [x if is_pydata_sparse(x) else as_pydata_coo(x) for x in args]
    elif mode == 'numpy':
        args = [as_numpy_array(x) for x in args]
    else:
        assert mode is None
        if default_mode == 'scipy-sparse':
            args = [x if is_any_array(x) else as_scipy_coo(x) for x in args]
        elif default_mode == 'pydata-sparse':
            args = [x if is_any_array(x) else as_pydata_coo(x) for x in args]
        elif default_mode == 'numpy':
            args = [x if is_any_array(x) else np.asarray(x) for x in args]
        else: # pragma: no cover
            assert default_mode is None

        if (homo_mode is not None and
            not all(isinstance(x, np.ndarray) for x in args) and
            not all(scipy.sparse.issparse(x) for x in args) and
            not (_config.has_package('sparse') and
                 all(is_pydata_sparse(x) for x in args))):
            args = validate_array_args(*args, mode=homo_mode)
            
    if len(args) == 1:
        args = args[0]
    return args


def validate_dir(path, default_path=None, create=False, create_default=True,
                 name='path'):
    """Validate dir path

    Args：

      path (path-like): Path to the directory which need to check.

      default_path (path-like): If path is None, use this as a
        fall-back value.

      create (bool): If path is not a valid dir, create it?

      create_default (bool): If path is None and default_path is not a
        valid dir, create default_path?

      name (str): Name of the argument. Default to 'path'.

    Returns:

    path-like: Validated path to the dir.

    >>> import tempfile
    >>> import uuid
    >>> validate_dir(None, default_path=None)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> p = Path(tempfile.gettempdir())/str(uuid.uuid4())
    >>> validate_dir(None, default_path=p, create_default=False)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> validate_dir(None, default_path=p, create_default=True) == p
    True
    >>> p = Path(tempfile.gettempdir())/str(uuid.uuid4())
    >>> validate_dir(p, create=False)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> validate_dir(p, create=True) == p
    True
    """
    if path is None:
        if default_path is None:
            raise ValueError('path and default_path are all None.')
        else:
            path = Path(default_path)
            if not path.is_dir():
                if create_default:
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    raise ValueError('default_path:', default_path,
                                     ' is not a valid dir.')
    else:
        path = Path(path)
        if not path.is_dir():
            if create:
                path.mkdir(parents=True)
            else:
                raise ValueError('path:', path, ' is not a valid dir.')
    return path


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
    return validate_dir(value, default_path=_config.DATA_HOME,
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
    return validate_dir(value, default_path=_config.MODEL_HOME,
                        create=create, name=name)


def is_any_array(a):
    """Check if a is any valid kind of array, including sparse arrays
    """
    return is_sparse(a) or isinstance(a, np.ndarray)



def assert_is_any_array(a):
    """Assert that a is an array or sparse array

    Examples:

    >>> assert_is_any_array(np.random.random((3, 4)))
    >>> assert_is_any_array(scipy.sparse.random(3, 4, 0.5))
    >>> import sparse
    >>> assert_is_any_array(sparse.random((3, 4), 0.5))
    >>> assert_is_any_array('abc')
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    if not is_any_array(a):
        raise TypeError('a is not an array or sparse array.')


def is_sparse(a):
    """Is a a sparse array/matrix type?

    Arguments:

      a: object to check for being a sparse array/matrix type
    
    Returns:

    bool: True if a is a sparse array/matrix, False otherwise

    Examples:
    
    >>> import sparse
    >>> is_sparse(sparse.random((3,4), 0.5))
    True
    >>> is_sparse(scipy.sparse.random(3, 4, 0.5))
    True
    >>> is_sparse([1,2])
    False

    >>> import tensorflow as tf
    >>> a = tf.SparseTensorValue(indices=[[0,0],[1,1]], values=[1,1],
    ...                          dense_shape=(2,2))
    >>> is_sparse(a)
    True
    """
    if _config.has_package('sparse') and is_pydata_sparse(a):
        return True
    if _config.has_package('tensorflow') and is_tf_sparse(a):
        return True
    return scipy.sparse.issparse(a)


def assert_is_sparse(a):
    """Raise an error if a is not a sparse array/matrix
    
    Raises:

      TypeError: if a is not a sparse array/matrix, False otherwise

    Examples:

    >>> assert_is_sparse(scipy.sparse.random(3, 4, 0.8))
    >>> assert_is_sparse([1, 2])
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    if not is_sparse(a):
        raise TypeError('a is not a sparse array/matrix.')



def is_scipy_sparse(a):
    """Alias of scipy.sparse.issparse

    Examples:

    >>> is_scipy_sparse(scipy.sparse.random(3, 4, 0.2))
    True
    """
    return scipy.sparse.issparse(a)


def assert_is_scipy_sparse(a):
    """Raise an error if a is not a scipy sparse matrix
    
    Raises:

      TypeError: if a is not a scipy sparse matrix

    Examples:

    >>> assert_is_scipy_sparse(scipy.sparse.random(3, 4, 0.5))
    >>> assert_is_scipy_sparse([1,2,3])
    Traceback (most recent call last):
      ...
    TypeError: a is not a scipy sparse matrix.
    """
    if not scipy.sparse.issparse(a):
        raise TypeError('a is not a scipy sparse matrix.')



    
def is_pydata_sparse(a):
    """Is a a pydata sparse array?

    Arguments:

      a: object to check for being a pydata sparse array

    Returns:

    bool: True if a is a pydata sparse array, False otherwise

    Raises:

      RuntimeError: if pydata sparse package is not installed

    Examples:

    >>> is_pydata_sparse([1,2])
    False
    >>> import sparse
    >>> is_pydata_sparse(sparse.random((1,2,3),0.5))
    True
    """
    _config.assert_has_package('sparse')
    import sparse
    return isinstance(a, sparse.SparseArray)


def assert_is_pydata_sparse(a):
    """Raise an error if a is not a pydata sparse array
    
    Raises:

      RuntimeError: if pydata sparse package is not installed

      TypeError: if a is not a pydata sparse array

    Examples:

    >>> import sparse
    >>> assert_is_pydata_sparse(sparse.random((1,2), 0.5))
    >>> assert_is_pydata_sparse([1,2])
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    if not is_pydata_sparse(a):
        raise TypeError('a is not a pydata sparse array')


    
def is_tf_sparse(a):
    """Is a a tf.SparseTensorValue object?
    
    Args:

      a: object to check for being a tf.SparseTensorValue

    Returns:

    bool: True if a is a tf.SparseTensorValue object, False otherwise.

    Raises:

      RuntimeError: if tensorflow is not installed

    """
    _config.assert_has_package('tensorflow')
    import tensorflow as tf
    return isinstance(a, tf.SparseTensorValue)


def assert_is_tf_sparse(a):
    """Raise an error if a is not a tf.SparseTensorValue object.

    Examples:

    >>> import tensorflow as tf
    >>> sess = tf.Session()
    >>> a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], 
    ...                     dense_shape=[3, 4])
    >>> a = sess.run(a)
    >>> assert_is_tf_sparse(a)

    >>> assert_is_tf_sparse([1,2])
    Traceback (most recent call last):
      ...
    TypeError: ...
    """
    if not is_tf_sparse(a):
        raise TypeError('a is not a tf.SparseTensorValue object.')




def as_numpy_array(a):
    """Convert to a numpy array.

    Examples:

    >>> a = np.array([1,0,0])
    >>> a2 = as_numpy_array(a)
    >>> np.all(a == a2)
    True
    >>> a2 = as_numpy_array(scipy.sparse.coo_matrix(a))
    >>> np.all(a == a2)
    True
    >>> isinstance(a2, np.ndarray)
    True
    >>> import sparse
    >>> a2 = as_numpy_array(sparse.COO(a))
    >>> np.all(a == a2)
    True
    """
    if _config.has_package('tensorflow') and is_tf_sparse(a):
        a = as_scipy_coo(a)
    
    if scipy.sparse.issparse(a):
        return a.toarray()
    elif _config.has_package('sparse') and is_pydata_sparse(a):
        return a.todense()
    else:
        return np.asarray(a)


def as_scipy_coo(a):
    """Convert a to scipy coo array

    Examples:

    >>> a = np.random.random((3,4))
    >>> a2 = as_scipy_coo(a)
    >>> isinstance(a2, scipy.sparse.coo_matrix)
    True
    >>> a3 = np.asarray(a2.todense())
    >>> np.testing.assert_array_equal(a, a3)

    >>> import sparse
    >>> a = sparse.random((3, 4), 0.5)
    >>> a2 = as_scipy_coo(a)
    >>> isinstance(a2, scipy.sparse.coo_matrix)
    True
    >>> a3 = np.asarray(a2.todense())
    >>> a4 = a.todense()
    >>> np.testing.assert_array_equal(a3, a4)
    """
    if _config.has_package('sparse') and is_pydata_sparse(a):
        a = a.to_scipy_sparse()
    
    if _config.has_package('tensorflow') and is_tf_sparse(a):
        a = scipy.sparse.coo_matrix((a.values, np.asarray(a.indices).T),
                                    shape=a.dense_shape)
    return scipy.sparse.coo_matrix(a)


def as_pydata_coo(a, shape=None, fill_value=None):
    """Convert a to pydata coo array
    
    Examples:

    >>> import sparse
    >>> a = np.random.random((3, 4, 5))
    >>> a2 = as_pydata_coo(a)
    >>> isinstance(a2, sparse.COO)
    True
    >>> np.testing.assert_array_equal(a, a2.todense())

    >>> a2 = as_pydata_coo(sparse.as_coo(a))
    >>> isinstance(a2, sparse.COO)
    True
    >>> np.testing.assert_array_equal(a, a2.todense())

    >>> import tensorflow as tf
    >>> indices = [[0,0], [1,1]]
    >>> values = [1.0, 2.0]
    >>> a = tf.SparseTensorValue(indices=indices, values=values,
    ...                          dense_shape=(2,2))
    >>> a2 = as_pydata_coo(a)
    >>> isinstance(a2, sparse.COO)
    True
    >>> np.testing.assert_array_equal(a2.todense(),
    ...                               np.array([[1.0, 0], [0, 2.0]]))
    """
    _config.assert_has_package('sparse')
    import sparse
    if not (scipy.sparse.issparse(a) or isinstance(a, np.ndarray) or
            is_pydata_sparse(a)):
        a = as_scipy_coo(a)
        
    return sparse.as_coo(a, shape, fill_value)


def as_tf_sparse(a):
    """Convert a to tf.SparseTensorValue

    Args:

      a: input array

    Returns:

    SparseTensorValue: converted object.

    Examples:

    >>> # numpy input
    >>> a = np.random.random((3,4))
    >>> a2 = as_tf_sparse(a)
    >>> import tensorflow as tf
    >>> isinstance(a2, tf.SparseTensorValue)
    True
    >>> np.testing.assert_array_equal(a, as_numpy_array(a2))

    >>> a2 is as_tf_sparse(a2)
    True
    """
    _config.assert_has_package('tensorflow')
    import tensorflow as tf
    if isinstance(a, tf.SparseTensorValue):
        return a

    a = as_scipy_coo(a)
    indices = np.asarray(np.mat([a.row, a.col]).transpose())
    return tf.SparseTensorValue(indices, a.data, a.shape)
    

def is_symmetric(a, axis1=-2, axis2=-1):
    """Check if a is symmetric matrix (matrices)

    Args:

      a (any-array): Input array

      axis1 (int): The first axis of the matrices

      axis2 (int): The second axis of the matrices

    Returns:

    bool: True if matrices defined by axis1 and axis2 are symmetric,
    False otherwise.

    """
    if not is_sparse(a):
        a = as_numpy_array(a)
    ndim = np.ndim(a)
    axis0, axis1 = validate_axis([axis1, axis2], ndim)
    if axis1 == axis2:
        raise ValueError('axis1 and axis2 must be different.')

    from . import ops as _ops
    return np.sum(_ops.swapaxes(a, axis0, axis1) != a) == 0



def assert_is_symmetric(a, axis1=-2, axis2=-1):
    """Assert a is symmetric matrix (matrices)

    Args:

      a (any-array): Input array

      axis1 (int): The first axis of the matrices

      axis2 (int): The second axis of the matrices

    Raises:

      ValueError: If some of the matrices defined by axis1 and axis2
        are not symmetric.

    Examples:

    >>> A = np.array([[1,2],[2,1]])
    >>> assert_is_symmetric(A)

    >>> import scipy.sparse
    >>> assert_is_symmetric(scipy.sparse.coo_matrix(A))

    >>> import sparse
    >>> assert_is_symmetric(sparse.as_coo(A))

    >>> assert_is_symmetric(A, axis1=1, axis2=1)
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> assert_is_symmetric(np.array([[1,2],[1,2]]))
    Traceback (most recent call last):
      ...
    ValueError: ...

    """
    if not is_symmetric(a, axis1, axis2):
        raise ValueError('a is not diagonal symmetric between axis: '+
                         str(axis1)+' and axis:'+str(axis2))


        

def download(url, fpath=None, chunk_size=None): #pragma: no cover
    """Download file from url to fpath

    Args:

      url (str): URL to download.

      fpath (path-like, NoneType): Path to store the file on disk. If
        it is None, file will be stored in CWD/file_name, where
        file_name is the file name contained in url. If it is an
        existing directory, store file in this directory with file
        name extracted from url. Otherwise, fpath will be treated as
        path to a file to store the downloaded content.

      chunk_size (int, NoneType): Read this much (in bytes) content
        and write to file each time, the whole file will be downloaded
        by multiple file writing. If None, will read the whole content
        from url and write to file once.

    """
    if fpath is None:
        fpath = './'
    fpath = pathlib.Path(fpath)
    if fpath.is_dir():
        fpath /= url.rpartition('/')[2]
        
    data = urllib.request.urlopen(url)
    with open(fpath, 'wb') as f:
        while True:
            content = data.read(chunk_size)
            if not content:
                break;
            f.write(content)
    return fpath


class SafeDict(collections.abc.MutableMapping):
    """A mapping class that checks keys and values during operation

    Args:

      key_checker (callable): Boolean function which will be applied
        to the key whenever a key is used. If the checker returns
        False, a KeyError will be raised. Note that key checking is
        performed before key normalization.

      value_checker (callable): Boolean function which will be applied
        to the value whenever inserting a value. If the checker
        returns False, a ValueError will be raised. Note that value
        checking is performed before value normalization.

      key_normalizer (callable): Normalizer which will be applied
        to the key whenever a key is used.

      value_normalizer (callable): Normalizer which will be applied to
        the value when insert to the dict.

      allow_update (bool): Whether allow updating items. If true,
        trying to updating items will raise an error.

      default_factory : If is not None, the dict will work as
        collections.defaultdict.

    Examples:

    >>> x = SafeDict(key_checker=lambda x: isinstance(x, int))
    >>> x[1.0] = 1
    Traceback (most recent call last):
      ...
    KeyError: ...
    >>> x = SafeDict(value_normalizer=str.lower)
    >>> x[1] = 'AbC'
    >>> x[1]
    'abc'
    >>> [y for y in x]
    [1]
    >>> x = SafeDict(allow_update=False)
    >>> x['a'] = 1
    >>> x['a'] = 2
    Traceback (most recent call last):
      ...
    KeyError: ...

    """
    def __init__(self, *args,
                 key_normalizer=None,
                 key_checker=None,
                 value_checker=None,                 
                 value_normalizer=None,
                 allow_update=True,
                 default_factory=None,
                 **kwargs):
        self._storage = dict(*args, **kwargs)
        self.key_normalizer = key_normalizer
        self.value_normalizer = value_normalizer
        self.key_checker = key_checker
        self.value_checker = value_checker
        self.allow_update = allow_update
        self.default_factory = default_factory


    def _process_key(self, key):
        if self.key_checker is not None and not self.key_checker(key):
            raise KeyError('Key checking failed.')
        if self.key_normalizer is not None:
            key = self.key_normalizer(key)
        return key

    
    def _process_value(self, value):
        if self.value_checker is not None and not self.value_checker(value):
            raise ValueError('Value checking failed.')
        if self.value_normalizer is not None:
            value = self.value_normalizer(value)
        return value

    
    def __contains__(self, key):
        key = self._process_key(key)
        return key in self._storage

    
    def __getitem__(self, key):
        key = self._process_key(key)
        if self.default_factory is not None and key not in self._storage:
            self._storage[key] = self.default_factory()
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    def __setitem__(self, key, value):
        key = self._process_key(key)
        value = self._process_value(value)
        if not self.allow_update and key in self._storage:
            raise KeyError('Try to update, but allow_update is False.')
        self._storage[key] = value


    def __delitem__(self, key):
        key = self._process_key(key)
        del self._storage[key]


