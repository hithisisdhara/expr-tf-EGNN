# -*- coding: utf-8 -*-
"""
Dataset manipulation utilities.
"""
import collections
import decoutils
import functools

from pathlib import Path

from .. import utils as _utils
from .. import config as _config


def normalize_dataset_name(name):
    """Normalize dataset name(s)

    The normalization performs the following

      1. Change each each name to lower-case

      2. Replacing all '_' with '-'

    Args:

      name (str): Input names

    Returns:

      str: Normalized name

    Examples:
    
    >>> normalize_dataset_name('hello_wOrld-abc')
    'hello-world-abc'

    """
    name = _utils.validate_str(name, 'name')
    return name.replace('_', '-')


def dataset_name_from_loader_name(loader_name):
    """Automatically deduce dataset names from loader

    Args:

      loader_name (str): The function name of the loader

    Return:

      str: Deduced dataset name from the name of loader

    Examples:
    
    >>> dataset_name_from_loader_name('load_abcxyz123abc')
    'abcxyz123abc'
    >>> dataset_name_from_loader_name('func_abcxyz123abc')
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    name = loader_name
    if not name.startswith('load_'):
        raise ValueError('Name of loader must starts with "load_". Received:'
                         +name)
    name = name[5:]
    return name

_dataset_loaders = _utils.SafeDict(key_normalizer=normalize_dataset_name,
                                   value_checker=callable)

@decoutils.decorator_with_args(return_original=True)
def register_dataset_loader(loader, name=None, update=False,
                            storage=_dataset_loaders):
    """Register dataset loader, can be used as decorators

    Two possible calling signatures:

      * register_dataset_loader(loader, name=None, update=False)
      * register_dataset_loader(name=None, update=False)

    The first one can be used as a regular function, and can also be
    used as a decorator. The second is a decorator with arguments.

    Args:

      loader (callable): Dataset loader.

      name (str, Seq[str]): dataset name(s). If None, the dataset name
        will be infered from the loader name. Default to None.
      
      update (bool): whether update the loader if it is already registered
    
    Returns:

      Return the *loader* for the first case, and return a decorator
      wrapper for the second case.

    Examples:

    >>> import uuid
    >>> def load_abcxyz123abc(): pass
    >>> register_dataset_loader(load_abcxyz123abc)
    <function load_abcxyz123abc at ...>
    >>> register_dataset_loader(load_abcxyz123abc, update=False)
    Traceback (most recent call last):
      ...
    KeyError: ...
    >>> register_dataset_loader()
    <function ...>
    >>> register_dataset_loader(name='load_abcxyz123abc')
    <function ...>
    >>> register_dataset_loader(loader=load_abcxyz123abc, update=True)
    <function load_abcxyz123abc at ...>
    >>> unregister_dataset_loader('abcxyz123abc')
    >>> dataset_has_loader('abcxyz123abc')
    False
    >>> register_dataset_loader(update=True)
    <function ...>
    >>> register_dataset_loader(loader=1.23, name='abc')
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> name1 = str(uuid.uuid4())
    >>> name2 = str(uuid.uuid4())
    >>> register_dataset_loader(loader=load_abcxyz123abc, name=[name1,name2])
    <function load_abcxyz123abc at ...>
    >>> get_dataset_loader(name1)
    <function load_abcxyz123abc at ...>
    >>> get_dataset_loader(name2)
    <function load_abcxyz123abc at ...>
    >>> unregister_dataset_loader([name1, name2])
    """
    global _dataset_loaders

    if name is None:
        name = dataset_name_from_loader_name(loader.__name__)

    name = _utils.validate_str_seq(name, name='name')
    for name1 in name:
        if not update and name1 in storage:
            raise KeyError('A loader was already registered with name:', name1)
        storage[name1] = loader
    

def unregister_dataset_loader(name, storage=_dataset_loaders):
    """Unregister a dataset loader
    
    Args:
        name (str): dataset name

    Returns:
        callable or NoneType : the loader if it was previously registered or None if no loader was registered

    Raises:

        ValueError: if `name` is not `str`
    
    Examples:

    See also:

      :func:`register_dataset_loader`
    """
    name = _utils.validate_str_seq(name, name='name')
    for name1 in name:
        del storage[name1]


def dataset_has_loader(name, storage=_dataset_loaders):
    """Check if a dataset has a registered loader
    
    Args:
      name (str): dataset name
    
    Return:
      bool: True if the dataset has a registered loader, False otherwise.

    Examples:
    
    >>> import uuid
    >>> dataset_has_loader(str(uuid.uuid4()))
    False
    """
    name = _utils.validate_str(name, 'name')
    return name in storage


def assert_dataset_has_loader(name, storage=_dataset_loaders):
    """Assert that a dataset has a registered loader
    
    Args:
      name (str): dataset name
    
    Raises:
      KeyError: if the dataset has no registered loader

    Examples:

    >>> import uuid
    >>> assert_dataset_has_loader(str(uuid.uuid4()))
    Traceback (most recent call last):
      ...
    KeyError: ...
    """
    if not dataset_has_loader(name, storage=storage):
        raise KeyError('Loader for dataset: '+name+' was not registered')


def get_dataset_loader(name, storage=_dataset_loaders):
    """Get a dataset loader by dataset name
    
    Args:
      name: dataset name

    Returns:
      callable: dataset loader
    
    Raises:
      KeyError: if no loader was registered for the dataset

    Examples:

    >>> def load_abcxyz123abc(): pass
    >>> register_dataset_loader(load_abcxyz123abc, update=True)
    <function load_abcxyz123abc at ...>
    >>> get_dataset_loader('abcxyz123abc')
    <function load_abcxyz123abc at ...>
    >>> unregister_dataset_loader('abcxyz123abc')
    >>> get_dataset_loader('abcxyz123abc')
    Traceback (most recent call last):
      ...
    KeyError: ...
    """
    assert_dataset_has_loader(name, storage=storage)
    return storage[name]


_dataset_tvt_loaders = _utils.SafeDict(
    key_checker=lambda x: isinstance(x, str),
    key_normalizer=normalize_dataset_name,
    value_checker=lambda x: isinstance(x, _utils.SafeDict),
    default_factory=functools.partial(
        _utils.SafeDict,
        key_checker=lambda x: isinstance(x, str),
        key_normalizer=normalize_dataset_name,
        value_checker=callable))

def dataset_splitting_name_from_loader_name(loader_name):
    """Automatically deduce dataset and splitting name from loader

    Args:

      loader (callable): Dataset splitting loader

    Return:

      tuple[list[str], str]: Dataset names and splitting names

    Example:

    >>> dataset_splitting_name_from_loader_name('load_abc_def_tvt_ghi')
    ('abc_def', 'ghi')
    >>> dataset_splitting_name_from_loader_name('load_abc_def_ghi')
    Traceback (most recent call last):
      ...
    ValueError: ...
    """
    name = loader_name
    name = name.split('_tvt_')
    if len(name) != 2:
        raise ValueError('loader names cannot be splitted to two parts.')
    name, splitting = name
    name = dataset_name_from_loader_name(name)
    return name, splitting

@decoutils.decorator_with_args(return_original=True)
def register_dataset_tvt_loader(loader, name=None, splitting=None,
                                update=False):
    """Register dataset tvt loader, can be used as decorators

    Two possible calling signatures:

      * register_dataset_tvt_loader(loader, name=None, splitting=None,
        update=False)
      * register_dataset_tvt_loader(name=None, splitting=None, update=False)

    The first one can be used as a regular function, or as a
    decorator. The second one is a decorator with arguments.

    Args:

      loader (callable): Dataset loader.

      name (str): dataset name

      splitting (str): dataset splitting name

      update (bool): whether update the loader if it is already registered
    
    Returns:

      Return the *loader* for the first case, and return a decorator
      wrapper for the second case.

    Examples:

    >>> def load_abc_tvt_abc(): pass
    >>> register_dataset_tvt_loader(load_abc_tvt_abc, update=True) # return the loader
    <function load_abc_tvt_abc at ...>
    >>> register_dataset_tvt_loader() # return a decorator
    <function ...>
    >>> register_dataset_tvt_loader(name='abc') # return a decorator
    <function ...>
    >>> register_dataset_tvt_loader(loader=load_abc_tvt_abc)
    Traceback (most recent call last):
      ...
    KeyError: ...
    >>> register_dataset_tvt_loader(name='abc') # return a decorator
    <function ...>
    >>> unregister_dataset_tvt_loader('abc', 'abc')
    >>> unregister_dataset_tvt_loader('abc') # No loaders
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> register_dataset_tvt_loader(loader=1.23, name='abc', splitting='abc')
    Traceback (most recent call last):
      ...
    ValueError: ...

    See also:

      :func:`unregister_dataset_tvt_loader`
      :func:`register_dataset_loader`

    """
    global _dataset_tvt_loaders

    if name is None or splitting is None:
        tmp = dataset_splitting_name_from_loader_name(loader.__name__)
        if name is None:
            name = tmp[0]
        if splitting is None:
            splitting = tmp[1]

    if splitting in _dataset_tvt_loaders[name] and not update:
        raise KeyError('loader for dataset:'+name+ ' and splitting:'+
                       splitting+' already registered.')
    else:
        _dataset_tvt_loaders[name][splitting] = loader

    
def unregister_dataset_tvt_loader(name, splitting=None):
    """Unregister a dataset splitting loader
    
    Args:

        name (str): Dataset name
    
        splitting (str, NoneType): Tvt name. If None, all splitting
          loaders associated with name will be reregistered.

    Returns:
        callable: the loader if it was previously registered
        None: if no loader was registered

    """
    name = normalize_dataset_name(name)
    splitting = _utils.validate_str(splitting, 'splitting', True)

    global _dataset_tvt_loaders
    if name in _dataset_tvt_loaders:
        if splitting is None:
            _dataset_tvt_loaders.pop(name, None)
        else:
            _dataset_tvt_loaders[name].pop(splitting, None)
        if not _dataset_tvt_loaders[name]:
            del _dataset_tvt_loaders[name]
    else:
        raise ValueError('No tvt load was registered for dataset: {}'
                         ' and splitting: {}'.format(name, splitting))


def dataset_tvt_has_loader(name, splitting):
    """Check if a dataset splitting has registered loader
    
    Args:

      name (str): Dataset name

      splitting (str): Splitting name
    
    Return:

    bool: True if the dataset splitting has a loader registered, False
    otherwise.

    Examples:

    >>> def load_abc_tvt_abc(): pass
    >>> register_dataset_tvt_loader(load_abc_tvt_abc)
    <function load_abc_tvt_abc at ...>
    >>> dataset_tvt_has_loader('abc', 'abc')
    True
    >>> unregister_dataset_tvt_loader('abc')
    >>> import uuid
    >>> name = str(uuid.uuid4())
    >>> dataset_tvt_has_loader(name, name)
    False
    """
    name = _utils.validate_str(name, 'name')
    splitting = _utils.validate_str(splitting, 'splitting')    
    return splitting in _dataset_tvt_loaders[name]


def assert_dataset_tvt_has_loader(name, splitting):
    """Assert that a dataset splitting has a loader registered
    
    Args:
      name (str): dataset name
      splitting (str): splitting name
    
    Raises:
      KeyValue: if the dataset splitting does not have a loader registered

    Examples:

    >>> import uuid
    >>> name = str(uuid.uuid4())
    >>> assert_dataset_tvt_has_loader(name, name)
    Traceback (most recent call last):
      ...
    KeyError: ...
    """
    if not dataset_tvt_has_loader(name, splitting):
        raise KeyError('Loader for dataset: '+name+' and splitting:'+splitting+' was not registered')


def get_dataset_tvt_loader(name, splitting):
    """Get a dataset loader by dataset name
    
    Args:

    name (str): dataset name

    splitting (str):  training-validation-testing splitting name

    Returns:
    
      callable: dataset loader
    
    Raises:

      KeyError: if no loader was registered for the dataset

    Examples:
    
    >>> def load_abc_tvt_abc():pass
    >>> register_dataset_tvt_loader(load_abc_tvt_abc, update=True)
    <function load_abc_tvt_abc at ...>
    >>> get_dataset_tvt_loader('abc', 'abc')
    <function load_abc_tvt_abc at ...>
    >>> unregister_dataset_tvt_loader('abc')
    """
    assert dataset_tvt_has_loader(name, splitting)
    return _dataset_tvt_loaders[name][splitting]



def get_file(fpath, url=None): # pragma: no cover
    if not fpath.exists() or not fpath.is_file():
        if url is not None:
            _utils.download(url, fpath)
        else:
            raise ValueError('file: '+str(fpath)+' does not exist.')


def validate_tvt(tvt, return_list=False,
                 training='training',
                 validation='validation',
                 test='test'):
    """Validate argument *tvt*: 'training', 'validation', 'test'.

    Args:

      tvt (str): Input value for tvt argument.

      return_list (bool): Whether return the result as list even input
        is a single str.

      training (str): Normalized name for 'training' split. 'train',
        'training', or 'tr' will be converted to this value.

      validation (str): Normalized name for 'validation' split. 'validation',
        'valid', or 'val' will be converted to this value.

      test (str): Normalized name for 'test' split. 'test', 'testing',
        or 'te' will be converted to this value.

    Examples:

    >>> validate_tvt('tr')
    'training'
    >>> validate_tvt(['tr', 'val', 'te'])
    ['training', 'validation', 'test']
    >>> validate_tvt(['train', 'valid', 'test'])
    ['training', 'validation', 'test']
    >>> validate_tvt(['training', 'validation', 'testing'])
    ['training', 'validation', 'test']
    >>> validate_tvt(['tr', 'train'])
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> validate_tvt(None) is None
    True

    """
    if tvt is None:
        return None
    is_single = False
    if not isinstance(tvt, (list, tuple)):
        is_single = True
        tvt = [tvt]
    for i, x in enumerate(tvt):
        if isinstance(x, str):
            x = x.lower()
            if x in ['train', 'tr', 'training']:
                tvt[i] = training
            elif x in ['val', 'validation', 'valid']:
                tvt[i] = validation
            elif x in ['test', 'te', 'testing']:
                tvt[i] = test
    if len(set(tvt)) < len(tvt):
        raise ValueError('Some of values in tvt have the same meaning.')
    if is_single and not return_list:
        assert len(tvt) == 1
        tvt = tvt[0]
    return tvt
    
