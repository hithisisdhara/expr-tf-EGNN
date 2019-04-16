# -*- coding: utf-8 -*-
"""
Configure this package.
"""
import importlib
import json

from pathlib import Path


CONFIG_HOME = Path('~').expanduser()/'.awesomeml'
CONFIG_PATH = CONFIG_HOME/'config.json'

if CONFIG_PATH.is_file(): #pragma no cover
    # load config file
    with open(CONFIG_PATH) as f:
        settings = json.load(f)
    keys_seen = set()
    for key in settings:
        key2 = key.lower()
        if key2 in keys_seen:
            raise KeyError(''.join([
                'Duplicate variale name:', key,
                ' in config file:', str(CONFIG_PATH)]))
        keys_seen.add(key2)
    settings = {key.upper(): Path(val) for key, val in settings.items()}
    names = ['HOME', 'DATA_HOME', 'MODEL_HOME', 'TF_DATA_HOME',
             'TF_MODEL_HOME', 'TORCH_DATA_HOME', 'TORCH_MODEL_HOME']
    names2 = list(set(settings.keys())-set(names))
    if len(names2) > 0:
        raise ValueError('Settings:', names2, ' not supported.')
    vars().update(settings)

if 'HOME' not in vars(): #pragma no cover
    HOME = CONFIG_HOME
if 'DATA_HOME' not in vars(): #pragma no cover
    DATA_HOME = HOME/'datasets'
if 'MODEL_HOME' not in vars(): #pragma no cover
    MODEL_HOME = HOME/'models'
if 'TF_DATA_HOME' not in vars(): #pragma no cover
    TF_DATA_HOME = DATA_HOME/'tensorflow'
if 'TF_MODEL_HOME' not in vars(): #pragma no cover
    TF_MODEL_HOME = MODEL_HOME/'tensorflow'
if 'TORCH_DATA_HOME' not in vars(): #pragma no cover
    TORCH_DATA_HOME = DATA_HOME/'torch'
if 'TORCH_MODEL_HOME' not in vars(): #pragma no cover
    TORCH_MODEL_HOME = MODEL_HOME/'torch'


def has_package(name):
    """Check whether a package is installed and can be importted
    
    Args:
      name (str): name of the package

    Return:
      bool: True if the package is installed, False otherwise

    Examples:

      >>> has_package('os')
      True
      >>> has_package('sys')
      True
      >>> import uuid
      >>> has_package(str(uuid.uuid4()))
      False

    """
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def assert_has_package(name):
    """Assert a package is installed and can be importted

    Args:

      name (str): Name of the package.

    Raises:

      RuntimeError: If the package is not installed or cannot be importted.

    Examples:

      >>> import uuid
      >>> assert_has_package(str(uuid.uuid4()))
      Traceback (most recent call last):
        ...
      RuntimeError: ...
    """
    if not has_package(name):
        raise RuntimeError('Package '+name+' is not installed or cannot '
                           'be importted.')

    

