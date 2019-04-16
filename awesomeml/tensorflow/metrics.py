# -*- coding: utf-8 -*-
"""
Collections of metrics for models implemented with TensorFlow.
"""
import os
import numpy as np
import tensorflow as tf
import arlib
from pathlib import Path
from scipy.stats import entropy
from .. import utils as _utils
from .. import datasets as _ds
from ..datasets import utils as _ds_utils
from .. import stats as _stats
from .. import config as _config
from . import utils as _tf_utils
from . import datasets as _tf_ds

def inception_load_graph(model_path=None, model_home=None): # pragma no cover
    """Load pretrained Inception network as GraphDef

    Args:

      model_path (path-like): Path to the model archive.

      model_home (path-like): Path to a folder where the archive will
        be searched for or downloaded to.

    """
    url = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
    if model_path is None:
        model_home = _tf_utils.validate_model_home(model_home)
        model_path = Path(model_home)/'frozen_inception_v1_2015_12_05.tar.gz'
        
    model_path = Path(model_path)
    print('model_path=', model_path)
    if not model_path.is_file():
        model_path = _utils.download(url, model_path)

    res = tf.GraphDef()
    with arlib.open(model_path, 'r:gz') as ar:
        with ar.open_member('inceptionv1_for_inception_score.pb', 'rb') as f:
            res.ParseFromString(f.read())
    
    return res


def inception_apply(data,
                    return_elements,
                    batch_size=32,
                    model_path=None,
                    model_home=None,
                    use_std_data=False,
                    std_data_path=None,
                    std_data_home=None,
                    std_data_subsets=None,
                    dtype=None): # pragma no cover
    """Apply Inception network to images

    Args:

      data (ndarray, path-like, str): Input images as ndarray, path
        to an image folder, or the name of a standard dataset. The
        value depends on the value of use_std_data. If use_std_data is
        True, images are assumed to be the name of a standard dataset;
        else if images are ndarray, images are assumed to be contents
        of images; otherwise, assumed to be path to an image
        folder. For ndarray image contents, pixel values must be in
        range [0, 255].

      return_elements (str, list[str]): A string or list of strings
        containing operation names in graph_def that will be returned
        as Operation objects; and/or tensor names in graph_def that
        will be returned as Tensor objects.

      batch_size (int): Size of mini-batch to run the network.

      model_path (path-like): Path to the pretrained inception model
        file. If it is a directory,
        model_path/'frozen_inception_v1_2015_12_05.tar.gz', will be
        used. If the model file is not existed, the pretrained
        inception model will be downloaded to the path. Default to
        CWD.

      model_home (path-like): Path to the folder where default model
        archive file will be searched and downloaded to.

      use_std_data (bool): Whether data is name of a standard dataset.

      std_data_path (path-like): Path to the standard dataset. Will be
        passed to dataset loader API as argument data_path.

      std_data_home (path-like): Path to the folder where the data
        will be searched and downloaded to. Will be passed to dataset
        loader API as argument data_home.

      std_data_subsets (Seq[str], str, NoneType): Subsets of standard
        dataset to use, e.g., 'train', 'test' or ['train', 'test']. If
        None, the whole dataset will be used.
      
    Returns:

    Tensor, list[Tensor]: A (or a list) of Operation and/or Tensor
    objects from the imported graph, corresponding to the names in
    return_elements.

    """
    if dtype is None:
        dtype = tf.float32
    assert dtype in [tf.float32, tf.float64]

    is_ndarray = False

    g = tf.Graph()
    with g.as_default():
        if use_std_data:
            extra_kwargs = dict()
            if 'imagenet' in data.lower() or 'ilsvrc' in data.lower():
                extra_kwargs['image_size'] = 299
            images = _tf_ds.load_dataset(
                data, subsets=std_data_subsets, data_path=std_data_path,
                data_home=std_data_home, **extra_kwargs)
        elif isinstance(data, np.ndarray):
            is_ndarray = True
            with g.as_default():
                images_placeholder = tf.placeholder(data.dtype, data.shape)
            images = tf.data.Dataset.from_tensor_slices(images_placeholder)
        elif isinstance(data, tf.Tensor):
            images = tf.data.Dataset.from_tensor_slices(data)
        elif isinstance(data, (str, bytes, os.PathLike)):
            images = _tf_ds.from_numpy_dataset(_ds.image.ImageFolder(data))
        else:
            assert isinstance(images, tf.data.Dataset)

    def map_func(*x):
        x = x[0]
        x = tf.image.convert_image_dtype(x, dtype)
        if x.get_shape().ndims in [2, 3] and x.get_shape()[-1] not in [1, 3]:
            # add a dimension for gray scale images
            x = tf.expand_dims(x, axis=-1)
        if x.get_shape()[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)
        x = (x-0.5) / 0.5
        x = tf.image.resize_images(x, (299, 299))
        return x
    
    with g.as_default():
        images = images.map(map_func)
        if images.output_shapes.ndims == 3:
            images = images.batch(batch_size)

        if is_ndarray:
            iterator = images.make_initializable_iterator()
        else:
            iterator = images.make_one_shot_iterator()
        next_element = iterator.get_next()

    if not isinstance(return_elements, (list, tuple)):
        single_output = True
        return_elements = [return_elements]

    # construct computing graph
    with g.as_default():
        graph_def = inception_load_graph(model_path, model_home)
        out = tf.import_graph_def(graph_def,
                                  input_map={'Mul:0': next_element},
                                  return_elements=return_elements)
    Y = [[] for _ in out]
    with tf.Session(graph=g) as sess:
        if is_ndarray:
            sess.run(iterator.initializer,
                     feed_dict={images_placeholder: data})
        while True:
            try:
                Y1 = sess.run(out)
                for i in range(len(Y)):
                    Y[i].append(Y1[i])
            except tf.errors.OutOfRangeError:
                break
    Y = [np.concatenate(x) for x in Y]
    if single_output:
        assert len(Y) == 1
        Y = Y[0]
    return Y


def inception_score(data, n_splits=10, batch_size=32,
                    model_path=None,
                    model_home=None,
                    use_std_data=False,
                    std_data_path=None,
                    std_data_home=None,
                    std_data_subsets=None): # pragma no cover
    """Calculate inception score from a collection of imgs

    Args:

      data (ndarray, path-like, str): Input images as ndarray, path
        to an image folder, or the name of a standard dataset. The
        value depends on the value of use_std_data. If use_std_data is
        True, images are assumed to be the name of a standard dataset;
        else if images are ndarray, images are assumed to be contents
        of images; otherwise, assumed to be path to an image folder.

      n_splits (int): Number of splits. Defaults to 10.

      batch_size (int): Number of examples per batch. Defaults to 100.

      model_path (path-like): Path to the pretrained inception model
        file. If it is a directory,
        model_path/'frozen_inception_v1_2015_12_05.tar.gz', will be
        used. If the model file is not existed, the pretrained
        inception model will be downloaded to the path. Default to
        CWD.

      model_home (path-like): Path to the folder where default model
        archive file will be searched and downloaded to.

      use_std_data (bool): Whether data is name of a standard dataset.

      std_data_path (path-like): Path to the standard dataset. Will be
        passed to dataset loader API as argument data_path.

      std_data_home (path-like): Path to the folder where the data
        will be searched and downloaded to. Will be passed to dataset
        loader API as argument data_home.

      std_data_subsets (Seq[str], str, NoneType): Subsets of standard
        dataset to use, e.g., 'train', 'test' or ['train', 'test']. If
        None, the whole dataset will be used.

    """
    logits = inception_apply(data,
                             return_elements='logits:0',
                             batch_size=batch_size,
                             model_path=model_path,
                             use_std_data=use_std_data,
                             std_data_path=std_data_path,
                             std_data_home=std_data_home,
                             std_data_subsets=std_data_subsets)
    logits_exp = np.exp(logits[:,:1000])
    Y = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
    Y = np.array_split(Y, n_splits)
    
    scores = [np.exp(np.mean(entropy(y.T, np.mean(y, 0, keepdims=True).T)))
              for y in Y]

    return np.mean(scores), np.std(scores)



def validate_std_stats_fpath(name, subsets, stats_home=None): # pragma no cover
    name = _dataset_utils.normalize_dataset_name(name)
    if subsets:
        subsets = _dataset_utils.validate_tvt(subsets, return_list=True)
        name = '_'.join([name]+subsets)

    stats_home = _utils.validate_dir(
        stats_home, default_path=_config.TF_DATA_HOME/'inception_stats')
        
    fpath = Path(stats_root) / (name+'.npz')
    if not fpath.is_file():
        url = 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_'
        if name == 'celeba':
            url += name
        elif name == 'cifar10_training':
            url += 'cifar10_train'
        elif name == 'svhn_training':
            url += 'svhn_train'
        elif name == 'lsun_training':
            url += 'lsun_train'
        elif name == 'imagenet_training':
            url += 'imagenet_train'
        elif name == 'imagenet_validation':
            url += 'imagenet_validation'
        else:
            raise ValueError(fpath, ' does not exist.')
        url += '.npz'
        fpath = download(url, fpath)
    return fpath



def inception_stats(data,
                    batch_size=32,
                    model_path=None,
                    model_home=None,
                    use_stats_file=False,
                    use_std_data=False,
                    std_data_path=None,
                    std_data_home=None,
                    std_data_subsets=None,
                    std_stats_home=None,
                    dtype=None): # pragma no cover
    """Calculate or load Inception statistics

    Args:

      data: Images, path to an image folder, statistics, or path to a
        precalculated statistics file. The value of data will be
        treated differently depends on different values of
        use_stats_file and use_std_data.

        * False, False: data are images represented as ndarray, or
          statistics represented as tuple or list of ndarray: (mu,
          sigma). 

        * False, True: data is name of a standard dataset, the
          dataset will be loaded to memory as ndarray for statistics
          calculation.

        * True, False: data is path a file contains the statistics
          which will be loaded from the file.

        * True, True: data is name of a standard dataset and
          precalculated statistics of this dataset will be loaded from
          a file. The file will be searched at stats_path, and will be
          donwloaded if not exist.

      batch_size (int): Size of minibatch to calculate Inception outputs.

      model_path (path-like): Path to the pretrained inception model
        file. If it is a directory,
        model_path/'frozen_inception_v1_2015_12_05.tar.gz', will be
        used. If the model file is not existed, the pretrained
        inception model will be downloaded to the path. Default to
        CWD.

      use_stats_file (bool): Whether load statistics from a file.

      use_std_data (bool): Whether data is name of a standard dataset.

      std_data_path (path-like): Path to the standard dataset. Will be
        passed to dataset loader API as argument data_path.

      std_data_home (path-like): Path to the folder where the data
        will be searched and downloaded to. Will be passed to dataset
        loader API as argument data_home.

      std_data_subsets (Seq[str], str, NoneType): Subsets of standard
        dataset to use, e.g., 'train', 'test' or ['train', 'test']. If
        None, the whole dataset will be used.


      std_stats_home (path-like, NoneType): Path to the directory contains
        the precalculated statistics of standard datasets. If None,
        use the same value of model_path. The precalculated statistics
        of standard datasets will be downloaded to this folder if they
        are not exist.

    Returns:

    tuple: (mu, sigma) as mean vectors and covariance matrix.

    """
    if use_std_data:
        data = _utils.validate_str(data, 'data')
        #std_data_home = _tf_utils.validate_data_home(std_data_home)
        std_data_subsets = _ds_utils.validate_tvt(std_data_subsets,
                                                  return_list=True)
    if use_stats_file:
        if use_std_data:
            # data = validate_std_stats_fpath(data, std_data_subsets,
            #                                 std_stats_home)
            std_stats_home = _utils.validate_dir(
                std_stats_home,
                _tf_utils.validate_data_home(std_data_home)/'inception_stats')
            # get path to std stats file
            name = data.lower()
            if 'imagenet' in name or 'ilsvrc' in name:
                name = 'ilsvrc2012'
            if std_data_subsets is not None:
                name = '-'.join([name] + std_data_subsets)
            fname = name + '.npz'
            std_stats_path = std_stats_home/fname
            if not std_stats_path.is_file():
                url = ('https://sourceforge.net/projects/std-inception'
                       '-stats/files/tensorflow/'+fname)
                _utils.download(url, std_stats_path)
            data = std_stats_path
        with np.load(data) as f:
            return f['mu'], f['sigma']

    Y = inception_apply(data, return_elements='pool_3:0',
                        model_path=model_path,
                        model_home=model_home,
                        batch_size=batch_size,
                        use_std_data=use_std_data,
                        std_data_path=std_data_path,
                        std_data_home=std_data_home,
                        std_data_subsets=std_data_subsets,
                        dtype=dtype)
    Y = np.reshape(Y, [Y.shape[0], -1])
    mu = np.mean(Y, axis=0)
    sigma = np.cov(Y, rowvar=False)
    return mu, sigma


def inception_distance(data1, data2, metric='frechet',
                       batch_size=32,
                       model_path=None,
                       model_home=None,
                       data1_use_stats_file=False,
                       data1_use_std=False,
                       data1_std_path=None,
                       data1_std_subsets=None,
                       data2_use_stats_file=False,
                       data2_use_std=False,
                       data2_std_path=None,
                       data2_std_subsets=None,
                       std_stats_home=None,
                       std_data_home=None,
                       dtype=None): # pragma no cover
    """Frechet Inception distance between generated and real images

    Args:

      data1: Images, path to an image folder, statistics, or path to a
        precalculated statistics file. The value of data1 will be
        treated differently depends on different values of
        data1_use_stats_file and data1_use_std:
    
        * False, False: data1 are images represented as ndarray, or
          statistics represented as tuple or list of ndarray: (mu,
          sigma). 

        * False, True: data1 is name of a standard dataset, the
          dataset will be loaded to memory as ndarray for statistics
          calculation.

        * True, False: data1 is path a file contains the statistics
          which will be loaded from the file.

        * True, True: data1 is name of a standard dataset and
          precalculated statistics of this dataset will be loaded from
          a file. The file will be searched at stats_path, and will be
          donwloaded if not exist.

      data2 : Same as data1, but for another data set.

      metric (str): Distance metric to use. Currently, only frechet
        distance is supported.

      batch_size (int): Size of minibatch to calculate Inception outputs.

      model_path (path-like): Path to the pretrained inception model
        archive. If None, will search for
        model_home/'frozen_inception_v1_2015_12_05.tar.gz', if this
        file does not exist, it will be downloaded to model_home.

      model_home (path-like): Path to the folder where default model
        archive file will be searched and downloaded to.

      data1_use_stats_file (bool): Whether load statistics from a
        file.

      data1_use_std (bool): Whether data1 is name of a standard dataset.

      data1_std_path (path-like): Path to the standard dataset for
        data1. Will be passed to dataset loader API as argument
        *data_path*.

      data1_std_subsets (Seq[str], str, NoneType): Subsets of standard
        dataset to use, e.g., 'train', 'test' or ['train', 'test']. If
        None, the whole dataset will be used.

      data2_use_stats_file (bool): Same as data1_use_stats_file, but
        for the second dataset.

      data2_use_std (bool): Same as data1_use_std, but for the second
        dataset.

      data2_std_path (path-like): Same as data1_std_path, but for the
        second dataset.

      data2_std_subsets (Seq[str], str): Same as data1_std_subsets, but
        for the second dataset.

      std_stats_home (path-like, NoneType): Path to the directory
        contains the precalculated statistics of standard datasets. If
        None, use the same value of model_path. The precalculated
        statistics of standard datasets will be downloaded to this
        folder if they are not exist.

      std_data_home (path-like): Path to the folder where the data
        will be searched and downloaded to. Will be passed to dataset
        loader API as argument data_home.

      dtype: DType of the computing graph.
    """
    mu1, sigma1 = inception_stats(data1,
                                  batch_size=batch_size,
                                  model_path=model_path,
                                  model_home=model_home,
                                  use_stats_file=data1_use_stats_file,
                                  use_std_data=data1_use_std,
                                  std_data_path=data1_std_path,
                                  std_data_home=std_data_home,
                                  std_data_subsets=data1_std_subsets,
                                  std_stats_home=std_stats_home,
                                  dtype=dtype)
    
    mu2, sigma2 = inception_stats(data2,
                                  batch_size=batch_size,
                                  model_path=model_path,
                                  model_home=model_home,
                                  use_stats_file=data2_use_stats_file,
                                  use_std_data=data2_use_std,
                                  std_data_path=data2_std_path,
                                  std_data_home=std_data_home,
                                  std_data_subsets=data2_std_subsets,
                                  std_stats_home=std_stats_home,
                                  dtype=dtype)
    
    return _stats.gaussian_frechet_distance(mu1, sigma1, mu2, sigma2)

if __name__ == '__main__': # pragma no cover
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Subcommands')

    # inception score
    def cmd_inception_score(**kwargs):
        output_decimal_length = kwargs.pop('output_decimal_length')
        score = inception_score(**kwargs)
        fmt = ''.join(('%.', str(output_decimal_length), 'f'))
        print(fmt % score)
        
    IS = subparsers.add_parser('inception-score', help='Inception score.')
    IS.add_argument('data', type=str)
    IS.add_argument('--n-splits', type=int, default=10)
    IS.add_argument('--batch-size', type=int, default=32)
    IS.add_argument('--model-path', type=Path, default=None)
    IS.add_argument('--model-home', type=Path, default=None)    
    IS.add_argument('--use-std-data', action='store_true')
    IS.add_argument('--std-data-path', type=Path, default=None)
    IS.add_argument('--std-data-home', type=Path, default=None)
    IS.add_argument('--std-data-subsets', type=str, default=None)
    IS.add_argument('--output-decimal-length', type=str, default=6,
                    help='Number of digits of the decimal.')
    IS.set_defaults(cmd=cmd_inception_score)

    # inception statistics
    def cmd_inception_stats(**kwargs):
        output_file = kwargs.pop('output-file')
        output_name_mean = kwargs.pop('output_name_mean')
        output_name_covariance = kwargs.pop('output_name_covariance')
        mu, sigma = inception_stats(**kwargs)
        kwargs2 = dict(zip([output_name_mean, output_name_covariance],
                           [mu, sigma]))
        np.savez(output_file, **kwargs2)
        
    STAT = subparsers.add_parser('inception-statistics',
                                 aliases=['inception-stats'],
                                 help='Inception statistics.')
    STAT.add_argument('data', type=str)
    STAT.add_argument('output-file', type=Path)
    STAT.add_argument('--batch-size', type=int, default=32)
    STAT.add_argument('--model-path', type=Path, default=None)
    STAT.add_argument('--model-home', type=Path, default=None)    
    STAT.add_argument('--use-std-data', action='store_true')
    STAT.add_argument('--std-data-path', type=Path, default=None)
    STAT.add_argument('--std-data-home', type=Path, default=None)
    STAT.add_argument('--std-data-subsets', type=str, default=None)
    STAT.add_argument('--output-name-mean', type=str, default='mu',
                      help='Name of mean vector in saved .npz file.')
    STAT.add_argument('--output-name-covariance', type=str, default='sigma',
                      help='Name of covariance matrix in saved .npz file.')
    STAT.set_defaults(cmd=cmd_inception_stats)

    # inception distance
    def cmd_inception_distance(**kwargs):
        output_decimal_length = kwargs.pop('output_decimal_length')
        d = inception_distance(**kwargs)
        fmt = ''.join(('%.', str(output_decimal_length), 'f'))
        print(fmt % d)

    DIST = subparsers.add_parser('inception-distance', help='Inception distance')
    DIST.add_argument('data1', type=str)
    DIST.add_argument('data2', type=str)
    DIST.add_argument('--metric', type=str, default='frechet')
    DIST.add_argument('--batch-size', type=int, default=32)
    DIST.add_argument('--model-path', type=Path, default=None)
    DIST.add_argument('--model-home', type=Path, default=None)
    DIST.add_argument('--data1-use-stats-file', action='store_true')
    DIST.add_argument('--data1-use-std', action='store_true')
    DIST.add_argument('--data1-std-path', type=Path, default=None)
    DIST.add_argument('--data1-std-subsets', type=str, default=None)
    DIST.add_argument('--data2-use-stats-file', action='store_true')
    DIST.add_argument('--data2-use-std', action='store_true')
    DIST.add_argument('--data2-std-path', type=Path, default=None)
    DIST.add_argument('--data2-std-subsets', type=str, default=None)
    DIST.add_argument('--output-decimal-length', type=str, default=6,
                    help='Number of digits of the decimal.')
    DIST.set_defaults(cmd=cmd_inception_distance)
    
    kwargs = vars(parser.parse_args())
    cmd = kwargs.pop('cmd')
    cmd(**kwargs)
