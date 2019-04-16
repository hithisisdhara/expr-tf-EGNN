# -*- coding: utf-8 -*-
"""
Graph/network datasets.
"""
import sys
import pickle
import arlib
import compfile
import scipy.sparse
import tarfile
import io

import networkx as nx
import numpy as np

from pathlib import Path

from . import utils as _ds_utils
from . import core as _ds_core
from .. import utils as _utils
from .. import config as _config


def load_citation_icml2016(name, data_path=None, data_home=None,
                           download=True):
    """Load the citation datasets preprocessed by the authors of Plantoid.
    
    Args:

      name (str): dataset name

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): Path to a folder where the dataset archive
        will be searched when data_path is None.
    
    Returns:

      Graph: The citation graph.

    Raises:
      ValueError: if `data_path` is not a valid directory, zip or tar file

    """
    name = _utils.validate_option(
        name, ['cora','citeseer','pubmed'], 'name',
        str_normalizer=str.lower)

    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        fname = 'citation-icml2016.tar.xz'
        url = ('https://sourceforge.net/projects/citation-network/files/'+
               fname)
        data_path = data_home/fname
        _ds_utils.get_file(data_path, url)

    arlib.assert_is_archive(data_path, 'r')

    with arlib.open(data_path, 'r') as ar:
        data = []    
        for ext_name in ['tx','ty','allx','ally','graph']:
            fname = next(x for x in ar.member_names
                         if x.endswith('ind.'+name+'.'+ext_name))
            with ar.open_member(fname, 'rb') as f:
                if sys.version_info > (3, 0):
                    data.append(pickle.load(f, encoding='latin1'))
                else:
                    data.append(pickle.load(f))

        tx, ty, allx, ally, graph = data
        fname = next(x for x in ar.member_names
                     if x.endswith('ind.'+name+'.test.index'))
        with ar.open_member(fname, 'rt') as f:
            idx_test = list(np.loadtxt(f, int))
    
    idx_test_sorted = np.sort(idx_test)
    if name == 'citeseer':
        idx_test_full = range(min(idx_test), max(idx_test)+1)
        tx_extended = scipy.sparse.lil_matrix((len(idx_test_full), tx.shape[1]))
        tx_extended[idx_test_sorted-min(idx_test_sorted),:] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(idx_test_full), ty.shape[1]))
        ty_extended[idx_test_sorted-min(idx_test_sorted),:] = ty
        ty = ty_extended

    features = scipy.sparse.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    idx_train = list(range(140))
    idx_val = list(range(140,140+500))

    # reorder test nodes
    features[idx_test,:] = features[idx_test_sorted,:]
    labels[idx_test,:] = labels[idx_test_sorted,:]

    # adjacency matrix
    N = features.shape[0]
    G = nx.from_dict_of_lists(graph)
    node_idx2id = dict(zip(range(N), [str(i) for i in range(N)]))
    nx.relabel_nodes(G, node_idx2id, copy=False)

    # nodes
    for i, row in enumerate(features.rows):
        if len(row) > 0:
            row = [str(x) for x in row]
            row_data = features.data[i]
            G.add_node(str(i), **dict(zip(row, row_data)))

    # node labels
    labels = scipy.sparse.lil_matrix(labels)
    for i in range(N):
        label = labels.rows[i]
        assert len(label) == 1 or len(label) == 0
        if len(label) == 1:
            label = label[0]
            G.nodes[str(i)]['category'] = label
        
    return G


@_ds_utils.register_dataset_loader
def load_cora_icml2016(data_path=None, data_home=None,
                       download=True):
    """Load the preprocessed version of the Cora dataset
    
    Args:

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): path to the root where the package looks
        for dataset

    Returns:

      Graph: The citation graph.

    """
    return load_citation_icml2016('cora', data_path=data_path,
                                  data_home=data_home, download=download)


@_ds_utils.register_dataset_loader
def load_citeseer_icml2016(data_path=None, data_home=None,
                           download=True):
    """Load the preprocessed version of the Citeseer dataset
    
    Args:

      data_path (path_like): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): Path to the root where the package looks
        for dataset

    Returns:

      Graph: The citation graph.

    """
    return load_citation_icml2016('citeseer', data_path=data_path,
                                  data_home=data_home, download=download)



@_ds_utils.register_dataset_loader
def load_pubmed_icml2016(data_path=None, data_home=None,
                         download=True):
    """Load the preprocessed version of the Pubmed dataset
    
    Args:

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (path_like): path to the root where the package looks
        for dataset

    Returns:

      Graph: The citation graph.

    """
    return load_citation_icml2016('pubmed', data_path=data_path,
                                  data_home=data_home,
                                  download=download)


def load_citation_icml2016_tvt(name, data_path=None, data_home=None,
                               download=True):
    """Load the training-validation-testing splitting of the citation
    graph data preprocessed by the authors of Plantoid
    
    Args:

      name (str): dataset name ('cora', 'citeseer' or 'pubmed')

      data_path (PathLike): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): path to the root where the package looks
        for dataset

    Returns:

    tuple[list]: IDs of training, validation and testing examples,
      respectively

    """
    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        url = None
        data_path = data_home/'citation-icml2016.tar.xz'
        _ds_utils.get_file(data_path, url)
    data_path = Path(data_path)
    with arlib.open(data_path, 'r') as ar:
        fname = next(x for x in ar.member_names
                     if x.endswith('ind.'+name+'.test.index'))
        with ar.open_member(fname, 'r') as f:
            id_test = list(np.loadtxt(f, str))

        fname = next(x for x in ar.member_names
                     if x.endswith('ind.'+name+'.y'))
        with ar.open_member(fname, 'rb') as f:
            if sys.version_info > (3, 0):
                y = pickle.load(f, encoding='latin1')
            else:
                y = pickle.load(f)

    id_train = [str(i) for i in range(len(y))]
    id_val = [str(i) for i in range(len(y), len(y)+500)]
        
    return id_train, id_val, id_test


@_ds_utils.register_dataset_tvt_loader
def load_cora_icml2016_tvt_icml2016(data_path=None, data_home=None,
                                    download=True):
    """Load the training-validation-testing splitting of the preprocessed Cora dataset

    Args:

      data_path (PathLike): path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): path to the root where the package looks for
        dataset

    """
    return load_citation_icml2016_tvt('cora', data_path=data_path,
                                      data_home=data_home, download=download)


@_ds_utils.register_dataset_tvt_loader
def load_citeseer_icml2016_tvt_icml2016(data_path=None, data_home=None,
                                        download=True):
    """Load the training-validation-testing splitting of the preprocessed Citeseer dataset

    Args:

      data_path (path_like): path to the dataset, should be a valid
        directory, zip, or tar file

      root (path_like): path to the root where the package looks for dataset

    """
    return load_citation_icml2016_tvt('citeseer', data_path=data_path,
                                      data_home=data_home, download=download)


@_ds_utils.register_dataset_tvt_loader
def load_pubmed_icml2016_tvt_icml2016(data_path=None, data_home=None,
                                      download=True):
    """Load the training-validation-testing splitting of the preprocessed Pubmed dataset

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

    """
    return load_citation_icml2016_tvt('pubmed', data_path=data_path,
                                      data_home=data_home, download=download)


def _parse_cora_citeseer(f_content, f_cites):
    # read contents
    G = nx.read_edgelist(f_cites, create_using=nx.DiGraph)
    D = None
    for line in f_content:
        line = line.split()
        node = line[0]
        assert node in G.nodes
        label = line[-1]
        G.add_node(node, category=label)
        for i, x in enumerate(line[1:-1]):
            if x == '1':
                G.nodes[node][i] = True
    return G


@_ds_utils.register_dataset_loader
def load_cora(data_path=None, data_home=None, download=True):
    """Load the Citeseer citation dataset
    
    See https://linqs.soe.ucsc.edu/data

    Args: 

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the file if it is not in the
        root directory.

    """
    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        data_path = data_home/'cora.tgz'
    url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    _ds_utils.get_file(data_path, url)
    assert data_path.is_file()

    with arlib.open(data_path, 'r') as ar:
        f_content = ar.open_member(next(x for x in ar.member_names
                                       if x.endswith('cora.content')))
        f_cites = ar.open_member(next(x for x in ar.member_names
                                       if x.endswith('cora.cites')), 'rb')
        G = _parse_cora_citeseer(f_content, f_cites)
    return G


@_ds_utils.register_dataset_loader
def load_citeseer(data_path=None, data_home=None, download=True):
    """Load the Citeseer citation dataset
    
    See https://linqs.soe.ucsc.edu/data

    Args: 

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the file if it is not in the
        root directory.
    """
    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        data_path = data_home/'citeseer.tgz'
    url = 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz'
    _ds_utils.get_file(data_path, url)

    with arlib.open(data_path, 'r') as ar:
        f_content = ar.open_member(next(x for x in ar.member_names
                                        if x.endswith('citeseer.content')))
        f_cites = ar.open_member(next(x for x in ar.member_names
                                       if x.endswith('citeseer.cites')),'rb')
        G = _parse_cora_citeseer(f_content, f_cites)

    # For citeseer dataset, there are a few nodes has neither node features
    # nor node labels, we need to remove them from the graph
    bad_nodes = [i for i in G.nodes if len(G.nodes[i])==0]
    G.remove_nodes_from(bad_nodes)
    return G


@_ds_utils.register_dataset_loader
def load_pubmed(data_path=None, data_home=None, download=True):
    """Load the Pubmed citation dataset
    
    See https://linqs.soe.ucsc.edu/data

    Args: 

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the file if it is not in the
        root directory.
    """
    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        data_path = data_home/'pubmed.tgz'
    url = 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz'
    _ds_utils.get_file(data_path, url)

    with arlib.open(data_path) as ar:
        # parse content file
        f_content = ar.open_member(next(
            x for x in ar.member_names if x.endswith('NODE.paper.tab')))
        # skip the first and second lines
        next(f_content)
        next(f_content)

        G = nx.DiGraph()
        for line in f_content:
            line = line.split()
            node = line[0]
            label = int(line[1].split('=')[1])
            G.add_node(node, category=label)
            for var in line[2:-1]:
                attr_id, attr_val = var.split('=')
                G.nodes[node][attr_id] = float(attr_val)
        
        # parse cites file
        f_cites = ar.open_member(next(
            x for x in ar.member_names if x.endswith('DIRECTED.cites.tab')))
        # skip the first and second lines
        next(f_cites)
        next(f_cites)
        for line in f_cites:
            line = line.split()
            cited = line[1].split(':')[1]
            citing = line[3].split(':')[1]
            G.add_edge(citing, cited)
    return G

def _load_citation_network_tvt_sf(fname, data_path=None, data_home=None,
                                  download=True):
    if data_path is None:
        data_home = _utils.validate_data_home(data_home)
        url = ('https://sourceforge.net/projects/citation-network/files/'+
               fname)
        data_path = data_home/fname
        _ds_utils.get_file(data_path, url)
    return _ds_core.load_dataset_tvt_txt(data_path)


@_ds_utils.register_dataset_tvt_loader
def load_cora_tvt_cvpr2019_051580(data_path=None, data_home=None,
                                  download=True):
    """Load the CVPR2019 (5%, 15% 80%) training-validation-testing
    splitting of the Cora dataset.

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the corresponding splitting
        file. Default to True
    """
    fname = 'cora-tvt-cvpr2019-051580.txt.xz'
    return _load_citation_network_tvt_sf(fname=fname, data_path=data_path,
                                         data_home=data_home,
                                         download=download)


@_ds_utils.register_dataset_tvt_loader
def load_cora_tvt_cvpr2019_602020(data_path=None, data_home=None,
                                  download=True):
    """Load the CVPR2019 (60%, 20%, 20%) training-validation-testing
    splitting of the Cora dataset.

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the corresponding splitting
        file. Default to True
    """
    fname = 'cora-tvt-cvpr2019-602020.txt.xz'
    return _load_citation_network_tvt_sf(fname=fname, data_path=data_path,
                                         data_home=data_home,
                                         download=download)



@_ds_utils.register_dataset_tvt_loader
def load_citeseer_tvt_cvpr2019_051580(data_path=None, data_home=None,
                                  download=True):
    """Load the CVPR2019 (5%, 15% 80%) training-validation-testing
    splitting of the Citeseer dataset.

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the corresponding splitting
        file. Default to True
    """
    fname = 'citeseer-tvt-cvpr2019-051580.txt.xz'
    return _load_citation_network_tvt_sf(fname=fname, data_path=data_path,
                                         data_home=data_home,
                                         download=download)


@_ds_utils.register_dataset_tvt_loader
def load_citeseer_tvt_cvpr2019_602020(data_path=None, data_home=None,
                                      download=True):
    """Load the CVPR2019 (60%, 20%, 20%) training-validation-testing
    splitting of the Citeseer dataset.

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the corresponding splitting
        file. Default to True
    """
    fname = 'citeseer-tvt-cvpr2019-602020.txt.xz'
    return _load_citation_network_tvt_sf(fname=fname, data_path=data_path,
                                         data_home=data_home,
                                         download=download)



@_ds_utils.register_dataset_tvt_loader
def load_pubmed_tvt_cvpr2019_051580(data_path=None, data_home=None,
                                  download=True):
    """Load the CVPR2019 (5%, 15% 80%) training-validation-testing
    splitting of the Pubmed dataset.

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the corresponding splitting
        file. Default to True
    """
    fname = 'pubmed-tvt-cvpr2019-051580.txt.xz'
    return _load_citation_network_tvt_sf(fname=fname, data_path=data_path,
                                         data_home=data_home,
                                         download=download)


@_ds_utils.register_dataset_tvt_loader
def load_pubmed_tvt_cvpr2019_602020(data_path=None, data_home=None,
                                      download=True):
    """Load the CVPR2019 (60%, 20%, 20%) training-validation-testing
    splitting of the Pubmed dataset.

    Args:

      data_path (PathLike): Path to the dataset, should be a valid
        directory, zip, or tar file

      data_home (PathLike): Path to the root where the package looks
        for dataset

      download (bool): Whether download the corresponding splitting
        file. Default to True
    """
    fname = 'pubmed-tvt-cvpr2019-602020.txt.xz'
    return _load_citation_network_tvt_sf(fname=fname, data_path=data_path,
                                         data_home=data_home,
                                         download=download)
    
