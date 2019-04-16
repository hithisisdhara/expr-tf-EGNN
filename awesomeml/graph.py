# -*- coding: utf-8 -*-
"""
Graph/network data manipulation.
"""
import scipy.sparse
import numpy as np
import networkx as nx
import collections.abc
import itertools

from . import ops as _ops
from . import utils as _utils
from . import config as _config

def normalize_adj(A, method='sym', *, axis1=-2, axis2=-1, 
                  assume_symmetric_input=True,
                  check_symmetry=False, eps=1e-10,
                  array_mode=None,
                  array_default_mode='numpy',
                  array_homo_mode=None):
    """Normalize adjacency matrix defined by axis1 and axis2 in an array

    Examples:

    >>> A = np.array([[1,2],[3,4]])
    >>> B = np.array([[1/3, 2/3], [3/7,4/7]])
    >>> B2 = normalize_adj(A, 'row')
    >>> np.testing.assert_array_almost_equal(B, B2)

    >>> B2 = normalize_adj(scipy.sparse.coo_matrix(A), 'row')
    >>> scipy.sparse.issparse(B2)
    True
    >>> import awesomeml.ops as _ops
    >>> np.testing.assert_array_almost_equal(B, B2.todense())

    >>> import sparse
    >>> B2 = normalize_adj(sparse.COO(A), 'row')
    >>> isinstance(B2, sparse.SparseArray)
    True
    >>> np.testing.assert_array_almost_equal(B, B2.todense())

    >>> B = np.array([[0.28867513,0.47140452],[0.56694671,0.6172134]])
    >>> B2 = normalize_adj(A, 'sym', assume_symmetric_input=False)
    >>> np.testing.assert_array_almost_equal(B, B2)

    >>> B2 = normalize_adj(A, 'sym', assume_symmetric_input=False,
    ...                    check_symmetry=True)
    >>> np.testing.assert_array_almost_equal(B, B2)

    >>> B = np.array([[0.33333333,0.43643578],[0.65465367,0.57142857]])
    >>> B2 = normalize_adj(A, 'sym', assume_symmetric_input=True)
    >>> np.testing.assert_array_almost_equal(B, B2)

    >>> normalize_adj(A, 'sym', assume_symmetric_input=True,
    ...               check_symmetry=True)
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> B = np.array([[0.50480769,0.49519231],[0.49519231,0.50480769]])
    >>> B2 = normalize_adj(A, 'ds')
    >>> np.testing.assert_array_almost_equal(B, B2)

    >>> B = np.reshape(B, (1,2,1,2,1))
    >>> B2 = normalize_adj(np.reshape(A,(1,2,1,2,1)), 'ds', axis1=1, axis2=3)
    >>> np.testing.assert_array_almost_equal(B, B2)

    """
    A = _utils.validate_array_args(A, mode=array_mode,
                                   default_mode=array_default_mode,
                                   homo_mode=array_homo_mode)
    dtype = A.dtype if np.issubdtype(A.dtype, np.floating) else np.float
    method = _utils.validate_option(
        method,
        ['row', 'column', 'col', 'sym', 'symmetric', 'ds', 'dsm',
         'doubly_stochastic'],
        'method', str_normalizer=str.lower)

    axis1,axis2 = _utils.validate_int_seq([axis1,axis2], 2)
    
    if method in ['row', 'col', 'column']:
        axis_to_sum = axis2 if method == 'row' else axis1
        norm = _ops.sum(A, axis_to_sum, dtype=dtype, keepdims=True)
        norm = _utils.as_numpy_array(norm)
        norm[norm==0] = eps
        norm = 1.0 / norm
        return _ops.multiply(A, norm)
    elif method in ['ds', 'dsm', 'doubly_stochastic']:
        # step 1: row normalize
        norm = _ops.sum(A, axis2, dtype=dtype, keepdims=True)
        norm = _utils.as_numpy_array(norm)
        norm[norm==0] = eps
        norm = 1.0 / norm
        P = _ops.multiply(A, norm)

        # step 2: P @ P^T / column_norm
        P = _ops.swapaxes(P, axis2, -1)
        P = _ops.swapaxes(P, axis1, -2)
        norm = _ops.sum(P, axis=-2, dtype=dtype, keepdims=True)
        norm = _utils.as_numpy_array(norm)
        norm[norm==0] = eps
        norm = 1.0 / norm
        PT = _ops.swapaxes(P, -1, -2)
        P = _ops.multiply(P, norm)
        T = _ops.matmul(P, PT)
        T = _ops.swapaxes(T, axis1, -2)
        T = _ops.swapaxes(T, axis2, -1)
        return T
    else:
        assert method in ['sym', 'symmetric']
        treat_A_as_sym = False
        if assume_symmetric_input:
            if check_symmetry:
                _utils.assert_is_symmetric(A, axis1, axis2)
            treat_A_as_sym = True
        else:
            if check_symmetry:
                treat_A_as_sym = _utils.is_symmetric(A, axis1, axis2)

        norm1 = np.sqrt(_ops.sum(A, axis2, dtype=dtype, keepdims=True))
        norm1 = _utils.as_numpy_array(norm1)
        norm1[norm1==0] = 1e-10
        norm1 = 1.0 / norm1
        if treat_A_as_sym:
            norm2 = _ops.swapaxes(norm1, axis1, axis2)
        else:
            norm2 = np.sqrt(_ops.sum(A, axis1, dtype=dtype, keepdims=True))
            norm2 = _utils.as_numpy_array(norm2)
            norm2[norm2==0] = 1e-10
            norm2 = 1.0 / norm2
        return _ops.multiply(_ops.multiply(A, norm1), norm2)


def adj_to_laplacian(A,  adj_norm='sym',
                     axis1=-2, axis2=-1,
                     array_mode=None, array_default_mode='numpy',
                     array_homo_mode=None,
                     **kwargs):
    """Calculate Laplacian(s) from adjacency matrix (matrices)

    >>> A = np.array([[1,2],[3,4]])
    >>> L = np.array([[2/3, -2/3], [-3/7,3/7]])
    >>> L2 = adj_to_laplacian(A, 'row')
    >>> np.testing.assert_array_almost_equal(L, L2)

    >>> adj_to_laplacian(np.array([1,2]), 'row')
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> import scipy.sparse
    >>> L2 = adj_to_laplacian(scipy.sparse.coo_matrix(A), 'row')
    >>> scipy.sparse.issparse(L2)
    True
    >>> np.testing.assert_array_almost_equal(L, L2.todense())

    >>> import sparse
    >>> L2 = adj_to_laplacian(sparse.as_coo(A), 'row')
    >>> np.testing.assert_array_almost_equal(L, L2.todense())
    """
    A = _utils.validate_array_args(A, mode=array_mode,
                                   default_mode=array_default_mode,
                                   homo_mode=array_homo_mode)
    if np.ndim(A) < 2 or A.shape[axis1] != A.shape[axis2]:
        raise ValueError('ndim of A must be greater than 2, and the two '
                         'corresponding axis should be equally long')
    K = A.shape[axis1]
    
    if not (adj_norm in [False, None] or
            isinstance(adj_norm, str) and
            adj_norm.lower() == 'none'):
        A = normalize_adj(A, method=adj_norm, axis1=axis1, axis2=axis2,
                          **kwargs)

    if scipy.sparse.issparse(A):
        return scipy.sparse.eye(K) - A
    elif _config.has_package('sparse') and _utils.is_pydata_sparse(A):
        I = _utils.as_pydata_coo(scipy.sparse.eye(K))
        shape = [1]*np.ndim(A)
        shape[axis1] = K
        shape[axis2] = K
        I = I.reshape(shape)
        return I - A
    else:
        I = np.eye(K)
        shape = [1]*np.ndim(A)
        shape[axis1] = K
        shape[axis2] = K
        I = I.reshape(shape)
        return I - A


def node_attr_ids(G, nodes=None):
    """Collecting the set of attribute ids associated with nodes

    Args:

      G (NetworkX Graph): Input Graph

      nodes (Iterable, NoneType): Nodes whose attribute ids will be
        collected. If it is None, collect attribute ids from all nodes
        of the graph. Default to :obj:`None`.

    Return:

      Set: The attribute id set of nodes

    Examples:
    
    >>> G = nx.Graph()
    >>> G.add_node(0, a=1)
    >>> G.add_node(1, b=2)
    >>> G.add_node(2, c=3)
    >>> sorted(node_attr_ids(G))
    ['a', 'b', 'c']
    >>> sorted(node_attr_ids(G, [0,1]))
    ['a', 'b']
    >>> node_attr_ids(G, [0])
    {'a'}

    """
    attr_ids = set()
    view = G.nodes(data=True)
    if nodes is None:
        nodes = G.nodes
    for i in nodes:
        attr_ids.update(view[i].keys())
    return attr_ids



def _validate_args_node_attr_func(G,
                                  include_attrs=None,
                                  exclude_attrs=None,
                                  nodes=None):
    if nodes is None:
        nodes = G.nodes
    elif not all(node in G for node in nodes):
        raise ValueError('Some nodes are not in the graph G.')
        

    if include_attrs is not None and exclude_attrs is not None:
        raise ValueError('include_attrs and exclude_attr cannot be specified'
                         ' together.')
    all_attrs = node_attr_ids(G)
    if include_attrs is None:
        attrs = all_attrs
    else:
        # check if attributes are in the graph
        attrs = include_attrs
        for a in attrs:
            if a not in all_attrs:
                raise KeyError('Graph has no attribute:', a)
    #attrs = node_attr_ids(G) if include_attrs is None else include_attrs
    if exclude_attrs is not None:
        exclude_attrs = set(exclude_attrs)
        attrs = [x for x in attrs if x not in exclude_attrs]

    return attrs, nodes


    
def node_attr_matrix(G, attrs=None, exclude_attrs=None, nodes=None,
                     mode='scipy-sparse', dtype=None, concat=True,
                     return_node_ids=False, return_attr_ids=False):
    """Return node attributes as a scipy.sparse.coo_matrix

    Notes:

    Specify either attrs or exclude_attrs, but not both.

    Args:

      G: Input graph.

      attrs (Iterable): Attributes to collect.

      exclude_attrs (Iterable): Attributes not to collect.

      nodes (Iterable): Nodes whose attributes will be collected.

      mode (str): Array mode of the output matrix (array). See also
        :func:`_utils.validate_array_mode`.

      dtype (DType): Data type of attribute matrix. If None, will be
        determined automatically according to the content. Default to
        None.

      concat (bool): If True, all node attribute values will be
        concatenated to a vector. Otherwise, will be stacked. Default
        to True.

      return_node_ids (bool): Whether return the corresponding node
        ids or not. Default to False.

      return_attr_ids (bool): Whether return the corresponding
        attribute ids or not. Default to False.

    Returns:

      (attr_matrix, node_ids, attr_ids) if both return_node_ids and
      return_attr_ids are True. Where M, nodes, attrs are. If only one
      is True, return a two element tuple. If none of them is True,
      return the attribute matrix only.

    Examples:

    >>> G = nx.Graph()
    >>> G.add_node(0, a=1)
    >>> G.add_node(1, b=1)
    >>> G.add_node(2, c=1)
    >>> G.nodes[2][1] = 'c'
    >>> a = node_attr_matrix(G, attrs=['a','b','c'])
    >>> isinstance(a, scipy.sparse.coo_matrix)
    True
    >>> np.all(a.todense() == np.eye(3))
    True
    >>> a = node_attr_matrix(G, nodes=[0,1], attrs=['a','b'])
    >>> np.all(a.todense() == np.eye(2))
    True

    >>> a, attrs = node_attr_matrix(G, nodes=[0,1], exclude_attrs=['c', 1],
    ...                             return_attr_ids=True)
    >>> b = node_attr_matrix(G, nodes=[0,1], attrs=attrs)
    >>> np.all(a.todense() == b.todense())
    True

    >>> a = node_attr_matrix(G, nodes=[0,1], attrs=['a','b'],
    ...                      exclude_attrs=['c'])
    Traceback (most recent call last):
      ...
    ValueError: ...
    
    >>> node_attr_matrix(G, nodes=[100011111])
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> node_attr_matrix(G, attrs=['xzykjk'])
    Traceback (most recent call last):
      ...
    KeyError: ...

    >>> G = nx.Graph()
    >>> G.add_node(0, a=[1,2,3])
    >>> G.add_node(1, b=[1,2,3])
    >>> a = np.array([[1,2,3,0,0,0],[0,0,0,1,2,3]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=True)
    >>> np.all(a == a2)
    True
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'],
    ...                       mode='scipy-sparse', concat=True)
    >>> np.all(a == a2.toarray())
    True
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'],
    ...                       mode='pydata-sparse', concat=True)
    >>> np.all(a == a2.todense())
    True

    >>> a = np.array([[[1,2,3],[0,0,0]],[[0,0,0],[1,2,3]]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=False)
    >>> np.all(a == a2)
    True
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'],
    ...                       mode='scipy-sparse', concat=False)
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'],
    ...                       mode='pydata-sparse', concat=False)
    >>> np.all(a == a2.todense())
    True

    >>> G = nx.Graph()
    >>> G.add_node(0, a=[1,2,3])
    >>> G.add_node(1, b=1)
    >>> a = np.array([[1,2,3,0],[0,0,0,1]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=True)
    >>> np.all(a == a2)
    True
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'],
    ...                       mode='scipy-sparse', concat=True)
    >>> np.all(a == a2.toarray())
    True
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'],
    ...                       mode='pydata-sparse', concat=True)
    >>> np.all(a == a2.todense())
    True
    >>> node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                  mode='numpy', concat=False)
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> G = nx.Graph()
    >>> G.add_node(0, a=[1,2,3])
    >>> G.add_node(1, a=1)
    >>> node_attr_matrix(G, nodes=[0,1], attrs=['a'], concat=True)
    Traceback (most recent call last):
      ...
    ValueError: ...

    >>> G = nx.Graph()
    >>> attr = np.array([[1,2],[3,4]])
    >>> G.add_node(0, a=attr)
    >>> G.add_node(1, b=attr)
    >>> a = np.array([[1,2,3,4,0,0,0,0],[0,0,0,0,1,2,3,4]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=True)
    >>> np.all(a == a2)
    True
    >>> a = np.array([[[[1,2],[3,4]],[[0,0],[0,0]]],
    ...               [[[0,0],[0,0]],[[1,2],[3,4]]]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=False)
    >>> np.all(a == a2)
    True

    >>> G = nx.Graph()
    >>> attr = scipy.sparse.coo_matrix(np.array([1,2,3]))
    >>> G.add_node(0, a=attr)
    >>> G.add_node(1, b=attr)
    >>> a = np.array([[1,2,3,0,0,0],[0,0,0,1,2,3]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=True)
    >>> np.all(a == a2)
    True

    >>> G = nx.Graph()
    >>> import sparse
    >>> attr = sparse.as_coo(np.array([1,2,3]))
    >>> G.add_node(0, a=attr)
    >>> G.add_node(1, b=attr)
    >>> a = np.array([[1,2,3,0,0,0],[0,0,0,1,2,3]])
    >>> a2 = node_attr_matrix(G, nodes=[0, 1], attrs=['a','b'], 
    ...                       mode='numpy', concat=True)
    >>> np.all(a == a2)
    True

    >>> _, nodes = node_attr_matrix(G, return_node_ids=True)
    >>> nodes
    [0, 1]
    
    >>> _, nodes, attrs = node_attr_matrix(G, return_node_ids=True,
    ...                                    return_attr_ids=True)
    >>> nodes
    [0, 1]
    >>> sorted(attrs)
    ['a', 'b']

    """
    attrs, nodes = _validate_args_node_attr_func(
        G, attrs, exclude_attrs, nodes)
    mode = _utils.validate_array_mode(mode)
    M, N = len(nodes), len(attrs)
    # collect and check attribute shape
    attr_shapes = dict(zip(attrs, [None] * N))
    for node in G.nodes:
        for attr in G.nodes[node]:
            s = np.shape(G.nodes[node][attr])
            if attr in attr_shapes:
                if attr_shapes[attr] is None:
                    attr_shapes[attr] = s
                elif attr_shapes[attr] != s:
                    raise ValueError('Shape of attribute:', attr, ' is not'
                                     'consistent.')
    attr_shapes = [attr_shapes[x] for x in attrs]
    if not concat and not len(set(attr_shapes)) == 1:
        raise ValueError('Shapes of all attributes must be the same'
                         ' if concat is False.')        

    # collect data
    data, I, J = [], [], []
    for node_idx, node in enumerate(nodes):
        node_data = G.nodes[node]
        for attr_idx, attr in enumerate(attrs):
            try:
                val = node_data[attr]
                val = val if _utils.is_any_array(val) else _utils.as_numpy_array(val)
            except KeyError:
                continue
            data.append(val)
            I.append(node_idx)
            J.append(attr_idx)

    # preprocess collected data
    all_scalar = all(x==() for x in attr_shapes)
    if not all_scalar:
        if concat:
            attr_len = [int(np.prod(x)) for x in attr_shapes]
            attr_len_cum = list(itertools.accumulate(attr_len))
            attr_j_start = {x: (0 if x==0 else attr_len_cum[x-1])
                            for x in range(N)}
            data2, I2, J2 = [], [], []
            for val, i, j in zip(data, I, J):
                if attr_shapes[j] == ():
                    data2.append(val.item())
                    I2.append(i)
                    J2.append(attr_j_start[j])
                else:
                    if _utils.is_scipy_sparse(val):
                        val = _utils.as_scipy_coo(val)
                        val = val.reshape((-1, 1))
                        for k in range(val.nnz):
                            data2.append(val.data[k])
                            I2.append(i)
                            J2.append(attr_j_start[j]+val.row[k])
                    elif (_config.has_package('sparse')
                        and _utils.is_pydata_sparse(val)):
                        val = _utils.as_pydata_coo(val)
                        val = val.reshape((-1,))
                        for k in range(val.nnz):
                            data2.append(val.data[k])
                            I2.append(i)
                            J2.append(attr_j_start[j]+val.coords[0,k])
                    else:      
                        val = _utils.as_numpy_array(val)
                        for k, val_elem in enumerate(val.flatten()):
                            if val_elem:
                                data2.append(val_elem)
                                I2.append(i)
                                J2.append(attr_j_start[j]+k)
            data, I, J, N = data2, I2, J2, attr_len_cum[-1]
        
    # construct array
    if concat or all_scalar:
        res = scipy.sparse.coo_matrix((data,(I,J)), shape=(M,N), dtype=dtype)
        if mode == 'numpy':
            res = _utils.as_numpy_array(res)
        elif mode == 'scipy-sparse':
            res.eliminate_zeros()
        elif mode == 'pydata-sparse':
            res = _utils.as_pydata_coo(res)
        else: #pragma no cover
            raise ValueError('mode:'+mode+' not supported.')
    else:
        if mode == 'scipy-sparse':
            raise ValueError('scipy-sparse mode works only when concat is '
                             'True or all attributes are scalar.')
        elif mode == 'numpy':
            shape = (M, N) + attr_shapes[0]
            dtype = np.result_type(*data) if dtype is None else dtype
            res = np.zeros(shape, dtype)
            for i, j, val in zip(I, J, data):
                res[i, j] = _utils.as_numpy_array(val)
        elif mode == 'pydata-sparse':
            _config.assert_has_package('sparse')
            import sparse
            shape = (M, N) + attr_shapes[0]
            coords = [[] for _ in range(len(shape))]
            data2 = []
            for i, j, val in zip(I, J, data):
                val = _utils.as_pydata_coo(val)
                for k in range(val.nnz):
                    coords[0].append(i)
                    coords[1].append(j)
                    for l in range(len(shape)-2):
                        coords[l+2].append(val.coords[l, k])
                    data2.append(val.data[k])
            data = np.array(data2, dtype=dtype)
            res = sparse.COO(coords, data, shape=shape)
        else: #pragma no cover
            raise ValueError('mode:'+mode+' not supported.')

    nodes, attrs = list(nodes), list(attrs)
    if return_node_ids:
        if return_attr_ids:
            return res, nodes, attrs
        else:
            return res, nodes
    else:
        if return_attr_ids:
            return res, attrs
        else:
            return res
            
