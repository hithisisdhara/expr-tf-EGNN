import sklearn.metrics.pairwise, scipy.sparse, pathlib, pickle, sparse
import sklearn.preprocessing as pp
import numpy as np
import networkx as nx
import awesomeml.datasets as aml_ds
import awesomeml.ops as aml_ops
import awesomeml.graph as aml_graph
import awesomeml.utils as aml_utils
from collections import Counter
import tensorflow as tf

def load_data(args):
    G = aml_ds.load_dataset(args.data, cache=False)
    id_train, id_val, id_test = aml_ds.load_dataset_tvt(
        args.data, args.data_splitting)

    node_ids = list(G.nodes)
    node_id2idx = dict(zip(node_ids, range(len(node_ids))))
    
    idx_train = [node_id2idx[i] for i in id_train]
    idx_val = [node_id2idx[i] for i in id_val]
    idx_test = [node_id2idx[i] for i in id_test]

    X = aml_graph.node_attr_matrix(G, exclude_attrs=['category'], nodes=node_ids)
    Y = [G.nodes[i]['category'] for i in node_ids]
    Y = pp.label_binarize(Y, list(set(Y)))
    A = nx.adjacency_matrix(G, nodelist=node_ids)
    if scipy.sparse.issparse(Y):
        Y = Y.tocsr()
    
    return X, Y, A, idx_train, idx_val, idx_test
    

def calc_node_features(X, A, args):
    vals = []
    if X is not None:
        X = X.astype(np.float32)
    K = A.shape[1] if X is None else X.shape[0]
    EYE = scipy.sparse.eye(K, dtype=np.float32, format='coo')
    if A is not None:
        A = A.astype(np.float32)
    for fea in args.node_features:
        if fea.lower() == 'x':
            # raw x
            assert X is not None
            vals.append(X)
        elif fea.lower() == 'xn':
            # normalized x
            assert X is not None
            rowsum = sparse.as_coo(X).sum(axis=-1, keepdims=True)
            #rowsum.data[rowsum.data==0] = 1e-10
            rowsum.data = 1.0 / rowsum.data
            vals.append((sparse.as_coo(X) * rowsum).to_scipy_sparse())
        elif fea.lower() == 'di':
            # in degree
            vals.append((A.transpose()>0).astype(np.float32).sum(-1, keepdims=True))
        elif fea.lower() == 'do':
            # out degree
            vals.append((A>0).astype(np.float32).sum(-1, keepdims=True))
        elif fea.lower() == 'idi':
            # in degree (including self)
            vals.append((A.transpose()+EYE>0).astype(np.float32).sum(-1, keepdims=True))
        elif fea.lower() == 'ido':
            # out degree (including self)
            vals.append((A+EYE>0).astype(np.float32).sum(-1, keepdims=True))
        elif fea.lower() == 'i':
            # eye, aka onehot encoding
            vals.append(EYE)
        elif fea.lower() == 'w':
            vals.append(A)
        elif fea.lower() == 'wn':
            tmp = sparse.as_coo(A)
            rowsum = tmp.sum(axis=-1, keepdims=True)
            rowsum.data = 1.0 / rowsum.data
            tmp = tmp * rowsum
            vals.append(tmp.to_scipy_sparse())
        else:
            raise ValueError('node feature '+fea+' not supported.')
    return scipy.sparse.hstack(vals)

def calc_edge_features(X, A, args):
    vals = []
    K = A.shape[0] if X is None else X.shape[0]
    if X is not None:
        X = np.asarray(X.todense()).astype(np.float32)
    if A is not None:
        A = A.astype(np.float32)
    EYE = scipy.sparse.eye(K, dtype=np.float32, format='coo')
    for fea in args.edge_features:
        if fea.lower() == 'i':
            vals.append(EYE)
        elif fea.lower() == 'w':
            vals.append(A)
        elif fea.lower() == 'wt':
            vals.append(A.transpose())
        elif fea.lower() == 'wwt':
            vals.append(A+A.transpose())
        elif fea.lower() == 'wi':
            vals.append(A+EYE)
        elif fea.lower() == 'wti':
            vals.append(A.transpose()+EYE)
        elif fea.lower() == 'wwti':
            vals.append(A+A.transpose()+EYE)
        elif fea.lower() == 'e':
            vals.append((A>0).astype(np.float32))
        elif fea.lower() == 'et':
            vals.append((A.transpose()>0).astype(np.float32))
        elif fea.lower() == 'eet':
            vals.append((A+A.transpose()>0).astype(np.float32))
        elif fea.lower() == 'ei':
            vals.append((A+EYE>0).astype(np.float32))
        elif fea.lower() == 'eti':
            vals.append((A.transpose()+EYE>0).astype(np.float32))
        elif fea.lower() == 'eeti':
            vals.append((A+A.transpose()+EYE>0).astype(np.float32))
        else:
            raise ValueError('edge feature '+fea+' not supported.')
    vals = [sparse.as_coo(x) for x in vals]
    vals = sparse.stack(vals, axis=0)
    vals = aml_graph.normalize_adj(vals, args.edge_norm, assume_symmetric_input=False)

    A = vals
    vals = [vals]
    A_order = A
    for _ in range(1, args.edge_order):
        A_order = aml_ops.matmul(A_order, A)
        #A_order = sparse.COO.from_numpy(np.matmul(A_order, A))
        vals.append(A_order)
    ret = sparse.concatenate(vals, 1)
    return np.transpose(ret, [1,2,0])

def calc_class_weights(Y):
    nC = Y.shape[-1]
    Y = Y.reshape((-1,nC))
    N = float(Y.shape[0])
    w = []
    for i in range(nC):
        w.append( N / nC / Y[:,i].sum())
    return w

def calc_loss(y_true, logits, idx, loss_func=tf.losses.softmax_cross_entropy, W=None):

    y_true = y_true.astype(np.float32)
    if aml_utils.is_sparse(y_true):
        y_true = aml_ops.as_numpy(y_true)
    logits = tf.gather(logits, idx)
    y_true = y_true[idx]
    if W is None:
        ret = loss_func(y_true, logits)
    else:
        W = tf.tensordot(y_true, W, 1)
        ret = loss_func(y_true, logits, weights=W)
    return ret


def calc_metrics(y_true, y_pred, idx, is_multi_label=False):
    nC = y_true.shape[-1]
    y_true = y_true[...,idx,:].reshape((-1,nC))
    y_pred = y_pred[...,idx,:].reshape((-1,nC))
    if is_multi_label:
        mf1 = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
        Mf1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
        return dict(mf1=mf1, Mf1=Mf1)
    else:
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
        kappa = sklearn.metrics.cohen_kappa_score(y_true.argmax(-1), y_pred.argmax(-1))
        return dict(acc=acc, f1=f1, kappa=kappa)



def calc_scores(Y, Yhat):
    Yhat = cluster_matching(Yhat, Y)
    acc = sklearn.metrics.accuracy_score(Y, Yhat)
    nmi = sklearn.metrics.normalized_mutual_info_score(Y, Yhat)
    f1 = sklearn.metrics.f1_score(Y, Yhat, average='macro')
    precision = sklearn.metrics.precision_score(Y, Yhat, average='macro')
    ari = sklearn.metrics.adjusted_rand_score(Y, Yhat)
    return dict(acc=acc, nmi=nmi, f1=f1, precision=precision, ari=ari)


def scores_to_str(scores):
    ret = ''
    for name in scores:
        ret += '{}={:.4f} '.format(name, scores[name])

    ret = ret[:-1]
    return ret
