import time,math,datetime,argparse,random,scipy.sparse,csv
import sklearn,sklearn.metrics,sklearn.cluster
import tensorflow as tf
import awesomeml.utils as aml_utils
import awesomeml.tensorflow.layers as aml_layers
import awesomeml.datasets as aml_ds
import awesomeml.tensorflow.ops as aml_tf_ops
import utils
from pathlib import Path
import sklearn.preprocessing as pp
import numpy as np

# ************************************************************
# args
# ************************************************************
parser = argparse.ArgumentParser()
# general
parser.add_argument('--ckpt-dir', type=Path, default=Path('./ckpt'))
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--no-test', action='store_true', default=False)
parser.add_argument('--no-sparse', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--node-dropout', type=float, default=0.6)
parser.add_argument('--edge-dropout', type=float, default=0.6)

# data
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--data-splitting', type=str, default='cvpr2019-602020')

# model
parser.add_argument('--layer-type', type=str, default='gcn')
parser.add_argument('--hidden-channels', type=int, default=[64], nargs='+')
parser.add_argument('--node-features', type=str, default=['xn'], nargs='+')
parser.add_argument('--edge-features', type=str, default=['eeti'], nargs='+')
parser.add_argument('--edge-norm', type=str, default='row')
parser.add_argument('--edge-order', type=int, default=1)
parser.add_argument('--adaptive', action='store_true', default=False)
parser.add_argument('--weighted', action='store_true', default=False)

if __name__ == '__main__' and '__file__' in globals():
    args = parser.parse_args()
else:
    args = parser.parse_args([])
print(args)

fname_prefix = args.data+'-'+args.data_splitting
fname_prefix += '-'+args.layer_type
fname_prefix += '-'+'-'.join(str(x) for x in args.hidden_channels)
fname_prefix += '-nfea-'+'-'.join([str(x) for x in args.node_features])
fname_prefix += '-efea-'+'-'.join([str(x) for x in args.edge_features])
fname_prefix += '-norm-'+args.edge_norm
fname_prefix += '-order-'+str(args.edge_order)
if args.layer_type.lower() == 'gat':
    if args.adaptive:
        fname_prefix += '-adaptive'
if args.weighted:
    fname_prefix += '-weighted'


# ************************************************************
# load data
# ************************************************************
X, Y, A, idx_train, idx_val, idx_test = utils.load_data(args)
K = A.shape[1] if X is None else X.shape[0]
nC = Y.shape[1]
Y = aml_utils.as_numpy_array(Y)
W = None
if args.weighted:
    W = utils.calc_class_weights(Y[...,idx_train,:])

# ************************************************************
# calculate node features
# ************************************************************
nodes = utils.calc_node_features(X, A, args)
print('nodes.shape=', nodes.shape)

# ************************************************************
# calculate edge features
# ************************************************************
edges = utils.calc_edge_features(X, A, args)
print('edges.shape=', edges.shape)

if args.no_sparse:
    nodes = aml_utils.as_numpy_array(nodes)
    edges = aml_utils.as_numpy_array(edges)

# ************************************************************
# construct model
# ************************************************************
def layer(layer_type, inputs, dim, training, args, **kwargs):
    """A wrapper to dispatch different layer construction
    """
    if layer_type.lower() == 'gcn':
        return aml_layers.graph_conv(
            inputs, dim, training,
            node_dropout=args.node_dropout,
            edge_dropout=args.edge_dropout,
            **kwargs)
    elif layer_type.lower() == 'gat':
        return aml_layers.graph_attention(
            inputs, dim, training,
            node_dropout=args.node_dropout,
            edge_dropout=args.edge_dropout,
            eps=1e-10,
            # we have a bug in the code for 'dsm' and 'sym' of gat
            #edge_normalize=args.edge_norm,
            adaptive=args.adaptive,
            **kwargs)
    else:
        raise ValueError('layer type:', layer_type, ' not supported.')

# reset computing graph
tf.reset_default_graph()

# input layer
training = tf.placeholder(dtype=tf.bool, shape=())
h, edges = nodes, edges

# hidden layers
for dim in args.hidden_channels:
    h, edges = layer(args.layer_type, (h, edges), dim, training, args,
                     activation=tf.nn.elu)
    
# classification layer
logits,_ = layer(args.layer_type, (h, edges), nC, training, args,
                 multi_edge_aggregation='mean')
Yhat = tf.one_hot(tf.argmax(logits, axis=-1), nC)
loss_train = utils.calc_loss(Y, logits, idx_train, W=W)
loss_val = utils.calc_loss(Y, logits, idx_val)
loss_test = utils.calc_loss(Y, logits, idx_test)

vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if
                   'bias' not in v.name and
                   'gamma' not in v.name]) * args.weight_decay
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
train_op = optimizer.minimize(loss_train + lossL2)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# ************************************************************
# training
# ************************************************************
(args.ckpt_dir/fname_prefix).mkdir(parents=True, exist_ok=True)
ckpt_path = args.ckpt_dir/fname_prefix/'checkpoint.ckpt'
print('ckpt_path=', ckpt_path)

bad_epochs = 0
scores_stop = dict(loss=math.inf,acc=-math.inf,f1=-math.inf,kappa=-math.inf)
saver = tf.train.Saver()
nan_happend = False
with tf.Session() as sess:
    sess.run(init_op)
                                     
    t0 = time.time()
    for epoch in range(args.epochs):
        t = time.time()
        # training step
        sess.run([train_op], feed_dict={training:True})

        # validation step
        [loss_train_np, loss_val_np, Yhat_np] = sess.run(
            [loss_train, loss_val, Yhat],
            feed_dict={training:False})
        scores_train = dict(loss=loss_train_np,
                            **utils.calc_metrics(Y, Yhat_np, idx_train))
        scores_val = dict(loss=loss_val_np,
                          **utils.calc_metrics(Y, Yhat_np, idx_val))
                                     
        if np.isnan(loss_train_np):
            nan_happend = True
            print('NaN loss, stop!')
            break

        if args.verbose:
            print('Epoch={} '.format(epoch) +
                  utils.scores_to_str(scores_train) +
                  ' | val:' + utils.scores_to_str(scores_val) +
                  ' t={:.4f}s'.format(time.time()-t))

        if (scores_val['loss'] <= scores_stop['loss']):
            bad_epochs = 0
            if not args.no_test:
                saver.save(sess, str(ckpt_path))
            scores_stop = scores_val
        else:
            bad_epochs += 1
            if bad_epochs == args.patience:
                print('Early stop - ' + utils.scores_to_str(scores_stop))
                print('totoal time {}'.format(
                    datetime.timedelta(seconds=time.time()-t0)))
                break

    # evaluation step
    # load check point
    if not args.no_test or nan_happend:
        saver.restore(sess, str(ckpt_path))
        [loss_val_np, Yhat_np] = sess.run(
            [loss_val, Yhat], feed_dict={training:False})
        scores_test = dict(loss=loss_val_np,
                           **utils.calc_metrics(Y, Yhat_np, idx_test))
        print('Testing - ' + utils.scores_to_str(scores_test))
        for name in scores_test:
            with open(fname_prefix+'-'+name+'.txt', 'a') as f:
                f.write('{} '.format(scores_test[name]))

