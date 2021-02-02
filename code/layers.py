import tensorflow as tf
from inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def dot3(x, y):
    """Wrapper for tf.matmul (sparse vs dense)."""
    print(x.shape)
    if len(x.shape) == len(y.shape):
        res = tf.matmul(x, y)
    else:
        k = tf.shape(x)[0]
        m = tf.shape(x)[1]
        n = tf.shape(x)[2]
        p = tf.shape(y)[1]
        reshape_x = tf.reshape(x, [k * m, n])
        res = tf.reshape(tf.matmul(reshape_x, y), [k, m, p])
    return res


def dot4(x, y):
    y = tf.expand_dims(y, axis=-1)
    res = tf.multiply(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution_mix1(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.leaky_relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_mix1, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.support = placeholders['support']
        self.support_mix = [placeholders['support'], placeholders['support1']]
        self.mask = placeholders['batch_mask']
        self.reverse_mask = placeholders['batch_reverse_mask']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_mask = placeholders["total_mask"]
        self.total_reverse_mask = placeholders["total_reverse_mask"]
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support_mix)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
                self.vars['weights_' + str(i) + str(i)] = glorot([output_dim, output_dim],
                                                                 name='weights_' + str(i) + str(i))
            self.vars['weights_atten_0'] = glorot([output_dim * 2, output_dim], name='weights_atten_0')
            self.vars['weights_atten_1'] = glorot([output_dim, 1], name='weights_atten_1')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _masked_fill(self, attention):
        condition = tf.equal(attention, 0.0)
        tmp = tf.fill([tf.shape(attention)[0]], -1e18)
        result = tf.where(condition, tmp, attention)
        return result

    def self_attention(self, x, y):
        output = tf.concat([x, y], axis=1)
        attention = dot(output, self.vars['weights_atten_0'])
        attention = tf.nn.leaky_relu(attention)
        attention = dot(attention, self.vars['weights_atten_1'])
        attention = tf.squeeze(attention)
        attention = self._masked_fill(attention)
        attention = tf.expand_dims(attention, -1)
        attention = tf.nn.softmax(attention)
        return attention

    def inter_gcn(self, supports):
        inter_features = []
        for i in range(len(supports)):
            supports[i] = dot(supports[i], self.vars['weights_' + str(i) + str(i)])
        attention = self.self_attention(supports[0], supports[1])
        attention_feature = tf.multiply(supports[1], attention)
        inter_features.append(tf.nn.leaky_relu(tf.add(supports[0], attention_feature)))
        attention_feature = tf.multiply(supports[0], attention)
        inter_features.append(tf.nn.leaky_relu(tf.add(supports[1], attention_feature)))
        return inter_features

    def _call(self, inputs):
        xx = inputs
        # dropout
        for i in range(len(xx)):
            if self.sparse_inputs:
                xx[i] = sparse_dropout(xx[i], 1 - self.dropout, self.num_features_nonzero)
            else:
                xx[i] = tf.nn.dropout(xx[i], 1 - self.dropout)

        intra_feature = list()
        for i in range(len(self.support_mix)):
            if not self.featureless:
                pre_sup = dot(xx[i], self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support_mix[i], pre_sup, sparse=True)
            support = tf.nn.leaky_relu(support)
            intra_feature.append(support)

        inter_feature = self.inter_gcn(intra_feature)

        supports = []
        for i in range(len(self.support_mix)):
            mask_intra_feature = dot4(intra_feature[i], self.total_reverse_mask)
            mask_inter_feature = dot4(inter_feature[i], self.total_mask)
            supports.append(tf.add(mask_intra_feature, mask_inter_feature))
        return supports
