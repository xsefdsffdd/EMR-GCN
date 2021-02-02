import tensorflow as tf
from layers import *
from metrics import *


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.prob = None
        self.pred_label = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def self_gcn(self, adj, main, act):
        with tf.variable_scope(self.name):
            weight = glorot([main.shape[2].value, self.output_dim])
            pre_sup = dot3(main, weight)
            main = dot3(adj, pre_sup)
            main = act(main)
            return main

    def inter_gcn(self, supports):
        with tf.variable_scope(self.name):
            for i in range(len(supports)):
                weight = glorot([supports[i].shape[2].value, self.output_dim])
                supports[i] = dot3(supports[i], weight)

            attention = glorot([self.config.batch_size, self.config.max_len, 1])
            attention_feature = tf.multiply(supports[1], attention)
            return tf.add(supports[0], attention_feature)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        self.activations.append(self.inputs)
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-2:])
            self.activations.extend(hidden)

        with tf.variable_scope(self.name):
            supp = tf.nn.embedding_lookup(tf.cast(self.activations[-1], tf.float32), self.placeholders["batch_query"])
            supp = self.self_gcn(self.placeholders["batch_support"], supp, tf.nn.leaky_relu)
            main = tf.nn.embedding_lookup(tf.cast(self.activations[-2], tf.float32), self.placeholders["batch_query"])
            main = self.self_gcn(self.placeholders["batch_support"], main, tf.nn.leaky_relu)
            main = self.inter_gcn([main, supp])
            self.outputs = tf.reduce_max(main, axis=1)

        # prepare for save model
        self.pred = self.outputs
        self.prob = tf.nn.softmax(self.outputs, name="prob")
        self.pred_label = tf.argmax(self.prob, axis=1, output_type=tf.int32, name="pred_label")

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        for key in list(self.vars.keys()):
            variables.append(self.vars[key])
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "../data/best_models/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "../data/best_models/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, config, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['batch_feature']
        self.input_dim = self.inputs.shape[1].value
        self.config = config
        self.output_dim = self.config.num_labels
        self.dropout = self.config.dropout
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.config.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['batch_label'])

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.placeholders['batch_label'], "accuracy")
        self.labels = tf.argmax(self.placeholders['batch_label'], 1)

    def _build(self):
        self.layers.append(GraphConvolution_mix1(input_dim=self.input_dim,
                                                 output_dim=self.config.hidden1,
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.leaky_relu,
                                                 dropout=self.dropout,
                                                 featureless=False,
                                                 sparse_inputs=False,
                                                 logging=self.logging))

        self.layers.append(GraphConvolution_mix1(input_dim=self.config.hidden1,
                                                 output_dim=self.config.hidden1,
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.leaky_relu,
                                                 dropout=self.dropout,
                                                 featureless=False,
                                                 sparse_inputs=False,
                                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
