from __future__ import division
from __future__ import print_function

import codecs
import configparser
import json
import os
import random
import time

import tensorflow as tf
from dataset import Dataset
from models import GCN
from utils import *
from metrics import *


def read_data(json_path):
    with codecs.open(json_path, 'r', 'utf8') as f:
        file = json.load(f)
    return file


def dict_fusion(dict1, dict2):
    for key in dict1.keys():
        z = dict1[key].copy()
        z.update(dict2[key])
        dict1[key] = z
    return dict1


class ModelConfig(object):
    def __init__(self, config_path):
        Config = configparser.ConfigParser()
        Config.read(config_path)
        self.learning_rate = float(Config.get('setting', 'learning_rate'))
        self.weight_decay = float(Config.get('setting', 'weight_decay'))
        self.epochs = int(Config.get('setting', 'epochs'))
        self.hidden1 = int(Config.get('setting', 'hidden1'))
        self.num_labels = int(Config.get('setting', 'num_labels'))
        self.batch_size = int(Config.get('setting', 'batch_size'))
        self.dropout = float(Config.get('setting', 'dropout'))
        self.max_len = int(Config.get('setting', 'max_len'))
        self.early_stopping = int(Config.get('setting', 'early_stopping'))
        self.vocab_size = int(Config.get('setting', 'vocab_size'))
        self.sequence_word_size = int(Config.get('setting', 'sequence_word_size'))

        if Config.get('setting', 'remove_stw') == u'True':
            self.remove_stw = True
        else:
            self.remove_stw = False

        if Config.get('setting', 'remove_punc') == u'True':
            self.remove_punc = True
        else:
            self.remove_punc = False

        print(self.remove_stw, self.remove_punc)


class Trainer:
    def __init__(self, dataset_train, dataset_valid, dataset_test):
        self.d_train = dataset_train
        self.d_valid = dataset_valid
        self.d_test = dataset_test
        self.current_best_loss, self.current_best_acc = 1e10, 0

    def train(self):
        step_time, loss, accuracy = [], [], []
        max_acc = 0.
        current_step = 0

        for epoch in range(config.epochs):
            for batch in iter(self.d_train.get_batch, 'EndOfEpoch'):
                if batch['processed']['target'].shape[0] < config.batch_size:
                    continue
                if current_step % 500 == 0 and current_step > 0:
                    self.evaluate(current_step, dataset=self.d_valid)

                start_time = time.time()
                step_loss, step_acc = self.train_step(batch['processed'])
                current_step += 1
                step_time.append(time.time() - start_time)
                loss.append(step_loss)
                accuracy.append(step_acc)

                if current_step % 50 == 0:
                    print("step {} step-time {:.2f} loss {:.4f}, acc {:.4f}".format(
                        current_step, np.mean(step_time), np.mean(loss), np.mean(accuracy)))
                    step_time, loss, accuracy = [], [], []
            else:
                print('Epoch {} finished'.format(epoch))

    def map(self, batch):
        batch_label = batch["target"]
        support = batch["graph_w20"]
        support1 = batch["graph_w201"]
        batch_support = batch["batch_graph_w20"]
        batch_support1 = batch["batch_graph_w201"]
        batch_feature = batch["feature"]
        batch_query = batch["batch_query"]
        batch_sequence_word_index = batch['batch_sequence_word_index']
        batch_sequence_main_len = batch['batch_sequence_main_len']
        batch_sequence_mask = batch['batch_sequence_mask']
        batch_sequence_reverse_mask = batch['batch_sequence_reverse_mask']
        total_mask = batch["total_mask"]
        total_reverse_mask = batch["total_reverse_mask"]
        batch_mask = batch["batch_mask"]
        batch_reverse_mask = batch["batch_reverse_mask"]
        return batch_label, support, support1, batch_support, batch_support1, batch_feature, batch_query, batch_sequence_word_index, batch_sequence_main_len, batch_sequence_mask, batch_sequence_reverse_mask, total_mask, total_reverse_mask, batch_mask, batch_reverse_mask

    def train_step(self, batch):
        batch_label, support, support1, batch_support, batch_support1, batch_feature, batch_query, batch_sequence_word_index, batch_sequence_main_len, batch_sequence_mask, batch_sequence_reverse_mask, total_mask, total_reverse_mask, batch_mask, batch_reverse_mask = self.map(
            batch)
        feed_dict = construct_feed_dict(batch_label, support, support1, batch_support, batch_support1, batch_feature,
                                        batch_query, batch_sequence_word_index, batch_sequence_main_len,
                                        batch_sequence_mask, batch_sequence_reverse_mask, total_mask,
                                        total_reverse_mask, batch_mask, batch_reverse_mask, placeholders)
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        return outs[1], outs[2]

    def test_step(self, batch):
        batch_label, support, support1, batch_support, batch_support1, batch_feature, batch_query, batch_sequence_word_index, batch_sequence_main_len, batch_sequence_mask, batch_sequence_reverse_mask, total_mask, total_reverse_mask, batch_mask, batch_reverse_mask = self.map(
            batch)
        feed_dict = construct_feed_dict(batch_label, support, support1, batch_support, batch_support1, batch_feature,
                                        batch_query, batch_sequence_word_index, batch_sequence_main_len,
                                        batch_sequence_mask, batch_sequence_reverse_mask, total_mask,
                                        total_reverse_mask, batch_mask, batch_reverse_mask, placeholders)
        outs = sess.run([model.loss, model.accuracy, model.pred], feed_dict=feed_dict)
        return outs[0], outs[1], outs[2]

    def evaluate(self, cur_step, dataset):
        all_accs, all_losses = [], []
        eval_losses, eval_accs = [], []
        eval_preds, eval_targets = [], []
        start_time = time.time()
        for batch in iter(dataset.get_batch, 'EndOfEpoch'):
            if batch['processed']['target'].shape[0] < config.batch_size:
                continue
            outs = self.test_step(batch['processed'])
            step_loss = outs[0]
            step_accs = outs[1]
            step_probs = outs[2]

            eval_losses.append(step_loss)
            eval_accs.append(step_accs)

            for i in range(len(step_probs)):
                eval_preds.append(int(np.argmax(step_probs[i])))
                eval_targets.append(int(np.argmax(batch['processed']['target'][i])))
        else:
            compute_scores(eval_targets, eval_preds)
            print('eval took {} sec, loss {}, accuracy {}'.format(time.time() - start_time, np.mean(eval_losses),
                                                                  np.mean(eval_accs)))
            all_losses.append(np.mean(eval_losses))
            all_accs.append(np.mean(eval_accs))

        if not istest:
            if all_accs[0] >= self.current_best_acc and cur_step > 0:
                self.current_best_acc = all_accs[0]
                print("current_best_acc:", self.current_best_acc)
                model.save(sess)


if __name__ == "__main__":
    istest = False
    config = ModelConfig('../data/ruku.ini')
    # Set random seed
    seed = random.randint(1, 200)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # Settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32, shape=(config.vocab_size, config.vocab_size)),
        'support1': tf.sparse_placeholder(tf.float32, shape=(config.vocab_size, config.vocab_size)),
        'batch_support': tf.placeholder(tf.float32, shape=(config.batch_size, config.max_len, config.max_len)),
        'batch_support1': tf.placeholder(tf.float32, shape=(config.batch_size, config.max_len, config.max_len)),
        'batch_feature': tf.placeholder(tf.float32, shape=(config.vocab_size, config.hidden1)),
        'batch_query': tf.placeholder(tf.int32, shape=[config.batch_size, config.max_len]),
        'batch_sequence_word_index': tf.placeholder(tf.int32, shape=(
            config.batch_size, config.max_len, config.sequence_word_size)),
        'batch_sequence_main_len': tf.placeholder(tf.int32, shape=(config.batch_size, config.max_len)),
        'batch_sequence_mask': tf.placeholder(tf.float32, shape=(config.batch_size, config.max_len)),
        'batch_sequence_reverse_mask': tf.placeholder(tf.float32, shape=(config.batch_size, config.max_len)),
        'total_mask': tf.placeholder(tf.float32, shape=(config.vocab_size)),
        'total_reverse_mask': tf.placeholder(tf.float32, shape=(config.vocab_size)),
        'batch_mask': tf.placeholder(tf.float32, shape=(config.batch_size, config.max_len)),
        'batch_reverse_mask': tf.placeholder(tf.float32, shape=(config.batch_size, config.max_len)),
        'batch_label': tf.placeholder(tf.float32, shape=((config.batch_size, config.num_labels))),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # Create model
    model = GCN(placeholders, config, logging=True)
    # Initialize session
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_conf)

    if not istest:
        train_data = read_data('../data/train.mask.json')
        valid_data = read_data('../data/test.mask.json')
        test_data = read_data('../data/test.mask.json')
        dataset_train = Dataset(train_data, 'train', config, shuffle=True, sequential=True)
        dataset_valid = Dataset(valid_data, 'valid', config, shuffle=False, sequential=True)
        dataset_test = Dataset(test_data, 'test', config, shuffle=True, sequential=True)
        # Init variables
        sess.run(tf.global_variables_initializer())
        # model.load_layers(sess)
        trainer = Trainer(dataset_train, dataset_valid, dataset_test)
        trainer.train()
        sess.close()
    else:
        test_data = read_data('../data/test.mask.json')
        dataset_test = Dataset(test_data, 'test', config, shuffle=False, sequential=True)
        model.load(sess)
        trainer = Trainer([], [], dataset_test)
        trainer.evaluate(trainer.d_test)
        sess.close()
