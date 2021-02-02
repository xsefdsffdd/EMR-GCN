# -*- coding:utf-8 -*-
# import tensorflow as tf
import codecs
import copy
import json
import os
import pickle
import random
from multiprocessing import JoinableQueue, Process

import numpy as np
import scipy.sparse as sp
from utils import register_interrupt

os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))


def get_reverse_mask(batch_mask_data):
    reverse = []
    for each in batch_mask_data:
        reverse.append([1 - x for x in each])
    return reverse


def seq_len(data):
    if len(data) > 0 and isinstance(data[0], list):
        return [seq_len(d) for d in data]
    else:
        return len(data)


def vectorize(data, vocab):
    if isinstance(data, list):
        return [vectorize(x, vocab) for x in data]
    else:
        try:
            vec = vocab[data]
        except KeyError:
            vec = vocab['UNK']
            vectorize.total_unk += 1
        return vec


vectorize.total_unk = 0


def remove_stw(data, mask, vocab):
    new_data = []
    new_mask = []
    for i in range(len(data)):
        if data[i] not in vocab:
            new_data.append(data[i])
            new_mask.append(mask[i])
    if not new_data:
        new_data = ['PAD']
    return new_data, new_mask


def read_data(json_path):
    with codecs.open(json_path, 'r', 'utf8') as f:
        file = json.load(f)
    return file


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


class Dataset:
    def __init__(self, data, prefix, config, shuffle=True, sequential=True):

        self.data = data
        self.prefix = prefix
        self.batch_size = config.batch_size
        self.config = config
        self.shuffle = shuffle
        self.sequential = sequential
        print("Dataset {} size: {}".format(self.prefix, len(self.data)))
        # load vocabulary
        word2id, id2word = pickle.load(open('../data/entity_vocab_emb.pkl', 'rb'))
        self.embedding = pickle.load(open('../data/entity_emb.pkl', 'rb'))
        self.vocab = word2id
        self.reverse_vocab = id2word
        self.vocab_size = len(word2id)
        # disease name vocab
        self.dnet2id = json.load(open('../data/label2id.json', 'rb'))
        # stop words
        self.stop_words = [w.strip() for w in open('../data/stw_and_punc/stopword.txt', encoding='utf-8')]
        self.stop_words = set(self.stop_words)
        # punctuations
        self.puncs = [w.strip() for w in open('../data/stw_and_punc/punc.txt', encoding='utf-8')]
        self.puncs = set(self.puncs)
        # load graphs
        self.name2graph = {}
        self._read_graphs('../data/graphs', prefix='w20')
        self.mask = json.load(open('../data/total_mask.json', 'rb'))
        self.total_mask = np.array([value for key, value in self.mask.items()])
        self.total_reverse_mask = np.array([1 - value for key, value in self.mask.items()])
        self.graph_w20 = preprocess_adj(self.name2graph['w20_aggregate'])
        self.graph_w201 = preprocess_adj(self.name2graph['w20_aggregate1'])
        self.dataset_ids = list(self.data.keys())
        self.id_queue = JoinableQueue(maxsize=20)
        self.batch_queue = JoinableQueue(maxsize=20)
        # processesr
        if not self.shuffle:
            num_batch_process = 1
        else:
            num_batch_process = 2
        self.batch_process = [Process(target=self._get_batch_ids, daemon=True)] + [
            Process(target=self._get_batch, daemon=True) for _ in range(num_batch_process)]
        for p in self.batch_process: p.start()

    def get_sequence(self, query, query_mask):
        sequence_word_index = []
        sequence = []
        for i in range(len(query_mask)):
            sequence.append(query[
                            max(0, i - self.config.sequence_word_size // 2):min(i + self.config.sequence_word_size // 2,
                                                                                len(query))])
            word_index = list(np.arange(max(0, i - self.config.sequence_word_size // 2),
                                        min(i + self.config.sequence_word_size // 2, len(query))))
            sequence_word_index.append(word_index)
        return sequence, sequence_word_index, query_mask

    def pad2(self, data, index, sequence_mask):
        new_data = []
        word_index = []
        sequence_main_len = []
        for i in range(len(data)):
            new_data.append(data[i] + ['PAD'] * (self.config.sequence_word_size - len(data[i])))
            word_index.append(index[i] + [0] * (self.config.sequence_word_size - len(data[i])))
            sequence_main_len.append(len(data[i]))
        for i in range(self.config.max_len - len(new_data)):
            new_data.append(['PAD'] * self.config.sequence_word_size)
            word_index.append([0] * self.config.sequence_word_size)
            sequence_main_len.append(0)
            sequence_mask.append(0)
        sequence_reverse_mask = [1 - x for x in sequence_mask]
        return new_data, word_index, sequence_main_len, sequence_mask, sequence_reverse_mask

    def _read_graphs(self, graph_dir, prefix=None):
        graphs = [os.path.join(graph_dir, graph) for graph in os.listdir(graph_dir) if 'npz' in graph]
        for graph in graphs:
            name = os.path.basename(graph).replace('entity_graph_', '').replace('.npz', '')
            self.name2graph[prefix + '_' + name] = sp.load_npz(graph)

    def pad(self, data):
        new_data = []
        for i in range(len(data)):
            new_data.append(data[i] + ['PAD'] * (self.config.max_len - len(data[i])))
        return new_data

    def _get_graphs(self, graph_name, batch_query):
        main_graph = self.name2graph[graph_name]
        batch_graphs = []
        for query in batch_query:
            graph = main_graph[query, :][:, query].toarray()
            graph = self._normalize_graph(graph)
            batch_graphs.append(graph)
        batch_graphs = np.array(batch_graphs)
        return batch_graphs

    def _normalize_graph(self, adj):
        np.fill_diagonal(adj, 1)
        sqrt_deg = np.diag(1.0 / np.sqrt(1e-10 + np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
        return adj

    def _one_hot(self, labels):
        y = []
        for label in labels:
            one_hot = [0 for l in range(len(self.dnet2id))]
            one_hot[label] = 1
            y.append(one_hot)
        return np.array(y)

    def _map_emb(self, data):
        batch_query_embed = []
        for i in range(len(data)):
            each = data[i]
            each_embed = []
            for word in each:
                word_embed = self.embedding[word]
                each_embed.append(word_embed)
            each_embed = np.array(each_embed, dtype=np.float64)
            batch_query_embed.append(each_embed)
        return np.array(batch_query_embed)

    def _get_batch_ids(self):
        register_interrupt()
        while True:
            if self.shuffle:
                random.shuffle(self.dataset_ids)
                print('shuffled dataset_jicheng')
            batches = []
            batch = []
            for i, example in enumerate(self.dataset_ids):
                batch.append(example)
                if len(batch) == self.batch_size or i == (len(self.dataset_ids) - 1):
                    batches.append(batch)
                    batch = []
            if self.shuffle:
                random.shuffle(batches)
                print('shuffled batches')
            for batch in batches:
                self.id_queue.put(batch)
            self.id_queue.join()
            self.batch_queue.join()
            self.batch_queue.put('EndOfEpoch')

    def _get_batch(self):
        register_interrupt()
        while True:
            batch_ids = self.id_queue.get()
            batch_query_data = []
            batch_mask_data = []
            batch_reverse_mask_data = []
            batch_target_data = []
            problem_list = []
            batch_sequence_word_index = []
            batch_sequence_main_len = []
            batch_sequence_mask = []
            batch_sequence_reverse_mask = []
            for id in batch_ids:
                problem = self.data[id]
                query = problem['mainSuit_map_concat'] + " " + problem['illnessHistory_map_concat']
                query_mask = problem["mainSuit_map_flag"] + problem["illnessHistory_map_flag"]
                query = query.split()
                problem_raw = copy.deepcopy(problem)
                problem_raw['query_raw'] = query
                problem_list.append(problem_raw)

                if self.config.remove_stw:
                    query = remove_stw(query, self.stop_words)

                if self.config.remove_punc:
                    query, query_mask = remove_stw(query, query_mask, self.puncs)

                if not self.sequential:
                    cc = list(zip(query, query_mask))
                    random.shuffle(cc)
                    query[:], query_mask[:] = zip(*cc)
                    if len(query) != len(query_mask):
                        print("mask出错！")

                if len(query) > self.config.max_len:
                    query = query[:self.config.max_len // 2] + query[-self.config.max_len // 2:]
                    query_mask = query_mask[:self.config.max_len // 2] + query_mask[-self.config.max_len // 2:]

                ################################
                batch_query_data.append(query)
                batch_mask_data.append(query_mask)
                batch_target_data.append(self.dnet2id[problem['diagnose']])

                sequence_word, sequence_word_index, sequence_mask = self.get_sequence(query, query_mask)
                sequence_word, sequence_word_index, sequence_main_len, sequence_mask, sequence_reverse_mask = self.pad2(
                    sequence_word, sequence_word_index, sequence_mask)
                batch_sequence_word_index.append(sequence_word_index)
                batch_sequence_main_len.append(sequence_main_len)
                batch_sequence_mask.append(sequence_mask)
                batch_sequence_reverse_mask.append(sequence_reverse_mask)
            batch_query_lens = seq_len(batch_query_data)
            batch_query_data = self.pad(batch_query_data)
            batch_reverse_mask_data = get_reverse_mask(batch_mask_data)
            batch_mask_data = np.array(batch_mask_data, dtype=np.float32)
            batch_reverse_mask_data = np.array(batch_reverse_mask_data, dtype=np.float32)
            batch_query_data = vectorize(batch_query_data, self.vocab)
            batch_query_array = np.array(batch_query_data, dtype=np.int32)
            batch_target_array = np.array(batch_target_data, dtype=np.int32)
            batch_target_array = self._one_hot(batch_target_array)

            batch_graph_array_w20 = self._get_graphs('w20_aggregate', batch_query_array)
            batch_graph_array_w201 = self._get_graphs('w20_aggregate1', batch_query_array)

            batch_sequence_word_index = np.array(batch_sequence_word_index, dtype=np.int32)
            batch_sequence_main_len = np.array(batch_sequence_main_len, dtype=np.int32)
            batch_sequence_mask = np.array(batch_sequence_mask, dtype=np.float32)
            batch_sequence_reverse_mask = np.array(batch_sequence_reverse_mask, dtype=np.float32)
            self.batch_queue.put({'processed': {
                'batch_query': batch_query_array,
                'batch_mask': batch_mask_data,
                'batch_reverse_mask': batch_reverse_mask_data,
                'batch_sequence_word_index': batch_sequence_word_index,
                'batch_sequence_main_len': batch_sequence_main_len,
                'batch_sequence_mask': batch_sequence_mask,
                'batch_sequence_reverse_mask': batch_sequence_reverse_mask,
                'target': batch_target_array,
                'query_len': batch_query_lens,
                'id': batch_ids,
                'feature': self.embedding,
                'total_mask': self.total_mask,
                'total_reverse_mask': self.total_reverse_mask,
                'graph_w20': self.graph_w20,
                'graph_w201': self.graph_w201,
                'batch_graph_w20': batch_graph_array_w20,
                'batch_graph_w201': batch_graph_array_w201},
                'raw': problem_list})
            self.id_queue.task_done()

    def get_batch(self):
        batch = self.batch_queue.get()
        self.batch_queue.task_done()
        return batch
