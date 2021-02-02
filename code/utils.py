import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import os
import scipy.sparse
#from numpy import random,mat
import random
import signal


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(batch_label, support, support1,batch_support,batch_support1, batch_feature,batch_query, batch_sequence_word_index, batch_sequence_main_len, batch_sequence_mask, batch_sequence_reverse_mask, total_mask,total_reverse_mask, batch_mask,batch_reverse_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['batch_label']: batch_label})
    feed_dict.update({placeholders['support']:support})
    feed_dict.update({placeholders['support1']:support1})
    feed_dict.update({placeholders['batch_support']: batch_support})
    feed_dict.update({placeholders['batch_support1']: batch_support1})
    feed_dict.update({placeholders['batch_feature']: batch_feature})
    feed_dict.update({placeholders['batch_query']:batch_query})
    feed_dict.update({placeholders['batch_sequence_word_index']: batch_sequence_word_index})
    feed_dict.update({placeholders['batch_sequence_main_len']: batch_sequence_main_len})
    feed_dict.update({placeholders['batch_sequence_mask']: batch_sequence_mask})
    feed_dict.update({placeholders['batch_sequence_reverse_mask']: batch_sequence_reverse_mask})
    feed_dict.update({placeholders['total_mask']:total_mask})
    feed_dict.update({placeholders['total_reverse_mask']: total_reverse_mask})
    feed_dict.update({placeholders['batch_mask']:batch_mask})
    feed_dict.update({placeholders['batch_reverse_mask']:batch_reverse_mask})
    feed_dict.update({placeholders['num_features_nonzero']: batch_feature[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def register_interrupt():
    def interrupt_handler(signal, frame):
        exit(0)
    signal.signal(signal.SIGINT, interrupt_handler)