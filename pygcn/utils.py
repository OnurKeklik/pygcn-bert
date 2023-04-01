import numpy as np
import scipy.sparse as sp
import torch
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
import json
import os, psutil

def getAdjLabels(path, file_name):
    edges_unordered = np.loadtxt("{}{}.cites".format(path, file_name),
                                    dtype=np.int32)
    idx_initial = []
    with open("{}{}.content".format(path, file_name)) as infile:
        for line in infile:
            idx_initial.append(line.split()[0])
    idx = np.array(np.asarray(idx_initial), dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    ll = list(map(idx_map.get, edges_unordered.flatten()));
    for i in range(0, len(ll)):
        if ll[i] is None:
            ll[i] = 0
    edges = np.array(ll,
                     dtype=np.int32).reshape(edges_unordered.shape)
    labels = getLabels(path, file_name)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, labels

def getLabels(path, file_name):
    labels_initial = []
    with open("{}{}.content".format(path, file_name)) as infile:
        for line in infile:
            labels_initial.append(line.split()[-1])
    labels = encode_onehot_efficient(np.asarray(labels_initial))
    return labels

def getFeatures(path, file_name):
    features = torch.empty((0, 768), dtype=torch.float32)
    with open("{}{}.content".format(path, file_name)) as infile:
        for line in infile:
            feature = sp.csr_matrix(np.asarray(line.split()[1:-1]), dtype=np.float32)
            feature = normalize(feature)
            feature = np.array(feature.todense())
            feature = torch.FloatTensor(feature)
            features = torch.cat((features, feature), 0)
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    return features


def dataset_split(dataset_name):
    tr = open("../data/" + dataset_name + "/train.json")
    v = open("../data/" + dataset_name + "/val.json")
    te = open("../data/" + dataset_name + "/test.json")
    train = json.load(tr)
    val = json.load(v)
    test = json.load(te)
    return len(train), len(val), len(test)


def encode_onehot_efficient(labels):
    values = array(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded;

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(args):
    path = "../data/" + args.dataset_name  + "/"
    file_name = "data"
    print('Loading {}'.format(path + file_name))

    train_size, val_size, test_size = dataset_split(args.dataset_name)
    idx_train = range(0, train_size)
    idx_val = range(train_size, train_size + val_size)
    idx_test = range(train_size + val_size, train_size + val_size + test_size)

    adj, labels = getAdjLabels(path, file_name)
    features = getFeatures(path, file_name)
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    mrr = 0
    recall_list = [0,0,0,0,0,0,0,0,0,0]
    top_preds = torch.topk(output, 50).indices;
    for i in range(0, len(labels)):
        top_preds_list = top_preds[i].tolist()
        label = labels[i].tolist()
        for j in range(0, len(top_preds_list)):
            if top_preds_list[j] == label:
                mrr = mrr + (1 / (j + 1))
                if j < 10:
                    for k in range(j, 10):
                        recall_list[k] = recall_list[k] + 1;
                break

    recall_scores = [x / len(labels) for x in recall_list]
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels), mrr / len(labels), recall_scores


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
