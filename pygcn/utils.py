import numpy as np
import scipy.sparse as sp
import torch
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
import json

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


#def load_data(path="../data/cora/", dataset="cora"):
def load_data(args):
    path = "../data/" + args.dataset_name  + "/"
    file_name = "data"
    print('Loading {}'.format(path + file_name))

    train_size, val_size, test_size = dataset_split(args.dataset_name)
    idx_train = range(0, train_size)
    idx_val = range(train_size, train_size + val_size)
    idx_test = range(train_size + val_size, train_size + val_size + test_size)

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, file_name),
                                        dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot_efficient(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, file_name),
                                    dtype=np.int32)

    ll = list(map(idx_map.get, edges_unordered.flatten()));
    for i in range(0, len(ll)):
        if ll[i] is None:
            ll[i] = 0

    edges = np.array(ll,
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    #idx_train = range(0, 9363)
    #idx_val = range(9363, 9855)
    #idx_test = range(9855, 16669)
    #idx_train = range(0, 30390)
    #idx_val = range(30390, 39771)
    #idx_test = range(39771, 49356)  

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

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
