from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch

from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor
from tqdm import tqdm
from datetime import datetime
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--max_test_size', type=int, default=200000,
                            help='max test size to use')
loss = torch.nn.NLLLoss()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    device = torch.device("cuda")
    torch.cuda.manual_seed(args.seed)
    print("using cuda..")
else:
    device = torch.device("cpu")

model_directory = "../model/" + args.dataset_name
model_path = os.path.join(model_directory, "model"+".pth")
model = torch.load(model_path)

adj, features, labels, indx_train, idx_test = load_data(args)

features = features.to(device)
labels = labels.to(device)

idx_output_test = torch.LongTensor(range(0, len(idx_test))).to(device)
adj_test = sparse_mx_to_torch_sparse_tensor(adj[idx_test.start:idx_test.stop, idx_test.start:idx_test.stop]).to(device)
idx_test = torch.LongTensor(idx_test).to(device)
        
def test():
    global top_mrr
    model.eval()
    output = model(features[idx_test], adj_test)
    loss_test = loss(output[idx_output_test], labels[idx_test])
    acc_test, mrr_test, recall_test = accuracy(output[idx_output_test], labels[idx_test])
    test_log = ("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
        'recall@3: {:.4f}'.format(recall_test[2]),
        'recall@5: {:.4f}'.format(recall_test[4]),
        'recall@7: {:.4f}'.format(recall_test[6]),
        'recall@10: {:.4f}'.format(recall_test[9]),
        "mrr= {:.4f}".format(mrr_test))
    print(test_log)

# Test model
t_total = time.time()
test()
print("Testing Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))