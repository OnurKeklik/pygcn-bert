from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--exclude_val', type=bool, default=True, 
                    help="Excludes validation dataset, can be useful for big datasets")

loss = torch.nn.NLLLoss()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    print("using cuda..")
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    if not args.exclude_val:
        idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #print(len(output))
    #print(len(labels))
    #raise("exception")
    loss_train = loss(output[idx_train], labels[idx_train])
    acc_train, mrr_train, recall_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'recall@1_train: {:.4f}'.format(recall_train[0]),
          'recall@3_train: {:.4f}'.format(recall_train[2]),
          'recall@5_train: {:.4f}'.format(recall_train[4]),
          'recall@7_train: {:.4f}'.format(recall_train[6]),
          'recall@10_train: {:.4f}'.format(recall_train[9]),
          'mrr_train: {:.4f}'.format(mrr_train),
          'time: {:.4f}s'.format(time.time() - t))
    if not args.exclude_val:
        val()
    test()

def val():
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = loss(output[idx_val], labels[idx_val])
    acc_val, mrr_val, recall_val = accuracy(output[idx_val], labels[idx_val])
    print("Validation set results:",
      "loss= {:.4f}".format(loss_val.item()),
      "accuracy= {:.4f}".format(acc_val.item()),
      'recall@1: {:.4f}'.format(recall_val[0]),
      'recall@3: {:.4f}'.format(recall_val[2]),
      'recall@5: {:.4f}'.format(recall_val[4]),
      'recall@7: {:.4f}'.format(recall_val[6]),
      'recall@10: {:.4f}'.format(recall_val[9]),
      "mrr= {:.4f}".format(mrr_val))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = loss(output[idx_test], labels[idx_test])
    acc_test, mrr_test, recall_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          'recall@1: {:.4f}'.format(recall_test[0]),
          'recall@3: {:.4f}'.format(recall_test[2]),
          'recall@5: {:.4f}'.format(recall_test[4]),
          'recall@7: {:.4f}'.format(recall_test[6]),
          'recall@10: {:.4f}'.format(recall_test[9]),
          "mrr= {:.4f}".format(mrr_test))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()