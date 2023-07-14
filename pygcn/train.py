from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor
from models import GCN
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--train_batch_size', type=int, default=1000,
                        help='batch size for train dataset')
parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='batch size for test dataset')
parser.add_argument('--test_epoch_gap', type=int, default=5,
                        help='run on test dataset between train epochs')
parser.add_argument('--test_batch_gap', type=int, default=100,
                        help='run on test dataset between train batches')
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

# Load data
adj, features, labels, indx_train, indx_test = load_data(args)
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

features = features.to(device)
labels = labels.to(device)
idx_output_train = torch.LongTensor(range(0, args.train_batch_size)).to(device)
train_batch_iterations = int(len(indx_train) / args.train_batch_size)

def train(epoch):
    t = time.time()
    total_labels = 0
    mrr = 0
    recall = [0,0,0,0,0,0,0,0,0,0]
    current_batch = indx_train.start if epoch % 2 == 0 else indx_train.stop
    model.train()
    for i in tqdm(range(0, train_batch_iterations)):
        optimizer.zero_grad()
        next_batch = current_batch + args.train_batch_size if epoch % 2 == 0 else current_batch - args.train_batch_size
        idx_train = range(current_batch, next_batch) if next_batch > current_batch else range(next_batch, current_batch)
        adj_train = sparse_mx_to_torch_sparse_tensor(adj[idx_train.start:idx_train.stop, idx_train.start:idx_train.stop]).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        output = model(features[idx_train], adj_train)
        loss_train = loss(output[idx_output_train], labels[idx_train])
        mrr_train, recall_train = accuracy(output[idx_output_train], labels[idx_train])
        mrr = mrr + mrr_train
        recall = [sum(x) for x in zip(recall, recall_train)]
        total_labels = total_labels + len(labels[idx_train])
        loss_train.backward()
        optimizer.step()
        current_batch = next_batch
        if (i + 1) % args.test_batch_gap == 0:
            test()
    recall = [x / total_labels for x in recall]
    mrr = mrr / total_labels

    print('Epoch: {:04d}'.format(epoch+1),
        'accuracy_train: {:.4f}'.format(recall[0]),
        'recall@3_train: {:.4f}'.format(recall[2]),
        'recall@5_train: {:.4f}'.format(recall[4]),
        'recall@7_train: {:.4f}'.format(recall[6]),
        'recall@10_train: {:.4f}'.format(recall[9]),
        'mrr_train: {:.4f}'.format(mrr),
        'time: {:.4f}s'.format(time.time() - t))
    if (epoch + 1) % args.test_epoch_gap == 0:
        test()
        

def test():
    t = time.time()
    total_labels = 0
    mrr = 0
    recall = [0,0,0,0,0,0,0,0,0,0]
    model.eval()
    current_batch = indx_test.start
    while current_batch < indx_test.stop:
        next_batch = min(current_batch + args.test_batch_size, indx_test.stop)
        idx_test = range(current_batch, next_batch)
        adj_test = sparse_mx_to_torch_sparse_tensor(adj[idx_test.start:idx_test.stop, idx_test.start:idx_test.stop]).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)
        output = model(features[idx_test], adj_test)
        idx_output_test = torch.LongTensor(range(0, next_batch - current_batch)).to(device)
        loss_test = loss(output[idx_output_test], labels[idx_test])
        mrr_test, recall_test = accuracy(output[idx_output_test], labels[idx_test])
        mrr = mrr + mrr_test
        recall = [sum(x) for x in zip(recall, recall_test)]
        total_labels = total_labels + len(labels[idx_test])
        current_batch = next_batch
    
    recall = [x / total_labels for x in recall]
    mrr = mrr / total_labels
    print("Test set results:",
        'accuracy: {:.4f}'.format(recall[0]),
        'recall@3: {:.4f}'.format(recall[2]),
        'recall@5: {:.4f}'.format(recall[4]),
        'recall@7: {:.4f}'.format(recall[6]),
        'recall@10: {:.4f}'.format(recall[9]),
        "mrr: {:.4f}".format(mrr),
        'time: {:.4f}s'.format(time.time() - t))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()