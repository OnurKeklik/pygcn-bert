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
from datetime import datetime
import os

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
parser.add_argument('--batch_size', type=int, default=1000,
                        help='batchsize for train')
parser.add_argument('--max_test_size', type=int, default=10000,
                            help='max test size to use')
parser.add_argument('--test_gap', type=int, default=5,
                        help='run on test dataset between epochs')
parser.add_argument('--test_batch_gap', type=int, default=100,
                        help='run on test dataset between batches')
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
if not os.path.isdir(model_directory):
    os.makedirs(model_directory)
model_path = os.path.join(model_directory, "model"+".pth")


logs = open("../data/" + args.dataset_name + "/logs.txt", "w")
# Load data
adj, features, labels, indx_train, idx_test = load_data(args)
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

top_mrr = 0
features = features.to(device)
labels = labels.to(device)
idx_output_train = torch.LongTensor(range(0, args.batch_size)).to(device)

idx_output_test = torch.LongTensor(range(0, len(idx_test))).to(device)
adj_test = sparse_mx_to_torch_sparse_tensor(adj[idx_test.start:idx_test.stop, idx_test.start:idx_test.stop]).to(device)
idx_test = torch.LongTensor(idx_test).to(device)
def train(epoch):
    loss_train_avg = 0
    recall_train_avg = [0,0,0,0,0,0,0,0,0,0]
    acc_train_avg = 0
    mrr_train_avg = 0
    current_batch = 0 if epoch % 2 == 0 else len(indx_train)
    t = time.time()
    model.train()
    total_batch_iterations = int(len(indx_train) / args.batch_size)
    for i in tqdm(range(0, total_batch_iterations)):
        optimizer.zero_grad()
        next_batch = current_batch + args.batch_size if epoch % 2 == 0 else current_batch - args.batch_size
        idx_train = range(current_batch, next_batch) if next_batch > current_batch else range(next_batch, current_batch)
        adj_train = sparse_mx_to_torch_sparse_tensor(adj[idx_train.start:idx_train.stop, idx_train.start:idx_train.stop]).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        output = model(features[idx_train], adj_train)
        loss_train = loss(output[idx_output_train], labels[idx_train])
        acc_train, mrr_train, recall_train = accuracy(output[idx_output_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        current_batch = next_batch
        loss_train_avg = loss_train_avg + loss_train.item()
        acc_train_avg = acc_train_avg + acc_train.item()
        mrr_train_avg = mrr_train_avg + mrr_train
        for j in range(0, len(recall_train_avg)):
            recall_train_avg[j] = recall_train_avg[j] + recall_train[j]
        batch_log= "batch time:" + str(datetime.now()) + ", total batch:" + str(total_batch_iterations) + ", current batch:" + str(i) + "\n"
        logs.write(batch_log)
        if i % args.test_batch_gap == 0:
            test()

    epoch_log = ('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train_avg / total_batch_iterations),
        'acc_train: {:.4f}'.format(acc_train_avg / total_batch_iterations),
        'recall@3_train: {:.4f}'.format(recall_train_avg[2] / total_batch_iterations),
        'recall@5_train: {:.4f}'.format(recall_train_avg[4] / total_batch_iterations),
        'recall@7_train: {:.4f}'.format(recall_train_avg[6] / total_batch_iterations),
        'recall@10_train: {:.4f}'.format(recall_train_avg[9] / total_batch_iterations),
        'mrr_train: {:.4f}'.format(mrr_train_avg / total_batch_iterations),
        'time: {:.4f}s'.format(time.time() - t))
    print(epoch_log)
    logs.write(str(epoch_log) + "\n")
    if epoch % args.test_gap == 0:
        test()
        

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
    logs.write(str(test_log) + "\n")
    if mrr_test > top_mrr:
        top_mrr = mrr_test
        torch.save(model, model_path)

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

logs.close()