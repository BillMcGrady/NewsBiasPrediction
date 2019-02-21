from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from rgcn.layers.graph import GraphConvolution
# from rgcn.layers.input_adj import InputAdj
# from rgcn.utils import *
from graph import GraphConvolution, InputLayer
from utils import *

import numpy as np
import pickle as pkl

import os
import sys
import time
import argparse

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="aifb",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=16,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--validation', dest='validation', action='store_true')
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=True)

args = vars(ap.parse_args())
print(args)

# Define parameters
DATASET = args['dataset']
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']
use_cuda = True

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
    data = pkl.load(f)

A = data['A']
y = data['y']
train_idx = data['train_idx']# [:135]
valid_idx = data['valid_idx']
# data['test_idx'] #
test_idx = data['test_idx'] #np.concatenate((data['train_idx'][135:], data['test_idx']))
all_labels = data['all_labels'] 
all_followees = data['all_followees'] 
all_nodes = data['all_nodes']
all_docs = data['all_docs']
        
print(len(all_labels), len(all_followees), len(all_nodes), len(all_docs))
print(train_idx, test_idx.shape)
y = np.array(y.todense())
labels = np.argmax(y, axis=1)

# print(type(train_idx), len(train_idx))
# print(type(test_idx), len(test_idx))
# print(type(A), type(y))
# print(len(A), y.shape)
# print(A[0], type(A[0]), A[0].shape)
# for i in range(len(A)):
#     print(A[i].shape)
# print('===')
# print(y[:500])
# exit()

# Get dataset splits
idx_train, idx_valid, idx_test = get_splits(y, train_idx, valid_idx, test_idx, VALIDATION)

num_nodes = A[0].shape[0]
support = len(A)

# Define empty dummy feature matrix (input is ignored as we set featureless=True)
# In case features are available, define them here and set featureless=False.
X = sp.eye(num_nodes).tocsr() 
# print(X.shape, X[:30])
# Normalize adjacency matrices individually
for i in range(len(A)):
    d = np.array(A[i].sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0.
    D_inv = sp.diags(d_inv)
    A[i] = D_inv.dot(A[i]).tocsr()
    # print('============== %d ===========' % i)
    # print(A[i][:50])

def get_rela_exp(x_pos, y_pos, x_range, y_range):
    x_neg, y_neg = [], []
    pos_set = set([])
    for x, y in zip(x_pos, y_pos):
        pos_set.add('%d_%d' % (x, y))

    for y in y_range:
        x_sample = np.random.choice(x_range, 25, replace=False)
        for x in x_sample:
            if ('%d_%d' % (x, y) in pos_set):
                continue
            x_neg.append(x)
            y_neg.append(y)

    x_all, y_all, v_all = list(x_pos) + x_neg, list(y_pos) + y_neg, [1] * len(x_pos) + [0] * len(x_neg)    
    print(len(x_pos), len(x_neg))

    x_all = torch.LongTensor(x_all)
    y_all = torch.LongTensor(y_all)
    v_all = torch.FloatTensor(v_all)
    if (use_cuda):
        x_all = x_all.cuda()
        y_all = y_all.cuda()
        v_all = v_all.cuda()
        
    return (x_all, y_all, v_all)

rela1 = get_rela_exp(A[0].tocoo().row, A[0].tocoo().col, all_followees, all_nodes)
rela2 = get_rela_exp(A[2].tocoo().row, A[2].tocoo().col, all_nodes, all_docs)
print(len(rela1[0]), len(rela2[0]))

# A_in = [InputAdj(sparse=True) for _ in range(support)]
# X_in = Input(shape=(X.shape[1],), sparse=True)

# Compile model
num_features = 16 # num_nodes
input_layer = InputLayer(num_features)
gc1 = GraphConvolution(num_features, HIDDEN, support, num_bases=BASES,
            activation='relu')
gc2 = GraphConvolution(HIDDEN, HIDDEN, support, num_bases=BASES,
            activation='relu')
clf_bias = nn.Linear(HIDDEN, y.shape[1])
clf_rela1, clf_rela2 = nn.Bilinear(HIDDEN, HIDDEN, 1), nn.Bilinear(HIDDEN, HIDDEN, 1)

# gc3 = GraphConvolution(HIDDEN, HIDDEN, support, num_bases=BASES,
#             activation='relu')
# gc4 = GraphConvolution(HIDDEN, y.shape[1], support, num_bases=BASES,
#             activation='softmax')

print(len(list(gc1.parameters())))
print(len(list(gc2.parameters())))

parameters = [p for p in list(input_layer.parameters()) + list(gc1.parameters()) + list(gc2.parameters()) # + list(gc3.parameters()) + list(gc4.parameters())
    + list(clf_bias.parameters()) + list(clf_rela1.parameters()) + list(clf_rela2.parameters()) 
    if p.requires_grad]
# for p in parameters:
#     print(p.size())
optimizer = optim.Adam(parameters,
                       lr=LR, weight_decay=L2)

cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
bce_loss = nn.BCELoss(reduction='mean')

inputs = [sparse_mx_to_torch_sparse_tensor(item) for item in [X] + A]
# print(len(inputs))
# for input in inputs:
#     print(input.size())
labels_train = torch.LongTensor(labels[idx_train])
labels_valid = torch.LongTensor(labels[idx_valid])
labels_test = torch.LongTensor(labels[idx_test])
    
if (use_cuda):
    inputs = [item.cuda() for item in inputs]
    labels_train = labels_train.cuda()
    labels_valid = labels_valid.cuda()
    labels_test = labels_test.cuda()
    gc1 = gc1.cuda()
    gc2 = gc2.cuda()
    clf_bias = clf_bias.cuda()
    clf_rela1 = clf_rela1.cuda()
    clf_rela2 = clf_rela2.cuda()
    
    # gc3 = gc3.cuda()
    # gc4 = gc4.cuda()

best_acc = 0
best_loss = 10000000
# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    gc1.zero_grad()
    gc2.zero_grad()
    # gc3.zero_grad()
    # gc4.zero_grad()

    # Single training iteration
    embeds_0 = input_layer()
    print(embeds_0.size())
    embeds_1 = gc1([embeds_0] + inputs[1:])
    embeds_2 = gc2([embeds_1] + inputs[1:])
    embeds_final = embeds_2
    # print(embeds_1.size())
    # print(embeds_1[:10])

    scores = clf_bias(embeds_2)
    # embeds_2 = gc2([embeds_1] + inputs[1:])
    # embeds_3 = gc2([embeds_2] + inputs[1:])
    # scores = gc3([embeds_3] + inputs[1:])
    # print(scores.size())
    # print(scores[:10])
    # exit()

    loss_train = cross_entropy_loss(scores[idx_train], labels_train)
    '''
    rela1_scores = torch.sigmoid(clf_rela1
        (embeds_final[rela1[0]], embeds_final[rela1[1]]).view(-1))
    rela2_scores = torch.sigmoid(clf_rela2
        (embeds_final[rela2[0]], embeds_final[rela2[1]]).view(-1))
    loss_rela1 = bce_loss(rela1_scores, rela1[2])
    loss_rela2 = bce_loss(rela2_scores, rela2[2])
    
    print(loss_train.item(), loss_rela1.item(), loss_rela2.item())
    alpha, beta = 0.5, 0.5
    loss_train += alpha * loss_rela1 + beta * loss_rela2
    '''
    loss_train.backward()
    optimizer.step()
    
    if epoch % 1 == 0:

        # Predict on full dataset
        embeds_0 = input_layer()
        embeds_1 = gc1([embeds_0] + inputs[1:])
        embeds_2 = gc2([embeds_1] + inputs[1:])
        # embeds_3 = gc2([embeds_2] + inputs[1:])
        scores = clf_bias(embeds_2)
        preds = torch.argmax(scores, dim=1)
        loss_train = cross_entropy_loss(scores[idx_train], labels_train)
        correct_train = torch.sum(preds[idx_train] == labels_train)
        loss_valid = cross_entropy_loss(scores[idx_valid], labels_valid)
        correct_valid = torch.sum(preds[idx_valid] == labels_valid)
        loss_test = cross_entropy_loss(scores[idx_test], labels_test)
        correct_test = torch.sum(preds[idx_test] == labels_test)

        loss_train_val = loss_train.item()
        loss_valid_val = loss_valid.item()
        loss_test_val = loss_test.item()
        
        train_acc = correct_train.item()/labels_train.size(0)
        valid_acc = correct_valid.item()/labels_valid.size(0)
        test_acc = correct_test.item()/labels_test.size(0)
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(loss_train_val),
              "train_acc= {:.4f}".format(train_acc),
              "val_loss= {:.4f}".format(loss_valid_val),
              "val_acc= {:.4f}".format(valid_acc),
              "test_loss= {:.4f}".format(loss_test_val),
              "test_acc= {:.4f}".format(test_acc),
              "time= {:.4f}".format(time.time() - t))
        if (valid_acc > best_acc):
        # if (loss_valid_val < best_loss):
            best_acc = valid_acc
            best_loss = loss_valid_val
            best_result = (epoch, loss_train_val, loss_valid_val, loss_test_val, 
                train_acc, valid_acc, test_acc)
        del loss_train, correct_train, loss_test, correct_test
        
    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

print(best_result)
fout = open('result_with_text_summary.txt', 'a')
fout.write(str(best_result) + '\n')
fout.close()

# Testing
# test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
# print("Test set results:",
#       "loss= {:.4f}".format(test_loss[0]),
#       "accuracy= {:.4f}".format(test_acc[0]))
