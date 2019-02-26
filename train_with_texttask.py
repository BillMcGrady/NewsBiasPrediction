from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from rgcn.layers.graph import GraphConvolution
# from rgcn.layers.input_adj import InputAdj
# from rgcn.utils import *
from graph import GraphConvolution, GCNModel
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
MODEL_NAME = DATASET + '_skipall_withtext_testttt'
# LOAD_MODEL_NAME = DATASET + '_Netonly'
SUPERVISE_FLAG = True
PRED_TYPE = 'all'   # select from 'net'/'netshareu'/'all'

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + DATASET + '.pickle', 'rb') as f:
    data = pkl.load(f)

A = data['A']
y = data['y']
train_idx = data['train_idx'] #[:135]
valid_idx = data['valid_idx']
# data['test_idx'] # 
test_idx = data['test_idx'] # np.concatenate((data['train_idx'][135:], data['test_idx']))
all_labels = data['all_labels'] 
all_followees = data['all_followees'] 
all_nodes = data['all_nodes']
all_docs = data['all_docs']
        
print(len(A), len(all_labels), len(all_followees), len(all_nodes), len(all_docs))
print(train_idx, test_idx.shape)
y = np.array(y.todense())
labels = np.argmax(y, axis=1)


# fin = open('%s_unsup_pred.pickle' % DATASET, 'rb')
# label_preds = pkl.load(fin)
# print(label_preds.shape)
# fin.close()


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
idx_train_set, idx_test_set = set(idx_train), set(idx_test)
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
    x_pos_f, y_pos_f = [], []
    pos_set = set([])
    for x, y in zip(x_pos, y_pos):
        if (x not in x_range or y not in y_range):
            continue
        x_pos_f.append(x)
        y_pos_f.append(y)
        pos_set.add('%d_%d' % (x, y))

    for y in y_range:
        x_sample = np.random.choice(x_range, 25, replace=False)
        for x in x_sample:
            if ('%d_%d' % (x, y) in pos_set):
                continue
            x_neg.append(x)
            y_neg.append(y)

    x_all, y_all, v_all = list(x_pos_f) + x_neg, list(y_pos_f) + y_neg, [1] * len(x_pos_f) + [0] * len(x_neg)    
    print(len(x_pos_f), len(x_neg))

    x_all = torch.LongTensor(x_all)
    y_all = torch.LongTensor(y_all)
    v_all = torch.FloatTensor(v_all)
    if (use_cuda):
        x_all = x_all.cuda()
        y_all = y_all.cuda()
        v_all = v_all.cuda()
        
    return (x_all, y_all, v_all)
'''
rela1 = get_rela_exp(A[0].tocoo().row, A[0].tocoo().col, all_followees, all_nodes)
rela2 = get_rela_exp(A[0].tocoo().row, A[0].tocoo().col, all_nodes, all_docs)
fout = open('relas_untyped.pickle', 'wb')
pkl.dump([rela1, rela2], fout)
fout.close()
'''
fin = open('relas_untyped.pickle', 'rb')
rela1, rela2 = pkl.load(fin)
fin.close()
print(rela1[1].size(), rela2[1].size())
# exit()
x_pos, y_pos = A[0].tocoo().row, A[0].tocoo().col
all_docs_set, all_nodes_set = set(all_docs), set(all_nodes)
node2adj = {}
for xi, yi in zip(x_pos, y_pos):
    if (xi not in all_docs_set or yi not in all_nodes_set):
        continue
    if xi not in node2adj:
        node2adj[xi] = []
    node2adj[xi].append(yi)
'''
fout = open('node2adj.pickle', 'wb')
pkl.dump(node2adj, fout)
fout.close()
fin = open('node2adj.pickle', 'rb')
node2adj = pkl.load(fin)
fin.close()
'''
print(len(node2adj))

# A_in = [InputAdj(sparse=True) for _ in range(support)]
# X_in = Input(shape=(X.shape[1],), sparse=True)

# Compile model
num_features = 16
model = GCNModel(num_features, num_nodes, HIDDEN, support, BASES, y.shape[1], 
    text_model='SkipThought', relation='distmult')
# model.load_state_dict(torch.load('result/%s' % (LOAD_MODEL_NAME)))
print(len(list(model.parameters())))

parameters = [p for p in list(model.parameters())
    if p.requires_grad]
for p in parameters:
    print(p.size())
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
    model = model.cuda()

best_acc, best_test_acc = 0, 0
best_loss = 10000000
num_docs = 12127-1742
text_batch_size = num_docs
best_result, best_test_result = None, None

def test_helper():
    # Predict on full dataset
    embeds_0 = inputs[0] # model.input_layer()
    embeds_1 = model.gc1([embeds_0] + inputs[1:])
    embeds_2 = model.gc2([embeds_1] + inputs[1:])
    scores = model.clf_bias(embeds_2)
    preds = torch.argmax(scores, dim=1)
    loss_train = cross_entropy_loss(scores[idx_train], labels_train)
    loss_valid = cross_entropy_loss(scores[idx_valid], labels_valid)
    loss_test = cross_entropy_loss(scores[idx_test], labels_test)
    correct_train = torch.sum(preds[idx_train] == labels_train)
    correct_valid = torch.sum(preds[idx_valid] == labels_valid)
    correct_test = torch.sum(preds[idx_test] == labels_test)
    train_acc_net = correct_train.item()/labels_train.size(0)
    valid_acc_net = correct_valid.item()/labels_valid.size(0)
    test_acc_net = correct_test.item()/labels_test.size(0)
    print(train_acc_net, valid_acc_net, test_acc_net)

    scores_shareu = []
    scores_value = scores.data.cpu().numpy()
    for node in range(12127):
        scores_t = np.zeros(3) 
        if (node not in node2adj):
            scores_shareu.append(scores_t)
            continue

        adj = list(node2adj[node])
        
        # this_doc_embed = doc_embed[node-1739]
        # adj_embed = output_embed[adj]
        # adj_coef = softmax(np.matmul(adj_embed, this_doc_embed).reshape(1, -1)).reshape(-1)
        adj_coef = [1/len(adj)] * len(adj)
        # print(adj_coef, adj_coef.shape)
        
        for user, coef in zip(adj, adj_coef):
            scores_t += coef * scores_value[user]#-135] # when use score from 2nd GCN
            # print(output_score[user])
        # print(scores)
        scores_shareu.append(scores_t) #/len(adj))
        # print(node, adj, gold[-1], pred[-1])
    scores_shareu = torch.FloatTensor(np.array(scores_shareu)).cuda()
    preds_shareu = torch.argmax(scores_shareu, dim=1)
    correct_train_shareu = torch.sum(preds_shareu[idx_train] == labels_train)
    correct_valid_shareu = torch.sum(preds_shareu[idx_valid] == labels_valid)
    correct_test_shareu = torch.sum(preds_shareu[idx_test] == labels_test)
    train_acc_shareu = correct_train_shareu.item()/labels_train.size(0)
    valid_acc_shareu = correct_valid_shareu.item()/labels_valid.size(0)
    test_acc_shareu = correct_test_shareu.item()/labels_test.size(0)
    print(train_acc_shareu, valid_acc_shareu, test_acc_shareu)

    scores_shareu += scores
    preds_shareu = torch.argmax(scores_shareu, dim=1)
    correct_train_shareu = torch.sum(preds_shareu[idx_train] == labels_train)
    correct_valid_shareu = torch.sum(preds_shareu[idx_valid] == labels_valid)
    correct_test_shareu = torch.sum(preds_shareu[idx_test] == labels_test)
    train_acc_netshareu = correct_train_shareu.item()/labels_train.size(0)
    valid_acc_netshareu = correct_valid_shareu.item()/labels_valid.size(0)
    test_acc_netshareu = correct_test_shareu.item()/labels_test.size(0)
    print(train_acc_netshareu, valid_acc_netshareu, test_acc_netshareu)

    text_idx_perm = [i for i in range(12127-1742)]
    scores_text = []
    for start in range(0, num_docs, text_batch_size):
        model.zero_grad()
        end = start + text_batch_size
        if (end > num_docs):
            end = num_docs
        doc_idx_list_raw = text_idx_perm[start:end]
        doctext_idx_list = torch.LongTensor(doc_idx_list_raw).cuda()
        batch_input = model.input_layer.get_doc_embed(doctext_idx_list)
            # torch.mm(model.gc2.W[0])
        scores_text.extend(list(model.clf_bias(batch_input).data.cpu().numpy()))
    scores_text = torch.FloatTensor(scores_text).cuda()
    preds_text = torch.argmax(scores_text, dim=1)
    # print(idx_train-1742)
    # print(preds_shareu[idx_train-1742])
    # exit()
    correct_train = torch.sum(preds_text[idx_train-1742] == labels_train)
    correct_valid = torch.sum(preds_text[idx_valid-1742] == labels_valid)
    correct_test = torch.sum(preds_text[idx_test-1742] == labels_test)
    train_acc_text = correct_train.item()/labels_train.size(0)
    valid_acc_text = correct_valid.item()/labels_valid.size(0)
    test_acc_text = correct_test.item()/labels_test.size(0)
    print(train_acc_text, valid_acc_text, test_acc_text)

    scores[1742:] += scores_text
    preds_nettext = torch.argmax(scores, dim=1)
    correct_train = torch.sum(preds_nettext[idx_train] == labels_train)
    correct_valid = torch.sum(preds_nettext[idx_valid] == labels_valid)
    correct_test = torch.sum(preds_nettext[idx_test] == labels_test)
    train_acc_nettext = correct_train.item()/labels_train.size(0)
    valid_acc_nettext = correct_valid.item()/labels_valid.size(0)
    test_acc_nettext = correct_test.item()/labels_test.size(0)
    print(train_acc_nettext, valid_acc_nettext, test_acc_nettext)

    scores_shareu[1742:] += scores_text
    preds_all = torch.argmax(scores_shareu, dim=1)
    correct_train = torch.sum(preds_all[idx_train] == labels_train)
    correct_valid = torch.sum(preds_all[idx_valid] == labels_valid)
    correct_test = torch.sum(preds_all[idx_test] == labels_test)
    train_acc_all = correct_train.item()/labels_train.size(0)
    valid_acc_all = correct_valid.item()/labels_valid.size(0)
    test_acc_all = correct_test.item()/labels_test.size(0)
    print(train_acc_all, valid_acc_all, test_acc_all)
    
    if (PRED_TYPE == 'net'):
        train_acc_sel, valid_acc_sel, test_acc_sel = train_acc_net, valid_acc_net, test_acc_net
    elif (PRED_TYPE == 'netshareu'):
        train_acc_sel, valid_acc_sel, test_acc_sel = train_acc_netshareu, valid_acc_netshareu, test_acc_netshareu
    elif (PRED_TYPE == 'all'):
        train_acc_sel, valid_acc_sel, test_acc_sel = train_acc_all, valid_acc_all, test_acc_all
    else:
        print('wrong PRED_TYPE')
        exit()
    result_table = [[train_acc_net, valid_acc_net, test_acc_net], 
        [train_acc_shareu, valid_acc_shareu, test_acc_shareu], 
        [train_acc_netshareu, valid_acc_netshareu, test_acc_netshareu], 
        [train_acc_text, valid_acc_text, test_acc_text], 
        [train_acc_nettext, valid_acc_nettext, test_acc_nettext], 
        [train_acc_all, valid_acc_all, test_acc_all]]
    return (loss_train.item(), loss_valid.item(), loss_test.item(), 
        train_acc_sel, valid_acc_sel, test_acc_sel, result_table)

# Fit
for epoch in range(1, NB_EPOCH + 1):
    # break

    # Log wall-clock time
    t = time.time()

    model.zero_grad()

    # Single training iteration
    embeds_0 = inputs[0] # model.input_layer()
    embeds_1 = model.gc1([embeds_0] + inputs[1:])
    embeds_2 = model.gc2([embeds_1] + inputs[1:])
    embeds_final = embeds_2
    # print(embeds_1.size())
    # print(embeds_1[:10])

    scores = model.clf_bias(embeds_2)
    # embeds_2 = model.gc2([embeds_1] + inputs[1:])
    # embeds_3 = model.gc2([embeds_2] + inputs[1:])
    # scores = model.gc3([embeds_3] + inputs[1:])
    # print(scores.size())
    # print(scores[:10])
    # exit()

    loss_train = cross_entropy_loss(scores[idx_train], labels_train)

    # supervised case
    # doctext_idx_list = torch.LongTensor(idx_train[135:]-1742).cuda()
    # batch_input = model.input_layer.get_doc_embed(doctext_idx_list)
    # scores_text = model.clf_bias(batch_input)
    # loss_text = cross_entropy_loss(scores_text, 
    #     labels_train[135:])
    # loss_train += loss_text

    # unsupervised case 
    # doctext_idx_list = torch.LongTensor(idx_test-1742).cuda()
    # batch_input = model.input_layer.get_doc_embed(doctext_idx_list)
    # scores_text = model.clf_bias(batch_input)
    # loss_text = cross_entropy_loss(scores_text, 
    #     torch.cuda.LongTensor(label_preds[idx_test]))
    # loss_train += loss_text

    # loss_train.backward()
    # optimizer.step()
    '''
    rela1_scores = torch.sigmoid(model.clf_rela1
        (embeds_final[rela1[0]], embeds_final[rela1[1]]).view(-1))
    rela2_scores = torch.sigmoid(model.clf_rela2
        (embeds_final[rela2[0]], embeds_final[rela2[1]]).view(-1))
    loss_rela1 = bce_loss(rela1_scores, rela1[2])
    loss_rela2 = bce_loss(rela2_scores, rela2[2])
    '''
    '''
    zero = torch.LongTensor([0]).cuda()
    one = torch.LongTensor([1]).cuda()
    rela1_scores = torch.sigmoid((model.clf_rela_embed(zero) * 
        embeds_final[rela1[0]] * embeds_final[rela1[1]]).sum(dim=1)).view(-1)
    rela2_scores = torch.sigmoid((model.clf_rela_embed(one) *
        embeds_final[rela2[0]] * embeds_final[rela2[1]]).sum(dim=1)).view(-1)
        
    loss_rela1 = bce_loss(rela1_scores, rela1[2])
    loss_rela2 = bce_loss(rela2_scores, rela2[2])
        
    print(loss_train.item(), loss_rela1.item(), loss_rela2.item())
    alpha, beta = 0.2, 0.2
    loss_train += alpha * loss_rela1 + beta * loss_rela2
    '''
    
    text_idx_perm = np.random.permutation(num_docs)
    for start in range(0, num_docs, text_batch_size):
        if (PRED_TYPE != 'all'):
            break
        model.zero_grad()
        end = start + text_batch_size
        if (end > num_docs):
            end = num_docs
        doc_idx_list_raw = text_idx_perm[start:end]

        embeds_0 = inputs[0] # model.input_layer()
        embeds_1 = model.gc1([embeds_0] + inputs[1:])
        embeds_2 = model.gc2([embeds_1] + inputs[1:])
        embeds_final = embeds_2
    
        doctext_idx_list, docnode_pos_idx, docnode_neg_idx, doctext_flag = [], [], [], []
        for tidx, doctext_idx in enumerate(doc_idx_list_raw):
            doctext_idx_list.append(doctext_idx)
            docnode_pos_idx.append(doctext_idx+1742)
            docnode_neg_idx.append(np.random.randint(1742, 12127, size=(5)))
            if (doctext_idx+1742 in idx_train_set):
                doctext_flag.append(tidx)
        docnode_pos_idx = np.array(docnode_pos_idx).reshape(-1, 1)
        docnode_neg_idx = np.array(docnode_neg_idx)
        docnode_idx_list = np.concatenate((docnode_pos_idx, docnode_neg_idx), axis=1)

        doctext_idx_list = torch.LongTensor(doctext_idx_list).cuda()
        batch_input = model.input_layer.get_doc_embed(doctext_idx_list)
        docnode_idx_list = torch.cuda.LongTensor(docnode_idx_list).cuda()
        # batch_target = model.gc1.W[0].index_select(0, docnode_idx_list.view(-1)).view(-1, 6, num_features)
        batch_target = embeds_final.index_select(0, docnode_idx_list.view(-1)).view(-1, 6, num_features)
   
        batch_scores = torch.bmm(batch_target, 
            batch_input.view(-1, num_features, 1)).squeeze()
        batch_labels = torch.LongTensor([0] * len(doctext_idx_list)).cuda()
        loss_text_node = cross_entropy_loss(batch_scores, batch_labels)
        
        if (len(doctext_flag) > 0):
            scores_text = model.clf_bias(batch_input)
            loss_text = cross_entropy_loss(scores_text[doctext_flag], 
                torch.LongTensor(labels[docnode_pos_idx][doctext_flag]).cuda().view(-1))
        else:
            loss_text = torch.FloatTensor([0]).cuda()
        # print(scores_text[doctext_flag].size())
        # print(loss_text)

        # print(loss_train.item(), loss_text_node.item())
        loss_train += 1.0 * loss_text_node + loss_text
        # print(loss_train)
        if (loss_train.item() == 0):
            continue
        loss_train.backward()
        optimizer.step()
        # break
    
    # loss_train.backward()
    # optimizer.step()
    
    if epoch % 1 == 0:

        (loss_train_val, loss_valid_val, loss_test_val, 
        train_acc_sel, valid_acc_sel, test_acc_sel, result_table) = test_helper()
        
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(loss_train_val),
              "train_acc= {:.4f}".format(train_acc_sel),
              "val_loss= {:.4f}".format(loss_valid_val),
              "val_acc= {:.4f}".format(valid_acc_sel),
              "test_loss= {:.4f}".format(loss_test_val),
              "test_acc= {:.4f}".format(test_acc_sel),
              "time= {:.4f}".format(time.time() - t))
        if (valid_acc_sel > best_acc):
        # if (loss_valid_val < best_loss):
            best_acc = valid_acc_sel
            best_loss = loss_valid_val
            best_result = (epoch, loss_train_val, loss_valid_val, loss_test_val, 
                result_table)
            torch.save(model.state_dict(), 'result/%s' % MODEL_NAME)
        if (test_acc_sel > best_test_acc):
            best_test_acc = test_acc_sel
            best_test_result = (epoch, result_table)
        
    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

print(best_result)
print(best_test_result)
fout = open('result_summary.txt', 'a')
fout.write(str(best_result) + '\n')
fout.write(str(best_test_result) + '\n')
fout.close()

# Testing
model.load_state_dict(torch.load('result/%s' % (MODEL_NAME)))
test_helper()

# fout = open('%s_unsup_pred.pickle' % DATASET, 'wb')
# preds_numpy = preds_shareu.data.cpu().numpy()
# print(preds_numpy.shape)
# pkl.dump(preds_numpy, fout)
# fout.close()
