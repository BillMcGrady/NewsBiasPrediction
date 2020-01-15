from model import GraphConvolution, GCNModel
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pkl
import os
import sys
import time
import argparse

class Trainer:
    def __init__(self, args):

        print(args)
        
        # Define Parameters
        self.DATASET = args['dataset']
        self.NB_EPOCH = args['epochs']
        self.LR = args['learnrate']
        self.L2 = args['l2norm']
        self.HIDDEN = args['hidden']
        self.BASES = args['bases']
        self.use_cuda = True
        self.TEXT_MODEL = args['text_model']
        self.SUPERVISE_FLAG = args['supervise_flag']   
        self.PRED_TYPE = args['pred_type']   
        self.USE_BIAS = False
        self.MODEL_NAME = '%s_%s_%s' % (self.DATASET, self.TEXT_MODEL, self.SUPERVISE_FLAG) # '_64hlstm30_withtext'
        self.num_features = 16
        self.log_step = 1

        # Load Data
        dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
        raw_data, data = load_data(
                args, dirname, self.use_cuda, self.SUPERVISE_FLAG)
        A = raw_data['A']
        y = raw_data['y']
        y = np.array(y.todense())
        self.labels = np.argmax(y, axis=1)

        self.idx_train = data['idx_train'] 
        self.idx_valid = data['idx_valid']  
        self.idx_test= data['idx_test'] 
        self.idx_train_set = data['idx_train_set'] 
        self.idx_test_set = data['idx_test_set'] 
        self.inputs = data['inputs']
        self.labels_train = data['labels_train'] 
        self.labels_valid = data['labels_valid']  
        self.labels_test = data['labels_test']  
        self.num_nodes = data['num_nodes']  
        self.num_docs = data['num_docs']  
        self.num_non_docs= data['num_non_docs']
        self.node2adj = data['node2adj']
        self.support = data['support']
        if (self.TEXT_MODEL == 'HLSTM'):
            self.text_batch_size = 30
        else:
            self.text_batch_size = self.num_docs
        if (self.SUPERVISE_FLAG == 'unsup2'):
            self.label_preds = data['label_preds']

        print('Data loaded successfully!')

        # Compile Model
        self.model = GCNModel(data, self.num_features, self.num_nodes, self.HIDDEN, self.support, self.BASES, y.shape[1], 
            text_model=self.TEXT_MODEL, bias_feature=self.USE_BIAS)
        if (self.use_cuda):
            self.model = self.model.cuda()

        parameters = [p for p in list(self.model.parameters())
            if p.requires_grad]
        # for p in parameters:
        #     print(p.size())
        self.optimizer = optim.Adam(parameters,
                            lr=self.LR, weight_decay=self.L2)

        if (self.TEXT_MODEL == 'HLSTM'):
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def _train(self):
        # Fit Model
        best_acc, best_test_acc = 0, 0
        best_loss = 10000000
        best_result, best_test_result = None, None
        for epoch in range(1, self.NB_EPOCH + 1):
            
            # Log wall-clock time
            t = time.time()

            self.model.zero_grad()

            # Single training iteration
            embeds_0 = self.inputs[0] 
            embeds_1 = self.model.gc1([embeds_0] + self.inputs[1:])
            embeds_2 = self.model.gc2([embeds_1] + self.inputs[1:])
            embeds_final = embeds_2
            scores = self.model.clf_bias(embeds_2)
            loss_train = self.cross_entropy_loss(scores[self.idx_train], self.labels_train) 
            if (self.SUPERVISE_FLAG == 'unsup2'):
                scores_comb = torch.cat((scores[self.idx_valid], scores[self.idx_test]), 0)
                labels_comb = torch.cat((
                    torch.LongTensor(self.label_preds[self.idx_valid]).cuda().view(-1),
                    torch.LongTensor(self.label_preds[self.idx_test]).cuda().view(-1)), 0)
                loss_train += self.cross_entropy_loss(scores_comb, labels_comb)
            if (self.TEXT_MODEL == 'HLSTM'):
                loss_train *= 10
                loss_train.backward()
                self.optimizer.step()
            
            text_idx_perm = np.random.permutation(range(self.num_non_docs, self.num_nodes))
            for start in range(0, self.num_docs, self.text_batch_size):
                if (self.PRED_TYPE not in ['text', 'all']):
                    break
                self.model.zero_grad()
                end = start + self.text_batch_size
                if (end > self.num_docs):
                    end = self.num_docs
                doc_idx_list_raw = text_idx_perm[start:end]

                embeds_0 = self.inputs[0] # model.input_layer()
                embeds_1 = self.model.gc1([embeds_0] + self.inputs[1:])
                embeds_2 = self.model.gc2([embeds_1] + self.inputs[1:])
                embeds_final = embeds_2
                
                doctext_idx_list, docnode_pos_idx, docnode_neg_idx, doctext_flag = [], [], [], []
                for tidx, doctext_idx in enumerate(doc_idx_list_raw):
                    doctext_idx_list.append(doctext_idx)
                    docnode_pos_idx.append(doctext_idx) 
                    docnode_neg_idx.append(np.random.randint(self.num_non_docs, self.num_nodes, size=(5)))
                    if (doctext_idx in self.idx_train_set):
                        doctext_flag.append(tidx)
                docnode_pos_idx = np.array(docnode_pos_idx).reshape(-1, 1)
                docnode_neg_idx = np.array(docnode_neg_idx)
                docnode_idx_list = np.concatenate((docnode_pos_idx, docnode_neg_idx), axis=1)

                doctext_idx_list = torch.LongTensor(doctext_idx_list).cuda()
                batch_input = self.model.input_layer.get_doc_embed(doctext_idx_list-self.num_non_docs)
                docnode_idx_list = torch.cuda.LongTensor(docnode_idx_list).cuda()
                batch_target = embeds_final.index_select(0, docnode_idx_list.view(-1)).view(-1, 6, self.num_features)
        
                batch_scores = torch.bmm(batch_target, 
                    batch_input.view(-1, self.num_features, 1)).squeeze()
                batch_labels = torch.LongTensor([0] * len(doctext_idx_list)).cuda()
                loss_text_node = self.cross_entropy_loss(batch_scores, batch_labels)

                if (len(doctext_flag) > 0):
                    scores_text = self.model.clf_bias(batch_input)
                    if (self.SUPERVISE_FLAG == 'supervise'):
                        loss_text = self.cross_entropy_loss(scores_text[doctext_flag], 
                            torch.LongTensor(self.labels[docnode_pos_idx][doctext_flag]).cuda().view(-1))
                    elif(self.SUPERVISE_FLAG == 'unsup2'):
                        loss_text = self.cross_entropy_loss(scores_text[doctext_flag], 
                            torch.LongTensor(self.label_preds[docnode_pos_idx][doctext_flag]).cuda().view(-1))
                else: 
                    loss_text = torch.tensor(0).cuda()

                if (self.TEXT_MODEL == 'HLSTM'):
                    loss_train = loss_text_node + loss_text
                    loss_train.backward()
                    self.optimizer.step()
                else:
                    loss_train += loss_text_node + loss_text
                break

            if (self.TEXT_MODEL != 'HLSTM'):
                loss_train.backward()
                self.optimizer.step()
            
            # Evaluate model and save the best one based on performance on validataion set
            if epoch % self.log_step == 0:

                (loss_train_val, loss_valid_val, loss_test_val, 
                train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel, result_table) = self._evaluate()
                
                print("Epoch: {:04d}".format(epoch),
                    "train_loss= {:.4f}".format(loss_train_val),
                    "train_acc= {:.4f}".format(train_acc_sel),
                    "val_loss= {:.4f}".format(loss_valid_val),
                    "val_acc= {:.4f}".format(valid_acc_sel),
                    "test_loss= {:.4f}".format(loss_test_val),
                    "test_acc= {:.4f}".format(test_acc_sel),
                    "time= {:.4f}".format(time.time() - t))
                if (valid_acc_sel > best_acc):
                    best_acc = valid_acc_sel
                    best_loss = loss_valid_val
                    best_result = (epoch, loss_train_val, loss_valid_val, loss_test_val, 
                        result_table)
                    torch.save(self.model.state_dict(), 'saved_models/%s' % self.MODEL_NAME)
                
            else:
                print("Epoch: {:04d}".format(epoch),
                    "time= {:.4f}".format(time.time() - t))
            break

        print(best_result)
        fout = open('logs/result_%s.txt' % self.DATASET, 'a')
        fout.write(str(best_result) + '\n')
        fout.close()

    def _evaluate(self, verbose=False):
        
        # Predict on full dataset
        embeds_0 = self.inputs[0] # model.input_layer()
        embeds_1 = self.model.gc1([embeds_0] + self.inputs[1:])
        embeds_2 = self.model.gc2([embeds_1] + self.inputs[1:])
        scores = self.model.clf_bias(embeds_2)
        preds = torch.argmax(scores, dim=1)
        loss_train = self.cross_entropy_loss(scores[self.idx_train], self.labels_train)
        loss_valid = self.cross_entropy_loss(scores[self.idx_valid], self.labels_valid)
        loss_test = self.cross_entropy_loss(scores[self.idx_test], self.labels_test)
        correct_train = torch.sum(preds[self.idx_train] == self.labels_train)
        correct_valid = torch.sum(preds[self.idx_valid] == self.labels_valid)
        correct_test = torch.sum(preds[self.idx_test] == self.labels_test)
        train_acc_net = correct_train.item()/self.labels_train.size(0)
        valid_acc_net = correct_valid.item()/self.labels_valid.size(0)
        test_acc_net = correct_test.item()/self.labels_test.size(0)
        if verbose:
            print('Graph:', train_acc_net, valid_acc_net, test_acc_net)

        scores_shareu = []
        scores_value = scores.data.cpu().numpy()
        for node in range(12127):
            scores_t = np.zeros(3) 
            if (node not in self.node2adj):
                scores_shareu.append(scores_t)
                continue

            adj = list(self.node2adj[node])
            
            adj_coef = [1/len(adj)] * len(adj)
            
            for user, coef in zip(adj, adj_coef):
                scores_t += coef * scores_value[user] 
            scores_shareu.append(scores_t) 
        scores_shareu = torch.FloatTensor(np.array(scores_shareu)).cuda()
        preds_shareu = torch.argmax(scores_shareu, dim=1)
        correct_train_shareu = torch.sum(preds_shareu[self.idx_train] == self.labels_train)
        correct_valid_shareu = torch.sum(preds_shareu[self.idx_valid] == self.labels_valid)
        correct_test_shareu = torch.sum(preds_shareu[self.idx_test] == self.labels_test)
        train_acc_shareu = correct_train_shareu.item()/self.labels_train.size(0)
        valid_acc_shareu = correct_valid_shareu.item()/self.labels_valid.size(0)
        test_acc_shareu = correct_test_shareu.item()/self.labels_test.size(0)
        if verbose:
            print('User:', train_acc_shareu, valid_acc_shareu, test_acc_shareu)

        scores_shareu += scores
        preds_netshareu = torch.argmax(scores_shareu, dim=1)
        correct_train_shareu = torch.sum(preds_netshareu[self.idx_train] == self.labels_train)
        correct_valid_shareu = torch.sum(preds_netshareu[self.idx_valid] == self.labels_valid)
        correct_test_shareu = torch.sum(preds_netshareu[self.idx_test] == self.labels_test)
        train_acc_netshareu = correct_train_shareu.item()/self.labels_train.size(0)
        valid_acc_netshareu = correct_valid_shareu.item()/self.labels_valid.size(0)
        test_acc_netshareu = correct_test_shareu.item()/self.labels_test.size(0)
        if verbose:
            print('G+U:', train_acc_netshareu, valid_acc_netshareu, test_acc_netshareu)

        text_idx_perm = [i for i in range(self.num_docs)]
        scores_text = []
        for start in range(0,self. num_docs, self.text_batch_size):
            self.model.zero_grad()
            end = start + self.text_batch_size
            if (end > self.num_docs):
                end = self.num_docs
            doc_idx_list_raw = text_idx_perm[start:end]
            doctext_idx_list = torch.LongTensor(doc_idx_list_raw).cuda()
            batch_input = self.model.input_layer.get_doc_embed(doctext_idx_list)
                # torch.mm(model.gc2.W[0])
            scores_text.extend(list(self.model.clf_bias(batch_input).data.cpu().numpy()))
        scores_text = torch.FloatTensor(scores_text).cuda()
        preds_text = torch.argmax(scores_text, dim=1)
        # print(idx_train-num_non_docs)
        # print(preds_shareu[idx_train-num_non_docs])
        # exit()
        correct_train = torch.sum(preds_text[self.idx_train-self.num_non_docs] == self.labels_train)
        correct_valid = torch.sum(preds_text[self.idx_valid-self.num_non_docs] == self.labels_valid)
        correct_test = torch.sum(preds_text[self.idx_test-self.num_non_docs] == self.labels_test)
        train_acc_text = correct_train.item()/self.labels_train.size(0)
        valid_acc_text = correct_valid.item()/self.labels_valid.size(0)
        test_acc_text = correct_test.item()/self.labels_test.size(0)
        if verbose:
            print('Text:', train_acc_text, valid_acc_text, test_acc_text)

        scores[self.num_non_docs:] += scores_text
        preds_nettext = torch.argmax(scores, dim=1)
        correct_train = torch.sum(preds_nettext[self.idx_train] == self.labels_train)
        correct_valid = torch.sum(preds_nettext[self.idx_valid] == self.labels_valid)
        correct_test = torch.sum(preds_nettext[self.idx_test] == self.labels_test)
        train_acc_nettext = correct_train.item()/self.labels_train.size(0)
        valid_acc_nettext = correct_valid.item()/self.labels_valid.size(0)
        test_acc_nettext = correct_test.item()/self.labels_test.size(0)
        if verbose:
            print('G+T:', train_acc_nettext, valid_acc_nettext, test_acc_nettext)

        scores_shareu[self.num_non_docs:] += scores_text
        preds_all = torch.argmax(scores_shareu, dim=1)
        correct_train = torch.sum(preds_all[self.idx_train] == self.labels_train)
        correct_valid = torch.sum(preds_all[self.idx_valid] == self.labels_valid)
        correct_test = torch.sum(preds_all[self.idx_test] == self.labels_test)
        train_acc_all = correct_train.item()/self.labels_train.size(0)
        valid_acc_all = correct_valid.item()/self.labels_valid.size(0)
        test_acc_all = correct_test.item()/self.labels_test.size(0)
        if verbose:
            print('G+U+T:', train_acc_all, valid_acc_all, test_acc_all)
        
        if (self.PRED_TYPE == 'net'):
            train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel = (
                train_acc_net, valid_acc_net, test_acc_net, preds.data.cpu().numpy())
        elif (self.PRED_TYPE == 'netshareu'):
            train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel = (
                train_acc_netshareu, valid_acc_netshareu, test_acc_netshareu, preds_netshareu.data.cpu().numpy())
        elif (self.PRED_TYPE == 'text'):
            train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel = (
                train_acc_text, valid_acc_text, test_acc_text, preds_text.data.cpu().numpy())
        elif (self.PRED_TYPE == 'all'):
            train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel = (
                train_acc_all, valid_acc_all, test_acc_all, preds_all.data.cpu().numpy())
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
            train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel, result_table)
        
    def run(self):
            
            # Train model
            self._train()
        
            # Evaluate trained model
            self.model.load_state_dict(torch.load('saved_models/%s' % (self.MODEL_NAME)), strict=False)
            (loss_train_val, loss_valid_val, loss_test_val, 
                train_acc_sel, valid_acc_sel, test_acc_sel, preds_sel, result_table) = self._evaluate()
            print(np.array(result_table)[:, 2])

            if (self.SUPERVISE_FLAG == 'unsup1'):
                fout = open('temp/%s_unsup_pred.pickle' % self.DATASET, 'wb')
                preds_numpy = preds_sel
                print(preds_numpy.shape)
                pkl.dump(preds_numpy, fout)
                fout.close()
            
            return np.array(result_table)[:, 2]


if __name__ == '__main__':
    # Hyper Parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default="newsbias_random",
                    help="Dataset string")
    ap.add_argument("-e", "--epochs", type=int, default=50,
                    help="Number training epochs")
    ap.add_argument("-hd", "--hidden", type=int, default=16,
                    help="Number hidden units")
    ap.add_argument("-b", "--bases", type=int, default=-1,
                    help="Number of bases used (-1: all)")
    ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
                    help="Learning rate")
    ap.add_argument("-l2", "--l2norm", type=float, default=5e-4,
                    help="L2 normalization of input weights")
    ap.add_argument("-tm", "--text_model", type=str, default="HLSTM",
                    help="text model: select from 'SkipThought'/'HLSTM'")
    ap.add_argument("-sf", "--supervise_flag", type=str, default="supervise",
                    help="supervise flag: select from 'supervise'/'unsup1'/'unsup2'")
    ap.add_argument("-pt", "--pred_type", type=str, default="all",
                    help="prediction method: select from 'net'/'netshareu'/'text'/all'")

    args = vars(ap.parse_args())
    dataset = args['dataset']

    num_folds = 3
    all_result = []
    for i in range(num_folds):
        print('------------------- fold: %d -------------------' % (i+1))
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        args['dataset'] = '%s_fold_%d' % (dataset, i+1)
        trainer = Trainer(args)
        fold_result = trainer.run()  
        all_result.append(fold_result)
    
    all_result = np.array(all_result)
    print('Average Accuracy of All Folds:', all_result.mean(0))