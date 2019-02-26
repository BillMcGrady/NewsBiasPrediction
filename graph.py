from __future__ import print_function

import pickle
import numpy as np
import json
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from HLSTM import HLSTM

def generate_examples_hlstm(text_inputs_raw, word_to_idx):
    all_text_input, all_text_lengths = [], []
    sent_length, doc_length = [], []
    for sentences in text_inputs_raw:
        text_input = []
        text_lengths = []
        for tokens in sentences:
            sent_input = [word_to_idx[token.lower()] for token in tokens if token.lower() in word_to_idx]
            sent_len = len(sent_input)
            if (sent_len != 0):
                text_input.append(sent_input)
                text_lengths.append(sent_len)
        doc_len = len(text_input)
        if (doc_len != 0):
            all_text_input.append(text_input)
            all_text_lengths.append(text_lengths)
            sent_length.extend(text_lengths)
            doc_length.append(doc_len)

    sent_length = np.array(sent_length)
    doc_length = np.array(doc_length)
    
    part = sent_length
    print(len(part), max(part), sum(part <= 128), sum(part <= 256), sum(part <= 512), sum(part <= 1024))
    part = doc_length
    print(len(part), max(part), sum(part <= 8), sum(part <= 16), sum(part <= 32), sum(part <= 64))

    return np.array(all_text_input), np.array(all_text_lengths)

def process_batch(all_text_input, all_text_lengths,
    batch_index, vocab_size, perm=None):
    
    MAX_SENT_LEN = 256
    if (perm is not None):
        batch_index = [perm[i] for i in batch_index]

    batch_input_raw = [all_text_input[i] for i in batch_index]
    batch_lengths_raw = [all_text_lengths[i] for i in batch_index]
    batch_sent_lengths = list(itertools.chain.from_iterable(batch_lengths_raw))
    batch_doc_lengths = [len(lengths) for lengths in batch_lengths_raw]
    max_sent_length = min(MAX_SENT_LEN, max(batch_sent_lengths))
    for i in range(len(batch_sent_lengths)):
        if (batch_sent_lengths[i] > max_sent_length):
            batch_sent_lengths[i] = max_sent_length
    # print(batch_sent_lengths, max_sent_length)

    batch_input = []
    for doc_input, sent_lengths in zip(batch_input_raw, batch_lengths_raw):
        for sent_input, sent_length in zip(doc_input, sent_lengths):
            if (sent_length > max_sent_length):
                batch_input.append(sent_input[:max_sent_length])
            else: 
                batch_input.append(sent_input + [vocab_size] * (max_sent_length - sent_length))
    
    batch_input = np.asarray(batch_input)
    batch_sent_lengths = np.asarray(batch_sent_lengths)
    batch_doc_lengths = np.asarray(batch_doc_lengths)
    
    return batch_input, batch_sent_lengths, batch_doc_lengths


class GCNModel(nn.Module):
    def __init__(self, num_features, num_nodes, hidden, support, num_bases, num_classes, 
        text_model='SkipThought', relation='distmult'):
        super(GCNModel, self).__init__()

        self.input_layer = InputLayer(num_features, text_model)
        self.gc1 = GraphConvolution(num_nodes, hidden, support, num_bases=num_bases,
            activation='relu')
        self.gc2 = GraphConvolution(hidden, hidden, support, num_bases=num_bases,
                    activation='relu')
        # gc3 = GraphConvolution(HIDDEN, HIDDEN, support, num_bases=BASES,
        #             activation='relu')
        # gc4 = GraphConvolution(HIDDEN, y.shape[1], support, num_bases=BASES,
        #             activation='softmax')

        self.clf_bias = nn.Linear(hidden, num_classes)
        if (relation == 'bilinear'):
            self.clf_rela1, self.clf_rela2 = nn.Bilinear(hidden, hidden, 1), nn.Bilinear(hidden, hidden, 1)
        elif (relation == 'distmult'):
            self.clf_rela_embed = nn.Embedding(2, hidden)
        else:
            print('relaton type not valid')
            exit()

class InputLayer(nn.Module):
    def __init__(self, num_feature, text_model='SkipThought'):

        super(InputLayer, self).__init__()

        self.node_embedding = nn.Embedding(12127, num_feature).cuda()
        # self.node_embedding.weight.requires_grad=False
        self.text_model = text_model

        if (text_model == 'SkipThought'):
            self.doc_embedding = nn.Embedding(12127-1742, 4800).cuda()
            self.doc_embedding.weight.requires_grad=False
            self.linear = nn.Linear(4800, num_feature).cuda()
            
            skip_doc = pickle.load(open('data/doc_embeddings_skipthought_py2.pkl', 'rb'))
                
            f=open("data/mygraph.mapping.supervised","r")
            embd_list = []
            while True:
                l=f.readline()
                if not l:
                    break
                l=l.strip().split(" ")
                id=int(l[0])
                if id<1742:
                    continue
                doc=l[1]
                embd=list(skip_doc[doc])
                embd_list.append(embd)
            embd_list=np.array(embd_list)
            print(embd_list.shape)
            self.doc_embedding.weight.data.copy_(torch.from_numpy(embd_list))
        
        elif (text_model == 'HLSTM'):
            with open('data/tokenized_docs.json') as fin:
                doc_inputs_raw = json.load(fin)
                fin.close()
            mapping, reverse_mapping = {}, {}
            fin = open('data/mygraph.mapping.supervised', 'r')
            for line in fin.readlines():
                vals = line.strip().split()
                key, idx = vals[1], int(vals[0])
                mapping[key] = idx-1742
                reverse_mapping[idx] = key
            fin.close()

            text_inputs_raw = []
            for doc_id in range(1742, 12127):
                text_inputs_raw.append(doc_inputs_raw[reverse_mapping[doc_id]])
            '''
            word_to_idx = {}
            word_embeddings = []
            fin = codecs.open('adj/glove.6B.300d.txt', 'r', 'utf8')
            idx = 0
            for line in fin.readlines():
                vals = line.strip().split()
                word, embed = vals[0], [float(v) for v in vals[1:]]
                word_to_idx[word] = idx
                idx += 1
                word_embeddings.append(embed)
            fin.close()
            word_embeddings.append([0] * 300)
            word_embeddings = np.array(word_embeddings)
            fout = open('glove.6b.300d.pickle', 'wb')
            pickle.dump(word_to_idx, fout)
            pickle.dump(word_embeddings, fout)
            fout.close()
            print(len(word_to_idx), word_embeddings.shape)
            '''
            fin = open('data/glove.6b.300d.pickle', 'rb')
            word_to_idx = pickle.load(fin)
            word_embeddings = pickle.load(fin)
            fin.close()
            print(len(word_to_idx), word_embeddings.shape)

            all_text_input, all_text_lengths = generate_examples_hlstm(text_inputs_raw, word_to_idx)
            HIDDEN_SIZE = 16
            self.doc_hlstm = HLSTM(HIDDEN_SIZE, word_to_idx, word_embeddings)
            self.all_text_input, self.all_text_lengths = all_text_input, all_text_lengths
            print('\t\t test', len(self.all_text_input), len(self.all_text_lengths))
            # exit()

    def forward(self):
        # index_1 = torch.LongTensor([i for i in range(1742)]).cuda()
        # index_2 = torch.LongTensor([i for i in range(12127-1742)]).cuda()
        # all_embed = torch.cat((self.node_embedding(index_1), 
        #     self.linear(self.doc_embedding(index_2))), 0)

        index_1 = torch.LongTensor([i for i in range(12127)]).cuda()
        all_embed = self.node_embedding(index_1)
        
        return all_embed

    def get_doc_embed(self, idx): 
        if (self.text_model == 'SkipThought'):
            text_embeds = self.linear(self.doc_embedding(idx))
        elif (self.text_model == 'HLSTM'):
            # print('\t\t --- doc embedding used')
            batch_input, batch_sent_lengths, batch_doc_lengths = process_batch(
                self.all_text_input, self.all_text_lengths, idx, 400000)
            text_embeds = self.doc_hlstm(batch_input, batch_sent_lengths, batch_doc_lengths)

        return text_embeds


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support=1, 
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., **kwargs):
        
        super(GraphConvolution, self).__init__(**kwargs)

        # self.init = initializations.get(init)
        if (activation == 'linear'):
            self.activation = None
        elif (activation == 'sigmoid'):
            self.activation = nn.Sigmoid()
        elif (activation == 'tanh'):
            self.activation = nn.Tanh()
        elif (activation == 'relu'):
            self.activation = nn.Tanh()
        elif (activation == 'softmax'):
            self.activation = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.dropout = nn.Dropout(dropout)

        assert support >= 1

        # self.W_regularizer = regularizers.get(W_regularizer)
        # self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.W = [Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.input_dim, self.output_dim)))
                for i in range(self.num_bases)]
            self.W_comp = Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.support, self.num_bases)))
        else:
            self.W = [Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.input_dim, self.output_dim)))
                for i in range(self.support)]
        for idx, item in enumerate(self.W):
            self.register_parameter('W_%d' % idx, item)

        print(self.W[0][:3])
        if self.bias:
            self.b = Parameter(torch.FloatTensor(self.output_dim, 1))

        # self.set_weights(self.initial_weights)
        
    # def get_output_shape_for(self, input_shapes):
    #     features_shape = input_shapes[0]
    #     output_shape = (features_shape[0], self.output_dim)
    #     return output_shape  # (batch_size, output_dim)

    def forward(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = []
        if self.num_bases > 0:
            V = []
            for i in range(self.support):
                basis_weights = []
                for j, a_weight in enumerate(self.W):
                    basis_weights.append(self.W_comp[i][j] * a_weight)
                V.append(torch.stack(basis_weights, dim=0).sum(0))
            for i in range(self.support):
                # print(V[i].size())
                supports.append(torch.spmm(features, V[i]))
        else:
            for a_weight in self.W:
                supports.append(torch.spmm(features, a_weight))
        
        for idx, support in enumerate(supports):
            flag = torch.sum(torch.isnan(support)) > 0
            if flag:
                print('nan in support %d' % idx)
                exit()

        outputs = []
        for i in range(self.support):
            # print(features.size(), A[i].size())
            outputs.append(torch.spmm(A[i], supports[i]))

        for idx, output in enumerate(outputs):
            flag = torch.sum(torch.isnan(output)) > 0
            if flag:
                print('nan in output %d' % idx)
                exit()
        output = torch.stack(outputs, dim=1).sum(1)            

        # if featureless add dropout to output, by elementwise multiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        # if self.featureless:
        #     output = self.dropout(output)

        if self.bias:
            output += self.b
        return self.activation(output)
