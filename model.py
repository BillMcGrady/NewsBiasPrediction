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
    def __init__(self, data, num_features, num_nodes, hidden, support, num_bases, num_classes, 
        text_model='SkipThought', bias_feature=False):
        super(GCNModel, self).__init__()

        self.input_layer = InputLayer(data, num_features, text_model, bias_feature)
        self.gc1 = GraphConvolution(num_nodes, hidden, support, num_bases=num_bases,
            activation='tanh')
        self.gc2 = GraphConvolution(hidden, num_features, support, num_bases=num_bases,
                    activation='tanh')
        self.clf_bias = nn.Linear(num_features, num_classes)
       

class InputLayer(nn.Module):
    def __init__(self, data, num_feature, text_model='SkipThought', bias_feature=False):

        super(InputLayer, self).__init__()

        num_nodes = data['num_nodes']  
        num_docs = data['num_docs']  
        num_non_docs= data['num_non_docs']
        
        # Not used, but kept to reproduce result
        self.node_embedding = nn.Embedding(num_nodes, num_feature).cuda()
        self.node_embedding.weight.requires_grad=False
        self.text_model = text_model
        self.bias_feature = bias_feature

        if (bias_feature):
            fin = open('data/news_article_feature.pickle', 'rb')
            feature_matrix = pickle.load(fin)
            self.feature_matrix = torch.FloatTensor(feature_matrix).cuda()
            fin.close()

        if (text_model == 'SkipThought'):
            self.doc_embedding = nn.Embedding(num_docs, 4800).cuda()
            self.doc_embedding.weight.requires_grad=False
            if (bias_feature):
                self.linear = nn.Linear(4800+141, num_feature).cuda()
            else:
                self.linear = nn.Linear(4800, num_feature).cuda()

            skip_doc = pickle.load(open('data/doc_embeddings_skipthought_py2.pkl', 'rb'))
                
            f=open("data/mygraph.mapping","r")
            embd_list = []
            while True:
                l=f.readline()
                if not l:
                    break
                l=l.strip().split(" ")
                id=int(l[0])
                if id < num_non_docs:
                    continue
                doc=l[1]
                embd=list(skip_doc[doc])
                embd_list.append(embd)
            embd_list=np.array(embd_list)
            print(embd_list.shape)
            self.doc_embedding.weight.data.copy_(torch.from_numpy(embd_list))
        
        elif (text_model == 'HLSTM'):
            # tokenized_docs.json contains the tokenized content of news articles
            with open('data/tokenized_docs.json') as fin:
                doc_inputs_raw = json.load(fin)
                fin.close()
            mapping, reverse_mapping = {}, {}
            fin = open('data/mygraph.mapping', 'r')
            for line in fin.readlines():
                vals = line.strip().split()
                key, idx = vals[1], int(vals[0])
                mapping[key] = idx-num_non_docs
                reverse_mapping[idx] = key
            fin.close()

            text_inputs_raw = []
            for doc_id in range(num_non_docs, num_nodes):
                text_inputs_raw.append(doc_inputs_raw[reverse_mapping[doc_id]])

            fin = open('data/glove.6b.300d.pickle', 'rb')
            word_to_idx = pickle.load(fin)
            word_embeddings = pickle.load(fin)
            fin.close()
            # print(len(word_to_idx), word_embeddings.shape)

            all_text_input, all_text_lengths = generate_examples_hlstm(text_inputs_raw, word_to_idx)
            HIDDEN_SIZE = 64
            BIAS_FEAT_SIZE = 141
            self.doc_hlstm = HLSTM(HIDDEN_SIZE, word_to_idx, word_embeddings)
            if (bias_feature):
                self.linear = nn.Linear(HIDDEN_SIZE+BIAS_FEAT_SIZE, num_feature).cuda()
            else:
                self.linear = nn.Linear(HIDDEN_SIZE, num_feature).cuda()
            self.all_text_input, self.all_text_lengths = all_text_input, all_text_lengths

    def get_doc_embed(self, idx): 
        if (self.bias_feature):
            bias_feat = self.feature_matrix.index_select(0, idx)

        if (self.text_model == 'SkipThought'):
            text_embeds = self.doc_embedding(idx)
            if (self.bias_feature):
                text_embeds = torch.cat((text_embeds, bias_feat), 1)
            text_embeds = self.linear(text_embeds)
        elif (self.text_model == 'HLSTM'):
            # print('\t\t --- doc embedding used')
            batch_input, batch_sent_lengths, batch_doc_lengths = process_batch(
                self.all_text_input, self.all_text_lengths, idx, 400000)
            text_embeds = self.doc_hlstm(batch_input, batch_sent_lengths, batch_doc_lengths)
            if (self.bias_feature):
                text_embeds = torch.cat((text_embeds, bias_feat), 1) 
            text_embeds = self.linear(text_embeds)
        return text_embeds


# This implementation is based on the code at https://github.com/tkipf/relational-gcn/blob/master/rgcn/layers/graph.py
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support=1, 
                 activation='linear', num_bases=-1, bias=False):
        
        super(GraphConvolution, self).__init__()

        if (activation == 'linear'):
            self.activation = None
        elif (activation == 'sigmoid'):
            self.activation = nn.Sigmoid()
        elif (activation == 'tanh'):
            self.activation = nn.Tanh()
        else:
            print('Error: activation function not available')
            exit()

        self.input_dim = input_dim
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights

        assert support >= 1

        self.bias = bias
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

        if self.bias:
            self.b = Parameter(torch.FloatTensor(self.output_dim, 1))
        
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

        outputs = []
        for i in range(self.support):
            # print(features.size(), A[i].size())
            outputs.append(torch.spmm(A[i], supports[i]))

        output = torch.stack(outputs, dim=1).sum(1)            

        if self.bias:
            output += self.b
        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output
