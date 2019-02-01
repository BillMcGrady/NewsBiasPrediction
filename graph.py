from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

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
