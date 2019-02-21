import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import six.moves.cPickle as pickle
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence #, pad_sequence, pack_sequence

from utils import *

class HLSTM(torch.nn.Module):
    def __init__(self, hidden_size, word_to_idx, word_embeddings):
        """
        In the constructor we construct instances that we will use
        in the forward pass.
        """
        super(HLSTM, self).__init__()
        rng = np.random.RandomState(1)
        
        word_count, embedding_size = word_embeddings.shape
        self.word_embeddings = nn.Embedding(word_count, embedding_size, padding_idx=-1).cuda()
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))   
        self.word_embeddings.weight.requires_grad = False     
        print(self.word_embeddings)

        self.sent_LSTM = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True).cuda()
        self.doc_LSTM = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True).cuda()
        self.word_to_idx = word_to_idx

        hidden_size2 = 100
        self.W_H_sent = nn.Linear(hidden_size, hidden_size2).cuda()
        self.v_sent = nn.Linear(hidden_size2, 1, bias=False).cuda()
        self.W_H = nn.Linear(hidden_size, hidden_size2).cuda()
        self.v = nn.Linear(hidden_size2, 1, bias=False).cuda()
        # print(self.v.weight)
        self.tanh = nn.Tanh()    
        self.dw_cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum') # elementwise_mean
        
    def get_sent_embeds(self, batch_input, sentence_lengths):
        max_sent_length = max(sentence_lengths)
        # print(max_sent_length)
        sent_out_mask = []
        for leng in sentence_lengths:
            sent_out_mask.append([1] * leng + [0] * (max_sent_length-leng))
        sent_out_mask_var = to_variable(sent_out_mask)
        
        sent_sort_index = sorted(range(len(sentence_lengths)), key=lambda k: sentence_lengths[k], reverse=True)
        sent_sort_index_map = {old: new for new, old in enumerate(sent_sort_index)}
        reverse_sent_index = [sent_sort_index_map[i] for i in range(len(sent_sort_index))]

        batch_input_var = to_variable(batch_input, 'Long')
        sent_sort_index_var = to_variable(sent_sort_index, 'Long')
        reverse_sent_index_var = to_variable(reverse_sent_index, 'Long')

        sentence_lengths_var = to_variable(sentence_lengths, 'Float')
        word_embeds = self.word_embeddings(batch_input_var)
        word_embeds_sort = torch.index_select(word_embeds, 0, sent_sort_index_var)
        sentence_lengths_sort = sorted(sentence_lengths, reverse=True)
        
        # print(word_embeds_sort.size(), sentence_lengths_sort)

        # compute sentence representation using LSTM
        sent_packed = pack_padded_sequence(word_embeds_sort, sentence_lengths_sort, batch_first=True)
        sent_out_packed, _ = self.sent_LSTM(sent_packed)

        sent_out_unpacked, _ = pad_packed_sequence(sent_out_packed, batch_first=True)
        # sent_embeds = torch.index_select(sent_out_unpacked, 0, reverse_sent_index_var).sum(1) / sentence_lengths_var.view((-1, 1))
        
        sent_out_h = torch.index_select(sent_out_unpacked, 0, reverse_sent_index_var)
        # review_users_sent_var = to_variable(review_users_sent, 'Long')
        # sent_out_u = self.user_embeddings(review_users_sent_var)
        # review_targets_sent_var = to_variable(review_targets_sent, 'Long')
        # sent_out_t = self.target_embeddings(review_targets_sent_var)
        
        sent_out_e_p1 = self.W_H_sent(sent_out_h) 
        # sent_out_e_p2 = self.W_U(sent_out_u).unsqueeze(1).expand_as(sent_out_e_p1)
        # sent_out_e_p3 = self.W_T(sent_out_t).unsqueeze(1).expand_as(sent_out_e_p1)
        sent_out_e = self.v_sent(self.tanh(sent_out_e_p1)) # + sent_out_e_p2 + sent_out_e_p3
        sent_out_eexp = sent_out_e.exp().squeeze() * sent_out_mask_var
        sent_out_alpha = sent_out_eexp / sent_out_eexp.sum(1, keepdim=True)
        sent_embeds = (sent_out_h * sent_out_alpha.unsqueeze(2).expand_as(sent_out_h)).sum(1)
        # print(sent_embeds.size())
        
        return sent_embeds, sent_out_alpha

    def get_user_embeds_loss(self, batch_dw_user_query, batch_dw_user_candis):
        batch_size, sample_size = batch_dw_user_candis.shape
        query_tensor, candis_tensor = torch.tensor(batch_dw_user_query, dtype=torch.long).cuda(), \
            torch.tensor(batch_dw_user_candis, dtype=torch.long).cuda()
        query_embeds, candis_embeds = self.user_embeddings(query_tensor), \
            self.user_embeddings(candis_tensor)
        scores = torch.bmm(candis_embeds, query_embeds.view(batch_size, -1, 1)).squeeze()
        label_tensor = torch.tensor([0] * batch_size, dtype=torch.long).cuda()
        loss = self.dw_cross_entropy_loss(scores, label_tensor)

        return loss

    def forward(self, batch_input, sentence_lengths, document_lengths):
        """

        """
        # print(sentence_lengths, document_lengths)
        # print(len(sentence_lengths), len(document_lengths))
        # print(sum(sentence_lengths), sum(document_lengths))
        
        # print(len(review_users_sent), review_users_sent)
        '''
        max_sent_length = max(sentence_lengths)
        sent_out_mask = []
        for leng in sentence_lengths:
            sent_out_mask.append([1] * leng + [0] * (max_sent_length-leng))
        sent_out_mask_var = to_variable(sent_out_mask)
        '''
        max_doc_length = max(document_lengths)
        # print(max_doc_length)
        doc_out_mask = []
        for leng in document_lengths:
            doc_out_mask.append([1] * leng + [0] * (max_doc_length-leng))
        doc_out_mask_var = to_variable(doc_out_mask)
        # print(max_sent_length, sent_out_mask_var)
        # print(max_doc_length, doc_out_mask_var)

        '''
        # _, sent_sort_index = torch.sort(sentence_lengths, descending=True)
        sent_sort_index = sorted(range(len(sentence_lengths)), key=lambda k: sentence_lengths[k], reverse=True)
        sent_sort_index_map = {old: new for new, old in enumerate(sent_sort_index)}
        reverse_sent_index = [sent_sort_index_map[i] for i in range(len(sent_sort_index))]

        batch_input_var = to_variable(batch_input, 'Long')
        sent_sort_index_var = to_variable(sent_sort_index, 'Long')
        reverse_sent_index_var = to_variable(reverse_sent_index, 'Long')

        sentence_lengths_var = to_variable(sentence_lengths, 'Float')
        word_embeds = self.word_embeddings(batch_input_var)
        word_embeds_sort = torch.index_select(word_embeds, 0, sent_sort_index_var)
        sentence_lengths_sort = sorted(sentence_lengths, reverse=True)
        
        # print(word_embeds_sort.size(), sentence_lengths_sort)

        # compute sentence representation using LSTM
        sent_packed = pack_padded_sequence(word_embeds_sort, sentence_lengths_sort, batch_first=True)
        sent_out_packed, _ = self.sent_LSTM(sent_packed)

        sent_out_unpacked, _ = pad_packed_sequence(sent_out_packed, batch_first=True)
        sent_out_h = torch.index_select(sent_out_unpacked, 0, reverse_sent_index_var)
        review_users_sent_var = to_variable(review_users_sent, 'Long')
        sent_out_u = self.user_embeddings(review_users_sent_var)
        review_targets_sent_var = to_variable(review_targets_sent, 'Long')
        sent_out_t = self.target_embeddings(review_targets_sent_var)
        
        sent_out_e_p1 = self.W_H(sent_out_h) 
        sent_out_e_p2 = self.W_U(sent_out_u).unsqueeze(1).expand_as(sent_out_e_p1)
        sent_out_e_p3 = self.W_T(sent_out_t).unsqueeze(1).expand_as(sent_out_e_p1)
        sent_out_e = self.v(self.tanh(sent_out_e_p1 + sent_out_e_p2 + sent_out_e_p3)) #  
        sent_out_eexp = sent_out_e.exp().squeeze() * sent_out_mask_var
        sent_out_alpha = sent_out_eexp / sent_out_eexp.sum(1, keepdim=True)
        sent_embeds = (sent_out_h * sent_out_alpha.unsqueeze(2).expand_as(sent_out_h)).sum(1)
        '''

        sent_embeds, sent_out_alpha = self.get_sent_embeds(batch_input, sentence_lengths)
        
        sent_embeds_list = []
        head = 0
        max_doc_length = max(document_lengths)
        for cur_len in document_lengths:
            tail = head + cur_len
            # print(head, tail)
            select_index = to_variable(np.asarray(range(head, tail)), 'Long')
            cur_sent_embeds = F.pad(torch.index_select(sent_embeds, 0, select_index), 
                (0, 0, 0, int(max_doc_length-cur_len)), "constant", 0)
            
            sent_embeds_list.append(cur_sent_embeds)
            head = tail

        doc_sort_index = sorted(range(len(document_lengths)), key=lambda k: document_lengths[k], reverse=True)
        doc_sort_index_map = {old: new for new, old in enumerate(doc_sort_index)}
        reverse_doc_index = [doc_sort_index_map[i] for i in range(len(doc_sort_index))]
        document_lengths_sort = sorted(document_lengths, reverse=True)

        document_lengths_var = to_variable(document_lengths, 'Float')
        reverse_doc_index_var = to_variable(reverse_doc_index, 'Long')
        # print(sent_sort_index, doc_sort_index, document_lengths, reverse_doc_index)

        sent_embeds_sort = torch.cat([sent_embeds_list[i].unsqueeze(0) for i in doc_sort_index], 0)
        # print(sent_embeds_sort.size())

        # compute document representation using LSTM
        doc_packed = pack_padded_sequence(sent_embeds_sort, document_lengths_sort, batch_first=True)
        doc_out_packed, _ = self.doc_LSTM(doc_packed)

        doc_out_unpacked, _ = pad_packed_sequence(doc_out_packed, batch_first=True)
        doc_out_h = torch.index_select(doc_out_unpacked, 0, reverse_doc_index_var)

        # sent_embeds_combi = torch.cat([sent_embeds_list[i].unsqueeze(0) for i in range(len(document_lengths))], 0)
        # doc_out_h = sent_embeds_combi
        # review_users_var = to_variable(review_users, 'Long')
        # doc_out_u = self.user_embeddings(review_users_var)
        # review_targets_var = to_variable(review_targets, 'Long')
        # doc_out_t = self.target_embeddings(review_targets_var)
        
        doc_out_e_p1 = self.W_H(doc_out_h) 
        # doc_out_e_p2 = self.W_U(doc_out_u).unsqueeze(1).expand_as(doc_out_e_p1)
        # doc_out_e_p3 = self.W_T(doc_out_t).unsqueeze(1).expand_as(doc_out_e_p1)
        doc_out_e = self.v(self.tanh(doc_out_e_p1)) # + doc_out_e_p2 + doc_out_e_p3
        doc_out_eexp = doc_out_e.exp().squeeze() * doc_out_mask_var
        doc_out_alpha = doc_out_eexp / doc_out_eexp.sum(1, keepdim=True)
        doc_embeds = (doc_out_h * doc_out_alpha.unsqueeze(2).expand_as(doc_out_h)).sum(1)
        # print(doc_embeds.size())

        return doc_embeds, sent_out_alpha, doc_out_alpha

    def get_scores(self, batch_input, sentence_lengths, document_lengths, label_embedding, alpha=False):
        doc_embeds, sent_out_alpha, doc_out_alpha = self.forward(batch_input, sentence_lengths, document_lengths)
        # review_users_var = to_variable(review_users, 'Long')
        # user_embeds = self.user_embeddings(review_users_var)
        # review_targets_var = to_variable(review_targets, 'Long')
        # target_embeds = self.target_embeddings(review_targets_var)
        final_embeds = doc_embeds # torch.cat((doc_embeds, user_embeds, target_embeds), 1)
        scores = label_embedding(final_embeds)
        
        # scores = torch.sigmoid(self.W_reg(doc_embeds)) * 4
        if (alpha):
            return scores, sent_out_alpha, doc_out_alpha
        else:
            return scores

    def predict(self, batch_input, sentence_lengths, document_lengths, label_embedding):
        scores = self.get_scores(batch_input, sentence_lengths, document_lengths, label_embedding)
        
        # pred_label = scores.view(-1)
        _, pred_label = torch.max(scores, 1)
        pred_label = to_value(pred_label)
        
        return pred_label

        

        


