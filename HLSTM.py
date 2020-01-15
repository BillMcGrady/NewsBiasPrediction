import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 

class HLSTM(torch.nn.Module):
    def __init__(self, hidden_size, word_to_idx, word_embeddings):
        """
        In the constructor we construct instances that we will use
        in the forward pass.
        """
        super(HLSTM, self).__init__()
        
        word_count, embedding_size = word_embeddings.shape
        self.word_embeddings = nn.Embedding(word_count, embedding_size, padding_idx=-1).cuda()
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))   
        self.word_embeddings.weight.requires_grad = False     
        print(self.word_embeddings)

        self.sent_LSTM = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True).cuda()
        self.doc_LSTM = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True).cuda()
        self.word_to_idx = word_to_idx

    def get_sent_embeds(self, batch_input, sentence_lengths):
        sent_sort_index = sorted(range(len(sentence_lengths)), key=lambda k: sentence_lengths[k], reverse=True)
        sent_sort_index_map = {old: new for new, old in enumerate(sent_sort_index)}
        reverse_sent_index = [sent_sort_index_map[i] for i in range(len(sent_sort_index))]

        batch_input_var = torch.LongTensor(batch_input).cuda()
        sent_sort_index_var = torch.LongTensor(sent_sort_index).cuda()
        reverse_sent_index_var = torch.LongTensor(reverse_sent_index).cuda()

        sentence_lengths_var = torch.LongTensor(sentence_lengths).cuda()
        word_embeds = self.word_embeddings(batch_input_var)
        word_embeds_sort = torch.index_select(word_embeds, 0, sent_sort_index_var)
        sentence_lengths_sort = sorted(sentence_lengths, reverse=True)
        
        # print(word_embeds_sort.size(), sentence_lengths_sort)

        # compute sentence representation using LSTM
        sent_packed = pack_padded_sequence(word_embeds_sort, sentence_lengths_sort, batch_first=True)
        sent_out_packed, _ = self.sent_LSTM(sent_packed)

        sent_out_unpacked, _ = pad_packed_sequence(sent_out_packed, batch_first=True)
        sent_embeds = torch.index_select(sent_out_unpacked, 0, reverse_sent_index_var).sum(1) / sentence_lengths_var.view((-1, 1))
        # print(sent_embeds.size())

        return sent_embeds

    def forward(self, batch_input, sentence_lengths, document_lengths):
        # _, sent_sort_index = torch.sort(sentence_lengths, descending=True)
        sent_sort_index = sorted(range(len(sentence_lengths)), key=lambda k: sentence_lengths[k], reverse=True)
        sent_sort_index_map = {old: new for new, old in enumerate(sent_sort_index)}
        reverse_sent_index = [sent_sort_index_map[i] for i in range(len(sent_sort_index))]

        batch_input_var = torch.LongTensor(batch_input).cuda()
        sent_sort_index_var = torch.LongTensor(sent_sort_index).cuda()
        reverse_sent_index_var = torch.LongTensor(reverse_sent_index).cuda()

        sentence_lengths_var = torch.FloatTensor(sentence_lengths).cuda()
        word_embeds = self.word_embeddings(batch_input_var)
        word_embeds_sort = torch.index_select(word_embeds, 0, sent_sort_index_var)
        sentence_lengths_sort = sorted(sentence_lengths, reverse=True)
        
        # print(word_embeds_sort.size(), sentence_lengths_sort)

        # compute sentence representation using LSTM
        sent_packed = pack_padded_sequence(word_embeds_sort, sentence_lengths_sort, batch_first=True)
        sent_out_packed, _ = self.sent_LSTM(sent_packed)

        sent_out_unpacked, _ = pad_packed_sequence(sent_out_packed, batch_first=True)
        sent_embeds = torch.index_select(sent_out_unpacked, 0, reverse_sent_index_var).sum(1) / sentence_lengths_var.view((-1, 1))
        # print(sent_embeds.size())

        sent_embeds_list = []
        head = 0
        max_doc_length = max(document_lengths)
        for cur_len in document_lengths:
            tail = head + cur_len
            # print(head, tail)
            select_index = torch.LongTensor(np.asarray(range(head, tail))).cuda()
            cur_sent_embeds = F.pad(torch.index_select(sent_embeds, 0, select_index), 
                (0, 0, 0, int(max_doc_length-cur_len)), "constant", 0)
            
            sent_embeds_list.append(cur_sent_embeds)
            head = tail

        # pad_sequence(sent_embeds_list).size()

        doc_sort_index = sorted(range(len(document_lengths)), key=lambda k: document_lengths[k], reverse=True)
        doc_sort_index_map = {old: new for new, old in enumerate(doc_sort_index)}
        reverse_doc_index = [doc_sort_index_map[i] for i in range(len(doc_sort_index))]
        document_lengths_sort = sorted(document_lengths, reverse=True)

        document_lengths_var = torch.FloatTensor(document_lengths).cuda()
        reverse_doc_index_var = torch.LongTensor(reverse_doc_index).cuda()
        # print(sent_sort_index, doc_sort_index, document_lengths, reverse_doc_index)

        sent_embeds_sort = torch.cat([sent_embeds_list[i].unsqueeze(0) for i in doc_sort_index], 0)
        # print(sent_embeds_sort.size())

        # compute document representation using LSTM
        doc_packed = pack_padded_sequence(sent_embeds_sort, document_lengths_sort, batch_first=True)
        doc_out_packed, _ = self.doc_LSTM(doc_packed)

        doc_out_unpacked, _ = pad_packed_sequence(doc_out_packed, batch_first=True)
        doc_embeds = torch.index_select(doc_out_unpacked, 0, reverse_doc_index_var).sum(1) / document_lengths_var.view((-1, 1))
        # print(doc_embeds.size())

        return doc_embeds

    def get_scores(self, batch_input, sentence_lengths, document_lengths, label_embedding):
        doc_embeds = self.forward(batch_input, sentence_lengths, document_lengths)
        scores = label_embedding(doc_embeds)
        # scores = torch.sigmoid(self.W_reg(doc_embeds)) * 4
        
        return scores

    def predict(self, batch_input, sentence_lengths, document_lengths, label_embedding):
        scores = self.get_scores(batch_input, sentence_lengths, document_lengths, label_embedding)
        # print(scores)

        # pred_label = scores.view(-1)
        _, pred_label = torch.max(scores, 1)
        pred_label = to_value(pred_label)
        
        return pred_label

        

        


