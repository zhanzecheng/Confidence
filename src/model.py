import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import os

class Confidence_lstm(nn.Module):
    def __init__(self):
        super(Confidence_lstm, self).__init__()
        self.encoder_lstm = nn.LSTM(50, 50, bidirectional=False, batch_first=True)
        self.project = nn.Sequential(nn.Linear(1 + 4 + 1,  50), nn.ReLU())
        self.sigmoid = F.sigmoid
        self.out = nn.Linear(50, 1)
        self.cuda()

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def encode(self, src_sents_var, src_sents_len):
        src_sents_len = np.array(src_sents_len)
        sort_perm = np.array(sorted(range(len(src_sents_len)),
                                    key=lambda k: src_sents_len[k], reverse=True), dtype=np.int32)
        sort_inp_len = src_sents_len[sort_perm]
        sort_perm_inv = np.argsort(sort_perm)
        if src_sents_var.is_cuda:
            sort_perm = torch.LongTensor(sort_perm).cuda()
            sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

        packed_src_token_embed = pack_padded_sequence(src_sents_var[sort_perm],
                                                      sort_inp_len, batch_first=True)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)

        src_encodings = pad_packed_sequence(src_encodings, batch_first=True)[0][sort_perm_inv]

        (last_state, last_cell) = (last_state[:, sort_perm_inv], last_cell[:, sort_perm_inv])
        # last_state = torch.cat([last_state[0], last_state[1]], -1)
        # last_cell = torch.cat([last_cell[0], last_cell[1]], -1)

        return src_encodings

    def forward(self, examples, examples_len, batch):
        input_type = self.input_type(batch.one_hot_type)

        examples = torch.cat([examples, input_type], dim=-1)
        # examples = examples.unsqueeze(-1)
        examples = self.project(examples)
        src_encodings = self.encode(examples, examples_len)
        project = self.out(src_encodings).squeeze(-1)

        project.data.masked_fill_(batch.src_token_mask, -float('inf'))

        logit = self.sigmoid(project)

        return logit.squeeze()

    def loss(self, col_score_logit, batch):

        label_val = self.input_type(batch.label).squeeze(-1)
        # print(label_val)
        # print('=====')
        # print(col_score_logit)
        # quit()

        loss = -sum(torch.mean(( label_val * \
                                       torch.log(col_score_logit + 1e-10)) + \
                                      (1 - label_val) * torch.log(1 - col_score_logit + 1e-10), 1))
        return loss

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
