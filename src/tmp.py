# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/4 下午8:04
# @Author  : zhanzecheng
# @File    : tmp.py
# @Software: PyCharm
"""
import torch
from torch import autograd
from torchvision import datasets,transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class PosTaggerData(data.Dataset):
    def __init__(self):
        self.training_data = [
            ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
            ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
        ]
        self.word_to_ix = {}
        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        print(self.word_to_ix)
        self.tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
        print(self.tag_to_ix)

        self.gen_all_data()

    def __len__(self):
        return len(self.items)

    def __getitem__(self,idx):
        return self.items[idx]

    def gen_all_data(self):
        self.items = []
        for sent,tags in self.training_data:
            data = self.prepare_sequence(sent,self.word_to_ix)
            targets = self.prepare_sequence(tags,self.tag_to_ix)

            print('data:',data)
            print('targets:',targets)

            self.items.append((data,targets))

    def prepare_sequence(self,seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        tensor = torch.LongTensor(idxs)
        return tensor


BATCH_SIZE = 1


class LSTMPosTagger(nn.Module):
    def __init__(self):
        super(LSTMPosTagger, self).__init__()

        self.vocab_size = 9
        self.embedding_size = 6
        self.lstm_hidden_size = 6
        self.target_size = 2

        self.lstm = nn.LSTM(1, self.lstm_hidden_size)  # seq_len,N,hidden_size
        self.fully = nn.Linear(self.lstm_hidden_size, self.target_size)

        self.hidden = self.init_hidden()

    def forward(self, input):
        input = input.unsqueeze(0)
        input = input.float().unsqueeze(-1)
        # print(input.shape)
        # quit()
        # embedded = self.word_embedding(input)
        lstm_out, self.hidden = self.lstm(input)  # seq_len,N,hidden_size
        tag_space = self.fully(
            lstm_out.view(-1, self.lstm_hidden_size))  # [seq_len*N,hidden_size] -> [seq_len*N,target_size]
        tag_score = F.log_softmax(tag_space)
        return tag_score

model = LSTMPosTagger()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.NLLLoss()

tagger = PosTaggerData()

for epoch in range(300):
    print('===========epoch===========', epoch)
    # for data,target in tagger.items:
    for data, target in tagger.items:
        model.zero_grad()
        data = torch.tensor([1.6099e-02, 1.0200e-06, 1.6354e-02, 3.3452e-04, 5.7853e-02, 9.0854e+00,
         4.4679e-04, 1.0094e-01, 1.2295e+01, 1.4483e-03, 8.8361e-02, 1.4398e+01,
         3.6589e-02, 6.5809e-06, 1.2800e-01, 1.4900e+01])
        target = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        data = Variable(data)
        target = Variable(target)
        tag_scores = model(data)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        # print(tag_scores.shape)
        # print(target.shape)
        loss = loss_function(tag_scores, target)
        print(loss)
        # quit()
        loss.backward()
        optimizer.step()

        # print(loss.data[0])
