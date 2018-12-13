# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/3 下午7:50
# @Author  : zhanzecheng
# @File    : utils.py
# @Software: PyCharm
"""
import numpy as np
import torch
from src.dataset import Example, Batch
from torch import nn

loss_function = nn.NLLLoss()

def to_batch_seq(datas, idxes, st, ed, is_train=True):

    examples = []
    for i in range(st, ed):
        feature = datas[idxes[i]]['feature']
        label = datas[idxes[i]]['label']
        decoder_pob = datas[idxes[i]]['decoder_pob']
        type = [x.split('.')[-1][:-2] for x in datas[idxes[i]]['type']]
        one_hot_type = np.zeros((len(feature), 4))
        for id_x, t_v in enumerate(type):
            if t_v == 'WikiSqlSelectColumnAction':
                one_hot_type[id_x][0] = 1
            elif t_v == 'ApplyRuleAction':
                one_hot_type[id_x][1] = 1
            elif t_v == 'GenTokenAction':
                one_hot_type[id_x][2] = 1
            elif t_v == 'ReduceAction':
                one_hot_type[id_x][3] = 1
            else:
                raise NotImplementedError("wrong type for ", t_v)



        assert len(feature) == len(label)
        example = Example(
            feature,
            np.expand_dims(np.array(label), axis=1),
            decoder_pob=decoder_pob,
            one_hot_type=one_hot_type
        )
        examples.append(example)

    if is_train:
        examples.sort(key=lambda e: -len(e.confidence))
        return examples
    else:
        return examples

def epoch_train(model, optimizer, batch_size, datas, args):
    model.train()
    # shuffe
    perm=np.random.permutation(len(datas))
    cum_loss = 0.0
    st = 0

    while st < len(datas):

        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples = to_batch_seq(datas, perm, st, ed)

        batch = Batch(examples, cuda=True)

        optimizer.zero_grad()

        input = torch.cat([batch.decoder_pob_car.unsqueeze(-1), batch.src_sents_var.unsqueeze(-1)], dim=-1)

        score = model.forward(input, batch.src_sents_len, batch).squeeze()

        loss = model.loss(score, batch)

        # TODO: what is the sup_attention?
        loss.backward()
        if args.clip_grad > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        # some records
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        st = ed

    return cum_loss / len(datas)

def epoch_acc(model, batch_size, datas):
    model.eval()
    perm = list(range(len(datas)))
    st = 0
    preds = []
    labels = []
    while st < len(datas):
        ed = st + batch_size if st + batch_size < len(perm) else len(perm)
        examples = to_batch_seq(datas, perm, st, ed, is_train=False)

        batch = Batch(examples, cuda=True)

        input = torch.cat([batch.decoder_pob_car.unsqueeze(-1), batch.src_sents_var.unsqueeze(-1)], dim=-1)

        score = model.forward(input, batch.src_sents_len, batch).squeeze().data.cpu().numpy()

        preds.extend(score.tolist())
        labels.extend(batch.label)

        st = ed

    pred_list = []
    label_list = []

    for b, y_label in enumerate(labels):
        pred = preds[b][:len(y_label)]
        pred = [0 if x < 0.5 else 1 for x in pred]
        y_label = y_label[:, 0]
        for p_val, l_val in zip(pred, y_label):
            pred_list.append(int(p_val))
            label_list.append(int(l_val))

    TP = sum((np.array(pred_list) == 1) & (np.array(label_list) == 1))
    TN = sum((np.array(pred_list) == 0) & (np.array(label_list) == 0))
    FN = sum((np.array(pred_list) == 0) & (np.array(label_list) == 1))
    FP = sum((np.array(pred_list) == 1) & (np.array(label_list) == 0))
    p = TP / float(TP + FP)
    r = TP / float(TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('acc:', acc)
    print('recall is ', r)
    print('precision is', p)
    print('F1：', F1)
    return acc, F1



