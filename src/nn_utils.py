# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/3 下午7:57
# @Author  : zhanzecheng
# @File    : nn_utils.py
# @Software: PyCharm
"""

import numpy as np
import torch

def input_transpose(sequences):
    max_len = max(len(s) for s in sequences)
    batch_size = len(sequences)
    data = np.zeros((batch_size, max_len), dtype=np.float32)
    for e_id in range(batch_size):
        for i in range(max_len):
            if len(sequences[e_id]) > i:
                data[e_id, i] = sequences[e_id][i]
            else:
                data[e_id, i] = 0.0
    return data


def to_input_variable(sequences, cuda=True):

    sents_t = input_transpose(sequences)
    with torch.no_grad():
        sents_var = torch.from_numpy(sents_t)

    if cuda:
        sents_var = sents_var.cuda()

    return sents_var

def length_array_to_mask_tensor(length_array, cuda=False):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.zeros((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask