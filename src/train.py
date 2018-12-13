# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/3 下午9:55
# @Author  : zhanzecheng
# @File    : train.py
# @Software: PyCharm
"""
import json
import torch
import os
import visdom
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from src.utils import epoch_train, epoch_acc
from src.model import Confidence_lstm

class Args():
    def __init__(self):
        pass

args = Args()
args.clip_grad = 5.0
args.optimizer = 'Adam'
args.lr = 0.01

def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            torch.nn.init.xavier_normal(p.data)

def main():
    save = 'all(-1)'
    print(save)
    # viz = visdom.Visdom(env=save)
    # x, y = 0, 0
    # win_acc = viz.line(
    #     X=np.array([x]),
    #     Y=np.array([y]),
    #     opts=dict(title='acc value'))
    #
    # win_F1 = viz.line(
    #     X=np.array([x]),
    #     Y=np.array([y]),
    #     opts=dict(title='F1 value'))
    epoch = 0

    with open('../data/confidence_trainf.json', 'r') as f:
        datas = json.load(f)

    # print(datas[0])
    # quit()

    model = Confidence_lstm()

    glorot_init(model.parameters())

    optimizer_cls = eval('torch.optim.%s' % "Adam")
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    max_f1 = 0
    for _ in range(100):
        epoch += 1
        print('epoch is -----> ', epoch)
        loss = epoch_train(model, optimizer, 64, datas[:50000], args)

        print('loss is :', loss)
        acc, F1 = epoch_acc(model, 64, datas[50000:])

        if F1 > max_f1:
            max_f1 = F1
            model.save('/home/v-zezhan/Confidence/save_model/' + save + 'best.bin')

        # viz.line(
        #             X=np.array([epoch]),
        #             Y=np.array([acc]),
        #             win=win_acc,
        #             update='append'
        #         )
        # viz.line(
        #     X=np.array([epoch]),
        #     Y=np.array([F1]),
        #     win=win_F1,
        #     update='append'
        # )

if __name__ == '__main__':

    main()