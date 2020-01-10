# -*- coding=utf-8 -*-


from collections import Counter
import numpy as np


class DataIter(object):
    
    def __init__(self, file_name, batch_size = None, split_char = ' '):
        self.file_name = file_name
        self.batch_size = batch_size
        self.split_char = split_char
        self.length = None

    def __iter__(self):
        with open(self.file_name, 'rt') as f:
            x, y = [], []
            for line in f:
                line = line.strip()
                items = line.split(self.split_char)
                x += [[float(feat) for feat in items[1:]]]
                y += [[int(items[0])]]
                if self.batch_size != None and len(x) == self.batch_size:
                    yield x, y
                    x, y = [], []

            if len(x) != 0:
                yield x, y
                x, y = [], []


    def __len__(self):
        if self.length == None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def cal_eer(label, score):
    # list 转置
    label = list(map(list, zip(*label)))[0]
    score = list(map(list, zip(*score)))[0]

    cnts = Counter(label)
    pos_cnt, neg_cnt = cnts[1], cnts[0]

    data = list(zip(label, score))
    data_sorted = sorted(data, key=lambda x: x[1])

    cur_score, cur_neg_cnt, cur_pos_cnt = None, 0, 0
    eers = []
    for i in range(len(data_sorted)):
        if data_sorted[i][1] != cur_score:
            neg_er = 1.0 * (neg_cnt - cur_neg_cnt) / neg_cnt
            pos_er = 1.0 * cur_pos_cnt / pos_cnt
            if abs(neg_er - pos_er) < 0.001:
                eers += [neg_er]
            cur_score = data_sorted[i][1]
        if data_sorted[i][0] == 0:
            cur_neg_cnt += 1
        else:
            cur_pos_cnt += 1

    return np.mean(eers) if len(eers) > 0 else 2.0


def cal_auc(label, score):
    # list 转置
    label = list(map(list, zip(*label)))[0]
    score = list(map(list, zip(*score)))[0]

    cnts = Counter(label)
    pos_cnt, neg_cnt = cnts[1], cnts[0]

    data = list(zip(label, score))
    data_sorted = sorted(data, key=lambda x: x[1])

    pair_cnt = 0
    for i in range(len(data_sorted)):
        if data_sorted[i][0] == 1:
            pair_cnt += i

    pair_cnt -= pos_cnt * (pos_cnt - 1)
    return 1.0 * pair_cnt / (pos_cnt * neg_cnt)



