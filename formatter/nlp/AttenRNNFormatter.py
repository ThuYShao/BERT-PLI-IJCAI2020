# -*- coding: utf-8 -*-
__author__ = 'yshao'

import json
import torch
import os

from formatter.Basic import BasicFormatter


class AttenRNNFormatter(BasicFormatter):
    def __init__(self, config, mode, *agrs, **params):
        super().__init__(config, mode, *agrs, **params)
        self.max_para_q = config.getint('model', 'max_para_q')
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        inputs = []
        guids = []
        # if mode != 'test':
        labels = []

        for temp in data:
            guid = temp['guid']
            emb_mtx = temp['res']
            assert (len(emb_mtx) == self.max_para_q)
            inputs.append(emb_mtx)
            guids.append(guid)

            # if mode != 'test':
            labels.append(temp['label'])

        inputs = torch.tensor(inputs)
        # if mode != 'test':
        labels = torch.LongTensor(labels)
        return {'guid': guids, 'input': inputs, 'label': labels}

        # if mode != 'test':
        #     return {'guid': guids, 'input': inputs, 'label': labels}
        # else:
        #     return {'guid': guids, 'input': inputs}

