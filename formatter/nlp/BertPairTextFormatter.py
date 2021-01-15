# -*- coding: utf-8 -*-
__author__ = 'yshao'

import json
import torch
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter
from .bert_feature_tool import example_item_to_feature


class BertPairTextFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.output_mode = config.get('model', 'output_mode')

    def process(self, data, config, mode, *args, **params):
        guids = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        if mode != 'test':
            labels = []

        for temp in data:
            res_dict = example_item_to_feature(temp, self.max_len, self.tokenizer, self.output_mode,
                                               cls_token_at_end=False, pad_on_left=False,
                                               cls_token_segment_id=0, pad_token_segment_id=0)
            input_ids.append(res_dict['input_ids'])
            attention_mask.append(res_dict['input_mask'])
            token_type_ids.append(res_dict['segment_ids'])
            guids.append(temp['guid'])

            if mode != 'test':
                labels.append(res_dict['label_id'])

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        if mode != 'test':
            labels = torch.LongTensor(labels)

        if mode != 'test':
            return {'guid': guids, 'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                    'label': labels}
        else:
            return {'guid': guids, 'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}





