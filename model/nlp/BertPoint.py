# -*- coding: utf-8 -*-
__author__ = 'yshao'

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from tools.accuracy_init import init_accuracy_function


class BertPoint(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPoint, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.output_mode = config.get('model', 'output_mode')

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(768, self.output_dim)
        if self.output_mode == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        _, y = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)

        if mode == 'test' and config.getboolean('output', 'pool_out'):
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}

        y = self.fc(y)
        y = y.view(y.size()[0], -1)

        if "label" in data.keys():
            label = data["label"]
            loss = self.criterion(y, label.view(-1))
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        else:
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}
    
