# -*- coding: utf-8 -*-
__author__ = 'yshao'


import logging
import torch
import json

from torch.autograd import Variable
from timeit import default_timer as timer

from tools.eval_tool import gen_time_str, output_value

logger = logging.getLogger(__name__)


def load_state_keywise(model, pretrained_dict):
    logger.info("load state keywise start ...")
    model_dict = model.state_dict()
    # print(model_dict.keys())
    # input("continue?")
    tmp_cnt = 0
    for k, v in pretrained_dict.items():
        # kk = k.replace("module.", "")
        # print('k=', k)
        # input("continue?")
        # print('kk=', kk)
        # input("continue?")
        if k in model_dict and v.size() == model_dict[k].size():
            model_dict[k] = v
            tmp_cnt += 1
        else:
            continue
    logger.info('tot #para=%d, load from pretrained #paras=%d' % (len(model_dict), tmp_cnt))
    model.load_state_dict(model_dict)
    return model


def pool_out(parameters, config, gpu_list, _outname):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = "Pool_Out"

    output_time = config.getint("output", "output_time")
    save_step = config.getint("output", "save_step")
    step = -1
    result = []

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "poolout")
        result = result + results["output"]
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(0, "poolout", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

        if save_step > 0 and step % save_step == 0:
            out_file = open(_outname, 'w', encoding='utf-8')
            for item in result:
                tmp_dict = {
                    'id_': item[0],
                    'res': item[1]
                }
                out_line = json.dumps(tmp_dict, ensure_ascii=False) + '\n'
                out_file.write(out_line)
            out_file.close()

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = "Pool_Out"
    output_value(0, "poolout", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    return result
