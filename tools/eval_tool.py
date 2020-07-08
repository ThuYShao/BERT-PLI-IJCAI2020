import logging
import os
import torch
import numpy as np
from collections import defaultdict
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def eval_micro_query(_result_list):
    label_dict = defaultdict(lambda: [])
    pred_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for item in _result_list:
        guid = item[0]
        label = int(item[1])
        pred = np.argmax(item[2])
        qid, cid = guid.split('_')
        if label > 0:
            label_dict[qid].append(cid)
        pred_dict[qid][cid] = pred
    assert (len(pred_dict) == len(label_dict))

    correct = 0
    label = 0
    predict = 0
    for qid in label_dict:
        label += len(label_dict[qid])
    for qid in pred_dict:
        for cid in pred_dict[qid]:
            if pred_dict[qid][cid] == 1:
                predict += 1
                if cid in label_dict[qid]:
                    correct += 1
    if correct == 0:
        micro_prec_query = 0
        micro_recall_query = 0
    else:
        micro_prec_query = float(correct) / predict
        micro_recall_query = float(correct) / label
    if micro_prec_query > 0 or micro_recall_query > 0:
        micro_f1_query = (2 * micro_prec_query * micro_recall_query) / (micro_prec_query + micro_recall_query)
    else:
        micro_f1_query = 0
    return micro_prec_query, micro_recall_query, micro_f1_query


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid"):
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"
    result = []

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "valid")

        loss, acc_result, output = results["loss"], results["acc_result"], results["output"]
        total_loss += float(loss)
        result = result + output
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)
    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                      epoch)

    # eval results based on query micro F1
    micro_prec_query, micro_recall_query, micro_f1_query = eval_micro_query(result)
    loss_tmp = total_loss / (step + 1)
    print('valid set: micro_prec_query=%.4f, micro_recall_query=%.4f, micro_f1_query=%.4f' %
          (micro_prec_query, micro_recall_query, micro_f1_query))

    model.train()
    return {'precision': micro_prec_query, 'recall': micro_recall_query, 'f1': micro_f1_query, 'loss': loss_tmp}
