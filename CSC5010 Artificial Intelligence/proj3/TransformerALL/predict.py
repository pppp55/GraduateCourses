# coding:utf-8

import torch
import numpy as np
MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'
tokenizer = lambda x: [y for y in x]


def load_dataset(text, vocab, pad_size=32):
    contents = []
    for line in text:
        lin = line.strip()
        if not lin:
            continue
        words_line = []
        token = tokenizer(line)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # 将预测文本转化为数字
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        contents.append((words_line, int(0), seq_len))
    return contents


def match_label(pred, config):
    label_list = config.class_list
    return label_list[pred]


def final_predict(config, model, data_iter):
    model.eval()
    predict_all = np.array([])
    with torch.no_grad():
        for texts, _ in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred_label = [match_label(i, config) for i in pred]
            predict_all = np.append(predict_all, pred_label)
    return predict_all
