import collections
import math

import torch

from src.predict import predict_seq2seq
from utils.data_loader import load_data_nlpcc, read_data_nlpcc, tokenize_nlpcc
from utils.utils import try_gpu


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


if __name__ == '__main__':
    contents ,titles= tokenize_nlpcc(read_data_nlpcc()[-4:])
    net = torch.load('../model/net.pt')
    batch_size, num_steps1,num_steps2 = 64, 1000,30
    device = try_gpu()
    train_iter, vocab = load_data_nlpcc(batch_size, num_steps1,num_steps2)
    for content, title in zip(contents, titles):
        summarization, attention_weight_seq = predict_seq2seq(
            net, content, vocab, num_steps1,num_steps2, device)
        print(f'{{\n  "content": "{"".join(content)}",\n  "summarization": "{"".join(summarization)}",\n  "bleu": {bleu(summarization, title, k=2):.3f}\n}}\n')

