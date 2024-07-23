import math

import torch

from utils.data_loader import truncate_pad

import torch


def predict_seq2seq(net, src_tokens, vocab, num_steps1, num_steps2, device, beam_size=2, alpha=0.75):
    """使用束搜索进行序列到序列模型的预测"""
    net.eval()
    src_tokens = vocab[src_tokens] + [vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps1, vocab['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)

    # 初始化束搜索的候选列表
    candidates = [{'dec_state': net.decoder.init_state(enc_outputs, enc_valid_len),
                   'seq': [vocab['<bos>']],
                   'score': 0.0}]
    all_candidates = []
    for _ in range(num_steps2):
        new_candidates = []
        for candidate in candidates:
            dec_X = torch.unsqueeze(torch.tensor(candidate['seq'], dtype=torch.long, device=device), dim=0)
            dec_state = candidate['dec_state']
            Y, dec_state = net.decoder(dec_X, dec_state)
            # 获取概率分布并选择最高beam_size个概率的词元
            topk_probs, topk_ids = torch.topk(Y.squeeze(0)[-1], beam_size)
            for prob, idx in zip(topk_probs, topk_ids):
                if (idx.item() == vocab['<eos>']):
                    new_seq = candidate['seq']
                    new_score = candidate['score'] + torch.log(prob / 100).item()
                    all_candidates.append({'dec_state': dec_state, 'seq': new_seq, 'score': new_score})
                    continue
                new_seq = candidate['seq'] + [idx.item()]
                new_score = candidate['score'] + torch.log(prob / 100).item()
                # 添加新的候选
                new_candidates.append({'dec_state': dec_state, 'seq': new_seq, 'score': new_score})

        # 从新的候选中选择得分最高的beam_size个候选
        new_candidates.sort(key=lambda x: x['score']/ (len(x['seq']) ** alpha), reverse=True)
        candidates = new_candidates[:beam_size]

    all_candidates += candidates
    # 选择得分最高的候选序列
    best_candidate = max(all_candidates, key=lambda x: x['score'] / (len(x['seq']) ** alpha))
    output_seq = best_candidate['seq']
    # 去掉序列开始的<bos>和序列结束的<eos>
    output_seq = output_seq[1:] if output_seq[0] == vocab['<bos>'] else output_seq
    # output_seq = output_seq[:-1] if output_seq[-1] == vocab['<eos>'] else output_seq

    return ' '.join(vocab.to_tokens(output_seq))

# 注意：truncate_pad函数和net.encoder/net.decoder的定义没有给出，需要根据实际情况定义。
