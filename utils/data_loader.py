import collections
import json
import os


import jieba
import torch
from matplotlib import pyplot as plt
from torch.utils import data

from utils.utils import use_svg_display


def read_data_nlpcc(num_examples=10000):
    '''
    仅取前10000(共5w)
    :return:
    '''
    data_dir = '../data/'
    with open(os.path.join(data_dir, 'nlpcc_data.json'), 'r') as f:
        return json.loads(f.read())[:num_examples]


def tokenize_nlpcc(dict_list,num_examples=None):
    if num_examples is not None and num_examples < len(dict_list):
        dict_list = dict_list[:num_examples]
    tokenized_list = [
        {'title': ' '.join(jieba.cut(item['title'], cut_all=False)),
         'content': ' '.join(jieba.cut(item['content'], cut_all=False))} for
        item in dict_list]
    titles = [item['title'].split(' ') for item in tokenized_list]
    contents = [item['content'].split(' ') for item in tokenized_list]
    return contents,titles
    # return tokenized_list


def set_figsize(figsize=(7, 5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    set_figsize()
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)
    plt.show()


def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`"""
    line=line
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array_nlpcc(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches.

    Defined in :numref:`subsec_mt_data_loading`"""

    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = reduce_sum(
        astype(array != vocab['<pad>'], int32), 1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nlpcc(batch_size, num_steps1,num_steps2, num_examples=1000,min_freq=0):
    """Return the iterator and the vocabularies of the translation dataset.

    Defined in :numref:`subsec_mt_data_loading`"""
    text = read_data_nlpcc()
    source, target = tokenize_nlpcc(text,num_examples)
    vocab = Vocab(source+target,reserved_tokens=['<pad>', '<bos>', '<eos>'],min_freq=min_freq)
    src_array, src_valid_len = build_array_nlpcc(source, vocab, num_steps1)
    tgt_array, tgt_valid_len = build_array_nlpcc(target, vocab, num_steps2)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, vocab


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
int32 = torch.int32

if __name__ == '__main__':
    train_iter, vocab = load_data_nlpcc(batch_size=2, num_steps1=1000,num_steps2=30)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break
    # print(len(read_data_nlpcc()))
    # contents, titles = tokenize_nlpcc(read_data_nlpcc())
    # count=[0,0]
    # for item1, item2 in zip(titles, contents):
    #     count[0]+=int(len(item1.split())<=30)
    #     count[1]+=int(len(item2.split())<=1000)
    # src_vocab = Vocab(titles + contents, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # print(len(src_vocab))
    # print(count)
    # show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
    #                         'count', contents, titles)
