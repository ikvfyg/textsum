import torch
from matplotlib import pyplot as plt
from torch import nn

from src.model import MaskedSoftmaxCELoss, Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder
from utils.data_loader import load_data_nlpcc
from utils.utils import Animator, Accumulator, grad_clipping, try_gpu
from utils.utils import Timer


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    animator.show()
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps1 ,num_steps2= 64, 1000,30
    lr, num_epochs, device = 0.005, 300, try_gpu()

    train_iter, vocab = load_data_nlpcc(batch_size, num_steps1,num_steps2)
    encoder = Seq2SeqEncoder(len(vocab), embed_size, num_hiddens, num_layers,
                             dropout)
    decoder = Seq2SeqDecoder(len(vocab), embed_size, num_hiddens, num_layers,
                             dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, vocab, device)
    torch.save(net, '../model/net.pt')
    #net = torch.load('../model/net.pt')
    plt.show()