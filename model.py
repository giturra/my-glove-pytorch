import torch
import torch.nn as nn
import torch.nn.init as init


class Glove(nn.Module):

    def __init__(self, vocab_size, emb_size, x_max=100, alpha=0.75):
        super(Glove, self).__init__()

        self.emb_u = nn.Embedding(vocab_size, emb_size)
        self.emb_v = nn.Embedding(vocab_size, emb_size)

        self.bias_u = nn.Embedding(vocab_size, 1)
        self.bias_v = nn.Embedding(vocab_size, 1)

        for param in self.parameters():
            init.xavier_uniform_(param, gain=.1)

        self.alpha = alpha
        self.x_max = x_max

    def forward(self, i, j, w):
        l_vecs = self.emb_u(i)
        r_vecs = self.emb_v(j)

        l_bias = self.bias_u(i)
        r_bias = self.bias_v(j)

        log_covals = torch.log(w)

        weight = torch.pow(w / self.x_max, self.alpha)
        weight[weight > 1] = 1

        sim = (l_vecs * r_vecs).sum(1).view(-1)
        x = (sim + l_bias + r_bias - log_covals) ** 2
        loss = torch.mul(x, weight)
        return loss.mean()

