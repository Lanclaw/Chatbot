import torch
from torch import nn
from DataProcess import *
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding_size = hidden_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, hidden_size)

    def forward(self, input, hidden):
        input = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedding = self.embedding(input).view(1, 1, -1)
        output = self.sigmoid(embedding)     # relu
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.softmax = nn.Softmax(dim=1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embeddded = self.embedding(input)               # [b_s, seq_len] -> [b_s, seq_len, embed_size]
        input = torch.cat((embeddded[0], hidden[0]), dim=1)  # [seq_len, e_s] + [seq_len, h_s] ->[seq_len, e_s + h_s]
        attn_weights = self.attn(input)         # [seq_len, max_length]
        attn_weights = self.softmax(attn_weights)
        attn_hidden = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))    # 1 * 1 * h_s
        output = self.attn_combine(torch.cat((embeddded[0], attn_hidden[0]), dim=1)).unsqueeze(0)        # [1, 1, 2*h_s]

        output = self.sigmoid(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


if __name__ == '__main__':
    pass