import torch
from torch import nn


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
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedding = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedding, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


if __name__ == '__main__':
    pass