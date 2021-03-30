import DataProcess
import model
import torch
import random
import eval
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
SOS_token = 0
EOS_token = 1


def train_sen(src_sen, tgt_sen, encoder, decoder, enc_optim, dec_optim, criterion, max_length=10):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    src_len = src_sen.size(0)
    tgt_len = tgt_sen.size(0)
    enc_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    enc_hidden = torch.zeros(1, 1, encoder.hidden_size, device=device)

    for i in range(src_len):
        enc_output, enc_hidden = encoder(src_sen[i], enc_hidden)
        enc_outputs[i] = enc_output[0, 0]

    loss = 0
    dec_input = torch.tensor([[SOS_token]], device=device)
    dec_hidden = enc_hidden

    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() > teacher_forcing_ratio else False
    if use_teacher_forcing:
        for i in range(tgt_len):
            dec_output, dec_hidden, attn = decoder(dec_input, dec_hidden, enc_outputs)
            loss += criterion(dec_output, tgt_sen[i])
            dec_input = tgt_sen[i].view(1, -1)

    else:
        for i in range(tgt_len):
            dec_output, dec_hidden, attn = decoder(dec_input, dec_hidden, enc_outputs)
            loss += criterion(dec_output, tgt_sen[i])
            _, topi = dec_output.topk(1)
            dec_input = topi.squeeze().detach().view(1, -1)

            if dec_input.item() == EOS_token:
                break


    loss.backward()
    enc_optim.step()
    dec_optim.step()

    return loss.item() / tgt_len


def train(pairs, encoder, decoder, input_lang, output_lang, n_iters=10000, lr=0.01):

    losses = []
    loss_total = 0
    train_pairs = [DataProcess.pair2tensor(input_lang, output_lang, random.choice(pairs)) for _ in range(n_iters)]
    enc_optim = torch.optim.SGD(encoder.parameters(), lr=lr)
    dec_optim = torch.optim.SGD(decoder.parameters(), lr=lr)
    critirion = torch.nn.NLLLoss()

    for iter in range(n_iters):
        train_pair = train_pairs[iter]
        input_tensor = train_pair[0]
        output_tensor = train_pair[1]
        loss = train_sen(input_tensor, output_tensor, encoder, decoder, enc_optim, dec_optim, critirion)

        loss_total += loss
        if iter % 100 == 0:
            print('iter:', iter, 'loss:', loss_total / (iter + 1))
            losses.append(loss_total / (iter + 1))
            loss_total = 0




import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == '__main__':
    input_lang, output_lang, pairs = DataProcess.readLangs('human', 'machine')
    if os.path.exists('./data/encoder.pkl'):
        encoder = torch.load('./data/encoder.pkl')
        decoder = torch.load('./data/decoder.pkl')
    else:
        encoder = model.EncoderRNN(input_lang.n_words, 256)
        decoder = model.AttnDecoderRNN(256, output_lang.n_words)
        train(pairs, encoder, decoder, input_lang, output_lang)
        torch.save(encoder, './data/encoder.pkl')
        torch.save(decoder, './data/decoder.pkl')
    eval.evaluateAndShowAttention(encoder, decoder, input_lang, output_lang, "我 爱 你")


